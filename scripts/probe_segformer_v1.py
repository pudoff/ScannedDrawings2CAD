"""Bounded synthetic CUDA training/inference qualification, no customer data."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(config_path, sha, output):
    from cad_recovery.dataset_layout_v1 import checked, encoded, read
    from cad_recovery.family_catalog_v2 import require
    config = read(checked(ROOT, {'path': config_path, 'sha256': sha}))
    for row in config['code']: checked(ROOT, row)
    sys.path.insert(0, str(ROOT / config['dependency_directory']))
    import torch
    import transformers
    from cad_recovery.segformer_model_v1 import build_model, partial_loss, restored_logits
    from cad_recovery.gpu_training_probe_v1 import state_digest
    require(torch.__version__ == '2.14.1+cu130' and transformers.__version__ == '4.57.1', 'Changed probe backend')
    require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.fp32_precision = 'ieee'; torch.backends.cudnn.conv.fp32_precision = 'ieee'
    destination = ROOT / output; require(not destination.exists(), 'Choose a fresh probe directory')
    destination.mkdir(parents=True)
    results = []; started = time.perf_counter()
    for repeat in (0, 1):
        torch.manual_seed(20261110); torch.cuda.manual_seed_all(20261110)
        model = build_model(ROOT, config['assets']).cuda().train()
        encoder_initial = state_digest(model.segformer.state_dict())
        initial = state_digest(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.01)
        # Fixed synthetic native grids; targets never come from a customer image.
        generator = torch.Generator().manual_seed(42)
        x = torch.randn((4, 3, 256, 256), generator=generator).cuda()
        target = torch.full((4, 256, 256), 255, dtype=torch.long)
        for c in range(10): target[:, c * 16, :] = c
        target = target.cuda(); weights = torch.ones(10, device='cuda')
        torch.cuda.reset_peak_memory_stats(); losses = []; checkpoint = None
        def update():
            optimizer.zero_grad(set_to_none=True)
            loss = partial_loss(model(x).logits, target, weights)
            require(bool(torch.isfinite(loss)), 'Nonfinite synthetic loss')
            loss.backward()
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Missing/nonfinite synthetic gradient')
            optimizer.step()
            return float(loss.detach())
        for step in range(2):
            losses.append(update())
            if step == 0:
                checkpoint = {'model': deepcopy(model.state_dict()), 'optimizer': deepcopy(optimizer.state_dict()),
                              'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        final = state_digest(model.state_dict()); require(final != initial, 'No parameter update')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['cpu_rng']); torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        update(); resume_equal = state_digest(model.state_dict()) == final
        require(resume_equal, 'Synthetic checkpoint resume differs')
        model.eval(); inference = []
        with torch.inference_mode():
            for side in (256, 512):
                z = torch.zeros((1, 3, side, side), device='cuda')
                logits = restored_logits(model(z).logits, (side, side))
                require(logits.shape == (1, 10, side, side) and bool(torch.isfinite(logits).all()), 'Invalid native logits')
                inference.append({'side': side, 'shape': list(logits.shape), 'finite': True})
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.max_memory_reserved() / 1024**2
        require(allocated <= 4096 and reserved <= 6144, 'CUDA memory ceiling exceeded')
        results.append({'repeat': repeat, 'initial': initial, 'encoder_initial': encoder_initial, 'final': final,
            'losses': losses, 'checkpoint_resume_equal': resume_equal, 'inference': inference,
            'allocated_peak_MiB': allocated, 'reserved_peak_MiB': reserved,
            'parameters': sum(p.numel() for p in model.parameters())})
        print('synthetic repeat', repeat, losses, allocated, reserved, flush=True)
        del model, optimizer, checkpoint, x, target, logits, z; torch.cuda.empty_cache()
    require(results[0]['final'] == results[1]['final'] and results[0]['losses'] == results[1]['losses'], 'Synthetic repeat differs')
    report = {'schema_version': 'segformer-qualification/1.0', 'qualified': True,
        'model': config['assets'], 'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
        'device_total_MiB': torch.cuda.get_device_properties(0).total_memory / 1024**2,
        'results': results, 'seconds': time.perf_counter() - started, 'customer_samples': 0,
        'dataset_training_started': False, 'synthetic_optimizer_updates': 6, 'quality_evaluation_executed': False,
        'production_admitted': False, 'VLM_capacity_qualified': False}
    (destination / 'report.json').write_bytes(encoded(report))


if __name__ == '__main__':
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(ROOT / 'artifacts/.cache/huggingface')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    p = argparse.ArgumentParser(__doc__)
    for name in ('config', 'sha256', 'output'): p.add_argument('--' + name, required=True)
    args = p.parse_args()
    from cad_recovery.gpu_inference_v1 import deny_python_network
    with deny_python_network(): main(args.config, args.sha256, args.output)
