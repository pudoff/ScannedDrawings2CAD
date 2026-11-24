"""One offline model process for a signed list of permitted crop requests."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(ROOT / 'artifacts/.cache/huggingface')


def main(args):
    from cad_recovery.dataset_layout_v1 import checked, encoded, read, inside
    from cad_recovery.vlm_adapter_v1 import POLICY, require, authorize_requests
    config = read(checked(ROOT, {'path': args.config, 'sha256': args.sha256}))
    require(config['policy'] == POLICY, 'Changed VLM inference policy')
    for record in config['code']: checked(ROOT, record)
    request = read(checked(ROOT, {'path': args.request, 'sha256': args.request_sha256}))
    cases = {r['id']: r for r in read(checked(ROOT, config['catalog']))}
    admission = read(checked(ROOT, request['admission'])) if 'admission' in request else None
    authorize_requests(cases, request, {'path': args.config, 'sha256': args.sha256}, admission)
    sys.path.insert(0, str(ROOT / config['dependency_directory']))
    import torch
    import transformers
    from cad_recovery.vlm_runtime_v1 import LocalVLM
    require(torch.__version__ == '2.14.1+cu130' and transformers.__version__ == '4.57.1', 'Changed VLM backend')
    output = inside(ROOT, args.output); require(not output.exists(), 'Choose new VLM output'); output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    model = LocalVLM(ROOT, config['assets'], len(request['requests']))
    results = []
    for row in request['requests']:
        result = model.infer(cases[row['case_id']], row['trigger']); results.append(result)
        print(row['case_id'], result['status'], result['reason'], result.get('runtime', {}).get('seconds'), flush=True)
    report = {'schema_version': 'vlm-worker-result/1.0', 'request': {'path': args.request, 'sha256': args.request_sha256},
        'results': results, 'load_seconds': model.load_seconds, 'seconds': time.perf_counter() - started,
        'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(), 'dtype': str(next(model.model.parameters()).dtype),
        'allocated_peak_MiB': torch.cuda.max_memory_allocated()/1024**2,
        'reserved_peak_MiB': torch.cuda.max_memory_reserved()/1024**2,
        'model_calls': model.generated_calls, 'request_attempts': model.budget.used,
        'generation': {'do_sample': model.generation.do_sample, 'num_beams': model.generation.num_beams,
            'mode': model.generation.get_generation_mode().value, 'max_new_tokens': model.generation.max_new_tokens},
        'training_executed': False, 'network_disabled': True,
        'production_admitted': False, 'graph_acceptance': False, 'physical_geometry_approved': False}
    (output / 'result.json').write_bytes(encoded(report))


if __name__ == '__main__':
    from cad_recovery.gpu_inference_v1 import deny_python_network
    p = argparse.ArgumentParser(__doc__)
    for name in ('config', 'sha256', 'request', 'request-sha256', 'output'): p.add_argument('--' + name, required=True)
    with deny_python_network(): main(p.parse_args())
