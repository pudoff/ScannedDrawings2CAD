"""Train-only batches and target-free native inference for the frozen B0 run."""
import time

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, encoded, pin, read
from .family_catalog_v2 import require
from . import segmentation_experiment_v1 as tiles
from . import segmentation_revision_v1 as revision


def training_arrays(root, records, expected_ids):
    require(len(records) == len(expected_ids) and {r['id'] for r in records} == set(expected_ids), 'Changed train identities')
    require(all(r['split'] == 'train' for r in records), 'Validation targets cannot enter training')
    result = {}
    for row in records:
        with Image.open(checked(root, row['image'])) as image: rgb = np.asarray(image.convert('RGB')).copy()
        with Image.open(checked(root, row['labels'])) as image: target = np.asarray(image).copy()
        require(rgb.shape[:2] == target.shape and np.isin(target, [*range(10), 255]).all(), 'Invalid native train grid')
        result[row['id']] = (rgb, target)
    return result


def batch(arrays, inventory, entry):
    rng = np.random.default_rng(entry['augmentation_seed']); result = []
    for index in entry['indices']:
        tile = inventory[index]
        require(tile['patch'] in arrays, 'Tile is outside admitted train set')
        rgb, target = tiles.padded_tile(*arrays[tile['patch']], tile['box'])
        rgb = revision.contrast(rgb, rng)
        result.append((tiles.preprocess(rgb, 'gray'), target.astype(np.int64)))
    return np.stack([r[0] for r in result]), np.stack([r[1] for r in result])


def average_logits(rgb, predict_tile, side=256, stride=192):
    """No target argument; native pixels, padded tiles and all-overlap logits."""
    h, w = rgb.shape[:2]
    sums = np.zeros((10, h, w), np.float32); counts = np.zeros((h, w), np.uint16)
    for box in tiles.boxes(w, h, side=side, stride=stride):
        x, y, right, bottom = box
        image = np.full((side, side, 3), 255, np.uint8)
        image[:bottom-y, :right-x] = rgb[y:bottom, x:right]
        values = predict_tile(image)
        require(values.shape == (10, side, side) and np.isfinite(values).all(), 'Invalid native tile logits')
        sums[:, y:bottom, x:right] += values[:, :bottom-y, :right-x]
        counts[y:bottom, x:right] += 1
    require(bool((counts > 0).all()), 'Native inference has an uncovered pixel')
    np.divide(sums, counts[None], out=sums)
    return sums.argmax(0).astype(np.uint8), int(counts.min()), int(counts.max())


def train(root, run_pin, variant, output):
    import sys
    config = read(checked(root, run_pin))
    for record in config['code']: checked(root, record)
    protocol = read(checked(root, config['experiment']))
    require(variant in protocol['variants'] and not output.exists(), 'Invalid variant/output')
    sys.path.insert(0, str(root / config['dependency_directory']))
    import torch
    import transformers
    from .segformer_model_v1 import build_model, partial_loss, restored_logits
    from .gpu_training_probe_v1 import state_digest
    require(torch.__version__ == '2.14.1+cu130' and transformers.__version__ == '4.57.1' and torch.cuda.is_available(), 'Changed CUDA backend')
    torch.set_num_threads(4); torch.set_default_dtype(torch.float32)
    torch.manual_seed(protocol['initial_seed']); torch.cuda.manual_seed_all(protocol['initial_seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.fp32_precision = 'ieee'; torch.backends.cudnn.conv.fp32_precision = 'ieee'
    arrays = training_arrays(root, config['training'], protocol['training_patch_ids'])
    counts = sum((np.bincount(y.ravel(), minlength=256)[:10] for _, y in arrays.values()), np.zeros(10, np.int64))
    require(counts.tolist() == protocol['train_class_counts'], 'Changed train class counts')
    require(revision.training_tiles(config['training'], arrays, True) == protocol['inventory'], 'Changed train tile inventory')
    model = build_model(root, protocol['assets']).cuda()
    qualification = read(checked(root, protocol['qualification']))
    initial = state_digest(model.state_dict()); encoder_initial = state_digest(model.segformer.state_dict())
    require(initial == qualification['results'][0]['initial'] and encoder_initial == qualification['results'][0]['encoder_initial'], 'Changed qualified initial tensors')
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
    weights = torch.tensor(protocol['class_weights'], dtype=torch.float32, device='cuda')
    model.train(); losses = []; torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True); torch.cuda.synchronize(); started = time.perf_counter()
    for step, entry in enumerate(protocol['schedule']):
        x, y = batch(arrays, protocol['inventory'], entry)
        require(len(x) == protocol['batch'], 'Changed effective batch')
        optimizer.zero_grad(set_to_none=True)
        loss = partial_loss(model(torch.from_numpy(x).cuda()).logits, torch.from_numpy(y).cuda(), weights)
        require(bool(torch.isfinite(loss)), 'Nonfinite dataset loss')
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Missing/nonfinite dataset gradient')
        optimizer.step(); losses.append(float(loss.detach()))
        if (step+1) % 30 == 0: print(variant, 'update', step+1, losses[-1], flush=True)
    torch.cuda.synchronize(); training_seconds = time.perf_counter() - started
    require(len(losses) == protocol['updates'], 'Incomplete fixed final update')
    final = state_digest(model.state_dict()); require(final != initial, 'Weights did not update')
    torch.save(model.state_dict(), output / 'final.pt')
    del arrays, optimizer, loss, x, y
    model.eval(); inference = []
    with torch.inference_mode():
        for row in config['inference']:
            require(set(row) == {'id', 'image', 'size_px'}, 'Inference must not receive labels')
            with Image.open(checked(root, row['image'])) as image: rgb = np.asarray(image.convert('RGB')).copy()
            require(list(rgb.shape[:2][::-1]) == row['size_px'], 'Changed native inference shape')
            def predict(image):
                value = tiles.preprocess(revision.contrast(image), 'gray')
                return restored_logits(model(torch.from_numpy(value[None]).cuda()).logits, (256, 256))[0].cpu().numpy()
            torch.cuda.synchronize(); started = time.perf_counter()
            prediction, low, high = average_logits(rgb, predict)
            torch.cuda.synchronize(); seconds = time.perf_counter() - started
            Image.fromarray(prediction).save(output / (row['id'] + '.png'))
            inference.append({'id': row['id'], 'size_px': row['size_px'], 'coverage_min': low, 'coverage_max': high,
                'seconds': seconds, 'prediction': pin(root, (output / (row['id'] + '.png')).relative_to(root).as_posix())})
            print(variant, 'native inference', row['id'], round(seconds, 3), 's', flush=True)
    require(state_digest(model.state_dict()) == final, 'Inference mutated model state')
    allocated, reserved = torch.cuda.max_memory_allocated()/1024**2, torch.cuda.max_memory_reserved()/1024**2
    require(allocated <= protocol['resources']['allocated_MiB'] and reserved <= protocol['resources']['reserved_MiB'], 'CUDA memory exceeded')
    result = {'schema_version': 'segformer-trained-result/1.0', 'variant': variant, 'run_protocol': run_pin,
        'experiment': config['experiment'], 'initial_parameters_sha256': initial, 'encoder_initial_sha256': encoder_initial,
        'final_parameters_sha256': final, 'losses': losses, 'updates': len(losses), 'training_seconds': training_seconds,
        'training_ids': protocol['training_patch_ids'], 'inference': inference,
        'allocated_peak_MiB': allocated, 'reserved_peak_MiB': reserved, 'torch': torch.__version__,
        'transformers': transformers.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
        'dataset_training_started': True, 'validation_targets_read_by_model': False, 'checkpoint_selected_before_validation': True,
        'closed_test_used': False, 'production_admitted': False, 'physical_geometry_approved': False}
    (output / 'result.json').write_bytes(encoded(result))
