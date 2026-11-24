"""Finite train-only sampling and contrast/decoder ablations; partial targets only."""
import numpy as np
from PIL import Image

from . import segmentation_experiment_v1 as tiles
from .family_catalog_v2 import require

POLICY = {'version': 'partial-segmentation-revision/1.0', 'task': 'P10.2b.3',
    'seed': 20261110, 'updates': 120, 'batch': 4, 'tile_side': 256, 'stride': 192,
    'balanced_tile_stride': 128, 'minimum_role_pixels': 24, 'minimum_background_pixels': 512,
    'variants': ['control-gray', 'balanced-contrast-transpose', 'balanced-contrast-resize',
                 'balanced-contrast-resize-repeat'], 'primary': 'balanced-contrast-resize',
    'checkpoint_selection': 'fixed final update only; diagnostic validation never selects',
    'checkpoint_step': 116, 'optimizer': 'AdamW', 'learning_rate': .0003, 'weight_decay': .01,
    'contrast': {'ink_below': 240, 'train_ink_range_inclusive': [0, 160],
        'train_paper_range_inclusive': [245, 255], 'inference_ink': 0, 'inference_paper': 255},
    'sampling': 'one verified-background-only tile, one axis tile, two cycling foreground roles; group then patch then tile uniform',
    'decoder': 'all five transpose upsamplers replaced by nearest resize x2 and 3x3 convolution; shared layers initialized identically',
    'AMP': False, 'TF32': False, 'deterministic_algorithms': True,
    'maximum_tiles': 4096, 'max_child_seconds': 900, 'max_total_child_seconds': 3600,
    'max_host_MiB': 6144, 'max_allocated_MiB': 4096, 'max_reserved_MiB': 6144,
    'max_device_MiB': 8192, 'independent_evaluation': False,
    'physical_geometry_approved': False, 'production_admitted': False, 'closed_test_used': False}


def training_tiles(rows, arrays, balanced):
    result = []
    for row in rows:
        if row['split'] != 'train': continue
        rgb, target = arrays[row['id']]
        for box in tiles.boxes(*row['size_px'], stride=POLICY['balanced_tile_stride'] if balanced else 256):
            _, mask = tiles.padded_tile(rgb, target, box)
            counts = np.bincount(mask.ravel(), minlength=256)[:10]
            if not balanced and counts[1:].sum() < POLICY['minimum_role_pixels']: continue
            if balanced and counts.sum() < POLICY['minimum_role_pixels']: continue
            result.append({'patch': row['id'], 'group_id': row['group_id'], 'box': box,
                           'supervised_counts': counts.tolist()})
    require(4 <= len(result) <= POLICY['maximum_tiles'], 'Invalid bounded train tile inventory')
    require({r['group_id'] for r in result} == {r['group_id'] for r in rows if r['split'] == 'train'},
            'Lost train source group')
    return result


def schedule(inventory, balanced):
    rng = np.random.default_rng(POLICY['seed']); result = []
    if balanced:
        pools = {0: [i for i, t in enumerate(inventory) if t['supervised_counts'][0] >=
                     POLICY['minimum_background_pixels'] and sum(t['supervised_counts'][1:]) == 0]}
        pools.update({c: [i for i, t in enumerate(inventory) if t['supervised_counts'][c] >=
                         POLICY['minimum_role_pixels']] for c in range(1, 10)})
        require(all(pools.values()), 'Missing supervised role/background sampling pool')
    for step in range(POLICY['updates']):
        roles = [0, 5, 1 + (step * 2) % 9, 1 + (step * 2 + 1) % 9] if balanced else [None] * 4
        indices = []
        for role in roles:
            if not balanced: break
            pool = pools[role]
            groups = sorted({inventory[i]['group_id'] for i in pool})
            group = str(rng.choice(groups))
            patches = sorted({inventory[i]['patch'] for i in pool if inventory[i]['group_id'] == group})
            patch = str(rng.choice(patches))
            indices.append(int(rng.choice([i for i in pool if inventory[i]['patch'] == patch])))
        if not balanced: indices = rng.choice(len(inventory), size=4, replace=False).tolist()
        result.append({'indices': indices, 'requested_roles': roles,
                       'augmentation_seed': POLICY['seed'] + step})
    return result


def contrast(rgb, rng=None):
    """Equal ink intensity within a tile, independent of annotated role or color."""
    gray = np.asarray(Image.fromarray(rgb).convert('L'))
    c = POLICY['contrast']
    ink, paper = c['inference_ink'], c['inference_paper']
    if rng is not None:
        ink = int(rng.integers(c['train_ink_range_inclusive'][0], c['train_ink_range_inclusive'][1] + 1))
        paper = int(rng.integers(c['train_paper_range_inclusive'][0], c['train_paper_range_inclusive'][1] + 1))
    return np.repeat(np.where(gray < c['ink_below'], ink, paper).astype(np.uint8)[:, :, None], 3, axis=2)


def replace_decoder(model):
    import torch
    nn = torch.nn
    for block in (model.up3, model.up2, model.up1, model.up0):
        old = block.up
        block.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(old.in_channels, old.out_channels, 3, padding=1))
    old = model.final[0]
    model.final[0] = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(old.in_channels, old.out_channels, 3, padding=1))
    return model
