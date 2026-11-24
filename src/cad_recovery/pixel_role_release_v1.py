"""Reviewed partial pixel supervision; never a dense mask or physical CAD target."""
from collections import Counter
from html import escape
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageColor, ImageFilter

from . import pixel_role_draft_v1 as draft
from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .source_coallocation_v1 import package

PARENT = {'path': 'artifacts/pixel-role-draft-v2/manifest.json',
          'sha256': '4e4acc98088dc22a5b8f3de3349c5bcafab171bf3b6377629c0274b215cda741'}
TASK = 'partial_pixel_role_supervision'
CODE = ['src/cad_recovery/pixel_role_release_v1.py', 'scripts/release_pixel_roles_v1.py']
POLICY = {'task': TASK, 'classes': ['background'] + draft.POLICY['roles'],
          'ignore': 255, 'foreground_gray_below': 240, 'background_gray_at_least': 250,
          'background_ink_clearance_px': 3, 'ambiguity_clearance_px': 2,
          'annotation_authority': 'assistant_visual_review', 'human_verified': False,
          'dense_ground_truth': False, 'physical_geometry_approved': False,
          'independent_evaluation': False, 'closed_test_used': False,
          'maximum_patches': 12, 'maximum_patch_side': 1024,
          'training_scope': 'fixed experimental partial supervision only; no production admission'}


def dilate(array, radius):
    return np.asarray(Image.fromarray(array.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(2*radius+1))) != 0


def reviewed_masks(image, annotations, unknown, background_regions):
    """Only reviewed ink islands and explicitly reviewed clear background are targets."""
    arrays = draft.masks(image, annotations, unknown)
    gray = np.asarray(image.convert('L'))
    labels = arrays['labels'].copy()
    ambiguous = dilate((arrays['state'] == 2) | (arrays['state'] == 3), POLICY['ambiguity_clearance_px'])
    labels[ambiguous] = 255
    background = np.zeros(gray.shape, bool)
    for region in background_regions:
        require(len(region.get('observation', '')) > 20, 'Background needs explicit visual evidence')
        background |= draft.rasterize(gray.shape, region)
    clear = ~dilate(gray < POLICY['background_gray_at_least'], POLICY['background_ink_clearance_px'])
    # Missing foreground annotation never turns ink/antialiasing into background.
    labels[background & clear & ~ambiguous & (arrays['bits'] == 0)] = 0
    return {**arrays, 'labels': labels, 'background_review_region': background, 'ambiguous': ambiguous}


def freeze(root, reading, output):
    require(not output.exists(), 'Choose new config')
    output.write_bytes(encoded({'schema_version': 'pixel-role-release-config/1.0',
        'task': 'P10.2a.2', 'parent': PARENT, 'policy': POLICY,
        'reading': pin(root, reading), 'code': [pin(root, x) for x in CODE]}))
    return pin(root, output.relative_to(root).as_posix())


def source_crop(root, source, box):
    """Decode only a pinned, already screened source; return original RGB crop."""
    path = checked(root, source)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        with Image.open(path) as image:
            width, height = image.size
            require(len(box) == 4 and all(type(v) is int for v in box)
                    and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height,
                    'Invalid original source crop')
            return image.crop(box).convert('RGB')


def materialize(root, record):
    config = read(checked(root, record))
    require(config['schema_version'] == 'pixel-role-release-config/1.0' and config['parent'] == PARENT
            and config['policy'] == POLICY and config['code'] == [pin(root, x) for x in CODE], 'Changed release recipe')
    _, parents = draft.verify(root, PARENT)
    parent_by_id = {p['id']: p for p in parents}
    parent_folder = (root/PARENT['path']).parent
    reading = read(checked(root, config['reading']))
    require(reading['reviewer'] == 'assistant' and reading['human_verified'] is False
            and reading['physical_geometry_approved'] is False and reading['task'] == TASK,
            'Changed review authority/task')
    proposals = reading['patches']
    require(0 < len(proposals) <= POLICY['maximum_patches'] and len({p['id'] for p in proposals}) == len(proposals), 'Invalid patch inventory')
    assets, records, stats = {}, [], []
    for proposal in proposals:
        require(proposal['parent_patch'] in parent_by_id and len(proposal['observations']) >= 2
                and all(len(s) > 20 for s in proposal['observations']), 'Missing review evidence')
        parent = parent_by_id[proposal['parent_patch']]
        require(parent['split'] in ('train', 'validation') and parent['group_id'], 'No JPEG/test admission')
        if proposal['source_box'] is None:
            image = Image.open(parent_folder/parent['image']).convert('RGB')
            annotations = [a for a in parent['annotations'] if a['id'] in proposal['accept_annotation_ids']]
            require(set(proposal['accept_annotation_ids']) == {a['id'] for a in annotations}, 'Unknown annotation decision')
            annotations += proposal['annotations']
            unknown = parent['unknown_regions'] + proposal['unknown_regions']
            matrix = parent['patch_to_source']
        else:
            require(not proposal['accept_annotation_ids'], 'New crop cannot import differently located annotations')
            image = source_crop(root, parent['source'], proposal['source_box'])
            annotations, unknown = proposal['annotations'], proposal['unknown_regions']
            x, y = proposal['source_box'][:2]
            matrix = [[1,0,x],[0,1,y],[0,0,1]]
        require(max(image.size) <= POLICY['maximum_patch_side'], 'Patch exceeds resource bound')
        require(len({a['id'] for a in annotations}) == len(annotations), 'Duplicate annotation identity')
        arrays = reviewed_masks(image, annotations, unknown, proposal['background_regions'])
        folder = 'patches/'+proposal['id']
        paths = {n: folder+'/'+n+'.png' for n in ['image', 'labels', 'role-bits', 'state', 'preview']}
        assets[paths['image']] = draft.png(image)
        for name, array in [('labels', arrays['labels']), ('role-bits', arrays['bits']), ('state', arrays['state'])]:
            assets[paths[name]] = draft.png(Image.fromarray(array))
        overlay = arrays['rgb'].copy()
        for item in read(parent_folder/'taxonomy.json'):
            selected = arrays['labels'] == item['id']
            color = np.asarray(ImageColor.getrgb(item['color']))
            overlay[selected] = (overlay[selected]*.25+color*.75).astype(np.uint8)
        overlay[arrays['labels'] == 0] = (overlay[arrays['labels'] == 0]*.75+[0,50,0]).astype(np.uint8)
        overlay[arrays['ambiguous'] & arrays['ink']] = [110,110,110]
        assets[paths['preview']] = draft.png(Image.fromarray(overlay))
        inverse = [[1,0,-matrix[0][2]],[0,1,-matrix[1][2]],[0,0,1]]
        row = {'id': proposal['id'], 'split': parent['split'], 'group_id': parent['group_id'],
               'source': parent['source'], 'parent_patch': proposal['parent_patch'], 'size_px': list(image.size),
               'patch_to_source': matrix, 'source_to_patch': inverse, 'paths': paths,
               'task': TASK, 'partial_targets': True, 'dense_ground_truth': False,
               'training_eligible': parent['split'] == 'train', 'diagnostic_only': parent['split'] == 'validation',
               'independent_evaluation': False, 'physical_geometry_approved': False,
               'review': proposal, 'annotations': annotations, 'unknown_regions': unknown}
        assets[folder+'/record.json'] = encoded(row)
        records.append(row)
        stats.append({'id': row['id'], 'split': row['split'], 'group_id': row['group_id'],
            'pixels': image.width*image.height, 'targets_by_class': [int(np.count_nonzero(arrays['labels'] == i)) for i in range(10)],
            'ignore_pixels': int(np.count_nonzero(arrays['labels'] == 255)),
            'multiple_candidate_pixels': int(np.count_nonzero(arrays['state'] == 2)),
            'explicit_unknown_pixels': int(np.count_nonzero(arrays['state'] == 3))})
    for split in ('train', 'validation'):
        require(any(r['split'] == split for r in records), 'Missing assigned split')
    require(not ({r['group_id'] for r in records if r['split'] == 'train'} &
                 {r['group_id'] for r in records if r['split'] == 'validation'}), 'Cross-split group leakage')
    coverage = {s: np.sum([r['targets_by_class'] for r in stats if r['split'] == s], axis=0).tolist() for s in ('train','validation')}
    require(all(n > 0 for n in coverage['train']), 'Not all train classes have reviewed targets')
    admission = {'task': TASK, 'train': 'experimental_partial_supervision_only',
        'validation': 'diagnostic_only_no_training_or_model_selection', 'test': 'not_used',
        'dense_segmentation_admitted': False, 'physical_geometry_approved': False,
        'production_admitted': False, 'human_verified': False, 'independent_evaluation': False,
        'limitations': reading['limitations'], 'parent_draft_training_admission_changed': False}
    report = {'schema_version': 'pixel-role-release-report/1.0', 'policy': POLICY, 'admission': admission,
        'patches': stats, 'patch_counts': dict(Counter(r['split'] for r in records)),
        'targets_by_split': coverage, 'class_names': POLICY['classes'], 'closed_test_content_read': False,
        'models_executed': False, 'training_started': False}
    assets.update({'records.json': encoded(records), 'report.json': encoded(report),
                   'admission.json': encoded(admission), 'reading.json': encoded(reading)})
    for split in ('train', 'validation'):
        assets[split+'.jsonl'] = b''.join(encoded(r).replace(b'\n', b'')+b'\n' for r in records if r['split'] == split)
    assets['test.jsonl'] = b''
    cards = ['<!doctype html><meta charset="utf-8"><title>Reviewed partial masks</title>',
        '<style>body{font:16px sans-serif;margin:2rem}img{max-width:48%;vertical-align:top}section{border-top:1px solid #bbb;padding:1rem 0}</style>',
        '<h1>Проверенные частичные маски</h1><p>Только ограниченный эксперимент. Фон зелёный; серый — исключения; без цвета — ignore или исходный цвет. Полная разметка не заявлена.</p>',
        '<p><a href="report.json">Отчёт</a> · <a href="admission.json">Допуск</a></p>']
    for row in records:
        cards.append('<section><h2>'+escape(row['id'])+' / '+row['split']+'</h2><img src="'+row['paths']['image']+'"><img src="'+row['paths']['preview']+'"><p><a href="patches/'+row['id']+'/record.json">Решения</a></p></section>')
    assets['index.html'] = ''.join(cards).encode('utf-8')
    return assets, report, records


def build(root, record, output):
    root = Path(root).resolve()
    target = inside(root, output)
    require(target.parent == root/'artifacts' and not target.exists(), 'Choose new artifacts directory')
    assets, report, _ = materialize(root, record)
    target.mkdir()
    for name, value in assets.items():
        path = inside(target, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    (target/'manifest.json').write_bytes(encoded({'schema_version': 'pixel-role-release-manifest/1.0',
        'config': record, 'files': [pin(target, name) for name in sorted(assets)]}))
    return {'manifest': pin(root, (target/'manifest.json').relative_to(root).as_posix()), 'coverage': report['targets_by_split']}


def verify(root, record):
    root = Path(root).resolve()
    folder, manifest = package(root, record, 'pixel-role-release-manifest/1.0')
    assets, report, rows = materialize(root, manifest['config'])
    require({r['path'] for r in manifest['files']} == set(assets), 'Changed release inventory')
    require(all((folder/n).read_bytes() == value for n,value in assets.items()), 'Changed partial-mask recipe')
    return report, rows


def load(root, record, *, split, task):
    require(task == TASK and split in ('train','validation'), 'Only experimental partial pixel supervision is admitted')
    _, rows = verify(root, record)
    return {'records': [r for r in rows if r['split'] == split], 'task': TASK,
            'training_eligible': split == 'train', 'diagnostic_only': split == 'validation',
            'independent_evaluation': False, 'dense_ground_truth': False}
