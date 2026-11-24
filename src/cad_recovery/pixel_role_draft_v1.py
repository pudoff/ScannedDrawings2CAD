"""Partial raster role proposals for review; no segmentation-training admission."""
from collections import Counter
from html import escape
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require
from .source_coallocation_v1 import package

BASE = {'path': 'artifacts/hard-raster-runs-final-v2/manifest.json', 'sha256': '43c0c58a0de2751ff216182c2b6590afb17c17540869883bd1d9f4c5ce57b03b'}
ROLES = {'path': 'artifacts/role-admission-labels-v1-final/manifest.json', 'sha256': '7819ee37f3ee6fdd1590e2993fd1a6086182a0a818ab4e6bc6fd224300f05ae5'}
GEOMETRY = {'path': 'artifacts/geometry-search-v3/manifest.json', 'sha256': '17fe1ab23441e365fc3ec44080b9faa11e23c910f067e3a972d66f824a21fcbe'}
POLICY = {'version': 'partial-pixel-role-review/1.0', 'ink_gray_below': 240, 'maximum_patch_side': 1024,
          'maximum_annotations_per_patch': 200, 'maximum_patches': 12,
          'roles': ['part_contour', 'dimension_line', 'extension_line', 'leader_line', 'axis_construction',
                    'text', 'sheet_frame', 'title_grid', 'hatching'],
          'background_id_reserved': 0, 'ignore_index': 255,
          'states': {'unassigned': 0, 'single_candidate': 1, 'multiple_candidates': 2, 'explicit_unknown': 3},
          'training_eligible': False, 'human_verified': False, 'physical_geometry_approved': False,
          'coordinate_regression_approved': False, 'independent_evaluation': False,
          'model_inference_executed': False, 'training_started': False,
          'source_pixels': 'unresized RGB crops of pinned previously screened contexts',
          'mask_authority': 'manual corridors/regions intersected with thresholded ink; weak visual proposals only',
          'unassigned_pixels': 'ignore, including white pixels; never assume background from missing annotations'}
CODE = ['src/cad_recovery/pixel_role_draft_v1.py', 'scripts/prepare_pixel_roles_v1.py']


def png(image):
    stream = BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def rasterize(shape, annotation):
    height, width = shape
    canvas = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    kind = annotation['kind']
    if kind == 'box':
        left, top, right, bottom = annotation['box']
        require(all(type(v) is int for v in (left, top, right, bottom)) and 0 <= left < right <= width and 0 <= top < bottom <= height, 'Box outside original patch')
        draw.rectangle((left, top, right-1, bottom-1), fill=255)
    elif kind == 'polygon':
        points = np.asarray(annotation['points'], dtype=float)
        require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 3 and np.isfinite(points).all()
                and (points >= 0).all() and (points <= [width-1, height-1]).all(), 'Invalid polygon')
        draw.polygon([tuple(p) for p in points], fill=255)
    elif kind in ('polyline', 'circle'):
        band = annotation['band_px']
        require(type(band) is int and 0 < band <= 20, 'Invalid corridor width')
        if kind == 'polyline':
            points = np.asarray(annotation['points'], dtype=float)
            require(points.ndim == 2 and points.shape[1] == 2 and len(points) >= 2 and np.isfinite(points).all()
                    and (points >= 0).all() and (points <= [width-1, height-1]).all(), 'Polyline outside patch')
            draw.line([tuple(np.rint(p).astype(int)) for p in points], fill=255, width=2*band+1)
        else:
            x, y = annotation['center']
            radius = annotation['radius']
            require(all(np.isfinite([x, y, radius])) and radius > band and 0 <= x-radius-band and x+radius+band < width
                    and 0 <= y-radius-band and y+radius+band < height, 'Circle corridor outside patch')
            outer, inner = radius+band, radius-band
            draw.ellipse((x-outer, y-outer, x+outer, y+outer), fill=255)
            draw.ellipse((x-inner, y-inner, x+inner, y+inner), fill=0)
    else:
        raise ValueError('Unsupported annotation geometry')
    return np.asarray(canvas) != 0


def masks(image, annotations, unknown_regions):
    rgb = np.asarray(image.convert('RGB'))
    ink = np.asarray(image.convert('L')) < POLICY['ink_gray_below']
    bits = np.zeros(ink.shape, np.uint16)
    for annotation in annotations:
        require(annotation['role'] in POLICY['roles'], 'Unknown pixel role')
        index = POLICY['roles'].index(annotation['role'])
        bits[rasterize(ink.shape, annotation) & ink] |= np.uint16(1 << index)
    unknown = np.zeros_like(ink)
    for region in unknown_regions:
        require(isinstance(region.get('reason'), str) and region['reason'], 'Unknown region needs reason')
        unknown |= rasterize(ink.shape, region)
    count = np.zeros(ink.shape, np.uint8)
    for index in range(len(POLICY['roles'])):
        count += ((bits >> index) & 1).astype(np.uint8)
    state = np.where(count == 0, 0, np.where(count == 1, 1, 2)).astype(np.uint8)
    state[unknown] = 3
    labels = np.full(ink.shape, 255, np.uint8)
    for index, role in enumerate(POLICY['roles']):
        selected = (bits & (1 << index)) != 0
        labels[selected & (state == 1)] = index+1
    return {'bits': bits, 'state': state, 'labels': labels, 'ink': ink, 'rgb': rgb}


def expanded_annotations(patch, references):
    annotations = list(patch['annotations'])
    unknown = list(patch['unknown_regions'])
    if patch.get('import_jpeg_guides'):
        require(patch['case_id'] == 'jpeg' and patch['box'] == [0, 0, 435, 600], 'JPEG reference coordinate mismatch')
        mapping = {'part_contour': 'part_contour', 'axis_or_construction': 'axis_construction', 'sheet_frame': 'sheet_frame',
                   'title_grid': 'title_grid', 'dimension_stroke': 'dimension_line', 'dimension_arc_guide': 'dimension_line', 'leader_stroke': 'leader_line'}
        for row in references['strokes']:
            item = {'id': 'P15.2a-'+row['id'], 'kind': 'polyline', 'points': row['points'], 'band_px': row['band_px']}
            if row['state'] != 'visible' or row['role'] == 'fold_artifact':
                unknown.append({**item, 'reason': 'preserved occlusion/fold exclusion from P15.2a'})
            else:
                annotations.append({**item, 'role': mapping[row['role']], 'origin': 'partial assigned corridor from P15.2a'})
        for row in references['texts']:
            item = {'id': 'P15.2a-'+row['id'], 'kind': 'box', 'box': row['box']}
            if row['state'] == 'readable':
                annotations.append({**item, 'role': 'text', 'origin': 'assigned readable field region, not character segmentation'})
            else:
                unknown.append({**item, 'reason': 'partial/unknown text remains unknown'})
    require(len(annotations) <= POLICY['maximum_annotations_per_patch'], 'Draft annotation budget exceeded')
    require(len({r['id'] for r in annotations}) == len(annotations), 'Repeated annotation ID')
    return annotations, unknown


def freeze(root, output):
    require(not output.exists(), 'Choose new draft config')
    config = {'schema_version': 'pixel-role-draft-config/1.0', 'task': 'P10.2a.1', 'policy': POLICY,
              'baseline': BASE, 'roles': ROLES, 'geometry': GEOMETRY,
              'reading': pin(root, 'configs/reviews/pixel-role-draft-reading-v1.json'),
              'jpeg_references': pin(root, 'configs/reviews/hard-raster-references-v1.json'),
              'approval_policy': pin(root, 'configs/reviews/data-label-approval-policy-v1.json'),
              'code': [pin(root, name) for name in CODE]}
    output.write_bytes(encoded(config))
    return pin(root, output.relative_to(root).as_posix())


def context(root, record):
    config = read(checked(root, record))
    require(config['policy'] == POLICY and config['code'] == [pin(root, name) for name in CODE], 'Changed draft implementation/policy')
    require((config['baseline'], config['roles'], config['geometry']) == (BASE, ROLES, GEOMETRY), 'Changed draft parents')
    baseline, manifest = package(root, BASE, 'hard-raster-runs-manifest/1.0')
    roles, _ = package(root, ROLES, 'role-admission-manifest/1.0')
    geometry, _ = package(root, GEOMETRY, 'geometry-search-manifest/2.0')
    old = read(checked(root, manifest['config']))
    reading = read(checked(root, config['reading']))
    for flag in ('human_verified', 'training_eligible', 'physical_geometry_approved', 'coordinate_regression_approved'):
        require(reading[flag] is False, 'Draft reading cannot approve training/physical targets')
    require(reading['reviewer'] == 'assistant' and [r['role'] for r in reading['taxonomy']] == POLICY['roles']
            and [r['id'] for r in reading['taxonomy']] == list(range(1, 10)), 'Changed draft taxonomy/authority')
    require(0 < len(reading['patches']) <= POLICY['maximum_patches'] and len({r['id'] for r in reading['patches']}) == len(reading['patches']), 'Invalid patch inventory')
    references = read(checked(root, config['jpeg_references']))
    approval = read(checked(root, config['approval_policy']))
    require(approval['decision'] == 'text_and_semantic_bindings_only' and approval['physical_geometry_approved'] is False, 'Unexpected user admission policy')
    return config, old, reading, references, baseline, roles, geometry


def materialize(root, record):
    config, old, reading, references, baseline, roles_folder, geometry_folder = context(root, record)
    cases = {c['id']: c for c in old['cases']}
    role_labels = read(roles_folder/'labels.json')
    groups = read(roles_folder/'groups.json')['groups']
    membership = {name: group for group in groups for name in group['members']}
    boundary = {row['id']: row for row in old['boundary']}
    registry = []
    for source in old['source_ledger']:
        group = membership[source['input']]
        require(group['id'] == source['group_id'] and not group['cross_split'] and not boundary[source['id']]['blocked'], 'Changed/closed source family boundary')
        require(source['declared_split'] in ('train', 'validation') and set(group['declared_splits']) == {source['declared_split']}, 'Mixed or closed source group')
        checked(root, {'path': 'dataset/'+source['input'], 'sha256': source['source_sha256']})
        require((source['declared_split'] == 'train' and source['input'].startswith(('clear_sum_dataset/vse_png/', 'pairs_pdf-dxf/pdf/')))
                or (source['declared_split'] == 'validation' and source['input'].startswith('valid_png/')), 'Current folder split differs')
        linked = [r for r in role_labels['roles'] if r['source_sha256'] == source['source_sha256']]
        registry.append({**source, 'existing_categorical_roles': linked, 'pixel_targets_released': False,
                         'semantic_family_approved': False, 'context_ids': [c['id'] for c in old['cases'] if c['source_id'] == source['id']]})
    registry.append({'id': 'jpeg', 'declared_split': 'viewed_development', 'group_id': None,
                     'source': references['source'], 'pixel_targets_released': False,
                     'context_ids': ['jpeg'], 'reason': 'viewed diagnostic photo, not train/validation'})
    assets = {}
    records = []
    statistics = []
    for patch in reading['patches']:
        require(patch['case_id'] in cases, 'Patch outside screened cohort')
        case = cases[patch['case_id']]
        checked(root, case['source'])
        source_path = baseline/'inputs'/(case['id']+'.png')
        box = patch['box']
        width, height = box[2]-box[0], box[3]-box[1]
        require(width > 0 and height > 0 and max(width, height) <= POLICY['maximum_patch_side'], 'Patch size budget exceeded')
        require(all(type(v) is int for v in box) and 0 <= box[0] < box[2] <= case['size_px'][0]
                and 0 <= box[1] < box[3] <= case['size_px'][1], 'Patch outside original context')
        with Image.open(source_path) as source:
            require(list(source.size) == case['size_px'], 'Changed original context size')
            image = source.convert('RGB').crop(box)
        annotations, unknown = expanded_annotations(patch, references)
        arrays = masks(image, annotations, unknown)
        folder = 'patches/'+patch['id']
        assets[folder+'/image.png'] = png(image)
        for name, array in [('role-bits', arrays['bits']), ('state', arrays['state']), ('class-index', arrays['labels'])]:
            assets[folder+'/'+name+'.png'] = png(Image.fromarray(array))
        for region in patch['regions']:
            rasterize((height, width), region)
        overlay = arrays['rgb'].copy()
        for item in reading['taxonomy']:
            selected = arrays['labels'] == item['id']
            color = np.array(ImageColor.getrgb(item['color']))
            overlay[selected] = (overlay[selected]*.25+color*.75).astype(np.uint8)
        overlay[arrays['state'] == 2] = [255, 0, 255]
        overlay[(arrays['state'] == 3) & arrays['ink']] = [100, 100, 100]
        assets[folder+'/review.png'] = png(Image.fromarray(overlay))
        source_group = None if case['id'] == 'jpeg' else membership[case['source']['path'][len('dataset/'):]]['id']
        left, top = case['source_box'][0]+box[0], case['source_box'][1]+box[1]
        row = {'id': patch['id'], 'case_id': case['id'], 'split': case['split'], 'group_id': source_group,
               'source': case['source'], 'context': pin(root, source_path.relative_to(root).as_posix()),
               'context_source_box': case['source_box'], 'patch_box_in_context': box, 'size_px': [width, height],
               'patch_to_source': [[1,0,left],[0,1,top],[0,0,1]], 'source_to_patch': [[1,0,-left],[0,1,-top],[0,0,1]],
               'image': folder+'/image.png', 'role_bits': folder+'/role-bits.png', 'state': folder+'/state.png',
               'class_index': folder+'/class-index.png', 'preview': folder+'/review.png', 'training_eligible': False,
               'physical_geometry_approved': False, 'annotations': annotations, 'unknown_regions': unknown, 'regions': patch['regions']}
        assets[folder+'/annotations.json'] = encoded(row)
        records.append(row)
        color_ink = arrays['ink'] & (np.max(arrays['rgb'], axis=2).astype(int)-np.min(arrays['rgb'], axis=2).astype(int) > 20)
        stats = {'id': patch['id'], 'split': case['split'], 'pixels': width*height,
                 'candidate_pixels_by_role': {r: int(np.count_nonzero(arrays['labels'] == i+1)) for i,r in enumerate(POLICY['roles'])},
                 'state_pixels': {name: int(np.count_nonzero(arrays['state'] == value)) for name,value in POLICY['states'].items()},
                 'non_neutral_ink_pixels': int(color_ink.sum()), 'source_color_role_cue_risk': patch['color_role_cue_risk'],
                 'ink_pixels': int(arrays['ink'].sum()), 'unassigned_background_targets': 0}
        statistics.append(stats)
    context_registry = []
    for case in old['cases']:
        geometry = read(geometry_folder/'runs'/case['id']/'regional_masked/result/page-001-geometry.json')
        context_registry.append({'id': case['id'], 'source_id': case['source_id'], 'split': case['split'],
                                 'patches': [r['id'] for r in records if r['case_id'] == case['id']],
                                 'draft_status': 'partial_draft' if any(r['case_id'] == case['id'] for r in records) else 'awaiting_pixel_annotation',
                                 'saved_CV_candidate_counts': dict(Counter(r['kind'] for r in geometry['candidates'])),
                                 'saved_CV_classified_roles': 0, 'CV_role_precision': None,
                                 'CV_used_as_mask_target': False})
    admission = {'schema_version': 'pixel-role-admission/1.0', 'status': 'draft_not_admitted',
                 'training_eligible': False, 'diagnostic_validation_only': True, 'independent_evaluation': False,
                 'physical_geometry_approved': False, 'coordinate_regression_approved': False,
                 'segmentation_training_indices_empty': True, 'accepted_categorical_roles_imply_pixel_targets': False,
                 'blockers': ['partial weak corridors/regions, not dense reviewed masks', 'unassigned pixels and all background remain ignore',
                              'class intersections and source-color cues require review', 'class coverage/family-independent evaluation insufficient'],
                 'native_exclusions_preserved': role_labels['new_role_exclusions']+role_labels['inherited_role_exclusions']}
    coverage = {split: {role: sum(s['candidate_pixels_by_role'][role] for s in statistics if s['split'] == split) for role in POLICY['roles']}
                for split in ('train', 'validation', 'viewed_development')}
    summary = {'schema_version': 'pixel-role-draft-report/1.0', 'policy': POLICY, 'source_rows': len(registry),
               'context_rows': len(context_registry), 'patch_rows': len(records), 'patches': statistics,
               'existing_categorical_roles': dict(Counter(r['split'] for r in role_labels['roles'])),
               'draft_patches_by_split': dict(Counter(r['split'] for r in records)), 'admission': admission,
               'candidate_pixels_by_role_and_split': coverage,
               'missing_train_candidate_roles': [role for role, count in coverage['train'].items() if not count],
               'models_executed': False, 'training_started': False, 'closed_test_content_read': False}
    assets.update({'registry.json': encoded(registry), 'contexts.json': encoded(context_registry),
                   'categorical-role-inventory.json': encoded(role_labels), 'admission.json': encoded(admission),
                   'report.json': encoded(summary), 'taxonomy.json': encoded(reading['taxonomy']), 'policy.json': encoded(POLICY)})
    for split in ('train','validation','test'):
        assets[split+'.jsonl'] = b''
    for split in ('train','validation','viewed_development'):
        assets['draft-'+split+'.jsonl'] = b''.join(encoded(r).replace(b'\n', b'')+b'\n' for r in records if r['split'] == split)
    cards = ['<!doctype html><meta charset="utf-8"><title>Pixel role draft</title>',
             '<style>body{font:16px sans-serif;margin:2rem;background:#fafafa}img{max-width:48%;vertical-align:top}section{border-top:1px solid #bbb;padding:1rem 0}pre{white-space:pre-wrap}</style>',
             '<h1>P10.2a.1 — черновые пиксельные роли</h1><p>Не допущены к обучению. Белые и неразмеченные пиксели — ignore. Пурпурный: несколько ролей; серый: явная неизвестность.</p>',
             '<p><a href="report.json">Отчёт</a> · <a href="admission.json">Допуск</a> · <a href="registry.json">Источники</a></p>']
    for row in records:
        cards.append('<section><h2>'+escape(row['id'])+'</h2><p>'+escape(row['split'])+'</p><img src="'+row['image']+'"><img src="'+row['preview']+'"><p><a href="patches/'+row['id']+'/annotations.json">Разметка и преобразования</a></p></section>')
    assets['index.html'] = ''.join(cards).encode('utf-8')
    return assets, summary, records


def build(workspace, record, output):
    root = Path(workspace).resolve()
    target = inside(root, output)
    require(target.parent == root/'artifacts' and not target.exists(), 'Choose new artifacts directory')
    assets, summary, _ = materialize(root, record)
    target.mkdir()
    for name, data in assets.items():
        path = inside(target, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (target/'manifest.json').write_bytes(encoded({'schema_version': 'pixel-role-draft-manifest/1.0', 'config': record,
                                                'files': [pin(target, name) for name in sorted(assets)]}))
    return {'manifest': pin(root, (target/'manifest.json').relative_to(root).as_posix()), 'report': summary}


def verify(workspace, record):
    root = Path(workspace).resolve()
    target, manifest = package(root, record, 'pixel-role-draft-manifest/1.0')
    assets, summary, records = materialize(root, manifest['config'])
    require(set(assets) == {row['path'] for row in manifest['files']}, 'Changed draft output inventory')
    for name, data in assets.items():
        require((target/name).read_bytes() == data, 'Changed pixel-role draft recipe: '+name)
    return summary, records


def load_review(workspace, record):
    _, records = verify(workspace, record)
    return {'records': records, 'training_eligible': False, 'task': 'pixel_role_review_only'}


def load_training(workspace, record, *, task='pixel_role_segmentation', split='train'):
    verify(workspace, record)
    raise ValueError('Draft pixel masks are not admitted to training or evaluation; release reviewed targets separately under P14.1b')
