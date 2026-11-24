"""Small train-only release of reviewed dimension/geometry roles.

Targets describe drawing semantics, not measured pixel or physical coordinates.
Native identities are target evidence and must never be inference inputs.
"""
from copy import deepcopy
import json
from pathlib import Path
import re

from PIL import Image

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .dataset_preparation_v1 import crop_source
from .family_catalog_v2 import require
from .ocr_release_v2 import load_ocr_training_v2

TASK = 'dimension_geometry_role_classification'
CODE = ['src/cad_recovery/'+name+'.py' for name in (
    'semantic_role_release_v1', 'dataset_preparation_v1', 'dataset_layout_v1', 'raster_regions')]
CODE += ['scripts/release_semantic_roles_v1.py']
ROLES = {
    'pitch_circle': {'pitch_circle_diameter_endpoint'},
    'outer_flange_contour': {'outer_flange_diameter_endpoint'},
    'profile_shoulder_span': {'lower_flat_contour_chamfer_junction', 'upper_flat_contour_chamfer_junction'},
    'hole_center_spacing': {'upper_hole_center', 'lower_hole_center'},
}


def reviewed_targets(field_report, decisions):
    objects = {(pair['source_id'], obj['identity']['handle']): (pair, obj)
               for pair in field_report['pairs'] for obj in pair['objects']}
    require(len(decisions) > 0, 'Empty role review')
    selected = set()
    rows = []
    for decision in decisions:
        key = (decision['source_id'], decision['dimension_handle'])
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', key[0]) is not None, 'Invalid source identifier')
        require(key in objects and key not in selected, 'Unknown or repeated reviewed object')
        selected.add(key)
        role = decision['role']
        require(role in ROLES and decision['decision'] == 'accepted_for_semantic_role_only', 'Unsupported role approval')
        pair, obj = objects[key]
        fields = obj['fields']
        require(fields['display_text']['source_matches_native'] is True, 'Source/native notation conflict')
        require(fields['numeric_consistency']['status'] == 'agrees_numerically', 'Conflicting native object cannot enter this role release')
        supports = fields['geometry_role_and_attachment']
        require(set(s['role_observation'] for s in supports.values()) == ROLES[role], 'Reviewed support roles differ')
        bindings = []
        for name, support in sorted(supports.items()):
            require(support['native_relations'], 'Missing explicit native support')
            for relation in support['native_relations']:
                require(relation['binding']['exact_finite_geometry_attachment'] is True, 'Infinite or approximate native support is not released')
                bindings.append({'definition_field': name, 'role': support['role_observation'],
                    'identity': deepcopy(relation['identity']), 'relation': relation['binding']['relation']})
        rows.append({'id': key[0]+'-'+key[1], 'source_id': key[0], 'source_sha256': pair['input_sha256'],
            'input': pair['input_path'], 'native_target': {'path': 'dataset/'+pair['target_path'], 'sha256': pair['target_sha256']},
            'dimension_handle': key[1], 'view_id': fields['view_membership']['view_id'],
            'target_role': role, 'native_target_evidence_only': bindings,
            'physical_geometry_approved': False, 'physical_scale_confirmed': False,
            'coordinate_regression_approved': False, 'reviewer': 'assistant', 'human_verified': False})
    return rows


def review_pairs(catalog, decisions):
    entries = {row['input']: row for row in catalog['entries']}
    seen = set()
    result = []
    for decision in decisions:
        key = decision['input']
        require(key in entries and key not in seen, 'Unknown or repeated pair review')
        seen.add(key)
        row = entries[key]
        require(row['declared_split'] == 'train', 'Manual review in this release is train-only')
        require(decision['input_sha256'] == row['input_sha256'] and decision['target_sha256'] == row['target_sha256'], 'Reviewed pair bytes differ')
        require(decision['decision'] == 'same_drawing_content' and len(decision['observations']) >= 3, 'Insufficient explicit pair decision')
        result.append({**deepcopy(decision), 'previous_visual_status': row['visual_status'],
            'correspondence_status': 'assistant_content_review_passed',
            'physical_target_approved': False, 'native_target_excluded': row['excluded_native_target']})
    return result


def context(workspace, config_pin):
    root = Path(workspace).resolve()
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'semantic-role-review-config/1.0', 'Unknown role config')
    require(config['task'] == TASK and config['split'] == 'train', 'Unsupported release task/split')
    require(config['physical_unit_policy'] == 'unconfirmed', 'Physical approval requires a separate release')
    require(config['code'] == [pin(root, name) for name in CODE], 'Role release code changed')
    paths = {key: checked(root, value) for key, value in config['dependencies'].items()}
    require(set(paths) == {'history_report', 'history_groups', 'catalog', 'field_report', 'ocr_manifest', 'ocr_labels', 'source_dispositions'}, 'Incomplete role dependencies')
    for decision in config['pair_reviews']:
        for image in decision['images']:
            checked(root, image)
    for source in config['source_reviews']:
        checked(root, source['context_evidence'])
    ocr = load_ocr_training_v2(root, config['dependencies']['ocr_manifest'])
    labels_path = paths['ocr_manifest'].parent/'labels.json'
    require(paths['ocr_labels'] == labels_path, 'Labels are not from the verified OCR release')
    labels = read(paths['ocr_labels'])
    sample_map = {row['id']: row for row in ocr['records']}
    annotations = {(row['source_id'], row['dimension_id']): row for row in labels['annotations']}
    field_report = read(paths['field_report'])
    rows = reviewed_targets(field_report, config['role_reviews'])
    groups = read(paths['history_groups'])['groups']
    membership = {name: group for group in groups for name in group['members']}
    history = read(paths['history_report'])
    require(history['groups'] == groups and history['summary']['cross_split_groups'] == 0, 'History group overlay differs or crosses splits')
    sources = {row['input_sha256']: row for row in read(paths['source_dispositions']) if row['source_kind'] == 'png'}
    source_reviews = {row['source_id']: row for row in config['source_reviews']}
    require(len(source_reviews) == len(config['source_reviews']), 'Repeated source review')
    require(set(source_reviews) == {row['source_id'] for row in rows}, 'Source review coverage differs')
    for row in rows:
        source = sources[row['source_sha256']]
        require(source['input'] == row['input'] and source['declared_split'] == 'train', 'Source is not current train')
        group = membership[row['input']]
        require(group['declared_splits'] == ['train'] and not group['cross_split'], 'Source group cannot enter train')
        checked(root, row['native_target'])
        row['group_id'] = group['id']
        annotation = annotations[(row['source_id'], row['dimension_handle'])]
        require(annotation['sample_ids'] and all(name in sample_map for name in annotation['sample_ids']), 'Query images missing')
        queries = [sample_map[name] for name in annotation['sample_ids']]
        require(all(q['source_sha256'] == row['source_sha256'] for q in queries), 'Query source mismatch')
        row['query_images'] = [{'id': q['id'], 'pin': pin(root, str(Path(q['image_path']).relative_to(root)).replace('\\', '/')),
                                'text': q['text']} for q in queries]
        row['printed_nominal'] = {'decimal': annotation['printed_components']['nominal']['decimal'], 'unit': 'printed_value'}
        row['scale_caption'] = source_reviews[row['source_id']]['scale_caption']
    pair_reviews = review_pairs(read(paths['catalog']), config['pair_reviews'])
    return root, config, rows, sources, pair_reviews


def build(workspace, config_pin, output):
    root, config, rows, sources, pair_reviews = context(workspace, config_pin)
    target = inside(root, output)
    require(not target.resolve().is_relative_to((root/'dataset').resolve()), 'Release cannot be inside source dataset')
    target.mkdir(parents=True, exist_ok=False)
    (target/'contexts').mkdir()
    (target/'queries').mkdir()
    contexts = {}
    for decision in config['source_reviews']:
        source_row = next(row for row in rows if row['source_id'] == decision['source_id'])
        rgb, provenance = crop_source(root, sources[source_row['source_sha256']], decision['context_box'])
        name = 'contexts/'+decision['source_id']+'.png'
        Image.fromarray(rgb).save(target/name)
        contexts[decision['source_id']] = {'path': name, 'sha256': pin(target, name)['sha256'],
            'source_box': decision['context_box'], 'provenance': provenance}
    for row in rows:
        row['context'] = contexts[row['source_id']]
        for query in row['query_images']:
            name = 'queries/'+query['id']+'.png'
            destination = target/name
            data = checked(root, query['pin']).read_bytes()
            with destination.open('xb') as stream:
                stream.write(data)
            query['image'] = pin(target, name)
    package = {'schema_version': 'semantic-role-labels/1.0', 'task': TASK, 'split': 'train',
        'config': config_pin, 'samples': rows, 'pair_reviews': pair_reviews,
        'training_eligible_for_task': True, 'human_verified': False,
        'physical_geometry_approved': False, 'physical_scale_confirmed': False,
        'coordinate_regression_approved': False, 'independent_evaluation': False,
        'target_evidence_policy': 'Native identities are labels/evidence only; inputs are source context and query raster.'}
    (target/'labels.json').write_bytes(encoded(package))
    with (target/'train.jsonl').open('xb') as stream:
        for row in rows:
            record = {'id': row['id'], 'context_image': row['context']['path'],
                'query_images': [q['image']['path'] for q in row['query_images']], 'target_role': row['target_role'],
                'source_sha256': row['source_sha256'], 'group_id': row['group_id'], 'split': 'train'}
            stream.write((json.dumps(record, ensure_ascii=True, sort_keys=True, allow_nan=False)+'\n').encode('utf-8'))
    for split in ('validation', 'test'):
        (target/(split+'.jsonl')).write_bytes(b'')
    summary = {'task': TASK, 'train': len(rows), 'validation': 0, 'test': 0,
        'sources': len(contexts), 'coallocation_groups': len({r['group_id'] for r in rows}),
        'manual_pair_correspondences': len(pair_reviews), 'physical_geometry_approved': False,
        'training_started': False, 'purpose': 'small assistant-reviewed role-classification training release; no physical regression targets'}
    (target/'report.json').write_bytes(encoded(summary))
    manifest = {'schema_version': 'semantic-role-manifest/1.0', 'config': config_pin,
        'files': [pin(target, path.relative_to(target).as_posix()) for path in sorted(target.rglob('*')) if path.is_file()]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return {'manifest': pin(root, (target/'manifest.json').relative_to(root).as_posix()), 'summary': summary}


def load_training(workspace, manifest_pin, *, task=TASK, split='train'):
    require(task == TASK and split == 'train', 'Only semantic role train is released; physical/coordinate tasks are prohibited')
    root = Path(workspace).resolve()
    manifest_path = checked(root, manifest_pin)
    manifest = read(manifest_path)
    require(set(manifest) == {'schema_version', 'config', 'files'} and manifest['schema_version'] == 'semantic-role-manifest/1.0', 'Unknown role manifest')
    folder = manifest_path.parent
    names = [entry['path'] for entry in manifest['files']]
    require(len(names) == len(set(name.casefold() for name in names)) and 'manifest.json' not in names, 'Duplicate/recursive manifest entries')
    require(set(names)|{'manifest.json'} == {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}, 'Role package inventory differs')
    for record in manifest['files']:
        checked(folder, record)
    _, config, expected, sources, pair_reviews = context(root, manifest['config'])
    labels = read(folder/'labels.json')
    require(set(labels) == {'schema_version', 'task', 'split', 'config', 'samples', 'pair_reviews',
        'training_eligible_for_task', 'human_verified', 'physical_geometry_approved', 'physical_scale_confirmed',
        'coordinate_regression_approved', 'independent_evaluation', 'target_evidence_policy'}, 'Unexpected package fields')
    require(labels['schema_version'] == 'semantic-role-labels/1.0', 'Unknown role labels schema')
    require(labels['task'] == TASK and labels['split'] == 'train' and labels['config'] == manifest['config'], 'Role task or config mismatch')
    require(labels['pair_reviews'] == pair_reviews, 'Pair decisions changed')
    for flag in ('human_verified', 'physical_geometry_approved', 'physical_scale_confirmed', 'coordinate_regression_approved', 'independent_evaluation'):
        require(labels[flag] is False, 'Unsupported package approval')
    require(labels['training_eligible_for_task'] is True, 'Task not admitted')
    require(len(labels['samples']) == len(expected), 'Role sample coverage differs')
    for actual, wanted in zip(labels['samples'], expected):
        context_record = actual['context']
        decision = next(d for d in config['source_reviews'] if d['source_id'] == wanted['source_id'])
        require(context_record['source_box'] == decision['context_box'], 'Context box changed')
        rgb, provenance = crop_source(root, sources[wanted['source_sha256']], decision['context_box'])
        with Image.open(checked(folder, {'path': context_record['path'], 'sha256': context_record['sha256']})) as image:
            require(image.mode == 'RGB' and image.size == (rgb.shape[1], rgb.shape[0]) and image.tobytes() == rgb.tobytes(), 'Context pixels differ from original preparation')
        require(context_record['provenance'] == provenance, 'Context provenance differs')
        wanted['context'] = context_record
        require(len(actual['query_images']) == len(wanted['query_images']), 'Query coverage differs')
        for query, expected_query in zip(actual['query_images'], wanted['query_images']):
            require(checked(folder, query['image']).read_bytes() == checked(root, expected_query['pin']).read_bytes(), 'Query pixels changed')
            expected_query['image'] = query['image']
        require(actual == wanted, 'Role fields differ from reviewed recipe')
    require((folder/'validation.jsonl').read_bytes() == (folder/'test.jsonl').read_bytes() == b'', 'Unexpected control samples')
    rows = [json.loads(line) for line in (folder/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    expected_rows = [{'id': r['id'], 'context_image': r['context']['path'], 'query_images': [q['image']['path'] for q in r['query_images']],
                     'target_role': r['target_role'], 'source_sha256': r['source_sha256'], 'group_id': r['group_id'], 'split': 'train'} for r in labels['samples']]
    require(rows == expected_rows, 'Training index differs from reviewed fields')
    return {'task': TASK, 'split': 'train', 'count': len(rows), 'records': rows,
            'physical_geometry_approved': False, 'coordinate_regression_approved': False}
