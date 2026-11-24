"""Expanded train-only categorical roles; immutable V1 recipes remain intact."""
from collections import Counter
from copy import deepcopy
from html import escape
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .dataset_preparation_v1 import crop_source
from .family_catalog_v2 import require
from .semantic_role_release_v1 import TASK, ROLES, context as parent_context
from .source_coallocation_v1 import load_overlay, package

CODE = ('src/cad_recovery/semantic_role_release_v2.py', 'scripts/release_semantic_roles_v2.py',
        'src/cad_recovery/semantic_role_release_v1.py', 'src/cad_recovery/dataset_preparation_v1.py',
        'src/cad_recovery/source_coallocation_v1.py')
FALSE_FLAGS = ('human_verified', 'physical_geometry_approved', 'physical_scale_confirmed',
               'coordinate_regression_approved', 'independent_evaluation')
ROLE_RULES = {name: (roles,) for name, roles in ROLES.items()}
ROLE_RULES.update({
    'flange_width': ({'flange_outer_right_corner', 'flange_outer_left_corner'},),
    'chamfer_size': ({'flange_chamfer_outer_corner', 'flange_chamfer_flat_corner'},
                     {'lower_left_chamfer_flat_junction', 'lower_left_chamfer_vertical_junction'}),
    'fillet_radius': ({'fillet_circle_center', 'fillet_finite_arc'},
                      {'upper_inner_fillet_circle_center', 'upper_inner_fillet_finite_arc'}),
    'drill_depth': ({'drill_full_diameter_depth_corner', 'hole_mouth_chamfer_endface_corner'},),
    'thread_depth': ({'hole_endface_at_axis', 'thread_depth_boundary_at_axis'},),
    'thread_major_diameter': ({'lower_thread_major_diameter_convention', 'upper_thread_major_diameter_convention'},),
    'plate_thickness': ({'side_view_right_edge_junction', 'side_view_left_edge_junction'},),
    'base_contour_span': ({'left_lower_contour_chamfer_junction', 'right_lower_contour_chamfer_junction'},),
    'hole_to_contour_offset': ({'lower_hole_center', 'lower_flat_contour_chamfer_junction'},),
    'hole_diameter': ({'upper_hole_diameter_endpoint'},),
})


def exclusion_reason(obj):
    fields = obj['fields']
    if not fields['display_text']['source_matches_native']:
        return 'source_native_text_conflict'
    if fields['numeric_consistency']['status'] != 'agrees_numerically':
        return fields['numeric_consistency']['status']
    if not fields['geometry_role_and_attachment']:
        return 'missing_supports'
    for support in fields['geometry_role_and_attachment'].values():
        if not support['native_relations'] or any(r['binding']['exact_finite_geometry_attachment'] is not True for r in support['native_relations']):
            return 'infinite_or_unverified_native_attachment'
    return None


def reviewed_rows(field_report, reading):
    require(reading['schema_version'] == 'semantic-role-review/2.0' and reading['reviewer'] == 'assistant', 'Unknown semantic review')
    require(all(reading[flag] is False for flag in FALSE_FLAGS), 'Unsupported review approval')
    objects = {(pair['source_id'], obj['identity']['handle']): (pair, obj)
               for pair in field_report['pairs'] for obj in pair['objects']}
    eligible = {key for key, (_, obj) in objects.items() if exclusion_reason(obj) is None}
    decisions = reading['decisions']
    require(len(decisions) == len(eligible) and {(d['source_id'], d['dimension_handle']) for d in decisions} == eligible,
            'Review must cover exactly all eligible objects')
    contexts = {c['id']: c for c in reading['contexts']}
    require(len(contexts) == len(reading['contexts']), 'Repeated context')
    rows = []
    for d in decisions:
        pair, obj = objects[(d['source_id'], d['dimension_handle'])]
        fields = obj['fields']
        require(d['decision'] == 'accepted_for_semantic_role_only' and d['role'] in ROLE_RULES, 'Unsupported role decision')
        supports = fields['geometry_role_and_attachment']
        require({s['role_observation'] for s in supports.values()} in ROLE_RULES[d['role']], 'Role does not match reviewed supports')
        require(len(d['observations']) >= 2 and all(isinstance(s, str) and len(s.strip()) >= 8 for s in d['observations']), 'Insufficient role observations')
        require(d['context_id'] in contexts, 'Missing reviewed context')
        c = contexts[d['context_id']]
        require(c['source_id'] == d['source_id'] and c['view_id'] == fields['view_membership']['view_id'], 'Context source/view differs')
        box = c['source_box']
        require(len(box) == 4 and all(type(v) is int for v in box)
                and 0 <= box[0] < box[2] and 0 <= box[1] < box[3]
                and (box[2]-box[0])*(box[3]-box[1]) <= 16_777_216, 'Invalid context box')
        # Approximate projections only reject obviously incomplete navigation crops;
        # they are not released as pixel labels or physical coordinates.
        for support in supports.values():
            x, y = support['projected_definition_point_xy']
            require(box[0] <= x < box[2] and box[1] <= y < box[3], 'Context does not cover support navigation point')
        bindings = [{'definition_field': name, 'role': support['role_observation'],
                     'identity': deepcopy(r['identity']), 'relation': r['binding']['relation']}
                    for name, support in sorted(supports.items()) for r in support['native_relations']]
        rows.append({'id': d['source_id']+'-'+d['dimension_handle'], 'source_id': d['source_id'],
            'input': pair['input_path'], 'source_sha256': pair['input_sha256'],
            'native_target_evidence_only': {'path': 'dataset/'+pair['target_path'], 'sha256': pair['target_sha256'], 'bindings': bindings},
            'context_id': d['context_id'], 'view_id': c['view_id'], 'target_role': d['role'],
            'dimension_handle': d['dimension_handle'], **{flag: False for flag in FALSE_FLAGS}})
    require({r['context_id'] for r in rows} == set(contexts), 'Unused context')
    excluded = [{'source_id': key[0], 'dimension_handle': key[1], 'reason': exclusion_reason(obj)}
                for key, (_, obj) in sorted(objects.items()) if key not in eligible]
    return rows, excluded


def context(workspace, config_pin):
    root = Path(workspace).resolve()
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'semantic-role-config/2.0' and config['task'] == TASK and config['split'] == 'train', 'Unsupported role task/split')
    require(config['code'] == [pin(root, name) for name in CODE], 'Semantic role code changed')
    require(set(config['dependencies']) == {'parent_config', 'source_overlay', 'approval_policy', 'reading'}, 'Incomplete role dependencies')
    paths = {key: checked(root, value) for key, value in config['dependencies'].items()}
    policy = read(paths['approval_policy'])
    require(policy['decision'] == 'text_and_semantic_bindings_only' and policy['scope'] == 'step1_palec_rebro_subset'
            and policy['allowed_tasks'][TASK]['split'] == 'train'
            and all(policy[key] is False for key in ('physical_geometry_approved', 'coordinate_regression_approved')), 'Changed approval scope')
    _, parent, previous, sources, _ = parent_context(root, config['dependencies']['parent_config'])
    _, prior_manifest = package(root, policy['allowed_tasks'][TASK]['manifest'], 'semantic-role-manifest/1.0')
    require(prior_manifest['config'] == config['dependencies']['parent_config'], 'Parent release uses a different recipe')
    overlay = load_overlay(root, config['dependencies']['source_overlay'])
    membership = {name: g for g in overlay['groups'] for name in g['members']}
    reading = read(paths['reading'])
    field_report = read(checked(root, parent['dependencies']['field_report']))
    rows, excluded = reviewed_rows(field_report, reading)
    old = {row['id']: row for row in previous}
    require(set(old) <= {r['id'] for r in rows}, 'Previous roles lost')
    ocr_path = checked(root, parent['dependencies']['ocr_manifest']).parent
    # parent_context has verified this complete OCR recipe and all 32 images.
    annotations = {(a['source_id'], a['dimension_id']): a for a in read(ocr_path/'labels.json')['annotations']}
    samples = {r['id']: r for r in [json.loads(line) for line in (ocr_path/'train.jsonl').read_text(encoding='utf-8').splitlines()]}
    for c in reading['contexts']:
        require(c['evidence'], 'Missing visual context evidence')
        for item in c['evidence']:
            checked(root, item)
    for row in rows:
        require(row['source_id'] in ('palec', 'rebro'), 'Source outside reviewed policy')
        group = membership[row['input']]
        require(group['declared_splits'] == ['train'] and group['cross_split'] is False, 'Current group cannot enter train')
        require(row['native_target_evidence_only']['path'][len('dataset/'):] in group['members'], 'Target not in source group')
        row['group_id'] = group['id']
        checked(root, {key: row['native_target_evidence_only'][key] for key in ('path', 'sha256')})
        source = sources[row['source_sha256']]
        require(source['input'] == row['input'] and source['declared_split'] == 'train', 'Changed source routing')
        annotation = annotations[(row['source_id'], row['dimension_handle'])]
        row['query_images'] = []
        for name in annotation['sample_ids']:
            sample = samples[name]
            require(sample['source_sha256'] == row['source_sha256'] and sample['split'] == 'train', 'Query source/split differs')
            original = ocr_path/sample['image']
            original_pin = {'path': original.relative_to(root).as_posix(), 'sha256': sample['image_sha256']}
            checked(root, original_pin)
            row['query_images'].append({'id': name, 'original': original_pin})
        require(row['query_images'], 'Empty notation query')
        row['parent_sample_id'] = row['id'] if row['id'] in old else None
        if row['id'] in old:
            require(row['target_role'] == old[row['id']]['target_role'], 'Inherited role changed')
    return root, rows, excluded, reading, sources


def materialize(workspace, config_pin):
    root, rows, excluded, reading, sources = context(workspace, config_pin)
    assets, contexts = {}, {}
    for c in reading['contexts']:
        row = next(r for r in rows if r['context_id'] == c['id'])
        rgb, provenance = crop_source(root, sources[row['source_sha256']], c['source_box'])
        stream = BytesIO()
        Image.fromarray(rgb).save(stream, format='PNG')
        name = 'contexts/'+c['id']+'.png'
        assets[name] = stream.getvalue()
        contexts[c['id']] = {'image': name, 'source_box': c['source_box'], 'provenance': provenance}
    for row in rows:
        row['context'] = contexts[row['context_id']]
        for query in row['query_images']:
            name = 'queries/'+query['id']+'.png'
            assets[name] = checked(root, query['original']).read_bytes()
            query['image'] = name
    indices = [{'id': row['id'], 'context_image': row['context']['image'],
                'query_images': [q['image'] for q in row['query_images']], 'target_role': row['target_role'],
                'group_id': row['group_id'], 'source_sha256': row['source_sha256'], 'split': 'train'} for row in rows]
    stats = {'task': TASK, 'train': len(rows), 'validation': 0, 'test': 0,
        'inherited_samples': sum(r['parent_sample_id'] is not None for r in rows),
        'new_samples': sum(r['parent_sample_id'] is None for r in rows), 'sources': len({r['source_id'] for r in rows}),
        'coallocation_groups': len({r['group_id'] for r in rows}), 'contexts': len(contexts),
        'query_png_files': sum(name.startswith('queries/') for name in assets),
        'unique_query_sha256': len({q['original']['sha256'] for r in rows for q in r['query_images']}),
        'roles': dict(sorted(Counter(r['target_role'] for r in rows).items())), 'excluded_objects': len(excluded),
        'training_started': False, **{flag: False for flag in FALSE_FLAGS}}
    assets['labels.json'] = encoded({'schema_version': 'semantic-role-labels/2.0', 'config': config_pin,
        'task': TASK, 'split': 'train', 'samples': rows, 'excluded_objects': excluded,
        'training_eligible_for_task': True, **{flag: False for flag in FALSE_FLAGS},
        'model_input_policy': 'Only context and query image pixels. Native evidence, identifiers, source/group metadata and crop provenance are not input features.'})
    assets['report.json'] = encoded(stats)
    assets['train.jsonl'] = ''.join(json.dumps(r, ensure_ascii=True, sort_keys=True)+'\n' for r in indices).encode('utf-8')
    assets['validation.jsonl'] = assets['test.jsonl'] = b''
    cards = ''.join('<section><h2>'+escape(r['id']+' — '+r['target_role'])+'</h2><p>'+escape(r['view_id'])+
        '</p>'+''.join('<img class="query" src="'+escape(q['image'], quote=True)+'" alt="Query">' for q in r['query_images'])+
        '<p><a href="'+r['context']['image']+'">Full-resolution context</a></p><img loading="lazy" class="context" src="'+r['context']['image']+'" alt="Source context"></section>' for r in rows)
    assets['index.html'] = ('<!doctype html><meta charset="utf-8"><title>Semantic roles V2</title>'
        '<style>body{font:16px sans-serif;margin:2rem}section{border-top:1px solid #aaa;padding:1rem}.query{max-height:110px;max-width:95%}.context{max-width:100%;max-height:650px;object-fit:contain}</style>'
        '<h1>P14.1b.9: train-only semantic roles</h1><p>Categorical role labels. Assistant-reviewed; no physical coordinates or independent evaluation.</p>'
        '<pre>'+escape(str(stats))+'</pre>'+cards).encode('utf-8')
    return assets, stats, indices


def build(workspace, config_pin, output):
    root = Path(workspace).resolve()
    target = inside(root, output)
    require(not target.exists() and not target.resolve().is_relative_to((root/'dataset').resolve()), 'Choose a new output outside dataset')
    assets, stats, _ = materialize(root, config_pin)
    target.mkdir(parents=True, exist_ok=False)
    for name, data in assets.items():
        path = inside(target, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    manifest = {'schema_version': 'semantic-role-manifest/2.0', 'config': config_pin,
                'files': [pin(target, name) for name in sorted(assets)]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return {'manifest': pin(root, (target/'manifest.json').relative_to(root).as_posix()), 'summary': stats}


def load_training(workspace, manifest_pin, *, task=TASK, split='train'):
    require(task == TASK and split == 'train', 'Only categorical semantic role train is approved')
    root = Path(workspace).resolve()
    folder, manifest = package(root, manifest_pin, 'semantic-role-manifest/2.0')
    assets, stats, indices = materialize(root, manifest['config'])
    require(set(assets) == {item['path'] for item in manifest['files']}, 'Released file scope differs')
    for name, expected in assets.items():
        require((folder/name).read_bytes() == expected, 'Released bytes differ from reviewed recipe: '+name)
    return {'task': TASK, 'split': 'train', 'count': stats['train'], 'records': indices,
            'physical_geometry_approved': False, 'coordinate_regression_approved': False}
