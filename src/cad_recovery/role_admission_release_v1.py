"""Additional text/role labels with an explicit diagnostic-only validation protocol."""
from collections import Counter
from html import escape
import json
from pathlib import Path

import ezdxf

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .dimension_field_native import inspect_drawing_native
from .family_catalog_v2 import require
from .role_admission_native_v1 import annotation_neighbours, compose_groups, inspect_slot
from .role_source_expansion_v1 import verify as verify_intake
from .semantic_role_release_v2 import load_training as load_parent_roles
from .source_coallocation_v1 import load_overlay, package

ROLE_TASK = 'dimension_geometry_role_classification'
TEXT_TASK = 'drawing_annotation_transcription'
FALSE_FLAGS = ('human_verified', 'physical_geometry_approved', 'physical_scale_confirmed',
               'coordinate_regression_approved', 'independent_evaluation', 'test_used')
CODE = tuple('src/cad_recovery/'+n+'.py' for n in (
    'role_admission_release_v1', 'role_admission_native_v1', 'role_source_expansion_v1',
    'dimension_field_decisions', 'dimension_field_native', 'dimension_attachment_candidates',
    'dxf_semantics', 'object_review', 'semantic_role_release_v2', 'dataset_layout_v1')) + (
    'scripts/release_role_admission_v1.py', 'scripts/survey_role_families_v1.py', 'requirements.lock')


def validate_reading(intake, reading):
    require(reading['schema_version'] == 'role-admission-reading/1.0'
            and reading['task'] == 'P14.1b.10b' and reading['reviewer'] == 'assistant', 'Unknown admission review')
    require(all(reading[f] is False for f in FALSE_FLAGS), 'Unsupported physical or evaluation approval')
    require(reading['native_resource_policy'] == 'standard_geometry_only_no_proxy_or_external_resources', 'Changed native admission policy')
    live = {r['id']: r for r in intake['slots'] if r['query_image']}
    decisions = reading['slot_decisions']
    require(len(decisions) == len(live) and {r['slot_id'] for r in decisions} == set(live), 'Every readable slot needs one decision')
    source_ids = {r['source_id'] for r in live.values()}
    pairs = reading['pair_reviews']
    require(len(pairs) == len(source_ids) and {p['source_id'] for p in pairs} == source_ids, 'Pair review coverage differs')
    for row in pairs + decisions + reading['family_constraints']:
        require(len(row['observations']) >= 2 and all(isinstance(s, str) and len(s.strip()) >= 10 for s in row['observations']),
                'Missing explicit admission observations')
    for p in pairs:
        require(p['decision'] == 'same_drawing_content_for_local_categorical_labels' and len(p['images']) == 2,
                'Missing explicit source/native pair decision')
    for d in decisions:
        require(d['text_decision'] == 'accepted_visible_raster_transcription', 'Unknown text decision')
        require(d['final_role_disposition'] in ('accepted_categorical_role', 'excluded_taxonomy', 'deferred_native_resources'),
                'Unknown final role disposition')
        slot = live[d['slot_id']]
        require((d['role_decision'] == 'accepted_categorical_role' and slot['state'] == 'reviewed_candidate'
                 and d['role'] == slot['candidate_role']) or
                (d['role_decision'] == 'excluded_taxonomy' and slot['state'] == 'unsupported_role_taxonomy' and d['role'] is None),
                'Changed role admission scope')
    protocol = reading['evaluation_protocol']
    require(protocol['kind'] == 'diagnostic_validation_only' and all(protocol[f] is False for f in
        ('independent_evaluation', 'training_use_of_validation', 'closed_ocr_tests_reused', 'model_selection_run')),
        'Only diagnostic validation is approved')
    return live


def context(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'role-admission-config/1.0' and config['task'] == 'P14.1b.10b', 'Unknown admission config')
    require(all(config[f] is False for f in FALSE_FLAGS), 'Changed admission scope')
    require(config['code'] == [pin(root, name) for name in CODE], 'Admission code changed')
    verify_intake(root, config['intake_manifest'])
    intake_dir, _ = package(root, config['intake_manifest'], 'role-source-expansion-manifest/1.0')
    selection, intake = read(intake_dir/'selection.json'), read(intake_dir/'review.json')
    require(config['parent_role_manifest'] == selection['parent_role_manifest'], 'Changed role parent')
    load_parent_roles(root, config['parent_role_manifest'])
    parent_dir, _ = package(root, config['parent_role_manifest'], 'semantic-role-manifest/2.0')
    overlay = load_overlay(root, selection['source_overlay'])
    survey_dir, survey_manifest = package(root, config['survey_manifest'], 'role-family-survey-manifest/1.0')
    require(survey_manifest['catalog'] == selection['catalog'] and survey_manifest['test_content_read'] is False
            and survey_manifest['independent_evaluation'] is False, 'Changed native survey scope')
    survey_code = ['src/cad_recovery/role_admission_native_v1.py', 'scripts/survey_role_families_v1.py', 'requirements.lock']
    require(survey_manifest['code'] == [pin(root, n) for n in survey_code], 'Survey recipe changed')
    survey = read(survey_dir/'survey.json')
    catalog = read(checked(root, selection['catalog']))
    expected = {}
    for c in catalog['entries']:
        if c['declared_split'] in ('train', 'validation'):
            expected.setdefault(c['target_sha256'], {'paths': set(), 'splits': set()})
            expected[c['target_sha256']]['paths'].add(c['target'])
            expected[c['target_sha256']]['splits'].add(c['declared_split'])
    require(len(survey) == len(expected) and {r['sha256'] for r in survey} == set(expected), 'Survey does not cover all non-test native files')
    for row in survey:
        require(row['paths'] == sorted(expected[row['sha256']]['paths'])
                and row['splits'] == sorted(expected[row['sha256']]['splits'])
                and row['status'] in ('read', 'unreadable'), 'Survey path/split scope differs')
    reading = read(checked(root, config['reading']))
    live = validate_reading(intake, reading)
    selected_sources = {s['id']: s for s in selection['sources']}
    for p in reading['pair_reviews']:
        source = selected_sources[p['source_id']]
        require(p['input_sha256'] == source['input_sha256'] and p['target_sha256'] == source['target_sha256'],
                'Reviewed pair bytes differ from selected source/target')
        for image in p['images']:
            checked(root, image)
    for neighbour in reading['neighbour_evidence']:
        checked(root, neighbour['overview'])
        require(neighbour['input'].startswith('valid_png/'), 'Neighbour outside reviewed split')
    for decision in reading['slot_decisions']:
        if decision['glyph_evidence']:
            require(decision['glyph_evidence'] == pin(root, (intake_dir/live[decision['slot_id']]['query_image']).relative_to(root).as_posix()),
                    'Glyph evidence is not the reviewed query')
    return config, selection, intake, intake_dir, parent_dir, overlay, survey, reading


def materialize(root, config_pin):
    config, selection, intake, intake_dir, parent_dir, overlay, survey, reading = context(root, config_pin)
    groups = compose_groups(overlay, survey, reading['family_constraints'])
    membership = {p: g for g in groups['groups'] for p in g['members']}
    sources = {s['id']: s for s in selection['sources']}
    slots = {s['id']: s for s in intake['slots']}
    assets = {'groups.json': encoded(groups), 'reviews.json': encoded(reading)}
    roles, texts, native_facts, resource_exclusions = [], [], [], []
    for row in read(parent_dir/'labels.json')['samples']:
        original_group = row['group_id']
        require(row['input'] in membership and membership[row['input']]['declared_splits'] == ['train'], 'Parent role routing changed')
        context_image = 'parent/'+row['context']['image']
        query_images = ['parent/'+q['image'] for q in row['query_images']]
        for original, name in [(row['context']['image'], context_image)] + [(q['image'], n) for q, n in zip(row['query_images'], query_images)]:
            assets[name] = (parent_dir/original).read_bytes()
        roles.append({'id': row['id'], 'split': 'train', 'group_id': membership[row['input']]['id'],
            'parent_group_id': original_group, 'source_sha256': row['source_sha256'],
            'context_image': context_image, 'query_images': query_images, 'target_role': row['target_role'],
            'origin': 'role-v2', 'evidence': {'parent_manifest': config['parent_role_manifest'], 'parent_sample_id': row['id']}})
    docs = {}
    for decision in reading['slot_decisions']:
        slot = slots[decision['slot_id']]
        source = sources[slot['source_id']]
        group = membership[source['input']]
        split = source['declared_split']
        require(split in ('train', 'validation') and group['declared_splits'] == [split]
                and source['target'] in group['members'], 'Source/target coallocation differs')
        if source['id'] not in docs:
            doc = ezdxf.readfile(checked(root/'dataset', {'path': source['target'], 'sha256': source['target_sha256']}))
            docs[source['id']] = (doc, inspect_drawing_native(doc))
        doc, resources = docs[source['id']]
        facts = inspect_slot(doc, slot, decision)
        native_facts.append({**facts, 'source_id': source['id'], 'native_target': {
            'path': 'dataset/'+source['target'], 'sha256': source['target_sha256']}, 'resource_inventory': resources})
        disposition = ('excluded_taxonomy' if decision['role_decision'] == 'excluded_taxonomy'
                       else 'accepted_categorical_role' if resources['known_native_external_resources_absent']
                       else 'deferred_native_resources')
        require(decision['final_role_disposition'] == disposition, 'Final disposition differs from native resource audit')
        if disposition == 'deferred_native_resources':
            resource_exclusions.append({'slot_id': slot['id'], 'state': disposition,
                'reason': 'proxy_or_external_native_resource_requires_separate_disposition'})
        query = 'additional/'+slot['query_image']
        assets[query] = (intake_dir/slot['query_image']).read_bytes()
        context_image = 'additional/contexts/'+slot['context_id']+'.png'
        assets[context_image] = (intake_dir/'contexts'/(slot['context_id']+'.png')).read_bytes()
        common = {'id': 'additional-'+slot['id'], 'split': split, 'group_id': group['id'],
                  'source_sha256': source['input_sha256'], 'origin': 'additional-reviewed-sources',
                  'evidence': {'slot_id': slot['id'], 'dimension_handle': decision['dimension_handle']}}
        texts.append({**common, 'query_image': query, 'target_text_lines': slot['transcription_lines']})
        if disposition == 'accepted_categorical_role':
            roles.append({**common, 'context_image': context_image, 'query_images': [query], 'target_role': decision['role']})
    for records in (roles, texts):
        require(len({r['id'] for r in records}) == len(records), 'Repeated label identity')
    roles.sort(key=lambda r: r['id'])
    texts.sort(key=lambda r: r['id'])
    for namespace, records in (('roles', roles), ('text', texts)):
        for split in ('train', 'validation', 'test'):
            # Native/group/crop evidence is intentionally absent from the inference features.
            task = ROLE_TASK if namespace == 'roles' else TEXT_TASK
            index = [model_record(r, task) for r in records if r['split'] == split]
            assets[namespace+'/'+split+'.jsonl'] = ''.join(json.dumps(r, ensure_ascii=True, sort_keys=True)+'\n' for r in index).encode('utf-8')
    excluded = [{'slot_id': s['id'], 'state': s['state'], 'reason': s['reason']} for s in intake['slots'] if not s['candidate_role']] + resource_exclusions
    parent_excluded = read(parent_dir/'labels.json')['excluded_objects']
    report = {'schema_version': 'role-admission-report/1.0', 'config': config_pin,
        'roles_by_split': dict(Counter(r['split'] for r in roles)),
        'additional_roles_by_split': dict(Counter(r['split'] for r in roles if r['origin'] != 'role-v2')),
        'text_fields_by_split': dict(Counter(r['split'] for r in texts)),
        'role_groups_by_split': {s: len({r['group_id'] for r in roles if r['split'] == s}) for s in ('train', 'validation')},
        'parent_groups': len(overlay['groups']), 'release_groups': len(groups['groups']),
        'survey_native_files': len(survey), 'survey_unreadable': sum(r['status'] == 'unreadable' for r in survey),
        'new_role_exclusions': len(excluded), 'inherited_role_exclusions': len(parent_excluded),
        'independent_evaluation': False, 'model_training_started': False, 'physical_geometry_approved': False}
    assets['labels.json'] = encoded({'schema_version': 'role-admission-labels/1.0', 'roles': roles, 'text': texts,
        'new_role_exclusions': excluded, 'inherited_role_exclusions': parent_excluded,
        'role_task': ROLE_TASK, 'text_task': TEXT_TASK, 'train_eligible_for_declared_tasks': True,
        'validation_protocol': reading['evaluation_protocol'], **{f: False for f in FALSE_FLAGS}})
    assets['native-evidence.json'] = encoded(native_facts)
    assets['family-screen.json'] = encoded({'neighbours': {s['id']: annotation_neighbours(survey, s['target_sha256'])
        for s in sources.values() if s['id'] in docs}, 'unreadable': [r for r in survey if r['status'] != 'read'],
        'independent_evaluation': False, 'method': 'annotation token shortlist plus explicit visual/project constraints'})
    assets['report.json'] = encoded(report)
    cards = []
    for row in roles:
        cards.append('<section><h2>'+escape(row['id']+' — '+row['target_role'])+'</h2><p>'+escape(row['split'])+'</p>'
            +''.join('<img class="query" src="'+p+'" alt="Query">' for p in row['query_images'])
            +'<p><a href="'+row['context_image']+'">Context</a></p></section>')
    assets['index.html'] = ('<!doctype html><meta charset="utf-8"><title>Reviewed additional labels</title>'
        '<style>body{font:16px sans-serif;margin:2rem}section{border-top:1px solid #aaa}.query{max-height:150px;max-width:95%}</style>'
        '<h1>P14.1b.10b — categorical labels</h1><p>Validation is diagnostic only. No independent score or physical targets.</p>'
        '<pre>'+escape(str(report))+'</pre>'+''.join(cards)).encode('utf-8')
    return assets, report, roles, texts


def build(workspace, config_pin, output):
    root = Path(workspace).resolve()
    target = inside(root, output)
    require(not target.exists() and not target.resolve().is_relative_to((root/'dataset').resolve()), 'Choose a new output outside dataset')
    assets, report, _, _ = materialize(root, config_pin)
    target.mkdir(parents=True, exist_ok=False)
    for name, data in assets.items():
        p = inside(target, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('xb') as stream:
            stream.write(data)
    with (target/'manifest.json').open('xb') as stream:
        stream.write(encoded({'schema_version': 'role-admission-manifest/1.0', 'config': config_pin,
                              'files': [pin(target, name) for name in sorted(assets)]}))
    return {'manifest': pin(root, (target/'manifest.json').relative_to(root).as_posix()), 'report': report}


def verify(workspace, manifest_pin):
    root = Path(workspace).resolve()
    folder, manifest = package(root, manifest_pin, 'role-admission-manifest/1.0')
    assets, report, roles, texts = materialize(root, manifest['config'])
    require(set(assets) == {r['path'] for r in manifest['files']}, 'Published label inventory differs')
    for name, data in assets.items():
        require((folder/name).read_bytes() == data, 'Published labels differ from reviewed recipe: '+name)
    return report, roles, texts


def load_training(workspace, manifest_pin, *, task=ROLE_TASK):
    require(task in (ROLE_TASK, TEXT_TASK), 'Physical/coordinate tasks are not approved')
    _, roles, texts = verify(workspace, manifest_pin)
    records = [model_record(r, task) for r in (roles if task == ROLE_TASK else texts) if r['split'] == 'train']
    return {'task': task, 'split': 'train', 'records': records, 'count': len(records),
            'input_policy': 'Only image pixels; source, group, native identity and crop metadata are never model input.',
            'physical_geometry_approved': False}


def load_diagnostic(workspace, manifest_pin, *, task=ROLE_TASK):
    require(task in (ROLE_TASK, TEXT_TASK), 'Unsupported diagnostic task')
    _, roles, texts = verify(workspace, manifest_pin)
    records = [model_record(r, task) for r in (roles if task == ROLE_TASK else texts) if r['split'] == 'validation']
    return {'task': task, 'split': 'validation', 'records': records, 'count': len(records),
            'protocol': 'diagnostic_validation_only', 'training_eligible': False, 'independent_evaluation': False}


def model_record(row, task):
    """Keep native proofs in evidence files; return only the declared index contract."""
    fields = ('id', 'split', 'group_id', 'source_sha256', 'context_image', 'query_images', 'target_role') if task == ROLE_TASK else (
        'id', 'split', 'group_id', 'source_sha256', 'query_image', 'target_text_lines')
    return {name: row[name] for name in fields}
