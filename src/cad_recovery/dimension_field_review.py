"""Bind a complete, explicit dimension field review to sealed raster/native evidence."""
from collections import Counter
from copy import deepcopy
import html
import json
import os
from pathlib import Path

import ezdxf
import numpy as np

from .clear_pairs import contained, digest, read_json
from .dimension_field_decisions import require, compare_display_contacts, decide_display_fields
from .dimension_field_native import extract_dimension_field_native, extract_text_instance, inspect_drawing_native, native_geometry_snapshot
from .dimension_field_raster import verify_field_raster_evidence
from .dimension_field_references import match_reference_anchors
from .instance_geometry import resolve_instance_entity
from .object_review import _binding
from .training_target_policy import build_exclusion_manifest, screen_training_target

MODULES = {'dimension_field_review.py', 'dimension_field_decisions.py', 'dimension_field_native.py',
           'dimension_field_raster.py', 'dimension_field_references.py', 'clear_pairs.py',
           'dxf_semantics.py', 'dimension_attachment_candidates.py', 'instance_geometry.py',
           'object_review.py', 'raster_regions.py', 'training_target_policy.py'}
CORE_DEPS = {'parent_run', 'object_corpus', 'point_comparison', 'point_references', 'point_seal',
             'raster_evidence', 'raster_seal', 'native_research', 'handoff', 'exclusions', 'exclusion_report'}


def _write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=True, indent=2, allow_nan=False)
        stream.write('\n')


def _index(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), 'Duplicate field-review identity: '+key)
    return result


def review_roles(document, native, recommendations, regions, matrix):
    """Check declared geometry instances; keep visual role assertions separate."""
    required = set(native['geometry_candidates']['points'])
    require(set(recommendations) == required, 'Every required support needs its own role review')
    results = {}
    geometry = {g['handle']: g for g in native['candidate_geometry']}
    for field, spec in recommendations.items():
        require(spec['cad_xyz'] == native['definition_points'][field], 'Role review moved a native support')
        region = regions[spec['source_region_id']]
        require(spec['source_region_pin'] == {'path': region['workspace_path'], 'sha256': region['png_sha256']},
                'Role review refers to a different raster region')
        relationships = []
        prior = native['geometry_candidates']['points'][field]['candidates']
        relations = spec['candidate_relations']
        require(relations == [{'handle': r['geometry_handle'], 'relation': r['relation'],
                               'endpoint': r.get('endpoint'), 'distance_cad': r['distance_cad']} for r in prior],
                'Complete role incidence inventory differs')
        if 'infinite_support_counterexample' in spec:
            require(not prior and not spec['recommended_geometry_handles'], 'Infinite support cannot hide finite candidates')
            infinite = spec['infinite_support_counterexample']
            identity = infinite['geometry_instance']
            entity, provenance = resolve_instance_entity(document, identity['handle'], identity['insert_path'])
            require(native_geometry_snapshot(entity) == infinite['native_geometry']
                    and provenance == infinite['instance_provenance'], 'Explicit support instance differs')
            binding = _binding(entity, spec['cad_xyz'], 'supporting_line', {})
            require(binding['exact_finite_geometry_attachment'] is False
                    and binding['distance_to_finite_segment_cad'] > 1e-6, 'Infinite support became a finite attachment')
            relationships.append({'identity': deepcopy(identity), 'native_geometry': deepcopy(infinite['native_geometry']),
                                  'instance_provenance': provenance, 'binding': binding})
        else:
            handles = spec['recommended_geometry_handles']
            require(handles and len(handles) == len(set(handles)), 'Explicit finite role needs distinct native features')
            require(set(handles) <= {r['handle'] for r in relations}, 'Role feature/relation inventory differs')
            for relation in relations:
                handle = relation['handle']
                if handle not in handles:
                    continue
                require(any(r['geometry_handle'] == handle and r['relation'] == relation['relation'] for r in prior),
                        'Role relation was not an exact retained incidence candidate')
                entity = document.entitydb.get(handle)
                require(entity is not None and entity.dxf.owner == document.modelspace().block_record_handle
                        and native_geometry_snapshot(entity) == geometry[handle], 'Role geometry differs from native snapshot')
                relationships.append({'identity': {'handle': handle, 'insert_path': []},
                    'native_geometry': deepcopy(geometry[handle]),
                    'binding': _binding(entity, spec['cad_xyz'], relation['relation'], {})})
        pixel = (np.asarray(matrix) @ [*spec['cad_xyz'][:2], 1])[:2].tolist()
        box = region['source_box_px']
        inside = box[0] <= pixel[0] < box[2] and box[1] <= pixel[1] < box[3]
        results[field] = {'status': 'explicit_role_and_native_relation_reviewed' if inside else 'qualified_context_not_covering_support',
            'role_observation': spec['observed_role_candidate'], 'reason': spec['reason'],
            'role_basis': 'assistant_visual_context_and_explicit_native_identity_not_layer_inference',
            'source_region': deepcopy(spec['source_region_pin']), 'source_region_id': region['id'],
            'projected_definition_point_xy': pixel, 'definition_point_inside_context_region': inside,
            'native_relations': relationships,
            'other_incidence_candidates': [deepcopy(r) for r in prior if r['geometry_handle'] not in spec['recommended_geometry_handles']],
            'finite_relations_only': all(r['binding']['exact_finite_geometry_attachment'] for r in relationships),
            'geometry_role_label_approved': False, 'measured_anchor_accuracy_approved': False}
    return results


def run_dimension_field_review(workspace, config_path, output):
    root, config_path, output = (Path(p).resolve() for p in (workspace, config_path, output))
    require(config_path.is_relative_to(root), 'Configuration must be in workspace')
    require(output.is_relative_to(root/'artifacts') and output != root/'artifacts', 'Choose a new artifacts directory')
    if output.exists():
        raise FileExistsError('Keep existing field review results; choose a new version')
    config_sha = digest(config_path)
    config = read_json(config_path)
    require(config.get('schema_version') == '1.0' and config.get('purpose') == 'explicit_dimension_field_review'
            and config.get('training_eligible') is False and config.get('split_modified') is False,
            'Expected a field review without training or split approval')
    require(set(config.get('implementation', {})) == MODULES, 'Incomplete implementation pins')
    dependencies = config.get('dependencies', {})
    require(CORE_DEPS <= set(dependencies), 'Missing field-review dependency')
    inputs = [(config_path, config_sha)]

    def pinned(record):
        path = contained(root, record['path'])
        require(digest(path) == record['sha256'], 'Pinned field evidence changed: '+record['path'])
        inputs.append((path, record['sha256']))
        return path

    paths = {name: pinned(record) for name, record in dependencies.items()}
    for name, sha in config['implementation'].items():
        path = Path(__file__).parent/name
        require(digest(path) == sha, 'Field review implementation changed')
        inputs.append((path, sha))
    for record in config.get('library_dependencies', {}).values():
        pinned(record)
    for name, sha in config.get('entrypoints', {}).items():
        pinned({'path': name, 'sha256': sha})
    dataset = contained(root, config['dataset'])
    sealed_dirs = {paths[name].parent for name in CORE_DEPS if paths[name].parent != root/'artifacts'}
    require(not output.is_relative_to(dataset) and not any(output.is_relative_to(p) for p in sealed_dirs),
            'Output cannot extend a sealed input directory')
    raster = verify_field_raster_evidence(root, dependencies['raster_evidence'], dependencies['raster_seal'])
    sources = _index(raster['data']['sources'], 'id')
    research = _index(read_json(paths['native_research'])['pairs'], 'pair_id')
    parent = read_json(paths['parent_run'])
    require(parent.get('complete') is True, 'Parent pair run must be complete')
    pairs = _index(parent['pairs'], 'input_path')
    corpus = _index(read_json(paths['object_corpus'])['pairs'], 'pair_id')
    catalog = _index(read_json(paths['handoff'])['entries'], 'input_path')
    points = read_json(paths['point_comparison'])
    require(points['parent_report_sha256'] == dependencies['parent_run']['sha256']
            and points['references_sha256'] == dependencies['point_references']['sha256']
            and points['seal_sha256'] == dependencies['point_seal']['sha256'], 'Old point evidence has different parents')
    exclusions = read_json(paths['exclusions'])
    require(exclusions == build_exclusion_manifest(dependencies['exclusion_report'], read_json(paths['exclusion_report'])['cases']),
            'Target policy differs from its pinned resource decisions')
    seen, reviewed_pairs, corrections = set(), [], []
    for case in config['cases']:
        identifier = case['pair_id']
        require(identifier not in seen, 'Repeated reviewed pair')
        seen.add(identifier)
        pair_path, object_path = pinned(case['pair_report']), pinned(case['objects'])
        pair, objects = read_json(pair_path), read_json(object_path)
        require(pair == pairs.get(pair['input_path']) and objects['pair_id'] == identifier
                and case['objects']['sha256'] == corpus[identifier]['objects_sha256']
                and object_path == contained(paths['object_corpus'].parent, corpus[identifier]['objects_path'])
                and pair_path == paths['parent_run'].parent/identifier/'pair.json', 'Parent pair/object inventory differs')
        for side in ('input', 'target'):
            path = contained(dataset, pair[side+'_path'])
            require(digest(path) == pair[side+'_sha256'], 'Reviewed original file changed')
            inputs.append((path, pair[side+'_sha256']))
        require(objects['target_sha256'] == pair['target_sha256'] and objects['source_sha256'] == pair['input_sha256']
                and objects['annotation_approved'] is False and objects['training_eligible'] is False,
                'Object source identity or approval differs')
        handed = catalog[pair['input_path']]
        require(handed['pair_report'] == case['pair_report'] and handed['target_sha256'] == pair['target_sha256']
                and handed['input_sha256'] == pair['input_sha256'] and handed['annotation_approved'] is False
                and handed['training_eligible'] is False and handed['split'] is None, 'Latest handoff no longer identifies this pair')
        source = sources[case['source_id']]
        require(source['source_sha256'] == pair['input_sha256']
                and contained(root, source['source_path']) == contained(dataset, pair['input_path']), 'Raster review source differs')
        regions = _index([r for r in raster['data']['regions'] if r['source_id'] == case['source_id']], 'id')
        regions = {key: {**region, 'workspace_path': contained(paths['raster_evidence'].parent, region['path']).relative_to(root).as_posix()}
                   for key, region in regions.items()}
        readings, views = _index(source['dimension_readings'], 'id'), _index(source['views'], 'id')
        specs = _index(case['objects_review'], 'handle')
        old_objects = {r['identity']['handle']: r for r in objects['objects']}
        require(len(old_objects) == len(objects['objects']) == objects['dimension_count'] == case['expected_dimension_count']
                and set(specs) == set(old_objects) and {s['reading_id'] for s in specs.values()} == set(readings)
                and len(specs) == len(readings), 'All dimensions and raster readings require one explicit association')
        target = contained(dataset, pair['target_path'])
        gate = screen_training_target(target, exclusions)
        require(gate['status'] == 'not_approved', 'Excluded native targets cannot enter this field review scope')
        doc = ezdxf.readfile(target)
        native_audit = inspect_drawing_native(doc)
        require(native_audit == research[identifier]['native_resource_audit'], 'Native resource inventory changed')
        title = {}
        for field, expectations in research[identifier]['title_instances'].items():
            require(isinstance(expectations, list) and expectations, 'Title evidence requires explicit instances')
            title[field] = []
            for expectation in expectations:
                fresh = extract_text_instance(doc, expectation['handle'], expectation['insert_path'])
                require(fresh == expectation, 'Title text instance changed')
                title[field].append(fresh)
        for field, raster_field in [('designation', 'drawing_identifier'), ('name', 'part_name'), ('scale', 'scale')]:
            require({row['plain_text'] for row in title[field]} == {source['title_readings'][raster_field]},
                    'Raster/native title readings disagree')
        native_rows = {r['identity']['handle']: r for r in research[identifier]['native_dimensions']}
        require(set(native_rows) == set(old_objects), 'Native research is incomplete')
        matrix = pair['alignment']['cad_to_source_pixel']
        records = []
        for handle, old in old_objects.items():
            spec = specs[handle]
            native = extract_dimension_field_native(doc, old)
            authored = native_rows[handle]
            require(all(authored[k] == v for k, v in native.items()), 'Native field research no longer reproduces')
            reading, view = readings[spec['reading_id']], views[spec['view_id']]
            require(authored['research_label_region_id'] == spec['reading_id']+'-label', 'Explicit native/raster association differs')
            require(case['view_mapping'][authored['view_id']] == view['id'], 'Explicit reviewed view differs')
            require(type(spec['broken_length_view']) is bool, 'Break scope must be explicit')
            has_break_review = (native['native_kind'] == 'linear' and 'break' in authored.get('research_comment', '').lower()
                                and any(r['id'] == 'broken-length-view' for r in source['ambiguous_points']))
            require(spec['broken_length_view'] == has_break_review, 'Broken-length qualification differs from authored review scope')
            if spec['broken_length_view']:
                require(spec['break_region_id'] in regions and regions[spec['break_region_id']]['role'] == 'curved_break_lines',
                        'Broken-length qualification needs its original raster evidence')
                require(native['native_kind'] == 'linear' and 'break' in authored['research_comment'].lower()
                        and any(r['id'] == 'broken-length-view' and spec['break_region_id'] in r['context_region_ids']
                                for r in source['ambiguous_points']),
                        'Broken-length scope requires independent source and explicit native review')
            contact = compare_display_contacts(native, matrix, reading)
            fields = decide_display_fields(native, reading, contact, title['scale'][0]['plain_text'],
                                           broken_length_view=spec['broken_length_view'])
            links = match_reference_anchors(old['source_evidence'], pair['input_sha256'], pair['target_sha256'], points)
            roles = review_roles(doc, native, authored['role_recommendations'], regions, matrix)
            fields['view_membership'] = {'status': 'explicit_reviewed_assignment', 'view_id': view['id'],
                'printed_label': view['printed_label'], 'source_box_px': view['source_box_px'],
                'physical_scale_confirmed': False, 'assignment_basis': 'authored_source_view_and_native_dimension_association'}
            fields['geometry_role_and_attachment'] = roles
            fields['independent_anchor_intervals'] = links
            own_corrections = []
            if native['formatting_conflict_can_be_reinterpreted'] and fields['display_text']['status'] == 'supported_observation':
                correction = {'pair_id': identifier, 'handle': handle, 'field': 'display_text_review_status',
                    'old_field_status': old['field_statuses']['text'], 'new_field_status': 'supported_observation',
                    'original_override': old['source_evidence']['text_override'],
                    'derived_plain_override': native['override_normalization']['plain'],
                    'reason': 'HWQ_formatting_only_native_display_and_independent_raster_text_agree',
                    'source_reading_id': reading['id'], 'source_modified': False,
                    'nominal_corrected': False, 'geometry_corrected': False, 'annotation_approved': False}
                own_corrections.append(correction)
                corrections.append(correction)
            records.append({'identity': deepcopy(old['identity']), 'parent_object': deepcopy(old),
                'source_reading': deepcopy(reading), 'native': native, 'display_contact_review': contact,
                'fields': fields, 'corrections': own_corrections,
                'annotation_approved': False, 'training_eligible': False, 'split': None})
        reviewed_pairs.append({'pair_id': identifier, 'source_id': case['source_id'],
            'input_path': pair['input_path'], 'input_sha256': pair['input_sha256'],
            'target_path': pair['target_path'], 'target_sha256': pair['target_sha256'],
            'pair_report': case['pair_report'], 'parent_objects': case['objects'],
            'parent_visual_status': pair['visual_status'], 'fixed_alignment': deepcopy(pair['alignment']),
            'native_resource_audit': native_audit, 'training_target_policy': gate,
            'train_only_constraint_inherited': handed['train_only_constraint_inherited'],
            'native_title_instances': title, 'source_title_readings': deepcopy(source['title_readings']),
            'filename_designation': research[identifier]['filename_designation'],
            'filename_designation_matches_native_title': research[identifier]['filename_designation_matches_native_title'],
            'views': deepcopy(source['views']), 'curve_and_thin_line_observations': deepcopy(source['curve_and_thin_line_observations']),
            'ambiguous_source_observations': deepcopy(source['ambiguous_points']), 'objects': records,
            'annotation_approved': False, 'training_eligible': False, 'split': None})
    all_objects = [o for p in reviewed_pairs for o in p['objects']]
    require(len(reviewed_pairs) == config['expected_pair_count'] and len(all_objects) == config['expected_dimension_count'],
            'Reviewed field scope differs from frozen configuration')
    for path, expected in inputs:
        require(digest(path) == expected, 'Evidence changed during field review')
    summary = {'pairs': len(reviewed_pairs), 'dimensions': len(all_objects),
        'display_text_status': dict(Counter(o['fields']['display_text']['status'] for o in all_objects)),
        'numeric_status': dict(Counter(o['fields']['numeric_consistency']['status'] for o in all_objects)),
        'display_contact_status': dict(Counter(o['display_contact_review']['status'] for o in all_objects)),
        'display_contact_windows': sum(len(o['display_contact_review']['source_windows_px']) for o in all_objects),
        'required_geometry_support_fields': sum(len(o['fields']['geometry_role_and_attachment']) for o in all_objects),
        'infinite_support_fields': sum(not r['finite_relations_only'] for o in all_objects for r in o['fields']['geometry_role_and_attachment'].values()),
        'support_fields_outside_declared_context': sum(not r['definition_point_inside_context_region'] for o in all_objects for r in o['fields']['geometry_role_and_attachment'].values()),
        'old_interval_field_links': sum(o['fields']['independent_anchor_intervals']['linked_fields'] for o in all_objects),
        'dimensions_with_old_interval_links': sum(o['fields']['independent_anchor_intervals']['linked_fields'] > 0 for o in all_objects),
        'formatting_only_derivative_corrections': len(corrections),
        'filename_designation_conflicts': sum(not p['filename_designation_matches_native_title'] for p in reviewed_pairs),
        'physical_scale_approvals': 0, 'training_eligible_dimensions': 0}
    report = {'schema_version': '1.0', 'date': config['date'], 'status': 'complete_explicit_field_review',
        'config': {'path': config_path.relative_to(root).as_posix(), 'sha256': config_sha},
        'dependencies': dependencies, 'implementation': config['implementation'], 'summary': summary,
        'raster_verification': raster['verification'], 'pairs': reviewed_pairs,
        'source_modified': False, 'split_modified': False, 'annotation_approved': False, 'training_eligible': False,
        'limitations': ['Source transcriptions and geometry roles are explicit assistant observations, not expert labels.',
            'Arrow-contact windows and independently reviewed measured-anchor intervals are distinct evidence.',
            'Printed scales and native DIMLFAC agreement do not establish a physical coordinate scale.',
            'Finite CAD incidence, infinite supports and thin construction graphics are recorded separately.',
            'Formatting corrections never repair a numeric mismatch, nominal or native geometry.',
            'The retained field corpus, all handoff records, source files and exclusion policy remain unchanged.']}
    output.mkdir(parents=True, exist_ok=False)
    _write(output/'report.json', report)
    _write(output/'corrections.json', {'schema_version': '1.0', 'corrections': corrections,
                                     'source_modified': False, 'training_eligible': False})
    _write(output/'next-review-queue.json', {'schema_version': '1.0', 'training_eligible': False,
        'items': [{'pair_id': p['pair_id'], 'handle': o['identity']['handle'],
            'numeric_status': o['fields']['numeric_consistency']['status'],
            'unlinked_anchor_fields': [key for key, r in o['fields']['independent_anchor_intervals']['points'].items() if r['status'] != 'agrees_with_authored_interval'],
            'required': ['physical_units_and_per_view_scale', 'measured_geometry_and_anchor_accuracy', 'explicit_label_approval']}
            for p in reviewed_pairs for o in p['objects']]})
    body = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Поля размеров</title>',
            '<style>body{font:15px system-ui;margin:24px}td,th{border:1px solid #aaa;padding:6px}table{border-collapse:collapse}img{max-width:95%;max-height:500px}pre{white-space:pre-wrap}</style>',
            '<h1>Проверка отдельных полей размеров</h1><p>Надписи, масштаб, опоры и координаты проверяются раздельно. Обучающие метки не утверждены.</p>',
            '<p><a href="report.json">Полный отчёт</a> · <a href="corrections.json">Производные исправления</a> · <a href="next-review-queue.json">Оставшиеся проверки</a></p>']
    for pair in reviewed_pairs:
        body.append('<h2>'+html.escape(pair['input_path'])+'</h2><p>Обозначение штампа: '+html.escape(pair['source_title_readings']['drawing_identifier'])
                    +'; совпадение с именем файла: '+str(pair['filename_designation_matches_native_title'])+'</p>')
        body.append('<p>Исходные фрагменты для проверки видов, кривых и тонких линий: ')
        label_ids = {obj['source_reading']['id']+'-label' for obj in pair['objects']}
        for region in raster['data']['regions']:
            if region['source_id'] != pair['source_id'] or region['id'] in label_ids:
                continue
            path = contained(paths['raster_evidence'].parent, region['path'])
            link = Path(os.path.relpath(path, output)).as_posix()
            body.append('<a href="'+html.escape(link, quote=True)+'">'+html.escape(region['id'])+'</a> · ')
        body.append('</p>')
        for obj in pair['objects']:
            reading = obj['source_reading']
            region = next(r for r in raster['data']['regions'] if r['source_id'] == pair['source_id'] and r['id'] == reading['id']+'-label')
            path = contained(paths['raster_evidence'].parent, region['path'])
            link = Path(os.path.relpath(path, output)).as_posix()
            fields = obj['fields']
            body.append('<details><summary>'+html.escape(obj['identity']['handle']+' · '+reading['text']+' · '+fields['numeric_consistency']['status'])
                        +'</summary><img src="'+html.escape(link, quote=True)+'"><pre>'
                        +html.escape(json.dumps(fields, ensure_ascii=False, indent=2))+'</pre></details>')
    body.append('</html>')
    (output/'index.html').write_text(''.join(body), encoding='utf-8')
    _write(output/'manifest.json', {'schema_version': '1.0', 'artifact_sha256': {
        p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return report
