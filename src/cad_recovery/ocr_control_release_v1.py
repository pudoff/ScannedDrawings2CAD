"""Pinned group-held-out OCR text, with the complete inherited train registry.

This release preserves every V3 train row and reserves whole source groups.
Validation/test mean an internal control under known, reviewed constraints;
they do not assert external semantic-family independence or approve CAD labels.
No model is invoked by the builder or consumer.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from html import escape
from pathlib import Path
import json

import numpy as np

from .coallocation_policy_v1 import build_coallocation_v1, validate_assignments_v1
from .coallocation_review_v1 import source_regions_v1
from .family_catalog_v2 import stable_digest
from .label_release_v1 import checked_pin
from .ocr_label_schema_v1 import make_ocr_transform_v1
from .ocr_line_schema_v3 import ReviewV3
from .ocr_release_v3 import load_ocr_training_v3, _package, _pixels, encoded, pin, read, require
from .clear_pairs import contained, digest
from .raster_regions import read_prepared_region

PARENT = {'path': 'artifacts/ocr-baseline-final-v2.json',
          'sha256': '6ee85d8133ce9758b590ad891890430ec6f8232e3b20cf3a9efa611d0b744dc6'}
PREVIOUS = {'path': 'artifacts/ocr-release-v3/manifest.json',
            'sha256': '40459498a211b577e6cdfbca9d8772b4c9b04a9768b2d9d842395995abb87fdd'}
SELECTION = {'path': 'artifacts/ocr-control-selection-v2.json',
             'sha256': 'c5debe99b3b15405d7e85be5efd356ef2287afbea46281e4e5d521b50abbf7bb'}
SCOPE = 'grouped_control_known_constraints_not_external_independence'
READINGS = ['artifacts/ocr-control-lines-'+letter+'-v1/review.json' for letter in ('a', 'b')]
DEPENDENCIES = {'parent_final', 'previous_ocr_manifest', 'catalog', 'policy', 'exclusions', 'overlay', 'prepared_catalog'}
CODE = ['src/cad_recovery/'+name+'.py' for name in ('ocr_control_release_v1', 'ocr_line_schema_v3',
    'ocr_release_v3', 'ocr_label_schema_v1', 'coallocation_policy_v1', 'coallocation_review_v1',
    'family_catalog_v2', 'ocr_lineage_v1', 'raster_regions', 'png_stream', 'clear_pairs', 'label_release_v1')
    ]+['scripts/release_ocr_control_v1.py']
FALSE_FLAGS = ('human_verified', 'OCR_run', 'training_started', 'semantic_family_approved', 'independent_evaluation')
PROTOCOL_FIELDS = {'schema_version', 'task', 'created_before_line_review_utc', 'selection', 'selection_outcomes',
                   'dependencies', 'sources', 'scope', *FALSE_FLAGS}
SOURCE_FIELDS = {'id', 'input_path', 'input_sha256', 'preparation', 'source_size_px', 'split', 'parent_regions', 'relationship_review'}
RELATION_FIELDS = {'decision', 'evidence', 'checked_related_source_paths', 'unresolved_cross_split_paths', 'notes'}
ASSIGNMENT_FIELDS = ('id', 'source_path', 'source_sha256', 'image_sha256', 'split')


def same(left, right):
    return stable_digest(left) == stable_digest(right)


def _timestamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None and parsed.utcoffset() is not None, 'Timestamp must include a timezone')
    return parsed


def _relationship_packet(root, record, protocol, selected_at):
    """Check authored review scope; hashes alone do not constitute admission."""
    value = read(checked_pin(root, record))
    fields = {'schema_version', 'selection', 'probes', 'reviewer', 'human_verified',
              'ocr_run', 'reviewed_at_utc', 'sources'}
    require(isinstance(value, dict) and fields <= set(value) <= fields | {'current_date'}
            and value['schema_version'] == 'ocr-control-relations/2.0'
            and value['selection'] == protocol['selection'] and value['reviewer'] == 'assistant'
            and value['human_verified'] is False and value['ocr_run'] is False,
            'Relationship evidence must be an explicit source review')
    require(selected_at <= _timestamp(value['reviewed_at_utc']) <=
            _timestamp(protocol['created_before_line_review_utc']), 'Relationship review is outside the selection/line-protocol interval')
    checked_pin(root, value['probes'])
    require(isinstance(value['sources'], list) and value['sources'], 'Missing relationship review sources')
    sources = {}
    for source in value['sources']:
        require(isinstance(source, dict) and {'id', 'input_path', 'input_sha256', 'title_image', 'decision'} <= set(source)
                and source['input_path'] not in sources and source['decision'] in
                ('known_train_link', 'needs_further_review', 'no_known_cross_split_link_in_reviewed_scope'),
                'Invalid or repeated relationship source decision')
        sources[source['input_path']] = source
    return sources


def _selection(root, protocol):
    require(protocol['selection'] == SELECTION, 'Use the authenticated canonical source selection')
    value = read(checked_pin(root, protocol['selection']))
    require(value.get('schema_version') == 'ocr-control-selection/2.0'
            and all(value[key] is False for key in ('images_inspected_in_this_step_before_selection',
                'OCR_run', 'training_started', 'independent_evaluation')) and value['split_assignment'] is None,
            'Changed frozen selection scope')
    selected_at = _timestamp(value['selected_at_utc'])
    queue = value['queue']
    require(queue.get('schema_version') == 'ocr-control-queue/2.0' and isinstance(queue.get('entries'), list)
            and isinstance(queue.get('components'), list), 'Selection must contain its complete source queue and components')
    entries = {row['id']: row for row in queue['entries']}
    require(len(entries) == len(queue['entries']), 'Repeated selection identity')
    outcomes = read(checked_pin(root, protocol['selection_outcomes']))
    require(set(outcomes) == {'schema_version', 'selection', 'ordered_candidates', 'admitted_ids'} and
            outcomes['schema_version'] == 'ocr-control-selection-outcomes/1.0' and
            outcomes['selection'] == protocol['selection'], 'Invalid selection outcome registry')
    ordered = outcomes['ordered_candidates']
    require(isinstance(ordered, list) and ordered, 'Missing ranked selection outcomes')
    components = sorted((row for row in queue['components'] if row['rank'] is not None), key=lambda row: row['rank'])
    require(all(type(row['rank']) is int for row in components) and
            [row['rank'] for row in components] == list(range(len(components))) and
            all(not row['exclusion_reasons'] for row in components), 'Eligible component ranks or exclusions differ')
    require([(row['id'], row['rank']) for row in ordered] == [(row['representative_id'], row['rank']) for row in components],
            'Outcomes must cover every ranked component representative without substitution')
    ranks, accepted, stopped, reviews = [], [], False, {}
    for row in ordered:
        require(set(row) == {'id', 'rank', 'decision', 'reason', 'review'}, 'Unexpected selection outcome fields')
        require(row['id'] in entries and type(row['rank']) is int and row['rank'] >= 0 and
                isinstance(row['reason'], str) and bool(row['reason'].strip()), 'Invalid ranked selection outcome')
        candidate = entries[row['id']]
        require(candidate.get('rank') == row['rank'], 'Outcome rank differs from frozen selection')
        require(row['decision'] in ('accepted_for_control', 'blocked_relation', 'not_reviewed'), 'Unsupported source disposition')
        if row['decision'] == 'not_reviewed':
            require(row['review'] is None, 'Unreviewed candidate cannot supply review evidence')
            stopped = True
        else:
            require(not stopped, 'Do not skip earlier unreviewed candidates')
            review_key = (row['review']['path'], row['review']['sha256'])
            if review_key not in reviews:
                reviews[review_key] = _relationship_packet(root, row['review'], protocol, selected_at)
            authored = reviews[review_key].get(candidate['input_path'])
            require(authored is not None and all(authored[key] == candidate[key] for key in
                    ('id', 'input_path', 'input_sha256')), 'Outcome review refers to a different candidate')
            checked_pin(root, authored['title_image'])
            allowed = ('no_known_cross_split_link_in_reviewed_scope',) if row['decision'] == 'accepted_for_control' else ('known_train_link', 'needs_further_review')
            require(authored['decision'] in allowed, 'Outcome contradicts the authored relationship decision')
        if row['decision'] == 'accepted_for_control':
            accepted.append(row['id'])
        ranks.append(row['rank'])
    require(ranks == sorted(set(ranks)) and len({r['id'] for r in ordered}) == len(ordered), 'Ranked outcomes must be unique and ordered')
    require(len(accepted) == 6 and outcomes['admitted_ids'] == accepted, 'Exactly the first six accepted candidates must be admitted')
    require([row['id'] for row in protocol['sources']] == accepted, 'Protocol source order differs from accepted candidates')
    for index, row in enumerate(protocol['sources']):
        candidate = entries[row['id']]
        require(all(same(row[key], candidate[key]) for key in ('input_path', 'input_sha256', 'preparation')),
                'Selected source revision differs')
        require(row['split'] == ('validation' if index % 2 == 0 else 'test'), 'Holdout split must alternate by accepted source order')
    return queue, outcomes


def _base(workspace, protocol_pin):
    root = Path(workspace).resolve()
    protocol = read(checked_pin(root, protocol_pin))
    require(set(protocol) == PROTOCOL_FIELDS and protocol['schema_version'] == 'ocr-control-lines-protocol/1.0'
            and protocol['task'] == 'P14.1b.7' and protocol['scope'] == SCOPE
            and all(type(protocol[key]) is bool and protocol[key] is False for key in FALSE_FLAGS), 'Changed control protocol scope')
    _timestamp(protocol['created_before_line_review_utc'])
    require(set(protocol['dependencies']) == DEPENDENCIES and protocol['dependencies']['parent_final'] == PARENT
            and protocol['dependencies']['previous_ocr_manifest'] == PREVIOUS, 'Changed parent release dependencies')
    values = {key: read(checked_pin(root, record)) for key, record in protocol['dependencies'].items()}
    require(values['parent_final']['release'] == PREVIOUS, 'Parent final proof identifies another OCR release')
    previous = load_ocr_training_v3(root, PREVIOUS)
    previous_dir = checked_pin(root, PREVIOUS).parent
    old_labels, old_registry = read(previous_dir/'labels.json'), read(previous_dir/'task-splits.json')
    require(previous['count'] == len(previous['records']) == 90 and len(old_registry['assignments']) == 90,
            'The complete previous 90-line registry is required')
    overlay = values['overlay']
    rebuilt = build_coallocation_v1(values['catalog'], values['policy'], values['exclusions'], overlay['constraints'])
    require(same(rebuilt, overlay), 'Control overlay differs from frozen dependencies')
    old_overlay = read(checked_pin(root, old_registry['overlay']))
    constraints = {row['id']: row for row in overlay['constraints']}
    require(all(row['id'] in constraints and same(row, constraints[row['id']]) for row in old_overlay['constraints']),
            'Previous coallocation constraints must be preserved')
    for constraint in overlay['constraints']:
        for evidence in constraint['evidence']:
            checked_pin(root, evidence)
    require(isinstance(protocol['sources'], list) and len(protocol['sources']) == 6 and
            len({row['id'].casefold() for row in protocol['sources']}) == 6 and
            len({row['input_path'].casefold() for row in protocol['sources']}) == 6, 'Expected six distinct control sources')
    queue, outcomes = _selection(root, protocol)
    for name, value in [('catalog', values['catalog']), ('policy', values['policy']), ('exclusions', values['exclusions']),
                        ('overlay', old_overlay), ('task_splits', old_registry)]:
        require(queue['input_content_sha256'][name] == stable_digest(value), 'Selection input snapshot differs from release context')
    require(same(queue['inherited_train_commitments'], old_registry['train_commitments']), 'Selection changed inherited train commitments')
    prepared = {row['input_path']: row for row in values['prepared_catalog']['entries']}
    require(len(prepared) == len(values['prepared_catalog']['entries']), 'Repeated prepared source')
    files = {row['path']: row for row in values['catalog']['files']}
    groups = {row['group_id']: row for row in overlay['groups']}
    old_groups = {row['group_id']: row for row in old_overlay['groups']}
    old_committed = set()
    for commitment in old_registry['train_commitments']:
        require(commitment['split'] == 'train' and commitment['member_paths'] == old_groups[commitment['coallocation_group_id']]['members'],
                'Parent train commitment is incomplete')
        old_committed.update(commitment['member_paths'])
    train_groups = {overlay['file_to_group'][path] for path in old_committed}
    require(len({overlay['file_to_group'][row['input_path']] for row in protocol['sources']}) == 6,
            'Six distinct final control groups are required')
    selected_at = _timestamp(read(checked_pin(root, protocol['selection']))['selected_at_utc'])
    outcome_reviews = {row['id']: row['review'] for row in outcomes['ordered_candidates'] if row['decision'] == 'accepted_for_control'}
    for source in protocol['sources']:
        require(set(source) == SOURCE_FIELDS, 'Unexpected control source fields')
        path = source['input_path']; relation = source['relationship_review']
        require(path in files and files[path]['kind'] == 'image' and files[path]['sha256'] == source['input_sha256'], 'Control source differs from catalog')
        group_id = overlay['file_to_group'][path]
        require(group_id not in train_groups and source['split'] in groups[group_id]['routing_allowed_splits'], 'Control overlaps inherited train or split restriction')
        component = next(row for row in queue['components'] if row['representative_id'] == source['id'])
        require(set(component['member_paths']) <= set(groups[group_id]['members']), 'Reserved overlay must include the entire selected filename-hint component')
        require(same(source['preparation'], prepared[path]['preparation']) and
                source['source_size_px'] == prepared[path]['source_size_px'] and
                source['input_sha256'] == prepared[path]['input_sha256'], 'Control preparation binding differs')
        prep = read(checked_pin(root, source['preparation']))
        require(prep['source_sha256'] == source['input_sha256'] and prep['source_size_px'] == source['source_size_px']
                and 'rgba_conversion' not in prep and 'container_repair' not in prep, 'Unexpected control preparation or conversion')
        checked_pin(root, {'path': 'dataset/'+path, 'sha256': source['input_sha256']})
        require(set(source['parent_regions']) == {'title', 'context'}, 'Control requires both fixed parent regions')
        for parent in source['parent_regions'].values():
            require(set(parent) == {'image', 'source_box_xyxy'}, 'Unexpected parent region fields')
            checked_pin(root, parent['image'])
        require(set(relation) == RELATION_FIELDS and relation['decision'] == 'no_known_cross_split_link_in_reviewed_scope'
                and relation['unresolved_cross_split_paths'] == [], 'Unresolved concrete cross-split relation blocks admission')
        relatives = sorted(p for p in groups[group_id]['members'] if files[p]['kind'] == 'image' and Path(p).suffix.lower() == '.png')
        require(relation['checked_related_source_paths'] == relatives, 'Review must cover every PNG in the complete reserved group')
        require(isinstance(relation['notes'], list) and relation['notes'] and
                all(isinstance(note, str) and note.strip() for note in relation['notes']), 'Missing relationship scope and limitations')
        require(isinstance(relation['evidence'], list) and relation['evidence'], 'Missing relationship review evidence')
        require(outcome_reviews[source['id']] in relation['evidence'], 'Accepted outcome review must remain in relationship evidence')
        reviewed_paths = set()
        for record in relation['evidence']:
            reviewed = _relationship_packet(root, record, protocol, selected_at)
            for relative in set(relatives).intersection(reviewed):
                authored = reviewed[relative]
                require(authored['input_sha256'] == files[relative]['sha256'], 'Related source review revision differs')
                checked_pin(root, {'path': 'dataset/'+relative, 'sha256': authored['input_sha256']})
                checked_pin(root, authored['title_image'])
                require(authored['decision'] == 'no_known_cross_split_link_in_reviewed_scope',
                        'Known or unresolved relationship evidence cannot approve control')
                reviewed_paths.add(relative)
        require(reviewed_paths == set(relatives), 'Evidence does not cover every related PNG')
    return protocol, values, previous, old_labels, old_registry, queue, outcomes


def prepare_control_config_v1(workspace, protocol_pin, output_path):
    root = Path(workspace).resolve(); output = contained(root, output_path)
    require(output.parent == root/'configs/reviews' and not output.exists(), 'Use a new configs/reviews file')
    _base(root, protocol_pin)
    config = {'schema_version': 'ocr-control-release-config/1.0', 'task': 'ocr_recognition', 'protocol': protocol_pin,
              'readings': [pin(root, path) for path in READINGS], 'code': [pin(root, path) for path in CODE]}
    output.write_bytes(encoded(config))
    return pin(root, output.relative_to(root).as_posix())


def _verify_inputs(workspace, config_pin):
    root = Path(workspace).resolve(); config = read(checked_pin(root, config_pin))
    require(set(config) == {'schema_version', 'task', 'protocol', 'readings', 'code'} and
            config['schema_version'] == 'ocr-control-release-config/1.0' and config['task'] == 'ocr_recognition', 'Wrong control release config')
    require(len(config['code']) == len(CODE) and {r['path'] for r in config['code']} == set(CODE), 'Missing control code pins')
    for record in config['code']:
        checked_pin(root, record)
    protocol, values, previous, old_labels, old_registry, queue, outcomes = _base(root, config['protocol'])
    require([row['path'] for row in config['readings']] == READINGS, 'Reading inventory or order changed')
    bindings = {row['id']: row for row in protocol['sources']}
    line_ids = {row['id'].casefold() for row in previous['records']}; sources, reviews, annotations = [], [], {}
    for batch, record in enumerate(config['readings']):
        review = ReviewV3.model_validate(read(checked_pin(root, record))).model_dump(mode='json')
        require(review['protocol'] == config['protocol'] and _timestamp(review['reviewed_at_utc']) >=
                _timestamp(protocol['created_before_line_review_utc']), 'Line review predates or changes protocol')
        expected = protocol['sources'][batch*3:(batch+1)*3]
        require(len(review['sources']) == len(expected) and {row['id'] for row in review['sources']} == {row['id'] for row in expected}, 'Every fixed source needs a reading outcome')
        for row in review['sources']:
            binding = bindings[row['id']]
            require(all(same(row[key], binding[key]) for key in ('input_path', 'input_sha256', 'preparation', 'source_size_px')), 'Line review changes source identity')
            for line in row['lines']:
                require(line['id'].casefold() not in line_ids, 'Repeated or case-aliased sample identity')
                line_ids.add(line['id'].casefold())
                owner = (row['id'], line['parent_kind'])
                require(line['annotation_id'] not in annotations or annotations[line['annotation_id']] == owner, 'Annotation reused across sources or parent regions')
                annotations[line['annotation_id']] = owner
                box = line['source_box_xyxy']; parent = binding['parent_regions'][line['parent_kind']]['source_box_xyxy']
                require(parent[0] <= box[0] < box[2] <= parent[2] and parent[1] <= box[1] < box[3] <= parent[3], 'Line lies outside its fixed parent region')
                for key in ('raw_crop_pin', 'image_pin'):
                    if line[key] is not None:
                        checked_pin(root, line[key])
            require(any(line['decision'] == 'accepted_for_ocr_transcription' for line in row['lines']), 'Each admitted source needs at least one reviewed line')
            sources.append({**row, 'review': record})
        reviews.append({'pin': record, 'data': review})
    return {'config': config, 'protocol': protocol, 'values': values, 'previous': previous, 'old_labels': old_labels,
            'old_registry': old_registry, 'queue': queue, 'outcomes': outcomes,
            'sources': sorted(sources, key=lambda row: row['id']), 'reviews': reviews}


def _documents(workspace, config_pin, verified):
    root = Path(workspace).resolve(); protocol = verified['protocol']; values = verified['values']
    bindings = {row['id']: row for row in protocol['sources']}
    overlay = values['overlay']; groups = {row['group_id']: row for row in overlay['groups']}
    files = {row['path']: row for row in values['catalog']['files']}
    images, samples, unresolved, evidence = {}, [], [], []
    for source in verified['sources']:
        binding = bindings[source['id']]
        boxes = {kind: parent['source_box_xyxy'] for kind, parent in binding['parent_regions'].items()}
        original = source_regions_v1(root/'dataset'/source['input_path'], source['input_sha256'], boxes)
        prep_dir = checked_pin(root, source['preparation']).parent
        parents = {}
        for kind, box in boxes.items():
            pixels, provenance = read_prepared_region(prep_dir, box)
            require(np.array_equal(pixels, original[kind]) and np.array_equal(pixels,
                    _pixels(root, binding['parent_regions'][kind]['image'], [box[2]-box[0], box[3]-box[1]])), 'Original/prepared/reviewed parent pixels differ')
            parents[kind] = {**binding['parent_regions'][kind], 'tile_provenance': provenance, 'original_png_stream_replay_equal': True}
        for line in source['lines']:
            box = line['source_box_xyxy']; parent = boxes[line['parent_kind']]
            raw = original[line['parent_kind']][box[1]-parent[1]:box[3]-parent[1], box[0]-parent[0]:box[2]-parent[0]]
            transform = make_ocr_transform_v1(box, line['rotation_cw_degrees'])
            oriented = np.rot90(raw, -(line['rotation_cw_degrees']//90)).copy()
            for key, pixels in [('raw_crop_pin', raw), ('image_pin', oriented)]:
                if line[key] is not None:
                    require(np.array_equal(pixels, _pixels(root, line[key], [pixels.shape[1], pixels.shape[0]])), 'Reviewed line crop or rotation differs')
            if line['decision'] == 'unresolved':
                unresolved.append({**line, 'source_id': source['id'], 'review': source['review'], 'reserved_split': binding['split']})
                continue
            yy, xx = np.where(np.any(oriented < 200, axis=2))
            require(len(xx) > 0, 'Accepted control line is blank')
            ink = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
            margins = [ink[0], ink[1], oriented.shape[1]-ink[2], oriented.shape[0]-ink[3]]
            require(min(margins) > 0, 'Accepted control text touches crop boundary')
            samples.append({**line, 'source_id': source['id'], 'review': source['review'], **transform,
                'split': binding['split'], 'white_margins_px_ltrb': margins, 'ink_box_image_xyxy': ink,
                'diagnostic_ink_threshold_exclusive': 200, 'physical_geometry_approved': False,
                'semantic_family_approved': False, 'independent_evaluation': False})
            images['images/'+line['id']+'.png'] = checked_pin(root, line['image_pin']).read_bytes()
        evidence.append({'source_id': source['id'], 'source': {'path': 'dataset/'+source['input_path'], 'sha256': source['input_sha256']},
                         'preparation': source['preparation'], 'parents': parents, 'relationship_review': binding['relationship_review']})
    samples.sort(key=lambda row: row['id']); unresolved.sort(key=lambda row: row['id'])
    rows = {split: [] for split in ('train', 'validation', 'test')}
    parent_dir = checked_pin(root, protocol['dependencies']['previous_ocr_manifest']).parent
    parent_rows = verified['previous']['records']; old_registry = verified['old_registry']
    old_assignments = {row['id']: row for row in old_registry['assignments']}
    require(set(old_assignments) == {row['id'] for row in parent_rows}, 'Parent sample registry is incomplete')
    assignments = []
    for row in parent_rows:
        require(row['split'] == 'train' and row['task'] == 'ocr_recognition', 'Inherited task or split changed')
        copied = {key: value for key, value in row.items() if key != 'image_path'}
        assignment = old_assignments[row['id']]
        require(assignment['split'] == 'train' and assignment['source_sha256'] == row['source_sha256']
                and assignment['image_sha256'] == row['image_sha256'] and assignment['base_group_id'] == row['group_id'], 'Inherited binding changed')
        image_path = contained(parent_dir, row['image'])
        require(digest(image_path) == row['image_sha256'], 'Inherited image changed')
        rows['train'].append(copied); images[row['image']] = image_path.read_bytes()
        assignments.append({key: assignment[key] for key in ASSIGNMENT_FIELDS})
    for sample in samples:
        source = bindings[sample['source_id']]; split = source['split']; path = source['input_path']
        rows[split].append({'id': sample['id'], 'source_id': sample['source_id'], 'image': 'images/'+sample['id']+'.png',
            'image_sha256': sample['image_pin']['sha256'], 'text': sample['text'], 'source_sha256': source['input_sha256'],
            'group_id': files[path]['group_id'], 'split': split, 'task': 'ocr_recognition'})
        assignments.append({'id': sample['id'], 'source_path': path, 'source_sha256': source['input_sha256'],
                            'image_sha256': sample['image_pin']['sha256'], 'split': split})
    all_rows = [row for part in rows.values() for row in part]
    require(len(rows['train']) == 90 and len({row['id'].casefold() for row in all_rows}) == len(all_rows), 'Inherited inventory or sample identities changed')
    by_hash = defaultdict(list)
    for row in all_rows:
        by_hash[row['image_sha256']].append(row)
    require(all(len({row['text'] for row in same_image}) == 1 for same_image in by_hash.values()), 'Identical images have conflicting texts')
    reservations = {}
    for commitment in old_registry['train_commitments']:
        require(commitment['split'] == 'train', 'Changed inherited commitment split')
        for path in commitment['member_paths']:
            reservations[overlay['file_to_group'][path]] = 'train'
    for source in protocol['sources']:
        identifier = overlay['file_to_group'][source['input_path']]
        require(identifier not in reservations or reservations[identifier] == source['split'], 'Whole-group reservation crosses train/validation/test')
        reservations[identifier] = source['split']
    commitments = [{'coallocation_group_id': identifier, 'split': split, 'member_paths': groups[identifier]['members'],
                    'reason': 'reserve_complete_related_group_for_pinned_ocr_split', 'semantic_family_approved': False,
                    'independent_evaluation': False} for identifier, split in sorted(reservations.items())]
    reserve_rows = [{'id': 'reservation-'+stable_digest([row['coallocation_group_id'], path])[:32],
                     'source_path': path, 'source_sha256': files[path]['sha256'], 'split': row['split']}
                    for row in commitments for path in row['member_paths']]
    combined = validate_assignments_v1(values['catalog'], values['policy'], values['exclusions'], overlay, assignments+reserve_rows)
    routing = validate_assignments_v1(values['catalog'], values['policy'], values['exclusions'], overlay, assignments)
    require(combined['assignment_count'] == len(assignments)+len(reserve_rows), 'Missing reservation rows')
    # Preserve the exact parent JSONL bytes; never reserialize inherited targets.
    parent_train = (parent_dir/'train.jsonl').read_bytes()
    require([json.loads(line) for line in parent_train.splitlines()] == rows['train'], 'Inherited train byte stream differs from parent consumer')
    labels = {'schema_version': 'ocr-control-labels/1.0', 'task': 'ocr_recognition', 'scope': SCOPE,
        'parent_release': protocol['dependencies']['previous_ocr_manifest'], 'inherited_labels': deepcopy(verified['old_labels']),
        'control_source_records': deepcopy(protocol['sources']), 'control_samples': samples, 'unresolved_candidates': unresolved,
        'physical_geometry_approved': False, 'native_target_approved': False, 'human_verified': False,
        'semantic_family_approved': False, 'independent_evaluation': False, 'training_started': False}
    registry = {'schema_version': 'ocr-control-task-splits/1.0', 'task': 'ocr_recognition', 'scope': SCOPE,
        'parent_registry': deepcopy(old_registry), 'overlay': protocol['dependencies']['overlay'],
        'assignments': routing['assignments'], 'group_commitments': commitments,
        'splits': {split: [row['id'] for row in part] for split, part in rows.items()},
        'global_split_policy_modified': False, 'independent_evaluation': False}
    report = {'schema_version': 'ocr-control-release/1.0', 'task': 'P14.1b.7', 'config': config_pin,
        'scope': SCOPE, 'status': 'released_for_internal_grouped_ocr_control', 'inherited_sample_count': 90,
        'control_source_count': 6, 'control_sample_count': len(samples), 'sample_count': len(all_rows),
        'unresolved_candidate_count': len(unresolved), 'candidate_count': len(samples)+len(unresolved),
        'task_split_counts': {split: len(part) for split, part in rows.items()},
        'control_source_split_counts': dict(Counter(row['split'] for row in protocol['sources'])),
        'coallocation_group_count': len(commitments), 'committed_member_count': len(reserve_rows),
        'committed_members_by_split': {split: sum(len(row['member_paths']) for row in commitments if row['split'] == split) for split in rows},
        'inherited_train_rows_images_and_labels_unchanged': True, 'full_registry_and_reservations_validated': True,
        'source_modified': False, 'global_split_policy_modified': False, 'training_started': False, 'OCR_run': False,
        'human_verified': False, 'semantic_family_approved': False, 'independent_evaluation': False,
        'physical_geometry_approved': False, 'native_target_approved': False,
        'limitations': ['Known whole-group, exact-content, filename and reviewed relationship constraints are enforced together.',
            'Unobserved external family links remain possible; this internal grouped control is not an independent external benchmark.',
            'Text is assistant-reviewed using explicit glyph conventions; original font code points are unknown.',
            'Unresolved candidates retain evidence without OCR targets. Crops do not approve detection, geometry or CAD labels.',
            'The complete inherited train and whole-group commitments must accompany every later release.',
            'Test is reserved for final evaluation, without parameter selection; this release invokes no OCR or training.']}
    documents = {'labels.json': labels, 'task-splits.json': registry, 'routing.json': routing,
        'reservation-routing.json': combined, 'report.json': report, 'evidence.json': evidence,
        'readings.json': verified['reviews'], 'selection-outcomes.json': verified['outcomes'],
        'review-schema.json': ReviewV3.model_json_schema()}
    output = {**{name: encoded(value) for name, value in documents.items()}, **images, 'train.jsonl': parent_train}
    for split in ('validation', 'test'):
        output[split+'.jsonl'] = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n' for row in rows[split]).encode('utf-8')
    html = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Контроль OCR</title>',
        '<style>body{font:17px system-ui;max-width:1100px;margin:2em auto}section{margin:2em 0}img{max-width:100%}</style>',
        '<h1>OCR: отдельный контроль по известным группам</h1><p>90 прежних train-строк сохранены. Внешняя независимость не утверждается. OCR и обучение не запускались.</p>',
        '<p><a href="report.json">Отчёт</a> · <a href="task-splits.json">Полный реестр</a> · <a href="labels.json">Метки и ограничения</a></p>']
    for split, part in rows.items():
        html.append('<h2>'+escape(split)+'</h2>')
        for row in part:
            html.append('<section><p>'+escape(row['source_id']+' / '+row['id'])+' — <b>'+escape(row['text'])+'</b></p><img loading="lazy" src="'+escape(row['image'], quote=True)+'" alt="'+escape(row['text'], quote=True)+'"></section>')
    for row in unresolved:
        html.append('<section><p>Неопределённо: '+escape(row['source_id']+' / '+row['id'])+' — '+escape(row['reason'])+'</p></section>')
    output['index.html'] = (''.join(html)+'</html>\n').encode('utf-8')
    return output, report, rows


def run_control_release_v1(workspace, config_pin, output_dir):
    root = Path(workspace).resolve(); output = contained(root, output_dir)
    require(output.parent == root/'artifacts' and not output.exists(), 'Use a new direct artifacts directory')
    files, report, _ = _documents(root, config_pin, _verify_inputs(root, config_pin))
    _verify_inputs(root, config_pin)
    output.mkdir()
    for name, data in files.items():
        path = contained(output, name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    manifest = {'schema_version': 'ocr-control-release-manifest/1.0', 'task': 'ocr_recognition', 'config': config_pin,
                'files': [pin(output, name) for name in sorted(files)]}
    (output/'manifest.json').write_bytes(encoded(manifest))
    return report


def load_ocr_control_v1(workspace, manifest_pin, *, split, task='ocr_recognition'):
    require(task == 'ocr_recognition' and split in ('train', 'validation', 'test'), 'Only explicit OCR train/validation/test requests are supported')
    root = Path(workspace).resolve(); path, manifest = _package(root, manifest_pin)
    require(set(manifest) == {'schema_version', 'task', 'config', 'files'} and
            manifest['schema_version'] == 'ocr-control-release-manifest/1.0' and manifest['task'] == task, 'Wrong control manifest')
    files, report, rows = _documents(root, manifest['config'], _verify_inputs(root, manifest['config']))
    require(len(manifest['files']) == len(files) and {row['path'] for row in manifest['files']} == set(files), 'Control package inventory differs')
    for record in manifest['files']:
        require(checked_pin(path.parent, record).read_bytes() == files[record['path']], 'Control package differs from authenticated recipe')
    _verify_inputs(root, manifest['config'])
    return {'task': task, 'split': split, 'count': len(rows[split]),
        'records': [{**row, 'image_path': str(contained(path.parent, row['image']))} for row in rows[split]],
        'training_started': False, 'scope': SCOPE, 'independent_evaluation': False, 'report': report}
