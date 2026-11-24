"""Complete prospective OCR queue with conservative whole-group hint closure.

Pure metadata logic: input snapshots and actual bytes must be authenticated by
the caller. Proposals do not approve labels, families or independent evaluation.
"""
from collections import Counter, defaultdict
from copy import deepcopy

from .coallocation_policy_v1 import build_coallocation_v1, validate_assignments_v1
from .family_catalog_v2 import canonical_sha, relative_path, require, stable_digest
from .ocr_control_selection_v1 import build_control_queue_v1, filename_hints_v1


PROTOCOL_V2 = {
    'id': 'ocr-control-source-selection/2.0',
    'scope': 'complete_added_PNG_queue_inherited_from_v1',
    'closure': 'whole_overlay_groups_plus_all_catalog_image_pdf_filename_hints_transitively',
    'filename_hints': 'frozen_v1_casefold_NFC_numeric_root_Bolti_series_MTSP_year_dotted_project_and_words',
    'hard_exclusions': ['development_train_commitments', 'historical_reservations', 'train_only',
                        'blocked_routing', 'previous_source_review'],
    'hint_effect': 'conservative_coallocation_and_exclusion_not_family_proof',
    'component_representative': 'first_input_sha256_then_input_path',
    'component_order': 'representative_input_sha256_then_input_path_then_component_id',
    'initial_wave_size': 6,
    'proposed_split_by_zero_based_rank_parity': {'even': 'validation', 'odd': 'test'},
    'future_waves': 'explicit_continuation_of_frozen_order_only',
    'quality_based_replacement_after_visual_inspection': False,
    'relation_blocked_continuation': 'retain_each_refusal_then_continue_frozen_order_without_skipping_unreviewed_candidates',
    'final_admission': 'first_six_relationship_accepted_representatives_in_frozen_order',
    'final_split_by_accepted_index_parity': {'even': 'validation', 'odd': 'test'},
    'final_split_freeze': 'outcome_registry_and_assignments_pinned_before_line_annotation_and_OCR',
    'filter_by_native_exclusion_or_visual_quality': False,
    'OCR_or_baseline_error_based_selection': False,
    'cross_relation_review': 'concrete_unresolved_or_confirmed_cross_split_relation_blocks_admission',
    'remaining_status': 'pending_relationship_and_transcription_review',
    'split_assignment': None, 'OCR_run': False, 'training': False, 'independent_evaluation': False,
}


def _same(left, right):
    return stable_digest(left) == stable_digest(right)


def _prior_sources(rows, files, old_queue):
    require(isinstance(rows, list), 'Expected prior source review inventory')
    result = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'path', 'sha256'}, 'Unexpected prior source review binding')
        path, sha = relative_path(row['path']), canonical_sha(row['sha256'])
        require(path not in result and path in files and files[path]['kind'] in ('image', 'pdf')
                and files[path]['sha256'] == sha, 'Prior source review identity or revision differs')
        result[path] = sha
    require(set(old_queue['selected_source_paths']) <= set(result), 'Prior review inventory omits V1 selected sources')
    return result


def _commitments(task_splits, catalog, policy, exclusions, overlay):
    require(isinstance(task_splits, dict) and set(task_splits) == {'assignments', 'global_split_policy_modified',
        'independent_evaluation', 'inherited_sample_ids', 'overlay', 'schema_version', 'scope', 'splits', 'task',
        'train_commitments'}, 'Unexpected parent OCR split fields')
    require(task_splits['schema_version'] == 'ocr-task-splits/3.0' and task_splits['task'] == 'ocr_recognition'
        and task_splits['global_split_policy_modified'] is False and task_splits['independent_evaluation'] is False,
        'Expected unmodified parent OCR train registry')
    require(isinstance(task_splits['overlay'], dict) and set(task_splits['overlay']) == {'path', 'sha256'}, 'Missing parent overlay pin')
    relative_path(task_splits['overlay']['path']); canonical_sha(task_splits['overlay']['sha256'])
    assignments = task_splits['assignments']
    require(isinstance(assignments, list) and assignments, 'Missing inherited OCR assignments')
    fields = {'id', 'source_path', 'source_sha256', 'image_sha256', 'split', 'base_group_id', 'coallocation_group_id'}
    require(all(isinstance(row, dict) and set(row) == fields and row['split'] == 'train' for row in assignments),
            'Unexpected inherited OCR assignment scope')
    core = [{key: row[key] for key in fields-{'base_group_id', 'coallocation_group_id'}} for row in assignments]
    verified = validate_assignments_v1(catalog, policy, exclusions, overlay, core)
    require(_same(sorted(assignments, key=lambda row: row['id']), verified['assignments']), 'Parent OCR routing differs')
    ids = {row['id'] for row in assignments}
    splits = task_splits['splits']
    require(isinstance(splits, dict) and set(splits) == {'train', 'validation', 'test'}
        and splits['validation'] == [] and splits['test'] == [] and isinstance(splits['train'], list)
        and len(splits['train']) == len(ids) and set(splits['train']) == ids, 'Parent split inventory differs')
    inherited = task_splits['inherited_sample_ids']
    require(isinstance(inherited, list) and len(inherited) == len(set(inherited)) and set(inherited) <= ids,
            'Inherited sample registry differs')
    groups = {row['group_id']: row for row in overlay['groups']}
    commitments = task_splits['train_commitments']
    require(isinstance(commitments, list), 'Missing train commitments')
    by_id = {}
    for row in commitments:
        require(isinstance(row, dict) and set(row) == {'coallocation_group_id', 'member_paths', 'reason',
                'semantic_family_approved', 'independent_evaluation', 'split'}, 'Unexpected train commitment fields')
        gid = row['coallocation_group_id']
        require(gid not in by_id and gid in groups and row['split'] == 'train'
                and row['semantic_family_approved'] is False and row['independent_evaluation'] is False
                and isinstance(row['reason'], str) and row['reason'].strip()
                and _same(row['member_paths'], groups[gid]['members']), 'Train commitment omits or changes whole group')
        by_id[gid] = row
    require(set(by_id) == {row['coallocation_group_id'] for row in assignments}, 'Train commitments differ from full registry')
    return by_id


def build_control_queue_v2(old_intake, new_intake, catalog, policy, handoff, exclusions,
                           overlay, task_splits, prior_reviewed_sources, protocol=PROTOCOL_V2):
    """Freeze the complete ordered component queue before viewing new source pixels.

    Every old train commitment survives, all source/dxf members remain together,
    and filename hints propagate through entire overlay groups. The first six
    components form the initial review wave; all later ranks are retained.
    No proposed split becomes an active assignment in this metadata-only step.
    """
    require(_same(protocol, PROTOCOL_V2), 'Selection protocol changed')
    queue = build_control_queue_v1(old_intake, new_intake, catalog, policy, handoff, exclusions)
    require(isinstance(overlay, dict) and isinstance(overlay.get('constraints'), list), 'Missing coallocation overlay')
    rebuilt = build_coallocation_v1(catalog, policy, exclusions, overlay['constraints'])
    require(_same(overlay, rebuilt), 'Coallocation overlay differs from frozen context')
    files = {row['path']: row for row in catalog['files']}
    groups = {row['group_id']: row for row in overlay['groups']}
    commitments = _commitments(task_splits, catalog, policy, exclusions, overlay)
    prior = _prior_sources(prior_reviewed_sources, files, queue)
    parents = {gid: gid for gid in groups}

    def find(gid):
        while parents[gid] != gid:
            parents[gid] = parents[parents[gid]]
            gid = parents[gid]
        return gid

    hint_groups, source_hints = defaultdict(set), {}
    for path, row in files.items():
        if row['kind'] in ('image', 'pdf'):
            hints = filename_hints_v1(path); source_hints[path] = hints
            for hint in hints:
                hint_groups[hint].add(overlay['file_to_group'][path])
    for identifiers in hint_groups.values():
        ordered = sorted(identifiers)
        for gid in ordered[1:]:
            left, right = find(ordered[0]), find(gid)
            if left != right:
                parents[max(left, right)] = min(left, right)
    memberships = defaultdict(list)
    for gid in groups:
        memberships[find(gid)].append(gid)
    candidates_by_component = defaultdict(list)
    for row in queue['entries']:
        candidates_by_component[find(overlay['file_to_group'][row['input_path']])].append(row)
    components, component_by_path = [], {}
    for representative, gids in memberships.items():
        gids.sort()
        members = sorted(path for gid in gids for path in groups[gid]['members'])
        sources = sorted(path for path in members if path in source_hints)
        reasons = []
        if set(gids) & set(commitments):
            reasons.append('development_train_commitment')
        if any(groups[gid]['train_only'] for gid in gids):
            reasons.append('train_only')
        reservations = sorted({split for gid in gids for split in groups[gid]['historical_reserved_splits']})
        if reservations:
            reasons.append('historical_reservation')
        if any(groups[gid]['status'] == 'blocked_restriction_conflict' or not groups[gid]['routing_allowed_splits'] for gid in gids):
            reasons.append('blocked_routing')
        reviewed = sorted(set(members) & set(prior))
        if reviewed:
            reasons.append('previous_source_review')
        candidates = sorted(candidates_by_component[representative], key=lambda row: (row['input_sha256'], row['input_path']))
        cid = 'ocr-hint-component-'+stable_digest(gids)[:24]
        row = {'id': cid, 'overlay_group_ids': gids, 'member_paths': members, 'source_paths': sources,
            'filename_hints': sorted({hint for path in sources for hint in source_hints[path]}),
            'candidate_ids': [r['id'] for r in candidates], 'representative_id': candidates[0]['id'] if candidates else None,
            'rank': None, 'proposed_split': None, 'exclusion_reasons': reasons,
            'train_commitment_group_ids': sorted(set(gids) & set(commitments)),
            'reviewed_source_paths': reviewed, 'historical_reserved_splits': reservations,
            'split': None, 'independent_evaluation': False}
        components.append(row)
        component_by_path.update({path: row for path in members})
    components.sort(key=lambda row: row['id'])
    require(len(component_by_path) == len(files) and len({row['id'] for row in components}) == len(components),
            'Hint components do not form a complete unique partition')
    entries_by_id = {row['id']: row for row in queue['entries']}
    ordered = sorted((row for row in components if row['candidate_ids'] and not row['exclusion_reasons']),
        key=lambda row: (entries_by_id[row['representative_id']]['input_sha256'],
                         entries_by_id[row['representative_id']]['input_path'], row['id']))
    require(len(ordered) >= PROTOCOL_V2['initial_wave_size'], 'Insufficient eligible components for initial wave')
    for rank, row in enumerate(ordered):
        row['rank'] = rank
        row['proposed_split'] = 'validation' if rank % 2 == 0 else 'test'
    selected = ordered[:PROTOCOL_V2['initial_wave_size']]
    selected_ids = [row['representative_id'] for row in selected]
    entries = []
    for row in queue['entries']:
        component = component_by_path[row['input_path']]
        entries.append({**deepcopy(row), 'coallocation_group_id': overlay['file_to_group'][row['input_path']],
            'component_id': component['id'], 'eligibility': 'blocked' if component['exclusion_reasons'] else 'pending_review',
            'exclusion_reasons': list(component['exclusion_reasons']), 'rank': component['rank'],
            'proposed_split': component['proposed_split'], 'selected_for_visual_review': row['id'] in selected_ids,
            'selection_reason': 'initial_component_representative' if row['id'] in selected_ids else
                ('component_excluded' if component['exclusion_reasons'] else 'ordered_reserve_or_component_sibling')})
    counts = Counter(row['eligibility'] for row in entries)
    return {'schema_version': 'ocr-control-queue/2.0', 'protocol': deepcopy(PROTOCOL_V2),
        'status': 'prospective_review_queue_not_evaluation_release', 'entries': entries, 'components': components,
        'ordered_component_ids': [row['id'] for row in ordered], 'selected_components': deepcopy(selected),
        'selected_ids': selected_ids, 'selected_source_paths': [entries_by_id[identifier]['input_path'] for identifier in selected_ids],
        'prior_reviewed_sources': [{'path': path, 'sha256': prior[path]} for path in sorted(prior)],
        'inherited_train_commitments': deepcopy(task_splits['train_commitments']),
        'input_content_sha256': {key: stable_digest(value) for key, value in [('old_intake', old_intake),
            ('new_intake', new_intake), ('catalog', catalog), ('policy', policy), ('handoff', handoff),
            ('exclusions', exclusions), ('overlay', overlay), ('task_splits', task_splits)]},
        'summary': {'new_source_count': len(entries), 'file_count': len(files), 'overlay_group_count': len(groups),
            'component_count': len(components), 'eligible_source_count': counts['pending_review'], 'blocked_source_count': counts['blocked'],
            'eligible_component_count': len(ordered), 'selected_count': len(selected), 'reserve_component_count': len(ordered)-len(selected),
            'prior_reviewed_source_count': len(prior), 'train_commitment_group_count': len(commitments),
            'train_committed_member_count': sum(len(row['member_paths']) for row in commitments.values()),
            'active_split_counts': {'train': 0, 'validation': 0, 'test': 0}},
        'split': None, 'training_eligible': False, 'annotation_approved': False, 'semantic_family_approved': False,
        'independent_evaluation': False, 'OCR_run': False, 'source_modified': False, 'global_split_policy_modified': False,
        'limitations': ['Filename hints conservatively coallocate sources, not establish semantic families.',
            'No observed hint is not proof of independence; concrete unresolved relationships block later admission.',
            'Every proposed split needs source/transcription/relationship review before a later frozen release.',
            'Native target status and visual/OCR quality do not affect metadata selection.',
            'Every source byte and input snapshot must be authenticated separately by the caller.']}
