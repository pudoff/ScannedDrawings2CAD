"""Pending OCR control-source queue; filename hints never establish independence.

Pure inventory logic. Callers authenticate files, freeze PROTOCOL_V1 before
selection, and review the selected source pixels separately. No OCR is run.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import PurePosixPath
import re
import unicodedata

from .family_catalog_v2 import canonical_sha, relative_path, require, stable_digest
from .ocr_lineage_v1 import _checked_context


PROTOCOL_V1 = {
    'id': 'ocr-control-source-selection/1.0',
    'scope': 'all_added_PNG_sources_from_new_intake_minus_old_intake',
    'hard_exclusions': ['train_only', 'historical_train', 'exact_content_link_to_historical_train'],
    'remaining_status': 'pending_family_review',
    'strata': ['153', '157', '260620', '346', '626', '964'],
    'strata_meaning': 'filename_prefix_coverage_not_semantic_families',
    'per_stratum': 1,
    'ordering': ['input_sha256', 'input_path'],
    'missing_stratum': 'fail_without_substitution',
    'selected_group_collision': 'fail_without_substitution',
    'replacement_after_visual_inspection': False,
    'filter_by_native_exclusion_or_visual_quality': False,
    'OCR_or_baseline_error_based_selection': False,
    'filename_hints': 'casefold_NFC_numeric_root_Bolti_series_MTSP_year_dotted_project_and_words',
    'name_hint_effect': 'review_warning_only_not_family_or_split_approval',
    'native_exclusion_scope': 'native_DXF_and_derived_labels_not_original_PNG_review',
    'split_assignment': None,
    'training': False,
    'independent_evaluation': False,
}


def _stem(path):
    relative_path(path)
    return ' '.join(unicodedata.normalize('NFC', PurePosixPath(path).stem).casefold().split())


def _series(path):
    stem = _stem(path)
    match = re.match(r'^(\d{5,})(?=\D|$)', stem)
    if match:
        return 'numeric:'+match[1]
    match = re.match(r'^bolti\s*(\d+\s*-\s*\d+)', stem)
    if match:
        return 'bolti:'+re.sub(r'\s+', '', match[1])
    match = re.match(r'^мтсп\.(\d+)(?=\D|$)', stem)
    if match:
        return 'namespace:мтсп.'+match[1]
    match = re.match(r'^([^\W\d_]+\.\d+\.\d+)(?=\D|$)', stem)
    if match:
        return 'project:'+match[1]
    token = re.search(r'[^\s_]+', stem)
    return 'name:'+(token[0] if token else stem)


def filename_hints_v1(path):
    """Return retrieval hints only, including broad shared words and namespaces."""
    stem = _stem(path)
    words = {word for word in re.findall(r'[^\W\d_]{3,}', stem) if word not in {'нов', 'new'}}
    return sorted({'series:'+_series(path), *('word:'+word for word in words)})


def _intake_index(value, name):
    require(value['schema_version'] == '1.0' and value['purpose'] == 'training_intake_not_split'
            and value['training_eligible'] is False, 'Expected pending '+name+' intake')
    require(isinstance(value['items'], list), 'Expected intake item list')
    result = {row['input_path']: row for row in value['items']}
    require(len(result) == len(value['items']), 'Duplicate '+name+' input')
    for path, row in result.items():
        relative_path(path); canonical_sha(row['input_sha256'])
        require(PurePosixPath(path).suffix.lower() == '.png' and row['split'] is None
                and row['training_eligible'] is False and row['rehash_matches_manifest'] is True,
                'Intake source must remain a pending PNG')
    return result


def _identity(row):
    target = row['candidate_target']
    return (row['input_path'], row['input_sha256'], row['input_size_bytes'],
            None if target is None else (target['path'], target['sha256']))


def build_control_queue_v1(old_intake, new_intake, catalog, policy, handoff, exclusions, protocol=PROTOCOL_V1):
    """Select fixed review candidates, keeping every new source and refusal.

    Catalog checks include the complete file/group partition, historical
    reservations, candidate pairs, exact-content coallocation and native gates.
    The caller must also hash actual bytes; this pure function does not do I/O.
    """
    require(protocol == PROTOCOL_V1, 'Selection protocol changed')
    files, groups, pairs = _checked_context(catalog, policy, exclusions)
    old = _intake_index(old_intake, 'old')
    new = _intake_index(new_intake, 'new')
    require(set(old) <= set(new), 'Old intake sources were removed')
    require(all(_identity(old[path]) == _identity(new[path]) for path in old), 'Old intake identity changed')
    require(handoff['training_eligible'] is False and handoff['split_modified'] is False,
            'Handoff cannot grant training or split approval')
    handoffs = {row['input_path']: row for row in handoff['entries']}
    require(len(handoffs) == len(handoff['entries']) and set(new) <= set(handoffs), 'Handoff source inventory differs')
    pair_by_paths = {(row['input_path'], row['target_path']): row for row in pairs.values()}
    require(len(pair_by_paths) == len(pairs), 'Repeated candidate source/target pair')
    by_hash = defaultdict(list)
    source_hints, hint_sources = {}, defaultdict(set)
    train_files = set()
    for path, row in files.items():
        by_hash[row['sha256']].append(path)
        group = groups[row['group_id']]
        if group['train_only'] or group['historical_reserved_split'] == 'train':
            train_files.add(path)
        if row['kind'] in ('image', 'pdf'):
            hints = filename_hints_v1(path)
            source_hints[path] = hints
            for hint in hints:
                hint_sources[hint].add(path)
    entries = []
    excluded_hashes = {row['target_sha256'] for row in exclusions['exclusions']}
    for path in sorted(new):
        item, retained = new[path], handoffs[path]
        require(path in files and files[path]['kind'] == 'image' and files[path]['role'] == 'drawing'
                and files[path]['sha256'] == item['input_sha256']
                and files[path]['size_bytes'] == item['input_size_bytes'], 'Intake revision differs from catalog')
        require(retained['input_sha256'] == item['input_sha256'] and retained['split'] is None
                and retained['annotation_approved'] is False and retained['training_eligible'] is False,
                'Handoff identity or scope changed')
        pair = pair_by_paths.get((path, retained['target_path']))
        require(pair is not None and pair['target_sha256'] == retained['target_sha256'], 'Handoff target differs from catalog')
        if item['candidate_target'] is not None:
            target = item['candidate_target']
            require(target['path'] == retained['target_path'] and target['sha256'] == retained['target_sha256'],
                    'Intake target differs from retained handoff')
        if path in old:
            continue
        source = files[path]
        group = groups[source['group_id']]
        source_same_hash = sorted(p for p in by_hash[source['sha256']] if p != path)
        target_same_hash = sorted(p for p in by_hash[pair['target_sha256']] if p != pair['target_path'])
        exact_train = sorted(train_files.intersection(source_same_hash + target_same_hash))
        reasons = []
        if group['train_only']:
            reasons.append('train_only_constraint')
        if group['historical_reserved_split'] == 'train':
            reasons.append('historical_train_reservation')
        if exact_train:
            reasons.append('exact_content_link_to_historical_train')
        blocked = bool(reasons)
        if not blocked:
            reasons.append('semantic_family_and_independence_require_review')
        hints = source_hints[path]
        related = sorted(set().union(*(hint_sources[h] for h in hints))-{path})
        train_matches = [{'input_path': p, 'input_sha256': files[p]['sha256'], 'group_id': files[p]['group_id'],
                          'shared_hints': sorted(set(hints).intersection(source_hints[p]))}
                         for p in related if p in train_files]
        if train_matches:
            reasons.append('filename_hint_matches_train_requires_review')
        if source_same_hash:
            reasons.append('exact_source_duplicates_retained')
        target_status = 'excluded' if pair['target_sha256'] in excluded_hashes else 'not_approved'
        if target_status == 'excluded':
            reasons.append('native_target_excluded_no_native_labels')
        stem = _stem(path)
        stratum = next((prefix for prefix in PROTOCOL_V1['strata'] if stem.startswith(prefix)), 'other')
        entries.append({
            'id': 'ocr-control-'+stable_digest([path, source['sha256']])[:16], 'input_path': path, 'input_sha256': source['sha256'],
            'input_size_bytes': source['size_bytes'], 'target_path': pair['target_path'],
            'target_sha256': pair['target_sha256'], 'group_id': group['group_id'],
            'train_only': group['train_only'], 'historical_reserved_split': group['historical_reserved_split'],
            'known_family_ids': sorted(group['known_family_ids']), 'semantic_family_approved': False,
            'queue_status': 'blocked_history_train' if blocked else 'pending_family_review',
            'selection_stratum': stratum, 'series_hint': _series(path), 'filename_hints': hints,
            'related_source_paths': related, 'name_hint_train_matches': train_matches,
            'group_members': sorted(group['members']),
            'exact_hash_matches': {'source': source_same_hash, 'native_target': target_same_hash},
            'hard_train_evidence': {'train_only': group['train_only'],
                                    'historical_reserved_split': group['historical_reserved_split'],
                                    'exact_content_train_paths': exact_train},
            'native_target_status': target_status, 'native_target_approved': False,
            'handoff_status': retained['handoff_status'], 'preparation': deepcopy(retained['preparation']),
            'selected_for_visual_review': False, 'selection_reason': 'blocked_history_train' if blocked else
                ('outside_fixed_six_strata' if stratum == 'other' else 'later_in_stratum_order'),
            'reason_codes': sorted(reasons), 'split': None, 'annotation_approved': False,
            'training_eligible': False, 'independent_evaluation': False,
        })
    require(len({row['id'] for row in entries}) == len(entries), 'Queue identity collision')
    selected = []
    for stratum in PROTOCOL_V1['strata']:
        candidates = [row for row in entries if row['selection_stratum'] == stratum
                      and row['queue_status'] == 'pending_family_review']
        require(candidates, 'No pending candidate in fixed stratum '+stratum)
        candidate = min(candidates, key=lambda row: (row['input_sha256'], row['input_path']))
        require(candidate['group_id'] not in {row['group_id'] for row in selected},
                'Selected strata intersect one lineage group; protocol review required')
        candidate['selected_for_visual_review'] = True
        candidate['selection_reason'] = 'first_sha_path_in_fixed_filename_stratum'
        selected.append(candidate)
    statuses = Counter(row['queue_status'] for row in entries)
    summary = {
        'old_source_count': len(old), 'current_source_count': len(new), 'new_source_count': len(entries),
        'new_group_count': len({row['group_id'] for row in entries}),
        'queue_status_counts': dict(sorted(statuses.items())), 'selected_count': len(selected),
        'unselected_count': len(entries)-len(selected),
        'pending_unselected_count': statuses['pending_family_review']-len(selected),
        'name_hint_train_match_source_count': sum(bool(row['name_hint_train_matches']) for row in entries),
        'native_target_status_counts': dict(sorted(Counter(row['native_target_status'] for row in entries).items())),
        'pending_stratum_counts': {key: sum(row['queue_status'] == 'pending_family_review' and
                                          row['selection_stratum'] == key for row in entries)
                                   for key in [*PROTOCOL_V1['strata'], 'other']},
        'active_split_counts': {'train': 0, 'validation': 0, 'test': 0},
    }
    return {'schema_version': 'ocr-control-queue/1.0', 'protocol': deepcopy(PROTOCOL_V1),
            'status': 'review_queue_not_evaluation_release', 'entries': entries,
            'selected_ids': [row['id'] for row in selected],
            'selected_source_paths': [row['input_path'] for row in selected], 'summary': summary,
            'split': None, 'training_eligible': False, 'annotation_approved': False,
            'independent_evaluation': False, 'source_modified': False,
            'global_split_policy_modified': False,
            'limitations': ['Filename hints and distinct hashes do not establish semantic independence.',
                            'All selected sources remain pending family and label review.',
                            'Native exclusions prohibit native-derived labels, not review of original PNG pixels.',
                            'The six filename strata cover a review sample, not a representative benchmark.',
                            'Actual source bytes must be authenticated by the caller.']}
