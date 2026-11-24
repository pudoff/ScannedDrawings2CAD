"""Conservative lineage groups for a combined corpus, without label approval.

Exact content and inherited families bind groups. Candidate pairs may bind a
group to prevent leakage, but never establish semantic family membership.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), 'Expected canonical SHA-256')
    return value


def relative_path(value):
    require(isinstance(value, str) and 0 < len(value) <= 4096 and '\\' not in value and ':' not in value,
            'Expected normalized relative POSIX path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value and path.parts
            and all(p not in ('.', '..') for p in path.parts), 'Unsafe relative path')
    return value


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def indexed(rows, key):
    require(isinstance(rows, list) and len(rows) <= 100000, 'Expected bounded inventory')
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), 'Duplicate inventory identity: '+key)
    return result


def build_family_catalog_v2(manifest, legacy_catalog, legacy_split, legacy_verification,
                            handoff, exclusions, field_review):
    """Build complete membership and historical reservations, not active splits."""
    files = indexed(manifest['files'], 'path')
    require(files, 'Empty dataset manifest')
    for path, row in files.items():
        relative_path(path)
        canonical_sha(row['sha256'])
        require(row['kind'] in ('dxf', 'image', 'pdf') and type(row['size_bytes']) is int
                and row['size_bytes'] >= 0, 'Invalid source inventory record')
    old_families = indexed(legacy_catalog['families'], 'target_path')
    require(legacy_verification.get('complete') is True and legacy_split['status'] == 'development_only',
            'Historical verification and reservations must be complete development evidence')
    require(set(legacy_split['splits']) == {'train', 'validation', 'test'}, 'Unexpected historical split names')
    require(handoff.get('training_eligible') is False and handoff.get('split_modified') is False,
            'Handoff must remain unapproved')
    handoffs = indexed(handoff['entries'], 'input_path')
    parent = {path: path for path in files}
    edges = []

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b, reason):
        require(a in files and b in files, 'Lineage endpoint absent from manifest')
        edges.append({'a': a, 'b': b, 'basis': reason, 'semantic_equivalence_approved': False})
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    by_hash = defaultdict(list)
    for path, row in files.items():
        by_hash[row['sha256']].append(path)
    for paths in by_hash.values():
        paths.sort()
        for path in paths[1:]:
            union(paths[0], path, 'identical_file_bytes')
    family_paths = defaultdict(list)
    family_hashes = defaultdict(list)
    for path, row in old_families.items():
        require(path in files and files[path]['kind'] == 'dxf'
                and files[path]['sha256'] == row['target_sha256'], 'Historical family target changed')
        require(row['status'] == 'grouped' and isinstance(row['family_id'], str) and row['family_id']
                and type(row['train_only']) is bool, 'Unresolved historical family entry')
        require(row['method'] in ('cad_title_label', 'conservative_real_pool'), 'Unknown historical family basis')
        family_paths[row['family_id']].append(path)
        family_hashes[row['target_sha256']].append(row)
    for paths in family_paths.values():
        paths.sort()
        for path in paths[1:]:
            union(paths[0], path, 'inherited_family_or_conservative_development_pool')

    pair_rows = {}

    def add_pair(row, basis, *, verified=False):
        source, target = row['input_path'], row['target_path']
        require(source in files and target in files and files[source]['kind'] in ('pdf', 'image')
                and files[target]['kind'] == 'dxf', 'Invalid candidate pair inventory')
        for side in ('input', 'target'):
            if side+'_sha256' in row:
                require(row[side+'_sha256'] == files[row[side+'_path']]['sha256'], 'Candidate pair source hash changed')
        key = (source, target)
        if key not in pair_rows:
            pair_rows[key] = {'pair_id': 'candidate-'+stable_digest(key)[:20],
                'input_path': source, 'input_sha256': files[source]['sha256'],
                'target_path': target, 'target_sha256': files[target]['sha256'],
                'evidence_origins': [], 'historical_visual_verified': False,
                'split': None, 'annotation_approved': False, 'training_eligible': False}
            union(source, target, 'candidate_correspondence_coallocation_only')
        record = pair_rows[key]
        if basis not in record['evidence_origins']:
            record['evidence_origins'].append(basis)
        record['historical_visual_verified'] |= verified
        return record

    for row in manifest.get('pair_candidates', []):
        add_pair(row, 'manifest_candidate_not_family_evidence')
    old_verified = {}
    for row in legacy_verification['pairs']:
        require(row['status'] in ('verified', 'quarantine', 'error'), 'Unknown historical visual state')
        record = add_pair(row, 'legacy_visual_verification', verified=row['status'] == 'verified')
        require((row['input_path'], row['target_path']) not in old_verified, 'Duplicate historical correspondence')
        old_verified[(row['input_path'], row['target_path'])] = row
    inherited_train_only = set()
    for source, row in handoffs.items():
        require(row['split'] is None and row['annotation_approved'] is False
                and row['training_eligible'] is False and type(row['train_only_constraint_inherited']) is bool,
                'Handoff cannot approve a pair or remove explicit family constraints')
        record = add_pair(row, 'review_handoff_v5')
        record['handoff'] = deepcopy(row)
        if row['train_only_constraint_inherited']:
            inherited_train_only.add(source)
        # Competing targets remain in one conservative group even when a given
        # correspondence was rejected: their semantic family is still unresolved.
        for competitor in row.get('competing_visual_targets', []):
            target = competitor['target_path']
            require(target in files and files[target]['kind'] == 'dxf'
                    and files[target]['sha256'] == competitor['target_sha256'], 'Competing target identity changed')
            union(source, target, 'reviewed_competitor_coallocation_only')

    historical = []
    for split, rows in legacy_split['splits'].items():
        for row in rows:
            key = (row['input_path'], row['target_path'])
            require(key in old_verified and old_verified[key]['status'] == 'verified',
                    'Historical reservation has no verified correspondence')
            record = add_pair(row, 'legacy_split_reservation')
            family = old_families[row['target_path']]
            require(row['family_id'] == family['family_id'] and row['train_only'] == family['train_only'],
                    'Historical split disagrees with family catalog')
            require(not row['train_only'] or split == 'train', 'Historical train-only leaked into evaluation')
            historical.append({'reserved_split': split, 'original_record': deepcopy(row),
                               'label_approval_inferred': False})
            require('historical_reserved_split' not in record, 'Duplicate historical assignment')
            record['historical_reserved_split'] = split

    identity_conflicts = []
    field_pairs = indexed(field_review['pairs'], 'input_path')
    require(field_review['training_eligible'] is False and field_review['annotation_approved'] is False,
            'Field observations cannot approve training labels')
    for source, row in field_pairs.items():
        require(source in handoffs and row['target_path'] == handoffs[source]['target_path']
                and row['input_sha256'] == files[source]['sha256']
                and row['target_sha256'] == files[row['target_path']]['sha256']
                and row['annotation_approved'] is False and row['training_eligible'] is False,
                'Field review is outside the retained handoff identity')
        if not row['filename_designation_matches_native_title']:
            conflict = {'input_path': source, 'target_path': row['target_path'],
                        'filename_designation': row['filename_designation'],
                        'source_title_designation': row['source_title_readings']['drawing_identifier'],
                        'status': 'identity_conflict_retained_not_renamed', 'family_approved': False}
            identity_conflicts.append(conflict)
            pair_rows[(source, row['target_path'])]['identity_conflict'] = deepcopy(conflict)

    members = defaultdict(list)
    for path in files:
        members[find(path)].append(path)
    group_for_path, groups = {}, []
    for paths in sorted(members.values(), key=lambda rows: min(rows)):
        paths.sort()
        group_id = 'lineage-'+stable_digest([[p, files[p]['sha256']] for p in paths])[:24]
        facts = [row for path in paths for row in family_hashes.get(files[path]['sha256'], [])]
        family_ids = sorted({row['family_id'] for row in facts})
        train_only = any(row['train_only'] for row in facts) or bool(inherited_train_only.intersection(paths))
        reserved = sorted({r['reserved_split'] for r in historical if r['original_record']['input_path'] in paths})
        require(len(reserved) <= 1, 'Conservative lineage component crosses historical holdouts')
        require(not train_only or not reserved or reserved == ['train'],
                'Train-only lineage component reaches a historical holdout')
        group = {'group_id': group_id, 'members': paths, 'known_family_ids': family_ids,
            'semantic_family_approved': False,
            'confirmed_native_template_family_ids': [x for x in family_ids if x.startswith('synthetic:')],
            'train_only': train_only, 'historical_reserved_split': reserved[0] if reserved else None,
            'split': None, 'training_eligible': False,
            'meaning': 'Conservative coallocation component; unknown members do not inherit semantic identity or label approval.'}
        groups.append(group)
        group_for_path.update({path: group for path in paths})

    excluded = {row['target_sha256']: row for row in exclusions['exclusions']}
    require(len(excluded) == len(exclusions['exclusions']) and exclusions['default_status'] == 'not_approved',
            'Duplicate or permissive exclusion manifest')
    target_rows = []
    file_rows = []
    for path, row in sorted(files.items()):
        group = group_for_path[path]
        file_rows.append({'path': path, 'sha256': row['sha256'], 'size_bytes': row['size_bytes'],
            'kind': row['kind'], 'role': row.get('role'), 'group_id': group['group_id'],
            'train_only': group['train_only'], 'historical_reserved_split': group['historical_reserved_split'],
            'split': None, 'training_eligible': False})
        if row['kind'] != 'dxf':
            continue
        prior = family_hashes.get(row['sha256'], [])
        ids = sorted({r['family_id'] for r in prior})
        require(len(ids) <= 1, 'Exact native bytes carry conflicting historical family claims')
        family_id = ids[0] if ids else None
        status = ('confirmed_template_from_pinned_native_label' if family_id and family_id.startswith('synthetic:') else
                  'conservative_development_pool_not_semantic_family' if family_id else 'semantic_family_unresolved')
        target_rows.append({'target_path': path, 'target_sha256': row['sha256'],
            'family_id': family_id, 'family_status': status, 'group_id': group['group_id'],
            'train_only': group['train_only'], 'historical_reserved_split': group['historical_reserved_split'],
            'inherited_family_records': deepcopy(prior),
            'target_status': 'excluded' if row['sha256'] in excluded else 'not_approved',
            'exclusion': deepcopy(excluded.get(row['sha256'])), 'split': None,
            'annotation_approved': False, 'training_eligible': False})
    target_index = {row['target_path']: row for row in target_rows}
    for key, record in sorted(pair_rows.items()):
        source, target = key
        group = group_for_path[source]
        require(group['group_id'] == group_for_path[target]['group_id'], 'Pair escaped its lineage group')
        facts = target_index[target]
        record.update(group_id=group['group_id'], family_id=facts['family_id'], family_status=facts['family_status'],
                      train_only=group['train_only'], group_reserved_split=group['historical_reserved_split'],
                      target_status=facts['target_status'])
        reasons = ['label_approval_missing']
        if facts['target_status'] == 'excluded': reasons.append('incomplete_native_target_excluded')
        if facts['family_status'] != 'confirmed_template_from_pinned_native_label': reasons.append('semantic_family_not_confirmed')
        if 'identity_conflict' in record: reasons.append('filename_title_identity_conflict')
        if not record['historical_visual_verified']: reasons.append('correspondence_and_coordinate_review_required')
        record['pending_reasons'] = reasons
    paired_inputs = {row['input_path'] for row in pair_rows.values()}
    unmatched = [row['path'] for row in file_rows if row['kind'] in ('pdf', 'image') and row['path'] not in paired_inputs]
    clear_rows = [r for r in pair_rows.values() if 'handoff' in r]
    return {'schema_version': '2.0', 'purpose': 'lineage_catalog_without_training_approval',
        'files': file_rows, 'targets': target_rows, 'pairs': list(sorted(pair_rows.values(), key=lambda r: r['pair_id'])),
        'groups': groups, 'lineage_edges': sorted(edges, key=lambda r: (r['a'], r['b'], r['basis'])),
        'historical_reservations': sorted(historical, key=lambda r: (r['reserved_split'], r['original_record']['input_path'])),
        'identity_conflicts': identity_conflicts, 'unmatched_inputs': unmatched,
        'summary': {'source_files': len(file_rows), 'native_target_files': len(target_rows),
            'unique_native_hashes': len({r['target_sha256'] for r in target_rows}),
            'candidate_pairs': len(pair_rows), 'lineage_groups': len(groups), 'clear_pairs': len(clear_rows),
            'clear_original_train_only_constraints': len(inherited_train_only),
            'clear_effective_train_only_constraints': sum(r['train_only'] for r in clear_rows),
            'clear_excluded_targets': sum(r['target_status'] == 'excluded' for r in clear_rows),
            'clear_unknown_semantic_targets': sum(r['family_id'] is None for r in clear_rows),
            'target_family_status': dict(Counter(r['family_status'] for r in target_rows)),
            'target_status': dict(Counter(r['target_status'] for r in target_rows)),
            'historical_pairs_by_reservation': dict(Counter(r['reserved_split'] for r in historical)),
            'identity_conflicts': len(identity_conflicts), 'unmatched_input_files': len(unmatched),
            'training_eligible_pairs': 0},
        'source_modified': False, 'historical_split_modified': False, 'training_eligible': False,
        'limitations': ['Candidate coallocation protects against leakage; it does not prove semantic family identity.',
            'Unresolved families stay unresolved, even if a candidate or competitor reaches an inherited group.',
            'Historical visual split reservations are retained as restrictions, not label approval.',
            'No pair currently has a complete label approval record; active training/validation/test remain empty.',
            'Derivatives must inherit every parent lineage component and reservation before any future release.']}
