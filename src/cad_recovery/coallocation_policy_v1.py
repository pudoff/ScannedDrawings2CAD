"""Additive whole-group coallocation, separate from family and label approval.

These functions consume caller-authenticated snapshots. They do not read files,
authenticate review evidence, assign global splits, or authorize model use.
``routing_allowed_splits`` expresses only compatibility with inherited mechanical
restrictions. In particular, an unknown family does not acquire evaluation
eligibility by satisfying those restrictions.
"""
from collections import Counter, defaultdict
from copy import deepcopy

from .family_catalog_v2 import canonical_sha, relative_path, require, stable_digest
from .ocr_lineage_v1 import _checked_context


SPLITS = ('train', 'validation', 'test')
BASES = frozenset({'reviewed_shared_reference', 'reviewed_document_series',
                   'unresolved_series_hold'})
ROUTING_SCOPE = 'mechanical_coallocation_only_labels_and_independence_unapproved'
CONSTRAINT_FIELDS = {'id', 'member_paths', 'basis', 'evidence', 'reason'}
ASSIGNMENT_FIELDS = {'id', 'source_path', 'source_sha256', 'split'}


def _nonempty(value, label, limit=4096):
    require(isinstance(value, str) and 0 < len(value) <= limit and value.strip(),
            'Expected nonempty '+label)


def _constraints(rows, files):
    require(isinstance(rows, list) and len(rows) <= 100000,
            'Expected bounded coallocation constraint list')
    result, identifiers, evidence_pins = [], set(), {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == CONSTRAINT_FIELDS,
                'Unexpected coallocation constraint fields')
        _nonempty(row['id'], 'constraint identity', 256)
        require(row['id'] not in identifiers, 'Duplicate coallocation constraint identity')
        identifiers.add(row['id'])
        require(isinstance(row['basis'], str) and row['basis'] in BASES,
                'Unknown coallocation basis')
        _nonempty(row['reason'], 'constraint reason')
        paths = row['member_paths']
        require(isinstance(paths, list) and 2 <= len(paths) <= len(files),
                'Constraint needs at least two registered member paths')
        for path in paths:
            relative_path(path)
            require(path in files, 'Unknown coallocation member path')
        require(len(set(paths)) == len(paths), 'Duplicate coallocation member path')
        require(isinstance(row['evidence'], list) and 0 < len(row['evidence']) <= 100000,
                'Constraint needs pinned review or hold evidence')
        pins, paths_seen = [], set()
        for pin in row['evidence']:
            require(isinstance(pin, dict) and set(pin) == {'path', 'sha256'},
                    'Unexpected coallocation evidence fields')
            path, sha = relative_path(pin['path']), canonical_sha(pin['sha256'])
            require(path not in paths_seen, 'Duplicate coallocation evidence path')
            require(path not in evidence_pins or evidence_pins[path] == sha,
                    'Conflicting coallocation evidence revision')
            paths_seen.add(path)
            evidence_pins[path] = sha
            pins.append({'path': path, 'sha256': sha})
        result.append({'id': row['id'], 'member_paths': sorted(paths), 'basis': row['basis'],
                       'evidence': sorted(pins, key=lambda pin: pin['path']), 'reason': row['reason']})
    return sorted(result, key=lambda row: row['id'])


def _unapproved():
    return {'semantic_family_approved': False, 'annotation_approved': False,
            'training_eligible': False, 'independent_evaluation': False, 'split': None}


def build_coallocation_v1(catalog, policy, exclusions, constraints):
    """Coallocate entire frozen groups and retain conflicting components blocked.

    Constraint members are dataset-relative; evidence paths are workspace-relative.
    Reviewed links and uncertainty holds both prevent cross-split allocation, but
    neither proves semantic family identity. The caller must authenticate every
    source snapshot and evidence pin, including the authority for each review.
    No actual bytes are checked by this pure builder.
    """
    require(isinstance(exclusions, dict), 'The frozen native exclusions are required')
    files, groups, _ = _checked_context(catalog, policy, exclusions)
    checked = _constraints(constraints, files)
    parent = {identifier: identifier for identifier in groups}

    def find(identifier):
        while identifier != parent[identifier]:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    for constraint in checked:
        identifiers = sorted({files[path]['group_id'] for path in constraint['member_paths']})
        for identifier in identifiers[1:]:
            left, right = find(identifiers[0]), find(identifier)
            if left != right:
                parent[max(left, right)] = min(left, right)
    components, by_component = defaultdict(list), defaultdict(list)
    for identifier in groups:
        components[find(identifier)].append(identifier)
    for constraint in checked:
        root = find(files[constraint['member_paths'][0]]['group_id'])
        by_component[root].append(constraint)

    overlays, file_to_group, base_group_to_overlay = [], {}, {}
    for root, identifiers in sorted(components.items()):
        identifiers.sort()
        members = sorted(path for identifier in identifiers for path in groups[identifier]['members'])
        group_id = 'coallocation-'+stable_digest(identifiers)[:24]
        train_only = any(groups[identifier]['train_only'] for identifier in identifiers)
        reservations = {groups[identifier]['historical_reserved_split'] for identifier in identifiers} - {None}
        reason_codes = []
        if len(reservations) > 1:
            reason_codes.append('conflicting_historical_reservations')
        if train_only and reservations - {'train'}:
            reason_codes.append('train_only_conflicts_with_historical_holdout')
        allowed = set(SPLITS)
        if train_only:
            allowed &= {'train'}
        for reservation in reservations:
            allowed &= {reservation}
        if reason_codes:
            allowed.clear()
        relations = by_component[root]
        holds = [row['id'] for row in relations if row['basis'] == 'unresolved_series_hold']
        reviewed = [row['id'] for row in relations if row['basis'] != 'unresolved_series_hold']
        overlays.append({'group_id': group_id, 'base_group_ids': identifiers, 'members': members,
            'constraint_ids': [row['id'] for row in relations],
            'reviewed_constraint_ids': reviewed, 'unresolved_hold_ids': holds,
            'train_only': train_only,
            'historical_reserved_splits': [split for split in SPLITS if split in reservations],
            'historical_group_reservations': [
                {'group_id': identifier, 'reserved_split': groups[identifier]['historical_reserved_split'],
                 'train_only': groups[identifier]['train_only']} for identifier in identifiers],
            'routing_allowed_splits': [split for split in SPLITS if split in allowed],
            'routing_scope': ROUTING_SCOPE,
            'status': 'blocked_restriction_conflict' if reason_codes else 'coallocation_only_pending_approval',
            'reason_codes': reason_codes, **_unapproved()})
        for path in members:
            file_to_group[path] = group_id
        for identifier in identifiers:
            base_group_to_overlay[identifier] = group_id
    overlays.sort(key=lambda row: row['group_id'])
    require(len({row['group_id'] for row in overlays}) == len(overlays), 'Coallocation identity collision')
    summary = {'file_count': len(files), 'base_group_count': len(groups),
        'retained_base_group_count': len(base_group_to_overlay), 'overlay_group_count': len(overlays),
        'merged_component_count': sum(len(row['base_group_ids']) > 1 for row in overlays),
        'merged_base_group_count': sum(len(row['base_group_ids']) for row in overlays if len(row['base_group_ids']) > 1),
        'constraint_count': len(checked),
        'reviewed_constraint_count': sum(row['basis'] != 'unresolved_series_hold' for row in checked),
        'unresolved_hold_count': sum(row['basis'] == 'unresolved_series_hold' for row in checked),
        'blocked_component_count': sum(bool(row['reason_codes']) for row in overlays),
        'train_only_component_count': sum(row['train_only'] for row in overlays)}
    return {'schema_version': '1.0', 'purpose': 'additional_coallocation_without_family_or_label_approval',
        'catalog_content_sha256': stable_digest(catalog), 'policy_content_sha256': stable_digest(policy),
        'exclusions_content_sha256': stable_digest(exclusions),
        'constraints_content_sha256': stable_digest(checked), 'constraints': checked,
        'groups': overlays, 'file_to_group': dict(sorted(file_to_group.items())),
        'base_group_to_overlay': dict(sorted(base_group_to_overlay.items())), 'summary': summary,
        'routing_scope': ROUTING_SCOPE, 'base_catalog_modified': False,
        'historical_split_modified': False, 'global_split_policy_modified': False,
        'actual_bytes_authenticated': False, **_unapproved()}


def validate_assignments_v1(catalog, policy, exclusions, overlay, assignments):
    """Check a complete proposed registry, without assigning or authorizing it.

    Each row has exactly id, source_path, source_sha256, split and optionally
    image_sha256 for a derivative. All catalog files can participate; this is not
    an OCR/native-label admission API. Pass every existing and proposed record in
    the relevant registry: separate calls cannot detect cross-release leakage.
    The caller must verify actual source/crop bytes and separately satisfy label,
    family and independent-evaluation requirements. Equality to a rebuilt overlay
    checks internal integrity, not the authenticity of a replacement constraint.
    """
    require(isinstance(overlay, dict) and isinstance(overlay.get('constraints'), list),
            'Expected coallocation overlay with explicit constraints')
    rebuilt = build_coallocation_v1(catalog, policy, exclusions, overlay['constraints'])
    # JSON digests also distinguish False from 0 and True from 1, unlike dict equality.
    require(stable_digest(rebuilt) == stable_digest(overlay),
            'Coallocation overlay differs from recomputed constraints or frozen dependencies')
    require(isinstance(assignments, list) and len(assignments) <= 100000,
            'Expected bounded complete assignment registry')
    files = {row['path']: row for row in catalog['files']}
    groups = {row['group_id']: row for row in rebuilt['groups']}
    seen, owners, rows, touched = set(), {}, [], set()
    counts = Counter({split: 0 for split in SPLITS})
    for row in assignments:
        require(isinstance(row, dict) and ASSIGNMENT_FIELDS <= set(row) <= ASSIGNMENT_FIELDS | {'image_sha256'},
                'Unexpected assignment fields')
        _nonempty(row['id'], 'assignment identity', 256)
        require(row['id'] not in seen, 'Duplicate assignment identity')
        seen.add(row['id'])
        path, sha = relative_path(row['source_path']), canonical_sha(row['source_sha256'])
        require(path in files, 'Unknown assignment source')
        require(files[path]['sha256'] == sha, 'Assignment source revision differs from catalog')
        split = row['split']
        require(isinstance(split, str) and split in SPLITS, 'Unknown assignment split')
        group_id = rebuilt['file_to_group'][path]
        group = groups[group_id]
        require(split in group['routing_allowed_splits'],
                'Assignment violates train-only, historical reservation or blocked coallocation restrictions')
        tokens = [('group', group_id), ('source', path), ('content', sha)]
        if 'image_sha256' in row:
            tokens.append(('content', canonical_sha(row['image_sha256'])))
        for token in tokens:
            require(token not in owners or owners[token] == split,
                    'Coallocation or exact-content leakage across splits')
            owners[token] = split
        counts[split] += 1
        touched.add(group_id)
        rows.append({**deepcopy(row), 'coallocation_group_id': group_id,
                     'base_group_id': files[path]['group_id']})
    return {'schema_version': '1.0',
        'status': 'routing_consistent_but_labels_and_independence_unapproved',
        'overlay_content_sha256': stable_digest(rebuilt), 'assignment_count': len(rows),
        'counts_by_split': dict(counts), 'coallocation_group_count': len(touched),
        'assignments': sorted(rows, key=lambda row: row['id']),
        'routing_scope': ROUTING_SCOPE, 'registry_scope': 'all_records_supplied_in_this_call_only',
        'actual_bytes_authenticated': False, 'global_split_policy_modified': False,
        **_unapproved()}
