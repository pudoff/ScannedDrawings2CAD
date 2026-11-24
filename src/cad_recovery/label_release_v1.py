"""Actual-byte lineage checks and fail-closed gates for labels/1.0."""
from collections import defaultdict
from pathlib import Path

from .clear_pairs import contained, digest
from .family_catalog_v2 import require, stable_digest
from .family_split_v2 import screen_pair_v2, validate_proposed_assignments_v2
from .label_schema_v1 import Pin, read_labels_v1, read_json_v1 as read_json


def checked_pin(workspace, pin):
    pin = Pin.model_validate(pin).model_dump()
    path = contained(Path(workspace).resolve(), pin['path'])
    require(digest(path) == pin['sha256'], 'Pinned bytes changed: '+pin['path'])
    return path


def build_derivative_index_v1(workspace, catalog, family_proof):
    """Rehash all prepared crops/overviews and bind them to original source bytes."""
    workspace = Path(workspace).resolve()
    files = {row['path']: row for row in catalog['files']}
    rows, seen, source_seen = [], set(), set()
    for binding in family_proof['prepared_lineage_bindings']:
        source = files[binding['source_path']]
        require(source['sha256'] == binding['source_sha256'] and source['group_id'] == binding['group_id'],
                'Preparation source/group differs from catalog')
        checked_pin(workspace, {'path': 'dataset/'+source['path'], 'sha256': source['sha256']})
        prep_path = checked_pin(workspace, binding['preparation'])
        prep = read_json(prep_path)
        require(prep['source_sha256'] == source['sha256'], 'Preparation content identifies a different source')
        require(source['path'] not in source_seen, 'Repeated prepared source')
        source_seen.add(source['path'])
        artifacts = [prep['overview'], prep['ink_overview'], *prep['tiles']]
        require(len(artifacts) == binding['derived_artifacts'], 'Preparation artifact count changed')
        for item in artifacts:
            artifact_path = contained(prep_path.parent, item['path']).relative_to(workspace).as_posix()
            require(artifact_path not in seen, 'A derivative path belongs to multiple preparation records')
            seen.add(artifact_path)
            checked_pin(workspace, {'path': artifact_path, 'sha256': item['sha256']})
            rows.append({'path': artifact_path, 'sha256': item['sha256'], 'source_path': source['path'],
                         'source_sha256': source['sha256'], 'group_id': source['group_id'],
                         'preparation': binding['preparation'], 'split': None, 'training_eligible': False})
    rows.sort(key=lambda r: r['path'])
    groups = defaultdict(set)
    for row in rows:
        groups[row['sha256']].add(row['group_id'])
    duplicates = [{'sha256': key, 'group_ids': sorted(value),
                   'disposition': 'deduplicate_or_review_joint_allocation_before_any_release'}
                  for key, value in sorted(groups.items()) if len(value) > 1]
    return {'schema_version': 'labels-derivatives/1.0', 'catalog_content_sha256': stable_digest(catalog),
            'artifacts': rows, 'cross_group_content': duplicates, 'source_count': len(source_seen),
            'training_eligible': False, 'augmentation_created': False}


def validate_derivative_index_v1(catalog, index):
    require(index['schema_version'] == 'labels-derivatives/1.0' and index['training_eligible'] is False
            and index['augmentation_created'] is False and index['catalog_content_sha256'] == stable_digest(catalog),
            'Derivative index is not bound to the current catalog')
    sources = {r['path']: r for r in catalog['files']}
    rows, groups = {}, defaultdict(set)
    for row in index['artifacts']:
        Pin.model_validate({'path': row['path'], 'sha256': row['sha256']})
        source = sources.get(row['source_path'])
        require(source is not None and source['sha256'] == row['source_sha256']
                and source['group_id'] == row['group_id'] and row['split'] is None
                and row['training_eligible'] is False and row['path'] not in rows,
                'Invalid derivative lineage record')
        rows[row['path']] = row
        groups[row['sha256']].add(row['group_id'])
    actual = {key: sorted(value) for key, value in groups.items() if len(value) > 1}
    declared = {row['sha256']: row['group_ids'] for row in index['cross_group_content']}
    require(actual == declared and len(declared) == len(index['cross_group_content']),
            'Cross-group duplicate hash list is incomplete or changed')
    return rows, set(actual)


def validate_derivative_assignments_v1(catalog, index, assignments):
    """Routing-only diagnostic. Duplicate derivatives cannot be silently released."""
    rows, cross_group = validate_derivative_index_v1(catalog, index)
    originals, seen, hash_splits = [], set(), {}
    for item in assignments:
        path = item['path']
        require(path in rows and path not in seen, 'Unknown or repeated derivative')
        seen.add(path)
        row = rows[path]
        require(row['sha256'] not in cross_group, 'Cross-group duplicate derivative requires deduplication or joint allocation review')
        split = item['split']
        require(row['sha256'] not in hash_splits or hash_splits[row['sha256']] == split, 'Exact derivative leakage')
        hash_splits[row['sha256']] = split
        originals.append({'input_path': row['source_path'], 'target_path': item['target_path'], 'split': split})
    unique = {(a['input_path'], a['target_path'], a['split']): a for a in originals}
    result = validate_proposed_assignments_v2(catalog, list(unique.values()))
    return {**result, 'meaning': 'Routing diagnostic only; label release is still unavailable in v1'}


def screen_label_package_v1(workspace, package, catalog, policy, exclusions, *, requested_split='train',
                            derivative_index=None, sample=None):
    """No path in v1 grants a training release, including analytical verified labels.

    The caller must authenticate the package/catalog/policy/index pins through its
    release config; all provenance and source/native/sample bytes are checked here.
    """
    workspace = Path(workspace).resolve()
    package = read_labels_v1(package)
    require(requested_split in {'train', 'validation', 'test'}, 'Unknown requested split')
    for pin in [package.source, *package.dependencies, *([package.native_target] if package.native_target else [])]:
        checked_pin(workspace, pin.model_dump())
    validate_label_derivation_v1(workspace, package)
    reasons = ['label_release_not_implemented_for_pending_v1_observations']
    if package.domain == 'analytical_control':
        reasons.append('analytical_control_is_not_customer_training_data')
        return {'status': 'control_only', 'reason_codes': reasons, 'requested_split': requested_split,
                'training_eligible': False, 'annotation_approved': False, 'split': None}
    actual_catalog = read_json(checked_pin(workspace, package.lineage.catalog.model_dump()))
    require(stable_digest(actual_catalog) == stable_digest(catalog), 'Package catalog pin differs from supplied catalog')
    require(package.source.path.startswith('dataset/') and package.native_target.path.startswith('dataset/'),
            'Customer labels must identify original dataset files')
    source_path = package.source.path.removeprefix('dataset/')
    target_path = package.native_target.path.removeprefix('dataset/')
    match = [p for p in catalog['pairs'] if p['input_path'] == source_path and p['target_path'] == target_path]
    require(len(match) == 1, 'Customer source/native pair is not in the catalog')
    row = match[0]
    require(row['input_sha256'] == package.source.sha256 and row['target_sha256'] == package.native_target.sha256
            and row['group_id'] == package.lineage.group_id and row['family_id'] == package.lineage.family_id
            and row['train_only'] == package.lineage.train_only
            and row['group_reserved_split'] == package.lineage.historical_reserved_split,
            'Customer package has changed source/target/group restrictions')
    result = screen_pair_v2(workspace/'dataset', catalog, policy, exclusions, source_path, target_path, requested_split)
    reasons.extend(result['reason_codes'])
    if sample is not None:
        sample = Pin.model_validate(sample)
        checked_pin(workspace, sample.model_dump())
        if sample != package.source:
            require(derivative_index is not None, 'A derived sample needs a pinned derivative index')
            rows, duplicate_hashes = validate_derivative_index_v1(catalog, derivative_index)
            derivative = rows.get(sample.path)
            require(derivative is not None and derivative['sha256'] == sample.sha256
                    and derivative['source_path'] == source_path and derivative['source_sha256'] == package.source.sha256
                    and derivative['group_id'] == package.lineage.group_id, 'Derived sample lineage does not match labels')
            if sample.sha256 in duplicate_hashes:
                reasons.append('cross_group_duplicate_derivative_requires_deduplication_or_joint_review')
            reasons.append('derived_label_transform_and_pixel_quality_require_review')
    return {**result, 'reason_codes': list(dict.fromkeys(reasons)), 'label_package_id': package.id,
            'physical_label_approval': False, 'augmentation_allowed': False}


def validate_label_derivation_v1(workspace, package):
    """Reconstruct from pinned evidence: scope changes cannot manufacture verified facts."""
    from .label_corpus_v1 import DEPENDENCIES, build_customer_labels_v1, build_analytical_labels_v1
    package = read_labels_v1(package)
    pins = {key: value.model_dump() for key, value in package.assembly_inputs.items()}
    require(set(pins) == DEPENDENCIES, 'Label package is missing its complete assembly recipe')
    for item in pins.values():
        checked_pin(workspace, item)
    if package.domain == 'analytical_control':
        expected = build_analytical_labels_v1(read_json(Path(workspace)/pins['control_graph']['path']),
                                             read_json(Path(workspace)/pins['control_spec']['path']), pins,
                                             read_json(Path(workspace)/pins['control_expectations']['path']))
    else:
        review = read_json(Path(workspace)/pins['field_review']['path'])
        matches = [p for p in review['pairs'] if package.source.path == 'dataset/'+p['input_path']
                   and package.source.sha256 == p['input_sha256'] and package.native_target.path == 'dataset/'+p['target_path']
                   and package.native_target.sha256 == p['target_sha256']]
        require(len(matches) == 1, 'Customer observation source/native identity is not in the pinned field review')
        expected = build_customer_labels_v1(matches[0], read_json(Path(workspace)/pins['catalog']['path']), pins,
                                            read_json(Path(workspace)/pins['point_report']['path'])['points'])
    require(stable_digest(package.model_dump(mode='json')) == stable_digest(expected.model_dump(mode='json')),
            'Label fields, statuses, scopes, coordinates or lineage differ from the pinned assembly recipe')
    return {'status': 'exact_pinned_derivation', 'training_eligible': False}


def authorize_label_augmentation_v1(*args, **kwargs):
    result = screen_label_package_v1(*args, **kwargs)
    if not result['training_eligible'] or result['split'] is None:
        raise ValueError('Augmentation requires an approved label release and a completed lineage split; both remain pending')
    # This guard is intentionally unconditional in the observation-only version.
    raise ValueError('No augmentation release exists in labels/1.0')
