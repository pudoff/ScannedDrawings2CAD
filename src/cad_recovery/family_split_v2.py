"""Write and consume pinned lineage restrictions without approving labels."""
from collections import Counter, defaultdict
from copy import deepcopy
import html
import json
from pathlib import Path

from .clear_pairs import contained, digest, read_json
from .family_catalog_v2 import build_family_catalog_v2, indexed, relative_path, require, stable_digest
from .training_target_policy import build_exclusion_manifest, screen_training_target

MODULES = {'family_catalog_v2.py', 'family_split_v2.py', 'clear_pairs.py', 'training_target_policy.py'}
DEPENDENCIES = {'dataset_manifest', 'legacy_manifest', 'legacy_catalog', 'legacy_split', 'legacy_verification',
                'handoff', 'exclusions', 'exclusion_report', 'field_review', 'field_final'}
SPLIT_NAMES = {'train', 'validation', 'test'}


def historical_digest(data):
    import hashlib
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_split_policy_v2(catalog, exclusions):
    """Reservations constrain future releases. No complete label approvals exist."""
    return {'schema_version': '2.0', 'purpose': 'split_policy_pending_label_approval',
        'catalog_content_sha256': stable_digest(catalog), 'exclusions_content_sha256': stable_digest(exclusions),
        'splits': {key: [] for key in sorted(SPLIT_NAMES)},
        'historical_reservations': deepcopy(catalog['historical_reservations']),
        'group_reservations': [{'group_id': g['group_id'], 'reserved_split': g['historical_reserved_split'],
                                'train_only': g['train_only']} for g in catalog['groups']],
        'candidate_dispositions': [{'pair_id': p['pair_id'], 'input_path': p['input_path'],
            'target_path': p['target_path'], 'group_id': p['group_id'], 'target_status': p['target_status'],
            'pending_reasons': list(p['pending_reasons']), 'split': None, 'training_eligible': False}
            for p in catalog['pairs']],
        'release_requirements': ['unchanged_actual_source_and_native_bytes', 'nonexcluded_native_target',
            'complete_explicit_label_approval', 'reviewed_semantic_family_or_train_only_development_scope',
            'whole_lineage_group_assignment_and_historical_reservations'],
        'historical_split_modified': False, 'training_eligible': False,
        'summary': {'eligible_pairs_by_split': {key: 0 for key in sorted(SPLIT_NAMES)},
                    'pending_pairs': len(catalog['pairs']), 'historical_reservations': len(catalog['historical_reservations'])}}


def validate_proposed_assignments_v2(catalog, assignments):
    """Validate routing only; this never accepts label quality or authorizes use."""
    pairs = {(p['input_path'], p['target_path']): p for p in catalog['pairs']}
    groups = indexed(catalog['groups'], 'group_id')
    require(len(pairs) == len(catalog['pairs']) and isinstance(assignments, list), 'Invalid routing inventory')
    owners, seen = {}, set()
    for assignment in assignments:
        key = (assignment['input_path'], assignment['target_path'])
        require(key in pairs and key not in seen, 'Unknown or repeated candidate pair')
        seen.add(key)
        split = assignment['split']
        require(split in SPLIT_NAMES, 'Unknown split assignment')
        pair = pairs[key]
        group = groups[pair['group_id']]
        require(not group['train_only'] or split == 'train', 'Train-only lineage cannot enter validation/test')
        require(group['historical_reserved_split'] in (None, split), 'Historical holdout reservation cannot change')
        tokens = [('group', group['group_id']), ('source', pair['input_path']), ('source', pair['target_path']),
                  ('hash', pair['input_sha256']), ('hash', pair['target_sha256'])]
        if pair['family_id']:
            tokens.append(('family', pair['family_id']))
        for token in tokens:
            require(token not in owners or owners[token] == split, 'Lineage or exact-content leakage across splits')
            owners[token] = split
    return {'status': 'routing_consistent_labels_unapproved', 'pair_count': len(assignments),
            'training_eligible': False, 'annotation_approved': False}


def _checked_policy(catalog, policy, exclusions):
    require(policy['purpose'] == 'split_policy_pending_label_approval'
            and policy['catalog_content_sha256'] == stable_digest(catalog)
            and policy['exclusions_content_sha256'] == stable_digest(exclusions)
            and policy['training_eligible'] is False
            and set(policy['splits']) == SPLIT_NAMES and all(not rows for rows in policy['splits'].values()),
            'Split policy/catalog changed or acquired unsupported approvals')
    return {(p['input_path'], p['target_path']): p for p in catalog['pairs']}


def _screen_actual_pair(dataset_root, catalog, pair_index, exclusions, input_path, target_path, requested_split):
    require(requested_split in SPLIT_NAMES, 'Unknown requested split')
    root = Path(dataset_root).resolve()
    source = contained(root, relative_path(input_path))
    target = contained(root, relative_path(target_path))
    current_source, native_gate = digest(source), screen_training_target(target, exclusions)
    pair = pair_index.get((input_path, target_path))
    reasons = []
    if native_gate['status'] == 'excluded':
        reasons.append('incomplete_native_target_excluded')
    if pair is None:
        reasons.append('unknown_candidate_pair')
    elif pair['input_sha256'] != current_source or pair['target_sha256'] != native_gate['target_sha256']:
        reasons.append('source_or_target_revision_changed')
    else:
        try:
            validate_proposed_assignments_v2(catalog, [{'input_path': input_path, 'target_path': target_path, 'split': requested_split}])
        except ValueError:
            reasons.append('lineage_or_historical_reservation_conflict')
        reasons.extend(pair['pending_reasons'])
    reasons = list(dict.fromkeys(reasons or ['label_approval_missing']))
    return {'input_path': input_path, 'input_sha256': current_source, 'target_path': target_path,
            'target_sha256': native_gate['target_sha256'], 'requested_split': requested_split,
            'status': 'excluded' if native_gate['status'] == 'excluded' else 'not_approved',
            'reason_codes': reasons, 'native_target_policy': native_gate,
            'split': None, 'training_eligible': False, 'annotation_approved': False}


def screen_pair_v2(dataset_root, catalog, policy, exclusions, input_path, target_path, requested_split):
    """Hash actual bytes and deny pending/excluded revisions; never filename-allow.

    Callers must pin these three artifacts to a completed run. Version 2 stores
    no label approvals, so a valid route is still unapproved for training.
    """
    pairs = _checked_policy(catalog, policy, exclusions)
    return _screen_actual_pair(dataset_root, catalog, pairs, exclusions, input_path, target_path, requested_split)


def inherit_derivative_v2(workspace, dataset_root, catalog, parent_paths, derivative_path, derivative_sha256):
    """Require exact source parents and one lineage for a crop/view/augmentation."""
    workspace, dataset_root = Path(workspace).resolve(), Path(dataset_root).resolve()
    require(dataset_root.is_relative_to(workspace), 'Dataset must remain in workspace')
    require(isinstance(parent_paths, list) and parent_paths and len(parent_paths) == len(set(parent_paths)),
            'Derivative needs distinct explicit source parents')
    files, groups = indexed(catalog['files'], 'path'), indexed(catalog['groups'], 'group_id')
    parents = []
    for path in parent_paths:
        relative_path(path)
        require(path in files, 'Unknown derivative parent')
        row = files[path]
        require(digest(contained(dataset_root, path)) == row['sha256'], 'Derivative parent bytes changed')
        parents.append({'path': path, 'sha256': row['sha256'], 'group_id': row['group_id']})
    ids = {p['group_id'] for p in parents}
    require(len(ids) == 1, 'Mixed lineage derivative needs a separately reviewed merged group')
    output = contained(workspace, relative_path(derivative_path))
    require(not output.is_relative_to(dataset_root) and digest(output) == derivative_sha256,
            'Derivative must be a matching separate artifact')
    group = groups[next(iter(ids))]
    return {'path': derivative_path, 'sha256': derivative_sha256, 'parents': parents,
            'group_id': group['group_id'], 'train_only': group['train_only'],
            'historical_reserved_split': group['historical_reserved_split'],
            'split': None, 'annotation_approved': False, 'training_eligible': False,
            'meaning': 'Lineage inheritance only; pixels, transforms and labels need separate review.'}


def _write(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def run_family_split_v2(workspace, config_path, output):
    workspace, config_path, output = (Path(p).resolve() for p in (workspace, config_path, output))
    require(config_path.is_relative_to(workspace), 'Config must remain in workspace')
    require(output.is_relative_to(workspace/'artifacts') and output != workspace/'artifacts', 'Choose a new artifacts directory')
    if output.exists():
        raise FileExistsError('Preserve the existing family/split result')
    config_hash = digest(config_path)
    config = read_json(config_path)
    require(config['schema_version'] == '2.0' and config['purpose'] == 'combined_family_split_policy'
            and config['training_eligible'] is False, 'Expected a lineage/split policy configuration')
    require(set(config['dependencies']) == DEPENDENCIES and set(config['implementation']) == MODULES,
            'Missing or unexpected pinned dependencies')
    pins = [(config_path, config_hash)]
    paths, values = {}, {}
    for name, pin in config['dependencies'].items():
        path = contained(workspace, pin['path'])
        require(digest(path) == pin['sha256'], 'Pinned lineage input changed: '+name)
        pins.append((path, pin['sha256']))
        paths[name], values[name] = path, read_json(path)
    for name, expected in config['implementation'].items():
        path = Path(__file__).parent/name
        require(digest(path) == expected, 'Lineage implementation changed')
        pins.append((path, expected))
    for relative, expected in config['entrypoints'].items():
        path = contained(workspace, relative)
        require(digest(path) == expected, 'Lineage CLI/builder changed')
        pins.append((path, expected))
    dataset = contained(workspace, config['dataset'])
    require(not output.is_relative_to(dataset) and not any(output.is_relative_to(p.parent) for p in paths.values()
                                                         if p.parent != workspace/'artifacts'),
            'Output cannot extend a frozen input directory')
    old_manifest, old_families, old_split, old_visual = (values[k] for k in
        ('legacy_manifest', 'legacy_catalog', 'legacy_split', 'legacy_verification'))
    require(old_families['manifest_sha256'] == historical_digest(old_manifest)
            and old_split['manifest_sha256'] == historical_digest(old_manifest)
            and old_split['catalog_sha256'] == historical_digest(old_families)
            and old_split['verification_sha256'] == historical_digest(old_visual), 'Historical evidence chain differs')
    exclusions = values['exclusions']
    require(exclusions == build_exclusion_manifest(config['dependencies']['exclusion_report'], values['exclusion_report']['cases']),
            'Exclusion manifest differs from explicit resource decisions')
    handoff = values['handoff']
    require(handoff['excluded_targets'] == config['dependencies']['exclusions'], 'Latest handoff refers to other exclusions')
    require(values['field_final']['report'] == config['dependencies']['field_review']
            and values['field_final']['training_eligible'] is False, 'Field review final proof differs')
    manifest = values['dataset_manifest']
    source_pins = []
    for row in manifest['files']:
        path = contained(dataset, relative_path(row['path']))
        require(path.stat().st_size == row['size_bytes'] and digest(path) == row['sha256'], 'Dataset file differs from registered bytes')
        source_pins.append((path, row['sha256']))
    current_files = {p.relative_to(dataset).as_posix() for p in dataset.rglob('*') if p.is_file()}
    require(current_files == {r['path'] for r in manifest['files']}, 'Dataset inventory changed; rebuild intake first')
    catalog = build_family_catalog_v2(manifest, old_families, old_split, old_visual,
                                      handoff, exclusions, values['field_review'])
    catalog['provenance'] = {'config': {'path': config_path.relative_to(workspace).as_posix(), 'sha256': config_hash},
                             'dependencies': deepcopy(config['dependencies']), 'date': config['date']}
    require(all(catalog['summary'][key] == count for key, count in config['expected_counts'].items()),
            'Combined lineage scope differs from explicit review configuration')
    policy = build_split_policy_v2(catalog, exclusions)
    routing = validate_proposed_assignments_v2(catalog, [
        {'input_path': r['original_record']['input_path'], 'target_path': r['original_record']['target_path'],
         'split': r['reserved_split']} for r in catalog['historical_reservations']])
    pair_index = _checked_policy(catalog, policy, exclusions)
    gates = [_screen_actual_pair(dataset, catalog, pair_index, exclusions, p['input_path'], p['target_path'],
                            p.get('group_reserved_split') or 'train') for p in catalog['pairs']]
    require(all(g['training_eligible'] is False for g in gates), 'Pending labels acquired training approval')
    family_pending = defaultdict(list)
    for row in catalog['targets']:
        if row['family_status'] != 'confirmed_template_from_pinned_native_label':
            family_pending[row['target_sha256']].append(row)
    family_queue = {'schema_version': '2.0', 'purpose': 'semantic_family_review_queue', 'training_eligible': False,
        'items': [{'target_sha256': sha, 'target_paths': [r['target_path'] for r in rows],
            'group_ids': sorted({r['group_id'] for r in rows}), 'inherited_family_id': rows[0]['family_id'],
            'target_status': rows[0]['target_status'], 'required': ['review_source_identity_and_semantic_family',
                'preserve_exact_duplicates_and_historical_reservations'], 'semantic_family_approved': False}
            for sha, rows in sorted(family_pending.items())]}
    label_queue = {'schema_version': '2.0', 'purpose': 'label_approval_queue', 'training_eligible': False,
                   'items': gates}
    report = {'schema_version': '2.0', 'status': 'complete_lineage_and_split_policy_pending_labels',
        'config': catalog['provenance']['config'], 'dependencies': deepcopy(config['dependencies']),
        'implementation': config['implementation'], 'summary': catalog['summary'],
        'verification': {'source_files_hashed_before_and_after': len(source_pins),
            'historical_routing': routing, 'actual_pair_gate_status': dict(Counter(g['status'] for g in gates)),
            'family_queue_unique_native_hashes': len(family_queue['items']),
            'preserved_handoff_rows': len(handoff['entries'])},
        'active_split_counts': policy['summary']['eligible_pairs_by_split'],
        'source_modified': False, 'historical_split_modified': False, 'annotation_approved': False,
        'training_eligible': False, 'limitations': deepcopy(catalog['limitations'])}
    for path, expected in pins+source_pins:
        require(digest(path) == expected, 'Source or pinned evidence changed during lineage review')
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('families.json', catalog), ('splits.json', policy), ('family-review-queue.json', family_queue),
                        ('label-review-queue.json', label_queue), ('report.json', report)]:
        _write(output/name, value)
    body = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Семейства и split v2</title>',
        '<style>body{font:15px system-ui;margin:24px}td,th{padding:5px;border:1px solid #aaa}table{border-collapse:collapse}</style>',
        '<h1>Каталог семейств и политика выборок v2</h1><p>Активные обучающие выборки пусты: метки не утверждены. Исторические назначения сохранены как ограничения.</p>',
        '<p><a href="families.json">Каталог</a> · <a href="splits.json">Политика split</a> · <a href="family-review-queue.json">Очередь семейств</a> · <a href="label-review-queue.json">Очередь меток</a> · <a href="report.json">Проверка</a></p>',
        '<table><tr><th>Источник</th><th>Семейство</th><th>Группа</th><th>Ограничения</th></tr>']
    for row in catalog['pairs']:
        body.append('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in (
            row['input_path'], row['family_id'] or 'не установлено', row['group_id'], ', '.join(row['pending_reasons'])))+'</tr>')
    body.append('</table></html>')
    (output/'index.html').write_text(''.join(body), encoding='utf-8')
    _write(output/'manifest.json', {'schema_version': '2.0', 'artifact_sha256': {
        p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return report
