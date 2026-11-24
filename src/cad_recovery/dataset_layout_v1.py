"""Explicit customer split folders, hash migration and conservative lineage audit.

A directory declares intended use. This registry does not approve labels, pairs
or independent evaluation, and never rewrites historical releases.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

from .family_catalog_v2 import canonical_sha, relative_path, require

SPLITS = ('train', 'validation', 'test')
CODE = ('src/cad_recovery/dataset_layout_v1.py', 'src/cad_recovery/family_catalog_v2.py',
        'scripts/register_dataset_layout_v1.py')
DEPENDENCIES = ('previous_manifest', 'relationships', 'ocr_registry', 'excluded_targets', 'previous_baseline_final')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inside(root, name):
    relative_path(name)
    path = root / name
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        require(not ancestor.is_symlink() and not getattr(ancestor, 'is_junction', lambda: False)(), 'Linked paths are unsupported')
    require(path.resolve().is_relative_to(root.resolve()), 'Path escapes workspace')
    return path


def pin(root, name):
    return {'path': name, 'sha256': digest(inside(root, name))}


def checked(root, record):
    require(set(record) == {'path', 'sha256'}, 'Invalid pinned dependency')
    canonical_sha(record['sha256'])
    path = inside(root, record['path'])
    require(digest(path) == record['sha256'], 'Pinned file changed: ' + record['path'])
    return path


def validate_layout(config):
    require(config['schema_version'] == 'dataset-layout/1.0', 'Unknown layout schema')
    relative_path(config['dataset_root'])
    require(config['split_authority'] == 'explicit_user_directory_assignment', 'Missing directory authority')
    require(config['training_eligible'] is False and config['independent_evaluation'] is False, 'Layout cannot approve training or independence')
    bundles = config['bundles']
    require(isinstance(bundles, list) and bundles and len({b['id'] for b in bundles}) == len(bundles), 'Invalid bundle identities')
    folders = []
    for b in bundles:
        require(set(b) == {'id', 'split', 'input_dir', 'input_extension', 'target_dir', 'target_extension'}, 'Invalid bundle fields')
        relative_path(b['id'])
        require('/' not in b['id'] and b['split'] in SPLITS and b['input_extension'] in ('.png', '.pdf')
                and b['target_extension'] == '.dxf', 'Invalid bundle format or split')
        for key in ('input_dir', 'target_dir'):
            folders.append(relative_path(b[key]))
    folded = [name.casefold() for name in folders]
    require(len(set(folded)) == len(folded), 'Repeated bundle directory')
    require(not any(a.startswith(b + '/') for a in folded for b in folded if a != b), 'Overlapping bundle directories')
    require(set(config['dependencies']) == set(DEPENDENCIES), 'Incomplete historical dependencies')


def scan_dataset(root, config):
    dataset = inside(root, config['dataset_root'])
    for bundle in config['bundles']:
        for key in ('input_dir', 'target_dir'):
            require(inside(dataset, bundle[key]).is_dir(), 'Required dataset directory is missing: ' + bundle[key])
    files = []
    for path in sorted(dataset.rglob('*')):
        require(not path.is_symlink() and not getattr(path, 'is_junction', lambda: False)(), 'Linked dataset entry')
        if not path.is_file():
            continue
        name = path.relative_to(dataset).as_posix()
        inside(dataset, name)
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Dataset changed during inventory')
        files.append({'path': name, 'size_bytes': before.st_size, 'sha256': sha})
        require(len(files) <= 100000, 'Dataset exceeds registry bound')
    return {'schema_version': 'dataset-layout-inventory/1.0', 'dataset_root': config['dataset_root'], 'files': files}


def audit_layout(inventory, config, previous_manifest, relationships, ocr_registry, excluded_targets):
    """Pure audit; accepts small synthetic fixtures as well as the current corpus."""
    validate_layout(config)
    rows = inventory['files']
    require(inventory['dataset_root'] == config['dataset_root'], 'Inventory root differs')
    require(len({r['path'] for r in rows}) == len(rows), 'Duplicate inventory path')
    old = {r['path']: r for r in previous_manifest['files']}
    require(len(old) == len(previous_manifest['files']), 'Duplicate historical path')
    by_hash = defaultdict(list)
    files = {}
    excluded = {r['target_sha256'] for r in excluded_targets['exclusions']}
    for row in sorted(rows, key=lambda r: r['path']):
        name = relative_path(row['path'])
        canonical_sha(row['sha256'])
        require(type(row['size_bytes']) is int and row['size_bytes'] >= 0, 'Invalid file size')
        matches = []
        for b in config['bundles']:
            for role in ('input', 'target'):
                if name.startswith(b[role + '_dir'] + '/') and Path(name).suffix.lower() == b[role + '_extension']:
                    matches.append((b['id'], b['split'], role))
        require(len(matches) <= 1, 'File belongs to several bundles')
        bundle, split, role = matches[0] if matches else (None, None, None)
        files[name] = {**row, 'bundle': bundle, 'declared_split': split, 'role': role,
                       'training_eligible': False, 'historically_excluded_native_target': row['sha256'] in excluded}
        by_hash[row['sha256']].append(name)
    parent = {name: name for name in files}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(names):
        names = sorted(set(names))
        for name in names[1:]:
            a, b = find(names[0]), find(name)
            parent[max(a, b)] = min(a, b)

    duplicates = []
    for sha, names in sorted(by_hash.items()):
        union(names)
        if len(names) > 1:
            splits = sorted({files[n]['declared_split'] for n in names if files[n]['declared_split']})
            duplicates.append({'sha256': sha, 'paths': names, 'declared_splits': splits, 'cross_split': len(splits) > 1})
    pairs, unmatched, ambiguous = [], [], []
    for b in config['bundles']:
        stems = defaultdict(lambda: {'input': [], 'target': []})
        for name, row in files.items():
            if row['bundle'] == b['id']:
                stem = unicodedata.normalize('NFC', Path(name).stem).strip().casefold()
                stems[stem][row['role']].append(name)
        for stem, roles in sorted(stems.items()):
            record = {'bundle': b['id'], 'declared_split': b['split'], 'normalized_stem': stem, **roles}
            if len(roles['input']) == len(roles['target']) == 1:
                pairs.append({**record, 'status': 'filename_candidate_only', 'target_approved': False,
                              'excluded_target': files[roles['target'][0]]['historically_excluded_native_target']})
                union(roles['input'] + roles['target'])
            elif len(roles['input']) > 1 or len(roles['target']) > 1:
                ambiguous.append(record)
                union(roles['input'] + roles['target'])
            else:
                unmatched.append(record)
    inherited = []
    for group in relationships['groups']:
        require(all(n in old for n in group['members']), 'Historical group member absent from inventory')
        names = sorted({n for p in group['members'] for n in by_hash.get(old[p]['sha256'], [])})
        union(names)
        inherited.append({'historical_group_id': group['group_id'], 'current_paths': names,
                          'unmapped_old_paths': [p for p in group['members'] if old[p]['sha256'] not in by_hash]})
    components = defaultdict(list)
    for name in files:
        components[find(name)].append(name)
    conflicts = []
    for names in components.values():
        splits = sorted({files[n]['declared_split'] for n in names if files[n]['declared_split']})
        if len(splits) > 1:
            conflicts.append({'paths': sorted(names), 'declared_splits': splits,
                              'status': 'requires_relationship_review_before_release'})
    migration = []
    for name, row in sorted(old.items()):
        matches = by_hash.get(row['sha256'], [])
        migration.append({'old_path': name, 'sha256': row['sha256'], 'current_paths': matches,
                          'status': 'same_path_same_bytes' if name in matches else 'content_found_at_other_paths' if matches else 'old_content_not_found',
                          'same_path_modified': name in files and files[name]['sha256'] != row['sha256']})
    historical_samples = []
    for sample in ocr_registry['assignments']:
        names = by_hash.get(sample['source_sha256'], [])
        splits = sorted({files[n]['declared_split'] for n in names if files[n]['declared_split']})
        historical_samples.append({'id': sample['id'], 'historical_split': sample['split'],
                                   'current_source_paths': names, 'current_declared_splits': splits,
                                   'historical_control_retired': sample['split'] in ('validation', 'test'),
                                   'new_label_release_required': True})
    # Names across separate bundles are only hints. Generic numbered names do not
    # establish byte identity or family membership.
    cross_names = defaultdict(list)
    for row in files.values():
        if row['role'] == 'input':
            cross_names[unicodedata.normalize('NFC', Path(row['path']).stem).strip().casefold()].append(row['path'])
    name_hints = [{'normalized_stem': stem, 'paths': names, 'semantic_equivalence_approved': False}
                  for stem, names in sorted(cross_names.items())
                  if len({files[n]['declared_split'] for n in names}) > 1]
    historical_hashes = {v['sha256'] for v in old.values()}
    mapped_paths = {p for group in inherited for p in group['current_paths']}
    summary = {
        'file_count': len(files), 'size_bytes': sum(r['size_bytes'] for r in rows),
        'files_by_split': {s: sum(r['declared_split'] == s for r in files.values()) for s in SPLITS},
        'files_by_bundle': {b['id']: sum(r['bundle'] == b['id'] for r in files.values()) for b in config['bundles']},
        'candidate_pairs_by_bundle': dict(sorted(Counter(r['bundle'] for r in pairs).items())),
        'candidate_pair_count': len(pairs), 'unmatched_count': len(unmatched), 'ambiguous_count': len(ambiguous),
        'unassigned_files': [n for n, r in files.items() if r['declared_split'] is None],
        'exact_duplicate_groups': len(duplicates), 'cross_split_exact_duplicate_groups': sum(r['cross_split'] for r in duplicates),
        'cross_split_relationship_components': len(conflicts), 'cross_split_name_hints': len(name_hints),
        'historical_migration_counts': dict(sorted(Counter(r['status'] for r in migration).items())),
        'current_files_with_new_hashes': sum(r['sha256'] not in historical_hashes for r in rows),
        'files_with_inherited_relationships_by_split': {s: sum(files[p]['declared_split'] == s for p in mapped_paths) for s in SPLITS},
        'historical_control_rows_retired': sum(r['historical_control_retired'] for r in historical_samples),
        'historically_excluded_target_files': sum(r['historically_excluded_native_target'] for r in files.values()),
        'training_eligible': False, 'independent_evaluation': False, 'training_started': False,
        'pair_content_verified': False, 'pdf_vector_content_verified': False,
        'model_retraining_required_by_path_change_alone': False,
        'historical_metrics_apply_to_new_controls': False,
    }
    return {'schema_version': 'dataset-layout-audit/1.0', 'summary': summary, 'files': list(files.values()),
            'candidate_pairs': pairs, 'unmatched': unmatched, 'ambiguous': ambiguous,
            'exact_duplicates': duplicates, 'inherited_relationships': inherited,
            'cross_split_relationships': conflicts, 'cross_split_name_hints': name_hints,
            'migration': migration, 'historical_ocr_samples': historical_samples}


def documents(root, config_path):
    config = read(config_path)
    validate_layout(config)
    dependencies = {key: read(checked(root, record)) for key, record in config['dependencies'].items()}
    inventory = scan_dataset(root, config)
    audit = audit_layout(inventory, config, *(dependencies[k] for k in DEPENDENCIES[:-1]))
    require(dependencies['previous_baseline_final']['training_started'] is False, 'Reassess migration of actually trained models')
    for record in config['dependencies'].values():
        checked(root, record)
    return {'inventory.json': inventory, 'audit.json': audit, 'summary.json': audit['summary']}


def build_layout_snapshot(workspace, config_path, output_dir):
    root = Path(workspace).resolve()
    config_file = inside(root, config_path)
    output = inside(root, output_dir)
    require(output.parent == root / 'artifacts' and not output.exists(), 'Choose a new direct artifacts directory')
    config_pin = pin(root, config_path)
    dataset = inside(root, read(config_file)['dataset_root'])
    require(not output.is_relative_to(dataset), 'Snapshot output must be outside source dataset')
    code_pins = [pin(root, name) for name in CODE]
    result = documents(root, config_file)
    checked(root, config_pin)
    require(code_pins == [pin(root, name) for name in CODE], 'Layout code changed during build')
    output.mkdir()
    for name, value in result.items():
        (output / name).write_bytes(encoded(value))
    manifest = {'schema_version': 'dataset-layout-manifest/1.0', 'config': config_pin,
                'code': code_pins, 'files': [pin(output, name) for name in sorted(result)]}
    (output / 'manifest.json').write_bytes(encoded(manifest))
    return result['summary.json']


def verify_layout_snapshot(workspace, manifest_pin):
    root = Path(workspace).resolve()
    path = checked(root, manifest_pin)
    manifest = read(path)
    require(set(manifest) == {'schema_version', 'config', 'code', 'files'}
            and manifest['schema_version'] == 'dataset-layout-manifest/1.0', 'Invalid layout manifest')
    require(manifest['code'] == [pin(root, name) for name in CODE], 'Layout code changed')
    config_path = checked(root, manifest['config'])
    result = documents(root, config_path)
    require(sorted(p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file())
            == sorted([*result, 'manifest.json']), 'Unexpected snapshot inventory')
    require(manifest['files'] == [pin(path.parent, name) for name in sorted(result)], 'Snapshot hashes differ')
    require(all(inside(path.parent, name).read_bytes() == encoded(value) for name, value in result.items()), 'Snapshot differs from current dataset or audit')
    return result['summary.json']
