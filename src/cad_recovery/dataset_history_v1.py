"""Reconcile historical unmatched inputs by bytes, preserving current exclusions.

This audit does not infer equivalence from filenames, approve geometry, or open
closed OCR controls. Origin-series grouping is a conservative coallocation rule.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from html import escape
from pathlib import Path

from .dataset_layout_v1 import checked, encoded, inside, pin, read, verify_layout_snapshot
from .family_catalog_v2 import canonical_sha, relative_path, require

CODE = ['src/cad_recovery/dataset_history_v1.py', 'scripts/reconcile_dataset_history_v1.py']


def indexed_files(records):
    result = {}
    folded = set()
    for row in records:
        name = relative_path(row['path'])
        canonical_sha(row['sha256'])
        require(name.casefold() not in folded, 'Duplicate inventory path')
        folded.add(name.casefold())
        result[name] = row
    return result


def reconcile(history, inventory, catalog, relationships):
    old = indexed_files(history['files'])
    current = indexed_files(inventory['files'])
    by_sha = defaultdict(list)
    for name, row in current.items():
        by_sha[row['sha256']].append(name)
    groups = deepcopy(relationships['groups'])
    membership = {}
    for group in groups:
        for name in group['members']:
            require(name in current and name not in membership, 'Invalid or repeated group member')
            membership[name] = group['id']
    require(set(membership) == set(current), 'Relationship coverage differs from inventory')
    pairs_by_file = defaultdict(list)
    split_by_file = {}
    for row in catalog['entries']:
        require(row['declared_split'] in ('train', 'validation', 'test'), 'Unknown declared split')
        for key in ('input', 'target'):
            name = row[key]
            require(name in current and current[name]['sha256'] == row[key+'_sha256'], 'Pair identity differs from inventory')
            require(name not in split_by_file or split_by_file[name] == row['declared_split'], 'Conflicting file split')
            split_by_file[name] = row['declared_split']
            pairs_by_file[name].append(row)
    records = []
    series = defaultdict(set)
    seen = set()
    for item in history['unmatched_files']:
        name = item['path']
        require(name in old and name not in seen, 'Unknown or repeated historical unmatched path')
        seen.add(name)
        before = old[name]
        matches = sorted(by_sha[before['sha256']])
        candidates = {entry['input']: entry for match in matches for entry in pairs_by_file[match]}
        records.append({'historical_path': name, 'sha256': before['sha256'], 'kind': before['kind'],
            'current_paths': matches, 'status': 'byte_identical_copy_found' if matches else 'no_identical_bytes_in_current_inventory',
            'filename_used_as_match': False, 'historical_reason': item['reason'],
            'current_group_ids': sorted({membership[path] for path in matches}),
            'current_candidates': [{'input': entry['input'], 'target': entry['target'],
                'declared_split': entry['declared_split'], 'visual_status': entry['visual_status'],
                'native_target_excluded': entry['excluded_native_target'],
                'physical_target_approved': False} for entry in sorted(candidates.values(), key=lambda v: v['input'])],
            'annotation_approved': False})
        # One legacy capture series is deliberately not claimed to be a semantic family.
        if before.get('set') == 'dataset4' and before['kind'] in ('pdf', 'image'):
            series['legacy-dataset4-origin-series'].update(matches)
    parent = {group['id']: group['id'] for group in groups}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    constraints = []
    for name, paths in sorted(series.items()):
        ids = sorted({membership[path] for path in paths})
        for gid in ids[1:]:
            parent[find(gid)] = find(ids[0])
        constraints.append({'id': name, 'members': sorted(paths), 'base_group_ids': ids,
            'basis': 'shared_historical_capture_series', 'semantic_family_approved': False})
    buckets = defaultdict(list)
    for group in groups:
        buckets[find(group['id'])].append(group)
    overlay = []
    for members in buckets.values():
        paths = sorted(path for group in members for path in group['members'])
        splits = sorted({split_by_file[path] for path in paths if path in split_by_file})
        identifier = 'history-group-'+hashlib.sha256(encoded(paths)).hexdigest()[:20]
        overlay.append({'id': identifier, 'members': paths, 'parent_group_ids': sorted(g['id'] for g in members),
            'declared_splits': splits, 'cross_split': len(splits) > 1, 'semantic_family_approved': False})
    overlay.sort(key=lambda value: value['id'])
    inputs = [row for row in records if row['kind'] in ('pdf', 'image')]
    targets = [row for row in records if row['kind'] == 'dxf']
    summary = {'historical_unmatched_files': len(records), 'historical_inputs': len(inputs),
        'historical_targets': len(targets), 'matched_inputs': sum(bool(r['current_paths']) for r in inputs),
        'inputs_without_identical_copy': sum(not r['current_paths'] for r in inputs),
        'matched_targets': sum(bool(r['current_paths']) for r in targets),
        'by_kind_and_status': dict(sorted(Counter(r['kind']+':'+r['status'] for r in records).items())),
        'base_groups': len(groups), 'conservative_groups': len(overlay),
        'cross_split_groups': sum(g['cross_split'] for g in overlay), 'independent_evaluation': False,
        'physical_labels_approved': 0, 'source_files_modified': False}
    return {'schema_version': 'dataset-history-reconciliation/1.0', 'task': 'P08.4',
        'summary': summary, 'records': records, 'coallocation_constraints': constraints,
        'groups': overlay, 'training_eligible': False,
        'limitations': ['No identical copy does not prove deletion or absence of a converted representation.',
            'Conservative coallocation is not a confirmed semantic family or independent evaluation.',
            'A prior visual pass does not approve native physical fields or the whole target.']}


def build(workspace, config_pin, output):
    root = Path(workspace).resolve()
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'dataset-history-config/1.0', 'Unknown config schema')
    require(config['code'] == [pin(root, name) for name in CODE], 'History reconciliation code changed')
    dependencies = {key: checked(root, value) for key, value in config['dependencies'].items()}
    require(set(dependencies) == {'history', 'layout_manifest', 'inventory', 'catalog', 'relationships'}, 'Incomplete inputs')
    # Existing verifier rehashes all current bytes and checks the declared split layout.
    verified_layout = verify_layout_snapshot(root, config['dependencies']['layout_manifest'])
    inventory_name = (dependencies['layout_manifest'].parent/'inventory.json').relative_to(root).as_posix()
    require(config['dependencies']['inventory'] == pin(root, inventory_name), 'Inventory is not the layout snapshot inventory')
    inventory = read(dependencies['inventory'])
    require(verified_layout['file_count'] == len(inventory['files']), 'Layout and inventory differ')
    report = reconcile(read(dependencies['history']), inventory, read(dependencies['catalog']), read(dependencies['relationships']))
    target = inside(root, output)
    require(not target.resolve().is_relative_to((root/'dataset').resolve()), 'Output cannot be inside dataset')
    target.mkdir(parents=True, exist_ok=False)
    report['config'] = config_pin
    (target/'report.json').write_bytes(encoded(report))
    (target/'groups.json').write_bytes(encoded({'schema_version': 'history-coallocation/1.0',
        'groups': report['groups'], 'constraints': report['coallocation_constraints'], 'independent_evaluation': False}))
    rows = ''.join('<tr><td>'+escape(row['historical_path'])+'</td><td>'+escape(row['kind'])+'</td><td>'+
        '<br>'.join(escape(name) for name in row['current_paths'])+'</td><td>'+escape(row['status'])+'</td></tr>'
        for row in report['records'])
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Historical data reconciliation</title>'
        '<style>body{font:16px sans-serif;margin:2rem}td,th{padding:.6rem;border:1px solid #bbb}table{border-collapse:collapse}</style>'
        '<h1>P08.4: прежняя очередь и актуальный датасет</h1><p>Совпадение SHA не утверждает физическую цель.</p>'
        '<table><tr><th>Исторический вход</th><th>Тип</th><th>Текущие копии</th><th>Решение</th></tr>'+rows+'</table>', encoding='utf-8')
    manifest = {'schema_version': 'dataset-history-manifest/1.0', 'config': config_pin,
        'files': [pin(target, name) for name in ('report.json', 'groups.json', 'index.html')]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return {'manifest': pin(root, str((target/'manifest.json').relative_to(root)).replace('\\', '/')), 'summary': report['summary']}
