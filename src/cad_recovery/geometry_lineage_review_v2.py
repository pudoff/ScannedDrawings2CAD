"""Replay P08.4d against frozen V1 fingerprints and current whole-group constraints."""
from collections import Counter
from html import escape
from importlib.metadata import version
from pathlib import Path

import ezdxf

from .dataset_layout_v1 import checked, encoded, inside, pin, read, verify_layout_snapshot
from .family_catalog_v2 import require
from .geometry_lineage_v1 import CODE as LEGACY_CODE, candidate_groups, fingerprint_document as legacy_fingerprint
from .geometry_lineage_v2 import fingerprint_document
from .role_admission_release_v1 import verify as verify_parent
from .source_coallocation_v1 import package

CODE = ('src/cad_recovery/geometry_lineage_v2.py', 'src/cad_recovery/geometry_lineage_review_v2.py',
        'scripts/audit_geometry_lineage_v2.py', 'src/cad_recovery/geometry_lineage_v1.py',
        'src/cad_recovery/dataset_layout_v1.py', 'src/cad_recovery/role_admission_release_v1.py',
        'src/cad_recovery/family_catalog_v2.py', 'requirements.lock')
FALSE_FLAGS = ('groups_changed', 'labels_changed', 'training_eligible', 'physical_geometry_approved',
               'semantic_family_approved', 'independent_evaluation', 'source_images_used_for_fingerprinting', 'closed_ocr_tests_executed')


def context(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'geometry-lineage-config/2.0' and config['task'] == 'P08.4d', 'Unknown lineage V2 config')
    require(config['code'] == [pin(root, n) for n in CODE] and config['ezdxf_version'] == version('ezdxf'), 'Lineage implementation/environment changed')
    require(all(config[f] is False for f in FALSE_FLAGS), 'Lineage audit cannot reassign or approve labels')
    folder, manifest = package(root, config['baseline_manifest'], 'geometry-lineage-manifest/1.0')
    baseline_config = read(checked(root, manifest['config']))
    require(baseline_config['code'] == [pin(root, n) for n in LEGACY_CODE], 'Legacy implementation changed')
    old = read(folder/'report.json')
    require(old['config'] == manifest['config'], 'Legacy report/config differ')
    verify_layout_snapshot(root, baseline_config['dependencies']['layout_manifest'])
    inventory = read(checked(root, baseline_config['dependencies']['inventory']))['files']
    catalog = read(checked(root, baseline_config['dependencies']['catalog']))['entries']
    expected, pairs = {}, {}
    files = {r['path']: r for r in inventory}
    for pair in catalog:
        require(all(files[pair[k]]['sha256'] == pair[k+'_sha256'] for k in ('input', 'target')), 'Catalog bytes differ')
        pairs.setdefault(pair['target'], []).append({'input': pair['input'], 'split': pair['declared_split'], 'visual_status': pair['visual_status']})
    for file in inventory:
        if file['path'].lower().endswith('.dxf'):
            require(file['path'] in pairs, 'Native target without routing')
            expected.setdefault(file['sha256'], []).append(file['path'])
    require(len(old['records']) == len(expected) and {r['sha256'] for r in old['records']} == set(expected), 'Legacy native SHA scope differs')
    for row in old['records']:
        require(row['paths'] == sorted(expected[row['sha256']]) and row['pairs'] == [p for n in row['paths'] for p in pairs[n]], 'Legacy path/pair lineage differs')
    verify_parent(root, config['constraint_manifest'])
    parent, _ = package(root, config['constraint_manifest'], 'role-admission-manifest/1.0')
    groups = read(parent/'groups.json')['groups']
    witnesses = read(checked(root, config['legacy_mtext_witnesses']))
    zero = {r['sha256'] for r in old['records'] if any(p['type'] == 'ZeroDivisionError' for p in r['problems'])}
    require(len(witnesses) == len(zero) and {w['sha256'] for w in witnesses} == zero, 'Legacy MTEXT witness scope differs')
    return config, old, groups, {w['sha256']: w for w in witnesses}


def compare(old, records):
    require(len({r['sha256'] for r in records}) == len(records)
            and {r['sha256'] for r in old['records']} == {r['sha256'] for r in records}, 'Comparison native scope differs')
    previous = {r['sha256']: r for r in old['records']}
    transitions, upgraded, preserved, regressions = Counter(), [], [], []
    for row in records:
        before = previous[row['sha256']]
        transitions[before['status']+' -> '+row['status']] += 1
        if before['status'] == 'complete_supported_geometry':
            same = all(row[k] == before[k] for k in ('status', 'signature', 'geometry_entities', 'geometry_types'))
            (preserved if same else regressions).append(row['sha256'])
        elif before['status'] == 'incomplete' and row['status'] == 'complete_supported_geometry':
            upgraded.append({'sha256': row['sha256'], 'paths': row['paths'], 'geometry_entities': row['geometry_entities'],
                             'geometry_types': row['geometry_types'], 'training_eligible': False})
    return {'status_transitions': dict(sorted(transitions.items())), 'unchanged_complete_sha256': sorted(preserved),
            'previously_complete_regressions': sorted(regressions), 'new_complete_records': upgraded,
            'source_images_reviewed_for_new_complete': False, 'new_complete_are_not_training_labels': True}


def review_queue(candidates, old):
    before = {tuple(sorted(c['target_sha256'])) for c in old['candidates']}
    result = []
    for c in candidates:
        if tuple(sorted(c['target_sha256'])) not in before or len(c['history_group_ids']) > 1:
            result.append({**c, 'decision': 'quarantine_for_source_review' if c['cross_split_candidate'] else 'requires_source_review'})
    return result


def materialize(root, config_pin, progress=None):
    config, old, groups, witnesses = context(root, config_pin)
    records, replays = [], []
    for index, baseline in enumerate(old['records'], 1):
        path = checked(root/'dataset', {'path': baseline['paths'][0], 'sha256': baseline['sha256']})
        try:
            document = ezdxf.readfile(path)
        except Exception as error:
            fresh = {'status': 'read_failed', 'signature': None, 'geometry_entities': 0, 'geometry_types': {},
                'ignored_annotations': {}, 'visited_entities': 0, 'conversions': {},
                'problems': [{'type': type(error).__name__, 'reason': str(error)[:300]}],
                'annotations_filtered_before_transform': True, 'geometry_signature_format': 'modelspace-native-tags/1.0',
                'physical_geometry_approved': False, 'semantic_family_approved': False}
        else:
            if baseline['sha256'] in witnesses:
                witness = witnesses[baseline['sha256']]
                require(witness['path'] == baseline['paths'][0] and witness['entity_type'] == 'MTEXT', 'Wrong legacy witness source')
                entity = document.entitydb.get(witness['native_handle'])
                require(entity is not None and entity.dxftype() == 'MTEXT' and entity.dxf.owner == witness['native_owner']
                        and entity.dxf.char_height == witness['char_height'] == 0.0 and entity.dxf.width == witness['width'], 'Legacy MTEXT identity/attributes differ')
                replay = legacy_fingerprint(document)
                require(replay == {k: baseline[k] for k in replay}, 'Legacy failed-case replay differs')
                replays.append(witness)
            fresh = fingerprint_document(document)
        records.append({**{k: baseline[k] for k in ('sha256', 'paths', 'pairs')}, **fresh})
        if progress and index % 100 == 0:
            progress(index, len(old['records']))
    comparison = compare(old, records)
    require(not comparison['previously_complete_regressions'], 'Previously complete signatures changed; separate review required')
    candidates = candidate_groups(records, groups)
    queue = review_queue(candidates, old)
    remaining = [{'sha256': r['sha256'], 'paths': r['paths'], 'status': r['status'], 'problems': r['problems']}
                 for r in records if r['status'] != 'complete_supported_geometry']
    per_document = Counter(kind for r in remaining for kind in {p.get('entity_type') or p['type'] for p in r['problems']})
    summary = {'native_files': sum(len(r['paths']) for r in records), 'unique_native_sha256': len(records),
        'by_status': dict(sorted(Counter(r['status'] for r in records).items())),
        'preserved_complete_signatures': len(comparison['unchanged_complete_sha256']),
        'new_complete': len(comparison['new_complete_records']), 'legacy_mtext_failures_replayed': len(replays),
        'remaining_problem_types_by_document': dict(sorted(per_document.items())),
        'different_sha_candidate_groups': len(candidates), 'cross_split_candidate_groups': sum(c['cross_split_candidate'] for c in candidates),
        'groups_inherited': len(groups), 'members_inherited': sum(len(g['members']) for g in groups),
        'review_queue_items': len(queue), **{f: False for f in FALSE_FLAGS}}
    report = {'schema_version': 'geometry-lineage-report/2.0', 'config': config_pin, 'summary': summary,
        'records': records, 'candidates': candidates, 'constraints': {'manifest': config['constraint_manifest'],
            'groups': groups, 'groups_modified': False},
        'method': {'scope': 'supported modelspace geometry; native blocks traversed before transforms',
            'coordinate_rounding_decimals': 9, 'signature_serialization_retained_from_v1': True,
            'transforms_applied_to_detached_geometry_leaves_only': True, 'paper_space_compared': False,
            'entity_order_ignored': True, 'multiplicity_retained': True, 'physical_scale_normalized': False,
            'native_split_scope': ['train', 'validation', 'test'], 'source_images_used_for_fingerprinting': False,
            'parent_label_images_replayed_for_verification': True,
            'models_run': False, 'closed_ocr_tests_executed': False},
        'limitations': ['A complete geometry-only fingerprint is not approval of the whole DXF or its physical labels.',
            'New complete records may be specifications or tables; no raster/text evidence is inferred.',
            'Unmatched signatures do not establish independent families. Incomplete files stay excluded.',
            'Unsupported resources, viewport content and solid geometry are not silently removed.']}
    assets = {'report.json': encoded(report), 'comparison.json': encoded(comparison),
        'remaining.json': encoded(remaining), 'legacy-mtext-replay.json': encoded(replays),
        'review-queue.json': encoded({'schema_version': 'geometry-lineage-review-queue/2.0', 'candidates': queue,
                                    **{f: False for f in FALSE_FLAGS}})}
    rows = ''.join('<tr><td>'+escape(c['id'])+'</td><td>'+'<br>'.join(escape(n) for n in c['targets'])+
                   '</td><td>'+escape(', '.join(c['declared_splits']))+'</td><td>'+str(len(c['history_group_ids']))+'</td></tr>' for c in candidates)
    assets['index.html'] = ('<!doctype html><meta charset="utf-8"><title>P08.4d geometry lineage V2</title>'
        '<style>body{font:16px sans-serif;margin:2rem}td{border:1px solid #aaa;padding:.5rem}pre{white-space:pre-wrap}</style>'
        '<h1>Geometry-only lineage V2</h1><p>No label approval, group or split changes. Native test geometry is hashed; '
        'fingerprints do not use source pixels, OCR labels or model predictions. Parent label images are replayed for integrity only.</p><pre>'+escape(str(summary))+'</pre>'
        '<p><a href="comparison.json">Compatibility</a> · <a href="remaining.json">Remaining exclusions</a> · '
        '<a href="review-queue.json">Source review queue</a></p><table>'+rows+'</table>').encode('utf-8')
    return assets, summary


def build(workspace, config_pin, output, *, progress=None):
    root = Path(workspace).resolve()
    folder = inside(root, output)
    require(not folder.exists() and not folder.resolve().is_relative_to((root/'dataset').resolve()), 'Choose a fresh output outside dataset')
    assets, summary = materialize(root, config_pin, progress)
    folder.mkdir(parents=True, exist_ok=False)
    for name, data in assets.items():
        with inside(folder, name).open('xb') as stream:
            stream.write(data)
    with (folder/'manifest.json').open('xb') as stream:
        stream.write(encoded({'schema_version': 'geometry-lineage-manifest/2.0', 'config': config_pin,
                              'files': [pin(folder, n) for n in sorted(assets)]}))
    return {'summary': summary, 'manifest': pin(root, (folder/'manifest.json').relative_to(root).as_posix())}


def verify(workspace, manifest_pin, *, progress=None):
    root = Path(workspace).resolve()
    folder, manifest = package(root, manifest_pin, 'geometry-lineage-manifest/2.0')
    assets, summary = materialize(root, manifest['config'], progress)
    require({r['path'] for r in manifest['files']} == set(assets), 'Lineage package inventory differs')
    for name, data in assets.items():
        require((folder/name).read_bytes() == data, 'Lineage package differs from source replay: '+name)
    return summary
