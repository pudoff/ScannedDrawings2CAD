"""Conservative candidates for repeated DXF exports, never family approval.

Only supported modelspace geometry participates. Text, dimensions and display
styles are ignored; coordinates are retained and quantized to nine decimals.
Unsupported or incomplete documents never enter signature matching.
"""
from collections import Counter, defaultdict
from html import escape
import math
from pathlib import Path

import ezdxf
from ezdxf.lldxf.tagwriter import TagCollector

from .dataset_layout_v1 import checked, encoded, inside, pin, read, verify_layout_snapshot
from .family_catalog_v2 import require, stable_digest

CODE = ('src/cad_recovery/geometry_lineage_v1.py', 'scripts/audit_geometry_lineage_v1.py')
GEOMETRY = {'LINE', 'POINT', 'CIRCLE', 'ARC', 'ELLIPSE', 'LWPOLYLINE',
            'POLYLINE', 'SPLINE', 'SOLID', 'TRACE', '3DFACE', 'HATCH', 'XLINE', 'RAY'}
ANNOTATIONS = {'TEXT', 'MTEXT', 'ATTRIB', 'ATTDEF', 'DIMENSION', 'LEADER',
               'MULTILEADER', 'TOLERANCE'}
DISPLAY_CODES = {5, 6, 8, 48, 60, 62, 67, 102, 105, 370, 410, 420, 430, 440}
DEPENDENCIES = {'layout_manifest', 'inventory', 'catalog', 'history_groups'}


def entity_record(entity):
    """Collect complete geometric tag data, including compound POLYLINE vertices."""
    collector = TagCollector(dxfversion='AC1032')
    entity.export_dxf(collector)
    # DXF document export writes these linked entities separately.
    if entity.dxftype() == 'POLYLINE':
        for vertex in entity.vertices:
            vertex.export_dxf(collector)
    record = []
    for tag in collector.tags:
        code, value = tag.code, tag.value
        if code in DISPLAY_CODES or 330 <= code <= 369 or code >= 1000:
            continue
        if code == 39 and value == 0:
            continue  # Explicit and omitted zero thickness are the same shape.
        if isinstance(value, float):
            require(math.isfinite(value), 'Non-finite geometric tag')
            value = round(value, 9)
            if value == 0:
                value = 0.0
        record.append([code, value])
    default_normal = [[210, 0.0], [220, 0.0], [230, 1.0]]
    # virtual_entities() can materialize the implicit world-Z normal.
    normalized = []
    index = 0
    while index < len(record):
        if record[index:index+3] == default_normal:
            index += 3
        else:
            normalized.append(record[index])
            index += 1
    record = normalized
    require(len(record) <= 200_000, 'Entity geometry tag limit')
    return record


def fingerprint_document(document, *, entity_limit=100_000, depth_limit=12):
    records, skipped, types, problems = [], Counter(), Counter(), []
    visited = 0

    def walk(entities, ancestors=()):
        nonlocal visited
        require(len(ancestors) <= depth_limit, 'Block nesting limit')
        for entity in entities:
            visited += 1
            require(visited <= entity_limit, 'Geometry entity limit')
            kind = entity.dxftype()
            if kind in ANNOTATIONS:
                skipped[kind] += 1
            elif kind == 'INSERT':
                name = entity.dxf.name
                require(name not in ancestors, 'Cyclic block reference')
                require(name in document.blocks and not document.blocks[name].block.is_xref,
                        'Missing or external block')
                require(entity.mcount <= entity_limit-visited+1, 'Block array instance limit')
                inserts = entity.multi_insert() if entity.mcount > 1 else (entity,)
                for insert in inserts:
                    def on_skip(child, reason):
                        problems.append({'type': child.dxftype(), 'reason': str(reason)})
                    walk(insert.virtual_entities(skipped_entity_callback=on_skip), ancestors+(name,))
            elif kind in GEOMETRY:
                records.append(entity_record(entity))
                types[kind] += 1
            else:
                problems.append({'type': kind, 'reason': 'unsupported_modelspace_entity'})

    try:
        walk(document.modelspace())
    except Exception as error:
        problems.append({'type': type(error).__name__, 'reason': str(error)[:300]})
    status = 'incomplete' if problems else ('empty' if not records else 'complete_supported_geometry')
    # Sorting ignores entity order, while retaining multiplicity and native coordinates.
    hashes = sorted(stable_digest(record) for record in records)
    signature = stable_digest({'method': 'modelspace-native-tags/1.0', 'entities': hashes}) if status == 'complete_supported_geometry' else None
    return {'status': status, 'signature': signature, 'geometry_entities': len(records),
            'geometry_types': dict(sorted(types.items())), 'ignored_annotations': dict(sorted(skipped.items())),
            'visited_entities': visited, 'problems': problems,
            'physical_geometry_approved': False, 'semantic_family_approved': False}


def candidate_groups(records, groups):
    require(len({r['sha256'] for r in records}) == len(records), 'Repeated native SHA record')
    membership = {}
    for group in groups:
        for name in group['members']:
            require(name not in membership, 'Repeated historical group member')
            membership[name] = group['id']
    by_signature = defaultdict(list)
    for row in records:
        if row['status'] == 'complete_supported_geometry':
            require(row['signature'] is not None and row['geometry_entities'] > 0, 'Missing geometry signature')
            by_signature[row['signature']].append(row)
    result = []
    for signature, rows in sorted(by_signature.items()):
        if len(rows) < 2:
            continue  # Records already contain one row per distinct source SHA.
        targets = sorted({path for row in rows for path in row['paths']})
        inputs = sorted({p['input'] for row in rows for p in row['pairs']})
        require(all(name in membership for name in targets+inputs), 'Unregistered candidate member')
        splits = sorted({p['split'] for row in rows for p in row['pairs']})
        result.append({'id': 'geometry-candidate-'+signature[:20], 'signature': signature,
            'target_sha256': sorted(row['sha256'] for row in rows), 'targets': targets, 'inputs': inputs,
            'declared_splits': splits, 'cross_split_candidate': len(splits) > 1,
            'history_group_ids': sorted({membership[name] for name in targets+inputs}),
            'decision': 'requires_source_and_annotation_review', 'coallocation_applied': False,
            'semantic_family_approved': False, 'pair_approved': False, 'physical_geometry_approved': False})
    return result


def build(workspace, config_pin, output, *, progress=None):
    root = Path(workspace).resolve()
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'geometry-lineage-config/1.0', 'Unknown geometry lineage config')
    require(config['code'] == [pin(root, name) for name in CODE], 'Geometry lineage code changed')
    require(set(config['dependencies']) == DEPENDENCIES, 'Incomplete lineage dependencies')
    paths = {key: checked(root, value) for key, value in config['dependencies'].items()}
    target = inside(root, output)
    require(not target.exists() and not target.resolve().is_relative_to((root/'dataset').resolve()),
            'Choose a new output outside dataset')
    verified = verify_layout_snapshot(root, config['dependencies']['layout_manifest'])
    require(config['dependencies']['inventory'] == pin(root, (paths['layout_manifest'].parent/'inventory.json').relative_to(root).as_posix()),
            'Inventory is not from the verified layout')
    inventory = read(paths['inventory'])
    require(verified['file_count'] == len(inventory['files']), 'Inventory differs')
    files = {row['path']: row for row in inventory['files']}
    require(len(files) == len(inventory['files']), 'Repeated inventory path')
    pairs = defaultdict(list)
    for pair in read(paths['catalog'])['entries']:
        require(pair['target'] in files and pair['input'] in files, 'Catalog path absent from inventory')
        require(pair['target_sha256'] == files[pair['target']]['sha256']
                and pair['input_sha256'] == files[pair['input']]['sha256'], 'Catalog bytes differ from inventory')
        require(pair['declared_split'] in ('train', 'validation', 'test'), 'Invalid declared split')
        pairs[pair['target']].append({'input': pair['input'], 'split': pair['declared_split'],
                                    'visual_status': pair['visual_status']})
    by_sha = defaultdict(list)
    for name, row in files.items():
        if name.lower().endswith('.dxf'):
            require(name in pairs, 'Unpaired native target requires explicit routing')
            by_sha[row['sha256']].append(name)
    records = []
    for index, (digest, names) in enumerate(sorted(by_sha.items()), 1):
        names.sort()
        path = checked(root, {'path': 'dataset/'+names[0], 'sha256': digest})
        try:
            result = fingerprint_document(ezdxf.readfile(path))
        except Exception as error:
            result = {'status': 'read_failed', 'signature': None, 'geometry_entities': 0,
                'geometry_types': {}, 'ignored_annotations': {}, 'visited_entities': 0,
                'problems': [{'type': type(error).__name__, 'reason': str(error)[:300]}],
                'physical_geometry_approved': False, 'semantic_family_approved': False}
        records.append({'sha256': digest, 'paths': names,
                        'pairs': [p for name in names for p in pairs[name]], **result})
        if progress and index % 100 == 0:
            progress(index, len(by_sha))
    groups = read(paths['history_groups'])['groups']
    candidates = candidate_groups(records, groups)
    summary = {'native_files': sum(len(r['paths']) for r in records), 'unique_native_sha256': len(records),
        'by_status': dict(sorted(Counter(r['status'] for r in records).items())),
        'different_sha_candidate_groups': len(candidates),
        'cross_split_candidate_groups': sum(c['cross_split_candidate'] for c in candidates),
        'new_history_group_links': sum(len(c['history_group_ids']) > 1 for c in candidates),
        'groups_changed': False, 'training_eligible': False, 'independent_evaluation': False}
    report = {'schema_version': 'geometry-lineage-report/1.0', 'config': config_pin, 'summary': summary,
        'method': {'scope': 'supported modelspace geometry, INSERT expanded; no units/scale normalization',
            'coordinate_rounding_decimals': 9, 'entity_order_ignored': True, 'multiplicity_retained': True,
            'annotations_and_display_styles_ignored': True, 'paper_space_not_compared': True,
            'translation_rotation_scale_invariance': False, 'incomplete_files_excluded_from_matching': True},
        'records': records, 'candidates': candidates,
        'limitations': ['Candidates can differ in text, dimensions, units, paper layouts or design intent.',
            'No signature match does not prove unrelated families or independent evaluation.',
            'No source images or OCR test predictions were used; no model tuning was performed.']}
    target.mkdir(parents=True, exist_ok=False)
    (target/'report.json').write_bytes(encoded(report))
    rows = ''.join('<tr><td>'+escape(c['id'])+'</td><td>'+'<br>'.join(escape(n) for n in c['targets'])+
        '</td><td>'+escape(', '.join(c['declared_splits']))+'</td><td>'+str(c['cross_split_candidate'])+'</td></tr>' for c in candidates)
    (target/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Geometry lineage candidates</title>'
        '<style>body{font:16px sans-serif;margin:2rem}td,th{padding:.6rem;border:1px solid #aaa}table{border-collapse:collapse}</style>'
        '<h1>P08.4b: кандидаты повторных DXF-экспортов</h1><p>Только проверка происхождения. '
        'Группы и split не менялись, семьи и физические метки не утверждены.</p><pre>'+escape(str(summary))+
        '</pre><p><a href="report.json">Полная опись, ошибки и ограничения</a></p>'
        '<table><tr><th>Кандидат</th><th>DXF</th><th>Split</th><th>Возможное пересечение</th></tr>'+rows+'</table>', encoding='utf-8')
    manifest = {'schema_version': 'geometry-lineage-manifest/1.0', 'config': config_pin,
                'files': [pin(target, name) for name in ('report.json', 'index.html')]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return {'summary': summary, 'manifest': pin(root, (target/'manifest.json').relative_to(root).as_posix())}
