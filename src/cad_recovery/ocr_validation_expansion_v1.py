"""Original-pixel validation extension, preserving the complete parent OCR release."""
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .dataset_preparation_v1 import crop_source
from .family_catalog_v2 import require
from .ocr_current_release_v1 import verify_current_ocr, validate_routing
from .ocr_release_v3 import _package

PARENT = {'path': 'artifacts/ocr-training-runtime-final-v1.json',
          'sha256': '066fd3e36785e782f3acf1acb3248ffd61d33ea183c0cd3ad74c5281e6514f0b'}
RELEASE = {'path': 'artifacts/ocr-current-release-v1/manifest.json',
           'sha256': 'faae846b483d94e5a6e4ab1c8cc77270f72598334ce64fae35e695e556f7928a'}
SELECTION = {'path': 'configs/reviews/ocr-validation-expansion-selection-v1.json',
             'sha256': 'bd959572ce4ced6021f8ac065b770a00b724a0cd6398bb36917cedc6beeb5033'}
CODE = ['src/cad_recovery/ocr_validation_expansion_v1.py', 'scripts/release_ocr_validation_expansion_v1.py']
DECISIONS = {'accepted_for_ocr_transcription', 'identifier_placeholder', 'absent', 'designation_in_name_cell', 'unresolved'}
CHECKS = {'all_61_title_contexts_reviewed', 'all_20_nonplaceholder_name_cells_reviewed',
          'all_24_dimension_contexts_reviewed', 'all_48_final_crops_reviewed',
          'missing_title_full_source_checked', 'missing_glyph_not_inferred',
          'failed_navigation_proposals_corrected', 'no_OCR_DXF_or_test_for_transcription',
          'original_pixels_only', 'train_test_preserved'}


def prepare_config(workspace, output):
    root = Path(workspace).resolve()
    checked(root, PARENT); checked(root, RELEASE); checked(root, SELECTION)
    config = {'schema_version': 'ocr-validation-expansion-config/1.0', 'task': 'P16.2a.3',
              'parent': PARENT, 'release': RELEASE, 'selection': SELECTION,
              'reading': pin(root, 'artifacts/ocr-validation-expansion-reading-v1.json'),
              'visual_review': pin(root, 'artifacts/ocr-validation-expansion-visual-review-v1.json'),
              'code': [pin(root, name) for name in CODE],
              'task_scope': 'validation recognition only; no physical geometry/native targets or independence approval',
              'training_started': False, 'OCR_run': False}
    path = inside(root, output); require(not path.exists(), 'Choose a new extension config')
    path.write_bytes(encoded(config)); return pin(root, output)


def _box(value, size, message):
    require(isinstance(value, list) and len(value) == 4 and all(type(v) is int for v in value)
            and 0 <= value[0] < value[2] <= size[0] and 0 <= value[1] < value[3] <= size[1]
            and (value[2]-value[0])*(value[3]-value[1]) <= 16777216, message)


def validate_reading(reading, selection, sources):
    require(reading['schema_version'] == 'ocr-validation-expansion-reading/1.0'
            and all(reading[k] is False for k in ('OCR_run', 'test_used', 'DXF_read_for_transcription',
                                                'human_verified', 'independent_evaluation')), 'Changed reading scope')
    validation = sorted((r for r in sources if r['declared_split'] == 'validation'), key=lambda r: (r['input_sha256'], r['input']))
    selected = [{'input': r['input'], 'input_sha256': r['input_sha256'], 'group_id': r['group_id']} for r in validation]
    require(selection['title_sources'] == selected and len(selected) == 61
            and selection['dimension_sources'] == [r['input'] for r in validation[:12]], 'Changed fixed validation source selection')
    source_by_name = {r['input']: r for r in validation}
    expected = {(r['input'], 'part_name') for r in validation}
    expected.update((r['input'], f'dimension_{i}') for r in validation[:12] for i in (1, 2))
    slots = reading['slots']
    require(len(slots) == len(expected) and {(s['source_path'], s['field']) for s in slots} == expected
            and len({s['id'] for s in slots}) == len(slots), 'Missing, duplicated or substituted field slot')
    ids = set()
    for slot in slots:
        source = source_by_name[slot['source_path']]
        require(type(slot['source_index']) is int and 1 <= slot['source_index'] <= len(validation), 'Invalid source index')
        require(slot['split'] == 'validation' and slot['source_sha256'] == source['input_sha256']
                and slot['group_id'] == source['group_id']
                and validation[slot['source_index']-1]['input'] == slot['source_path'], 'Field rebound to another source/group/split')
        require(slot['decision'] in DECISIONS and slot['reason'], 'Unknown or unexplained field decision')
        require(slot['id'] == 'valexpand-'+slot['source_sha256'][:16]+'-'+slot['field'], 'Changed field identity')
        require(slot['name_is_identifier_placeholder'] == (slot['decision'] == 'identifier_placeholder'), 'Changed placeholder meaning')
        _box(slot['context_box_xyxy'], source['source_size_px'], 'Invalid source context box')
        if slot['decision'] != 'accepted_for_ocr_transcription':
            require(slot['text'] is None and slot['lines'] == [], 'Rejected field has hidden OCR targets')
            require(slot['decision'] != 'identifier_placeholder' or slot['field'] == 'part_name', 'Dimension relabelled as placeholder')
            continue
        require(isinstance(slot['lines'], list) and 1 <= len(slot['lines']) <= 3
                and slot['text'] == ' '.join(l['text'] for l in slot['lines']), 'Field text differs from physical components')
        for i, line in enumerate(slot['lines']):
            require(line['id'] not in ids and line['line_index'] == i, 'Repeated/unordered physical line')
            require(line['id'] == slot['id']+f'-{i+1}' and type(line['crop_contains_rule_pixels']) is bool, 'Changed line identity/rule flag')
            ids.add(line['id'])
            require(isinstance(line['text'], str) and line['text'] and line['text'].strip() == line['text']
                    and not any(c in line['text'] for c in '\r\n\t\x00'), 'Empty or multiline recognition target')
            require(type(line['rotation_cw_degrees']) is int and line['rotation_cw_degrees'] in (0, 90, 180, 270), 'Invalid line orientation')
            _box(line['source_box_xyxy'], source['source_size_px'], 'Invalid source line box')
            a = slot['context_box_xyxy']; b = line['source_box_xyxy']
            require(a[0] <= b[0] < b[2] <= a[2] and a[1] <= b[1] < b[3] <= a[3], 'Line is outside reviewed context')
            if slot['field'] == 'part_name': require(line['component_role'] == 'physical_name_line', 'Name component role changed')
            else: require(line['component_role'] in ('physical_dimension_line', 'nominal', 'upper_deviation', 'lower_deviation'), 'Dimension component role changed')
    return slots


def extend_rows(old_rows, slots, sources):
    source_map = {r['input']: r for r in sources}; new = []
    for slot in slots:
        for line in slot['lines']:
            source = source_map[slot['source_path']]
            new.append({'id': line['id'], 'source_id': 'layout1-'+slot['source_sha256'][:16],
                'image': 'images/'+line['id']+'.png', 'image_sha256': line['image']['sha256'], 'text': line['text'],
                'source_path': slot['source_path'], 'source_sha256': slot['source_sha256'],
                'group_id': source['group_id'], 'split': 'validation', 'task': 'ocr_recognition',
                'field': slot['field'], 'slot_id': slot['id'], 'line_index': line['line_index'],
                'historical_split': None, 'name_is_identifier_placeholder': False,
                'text_kind': 'name_literal' if slot['field'] == 'part_name' else 'dimension_literal',
                'component_role': line['component_role'], 'crop_contains_rule_pixels': line['crop_contains_rule_pixels']})
    rows = old_rows+new
    require(len({r['id'] for r in rows}) == len(rows), 'Extension overwrites an inherited OCR row')
    validate_routing(rows)
    hashes = defaultdict(list)
    for r in rows: hashes[r['image_sha256']].append(r)
    require(all(len({r['text'] for r in group}) == 1 for group in hashes.values()), 'Identical OCR pixels have conflicting labels')
    return rows, new


def _replay(root, source, box, rotation, image_pin):
    rgb, proof = crop_source(root, source, box)
    if rotation: rgb = np.rot90(rgb, -rotation//90).copy()
    path = checked(root, image_pin)
    with Image.open(path) as image:
        require(image.mode == 'RGB' and np.array_equal(np.array(image), rgb), 'Released crop differs from original RGB pixels')
    return proof


def _documents(workspace, config_pin):
    root = Path(workspace).resolve(); config = read(checked(root, config_pin))
    require(config['schema_version'] == 'ocr-validation-expansion-config/1.0' and config['task'] == 'P16.2a.3'
            and config['parent'] == PARENT and config['release'] == RELEASE and config['selection'] == SELECTION
            and config['training_started'] is False and config['OCR_run'] is False, 'Changed extension protocol')
    require([p['path'] for p in config['code']] == CODE, 'Incomplete extension code pins')
    for p in config['code']: checked(root, p)
    checked(root, PARENT)
    selection = read(checked(root, SELECTION)); reading = read(checked(root, config['reading']))
    require(reading['selection'] == SELECTION, 'Reading has another source selection')
    review = read(checked(root, config['visual_review']))
    require(review['reading'] == config['reading'] and review['selection'] == SELECTION
            and set(review['checks']) == CHECKS and all(review['checks'].values())
            and review['human_verified'] is False and review['OCR_run'] is False and review['test_used'] is False,
            'Missing or misrepresented visual review')
    require(len(review['title_sheets']) == 11 and len(review['localized_field_sheets']) == 6
            and len(review['accepted_line_sheets']) == 4, 'Incomplete visual contact-sheet review')
    for key in ('title_sheets', 'localized_field_sheets', 'accepted_line_sheets'):
        for image_pin in review[key]: checked(root, image_pin)
    old_report, old_rows, base = verify_current_ocr(root, RELEASE)
    sources = read(base/'source-dispositions.json'); source_map = {r['input']: r for r in sources}
    slots = validate_reading(reading, selection, sources)
    require(review['accepted_lines'] == [l['image'] for s in slots for l in s['lines']], 'Visual review crop inventory differs')
    evidence = read(checked(root, reading['review_evidence']))
    localization = read(checked(root, reading['localization']))
    require(evidence['selection'] == SELECTION and localization['selection'] == SELECTION, 'Navigation has another source selection')
    files = {r['image']: (base/r['image']).read_bytes() for r in old_rows}
    replays = []
    context_names = {}
    def add_context(image_pin, source, box, name):
        proof = _replay(root, source, box, 0, image_pin)
        require(name not in files, 'Repeated context asset name')
        files[name] = checked(root, image_pin).read_bytes()
        context_names[image_pin['path']] = name
        replays.append({'image': name, 'source_path': source['input'], 'source_sha256': source['input_sha256'],
                        'source_box_xyxy': box, 'original_image': image_pin, 'pixel_provenance': proof})
    require(len(evidence['sources']) == 61
            and [e['source']['input'] for e in evidence['sources']] == [s['input'] for s in selection['title_sources']], 'Incomplete title review inventory')
    for e in evidence['sources']:
        source = source_map[e['source']['input']]
        require(e['source']['input_sha256'] == source['input_sha256'] and e['source']['group_id'] == source['group_id']
                and selection['title_sources'][e['index']-1]['input'] == source['input'], 'Review context rebound')
        prep_path = checked(root, source['preparation']); prep = read(prep_path)
        overview_pin = {'path': (prep_path.parent/prep['overview']['path']).relative_to(root).as_posix(),
                        'sha256': prep['overview']['sha256']}
        require(e['overview'] == overview_pin, 'Overview differs from source preparation')
        add_context(e['title_context'], source, e['context_box_xyxy'], f"contexts/title-{e['index']:02d}.png")
        files[f"overviews/source-{e['index']:02d}.png"] = checked(root, e['overview']).read_bytes()
    corrected = {(c['index'], c['rank']): c for c in reading['corrected_localizations']}
    for c in localization['contexts']:
        if c['kind'] == 'dimension':
            c = corrected.get((c['index'], c['rank']), c)
            add_context(c['context'], source_map[c['source']['input']], c['context_box_xyxy'], f"contexts/dimension-{c['index']:02d}-{c['rank']}.png")
        elif c['index'] == 15:
            add_context(c['context'], source_map[c['source']['input']], c['context_box_xyxy'], 'contexts/title-15-relocated.png')
    for slot in slots:
        require(slot['context']['path'] in context_names, 'Unreviewed field context')
        for line in slot['lines']:
            proof = _replay(root, source_map[slot['source_path']], line['source_box_xyxy'], line['rotation_cw_degrees'], line['image'])
            name = 'images/'+line['id']+'.png'; files[name] = checked(root, line['image']).read_bytes()
            replays.append({'image': name, 'slot_id': slot['id'], 'source_path': slot['source_path'],
                            'source_box_xyxy': line['source_box_xyxy'], 'rotation_cw_degrees': line['rotation_cw_degrees'],
                            'original_image': line['image'], 'pixel_provenance': proof})
    rows, new = extend_rows(old_rows, slots, sources)
    relationships = read(base/'relationships.json'); touched = {r['group_id'] for r in rows}
    commitments = [g for g in relationships['groups'] if g['id'] in touched]
    require({g['id'] for g in commitments} == touched and all(len(g['declared_splits']) == 1 and not g['cross_split'] for g in commitments), 'Changed or missing group commitment')
    registry = {'schema_version': 'current-ocr-splits/2.0', 'task': 'ocr_recognition', 'assignments': rows,
                'splits': {s: [r['id'] for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')},
                'group_commitments': commitments, 'split_authority': 'current_user_directory_layout', 'independent_evaluation': False}
    duplicates = defaultdict(list)
    for row in rows: duplicates[row['image_sha256']].append(row['id'])
    report = {'schema_version': 'ocr-validation-expansion-release/1.0', 'task': 'P16.2a.3', 'config': config_pin,
        'task_split_counts': dict(Counter(r['split'] for r in rows)), 'new_validation_line_count': len(new),
        'new_validation_lines_by_kind': dict(Counter(r['text_kind'] for r in new)),
        'new_dimension_lines_by_component': dict(Counter(r['component_role'] for r in new if r['text_kind'] == 'dimension_literal')),
        'source_counts_by_split': {s: len({r['source_path'] for r in rows if r['split'] == s}) for s in ('train', 'validation', 'test')},
        'new_slot_count': len(slots), 'new_slot_decisions': dict(Counter(s['decision'] for s in slots)),
        'new_real_name_fields': sum(s['field'] == 'part_name' and bool(s['lines']) for s in slots),
        'new_dimension_fields': sum(s['field'] != 'part_name' and bool(s['lines']) for s in slots),
        'title_sources_reviewed': 61, 'dimension_sources_reviewed': 12, 'dimension_sources_not_reviewed': 49,
        'new_rotated_lines': sum(l['rotation_cw_degrees'] != 0 for s in slots for l in s['lines']),
        'new_crops_with_rule_pixels': sum(r['crop_contains_rule_pixels'] for r in new),
        'original_rows_preserved': len(old_rows), 'train_test_jsonl_unchanged': True,
        'historical_pending_control_source_count': old_report['pending_control_source_count'],
        'pending_field_scope': {'validation_drawing_id_not_added': 53, 'validation_dimension_sources_not_reviewed': 49,
                                'test_pending_sources_unchanged': 112},
        'duplicate_image_groups': [ids for ids in duplicates.values() if len(ids) > 1],
        'contexts_replayed': len(replays)-len(new), 'new_line_crops_replayed': len(new),
        'training_started': False, 'OCR_run': False, 'human_verified': False, 'independent_evaluation': False,
        'physical_geometry_approved': False, 'native_target_approved': False,
        'training_ready': False, 'remaining': ['P16.2a.4 rec-only baseline and experiment freeze', 'P16.2b corpus training'],
        'limitations': ['Visual transcription by implementing assistant; no independent human annotation review.',
          'All current names reviewed, only two selected dimension fields on first twelve SHA-ranked sources.',
          'Physical components of a tolerance remain separate; no synthetic stitched image or inferred nominal value.',
          'Five accepted crops retain rule/arrow pixels; preserve them in subsequent evaluation.',
          'Conservative groups, weak relatedness hints and all native/geometry quarantines remain; independence is unproven.',
          'Parent test uncertainty and incomplete field coverage remain unchanged; historical pending count is not a full-sheet annotation claim.']}
    labels = {'schema_version': 'current-ocr-labels/2.0', 'previous_labels': read(base/'labels.json'),
              'validation_expansion': reading, 'visual_review': review, 'new_field_ledger': slots,
              'training_eligible_for_task': 'ocr_recognition_only', 'human_verified': False,
              'independent_evaluation': False, 'physical_geometry_approved': False, 'native_target_approved': False}
    files.update({'report.json': encoded(report), 'labels.json': encoded(labels), 'task-splits.json': encoded(registry),
                  'source-dispositions.json': (base/'source-dispositions.json').read_bytes(),
                  'relationships.json': (base/'relationships.json').read_bytes(), 'pixel-evidence.json': encoded(replays),
                  'train.jsonl': (base/'train.jsonl').read_bytes(), 'test.jsonl': (base/'test.jsonl').read_bytes(),
                  'validation.jsonl': (base/'validation.jsonl').read_bytes()+b''.join(encoded(r).replace(b'\n', b'')+b'\n' for r in new)})
    page = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Validation: названия и размеры</title>',
        '<style>body{font:17px system-ui;max-width:1150px;margin:auto}img{max-width:100%;max-height:260px}section{margin:2em 0}</style>',
        '<h1>Validation: названия и размерные строки</h1><p>Исходные train/test сохранены. Новые цели — визуальное чтение растра; независимость семейств и физическая геометрия не подтверждены.</p>',
        '<p><a href="report.json">Сводка</a> · <a href="labels.json">Все решения и неопределённости</a></p>']
    for slot in slots:
        page.append('<section><h3>'+escape(slot['id'])+'</h3><p>'+escape(slot['decision']+' / '+slot['reason'])+'</p>')
        page.append('<a href="'+escape(context_names[slot['context']['path']], quote=True)+'">Контекст исходного растра</a>')
        for line in slot['lines']:
            page.append('<p>'+escape(line['text'])+'</p><img loading="lazy" src="images/'+escape(line['id'], quote=True)+'.png">')
        page.append('</section>')
    files['index.html'] = (''.join(page)+'</html>').encode('utf-8')
    return files, report, rows


def build_release(workspace, config_pin, output):
    root = Path(workspace).resolve(); target = inside(root, output)
    require(target.parent == root/'artifacts' and not target.exists(), 'Choose a new artifacts output')
    files, report, _ = _documents(root, config_pin); target.mkdir()
    for name, value in files.items():
        path = inside(target, name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)
    (target/'manifest.json').write_bytes(encoded({'schema_version': 'ocr-validation-expansion-manifest/1.0',
        'config': config_pin, 'files': [pin(target, name) for name in sorted(files)]}))
    return report


def verify_release(workspace, manifest_pin):
    root = Path(workspace).resolve(); path, manifest = _package(root, manifest_pin)
    require(manifest['schema_version'] == 'ocr-validation-expansion-manifest/1.0', 'Unknown extension manifest')
    files, report, rows = _documents(root, manifest['config'])
    require(manifest['files'] == [pin(path.parent, name) for name in sorted(files)]
            and all(inside(path.parent, name).read_bytes() == value for name, value in files.items()), 'Extension package differs from replay')
    return report, rows, path.parent


def load_release(workspace, manifest_pin, *, split, task='ocr_recognition'):
    require(split in ('train', 'validation', 'test') and task == 'ocr_recognition', 'Explicit recognition split/task required')
    report, rows, base = verify_release(workspace, manifest_pin)
    return {'task': task, 'split': split, 'records': [{**r, 'image_path': str(inside(base, r['image']))} for r in rows if r['split'] == split],
            'report': report, 'training_ready': False, 'independent_evaluation': False}
