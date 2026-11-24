"""Current-layout OCR release: unchanged historical text and fixed new controls."""
from collections import Counter, defaultdict
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .dataset_preparation_v1 import verify_preparation, crop_source
from .family_catalog_v2 import require
from .ocr_release_v3 import _package

CODE = ['src/cad_recovery/'+n+'.py' for n in ('ocr_current_release_v1', 'dataset_preparation_v1',
        'dataset_layout_v1', 'raster_regions', 'family_catalog_v2')]+['scripts/release_current_ocr_v1.py']
SPLITS = ('train', 'validation', 'test')
VISUAL_CHECKS = {'all_contexts_viewed', 'all_final_line_crops_viewed', 'literal_text_matches', 'original_pixels_only',
                 'no_OCR_or_DXF_reading', 'no_source_replacement', 'multiline_fields_split_into_physical_lines',
                 'low_contrast_source_remains_unresolved', 'identifier_placeholders_retained_literally'}


def prepare_config(workspace, output):
    root = Path(workspace).resolve()
    config = {'schema_version': 'current-ocr-release-config/1.0', 'task': 'P08.9c',
              'preparation': pin(root, 'artifacts/current-prepared-v1/manifest.json'),
              'previous_ocr': pin(root, 'artifacts/ocr-control-release-v2/manifest.json'),
              'reading': pin(root, 'artifacts/ocr-current-reading-v3.json'),
              'visual_review': pin(root, 'artifacts/ocr-current-visual-review-v1.json'),
              'code': [pin(root, name) for name in CODE],
              'scope': '121 inherited accepted lines in current train; first eight SHA-ranked sources per control split; title fields only',
              'cross_split_duplicates': 'reject identical OCR image or source or conservative group across splits',
              'training_started': False, 'independent_evaluation': False}
    path = inside(root, output)
    require(not path.exists(), 'Use a new release config')
    path.write_bytes(encoded(config))
    return pin(root, output)


def validate_reading(reading, protocol, prepared):
    require(reading['schema_version'] == 'current-ocr-reading/1.0' and reading['human_verified'] is False
            and reading['OCR_run'] is False, 'Unknown or misrepresented visual reading')
    selected = {r['input']: r for rows in protocol['control_selection'].values() for r in rows}
    expected = {(source, field) for source in selected for field in ('drawing_id', 'part_name')}
    slots = reading['slots']
    require(len(slots) == len(expected) and {(r['source_path'], r['field']) for r in slots} == expected,
            'Missing, replaced or duplicated control field slot')
    require(len({r['id'] for r in slots}) == len(slots), 'Repeated control slot ID')
    line_ids = set()
    for slot in slots:
        source = selected[slot['source_path']]
        require(slot['source_sha256'] == source['input_sha256'] and slot['split'] == source['declared_split']
                and prepared[slot['source_path']]['declared_split'] == slot['split'], 'Control field rebound to another source or split')
        require(slot['decision'] in ('accepted_for_ocr_transcription', 'unresolved'), 'Unknown field decision')
        if slot['decision'] == 'unresolved':
            require(slot['text'] is None and slot['lines'] == [], 'Unresolved field has a hidden OCR target')
            continue
        require(1 <= len(slot['lines']) <= 4 and slot['text'] == ' '.join(r['text'] for r in slot['lines']), 'Field text differs from physical lines')
        for i, line in enumerate(slot['lines']):
            require(line['id'] not in line_ids and line['line_index'] == i and type(line['rotation_cw_degrees']) is int
                    and line['rotation_cw_degrees'] in (0, 90, 180, 270), 'Duplicate line or invalid orientation/order')
            line_ids.add(line['id'])
            require(isinstance(line['text'], str) and line['text'].strip() == line['text']
                    and line['text'] and '\n' not in line['text'] and '\r' not in line['text'], 'Expected one literal physical text line')
    return slots


def validate_routing(rows):
    require(len({r['id'] for r in rows}) == len(rows), 'Repeated OCR sample identity')
    owners, targets = {}, defaultdict(set)
    for row in rows:
        require(row['split'] in SPLITS and row['task'] == 'ocr_recognition', 'OCR task or split changed')
        for kind, value in (('image', row['image_sha256']), ('source', row['source_sha256']), ('group', row['group_id'])):
            require((kind, value) not in owners or owners[kind, value] == row['split'], 'Cross-split OCR image/source/group leakage')
            owners[kind, value] = row['split']
        targets[row['image_sha256']].add(row['text'])
    require(all(len(values) == 1 for values in targets.values()), 'Identical OCR pixels have conflicting text')


def _replay(root, record, box, rotation, image_pin):
    require(type(rotation) is int and rotation in (0, 90, 180, 270), 'Only exact right-angle OCR rotations are allowed')
    pixels, provenance = crop_source(root, record, box)
    rotated = np.rot90(pixels, -(rotation//90)).copy()
    require(np.count_nonzero(np.min(rotated, axis=2) < 200) >= 8, 'OCR/context crop has insufficient visible ink')
    with Image.open(checked(root, image_pin)) as image:
        require(np.array_equal(rotated, np.array(image.convert('RGB'))), 'OCR crop differs from current original source pixels')
    return provenance


def _documents(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'current-ocr-release-config/1.0' and config['task'] == 'P08.9c'
            and config['code'] == [pin(root, p) for p in CODE]
            and config['training_started'] is False and config['independent_evaluation'] is False, 'Changed release protocol/code')
    prepared_rows, protocol = verify_preparation(root, config['preparation'], all_assets=False)
    prepared = {r['input']: r for r in prepared_rows}
    reading = read(checked(root, config['reading']))
    require(reading['preparation_config'] == read(checked(root, config['preparation']))['config'], 'Reading used another selection protocol')
    review = read(checked(root, config['visual_review']))
    require(review['reading'] == config['reading'] and review['human_verified'] is False
            and review['reviewer'] == 'assistant' and set(review['checks']) == VISUAL_CHECKS
            and all(v is True for v in review['checks'].values()), 'Missing visual reading checks')
    checked(root, review['generator'])
    slots = validate_reading(reading, protocol, prepared)
    contexts = read(checked(root, reading['context_evidence']))
    require(contexts['protocol'] == reading['preparation_config'], 'Contexts differ from fixed selection')
    context_by_source = {r['source']['input']: r for r in contexts['sources']}
    require(set(context_by_source) == {r['source_path'] for r in slots}
            and len(context_by_source) == len(contexts['sources']), 'Incomplete source contexts')
    evidence = []
    for source, context in context_by_source.items():
        require(context['source']['input_sha256'] == prepared[source]['input_sha256'], 'Context source changed')
        provenance = _replay(root, prepared[source], context['box_source_xyxy'], 0, context['context'])
        evidence.append({'source_path': source, 'context': context['context'], 'provenance': provenance})
    previous_path, _ = _package(root, config['previous_ocr'])
    previous_dir = previous_path.parent
    historical_labels = read(previous_dir/'labels.json')
    old_samples = historical_labels['inherited_labels']['inherited_samples'] + historical_labels['inherited_labels']['new_samples'] + historical_labels['control_samples']
    samples_by_id = {r['id']: r for r in old_samples}
    prior_rows = [json.loads(line) for split in SPLITS for line in (previous_dir/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
    old_registry = {r['id']: r for r in read(previous_dir/'task-splits.json')['assignments']}
    require(len(prior_rows) == len(samples_by_id) == 121 and {r['id'] for r in prior_rows} == set(samples_by_id), 'Inherited accepted sample inventory differs')
    images, rows, replay = {}, [], []
    for previous in prior_rows:
        binding = old_registry[previous['id']]
        source = binding['source_path']
        require(source in prepared and prepared[source]['input_sha256'] == previous['source_sha256']
                and prepared[source]['declared_split'] == 'train', 'Historical OCR source is not in current train')
        sample = samples_by_id[previous['id']]
        require(sample['text'] == previous['text'], 'Inherited OCR text changed')
        image_pin = {'path': (previous_dir/previous['image']).relative_to(root).as_posix(), 'sha256': previous['image_sha256']}
        proof = _replay(root, prepared[source], sample['source_box_xyxy'], sample['rotation_cw_degrees'], image_pin)
        row = {**previous, 'split': 'train', 'group_id': prepared[source]['group_id'], 'source_path': source,
               'field': sample.get('field', sample.get('role', 'inherited_literal_text')), 'line_index': sample.get('line_index', 0),
               'historical_split': previous['split'], 'name_is_identifier_placeholder': False}
        rows.append(row); images[row['image']] = checked(root, image_pin).read_bytes()
        replay.append({'id': row['id'], 'source_box_xyxy': sample['source_box_xyxy'], 'rotation_cw_degrees': sample['rotation_cw_degrees'],
                       'original_image': image_pin, 'provenance': proof})
    for slot in slots:
        checked(root, slot['context'])
        require(slot['context'] == context_by_source[slot['source_path']]['context'], 'Field uses a different context')
        for line in slot['lines']:
            source = prepared[slot['source_path']]
            context_box = context_by_source[slot['source_path']]['box_source_xyxy']
            box = line['source_box_xyxy']
            require(len(box) == 4 and all(type(v) is int for v in box)
                    and context_box[0] <= box[0] < box[2] <= context_box[2]
                    and context_box[1] <= box[1] < box[3] <= context_box[3], 'Line crop is outside its reviewed context')
            proof = _replay(root, source, line['source_box_xyxy'], line['rotation_cw_degrees'], line['image'])
            row = {'id': line['id'], 'source_id': 'layout1-'+slot['source_sha256'][:16], 'image': 'images/'+line['id']+'.png',
                   'image_sha256': line['image']['sha256'], 'text': line['text'], 'source_sha256': slot['source_sha256'],
                   'source_path': slot['source_path'], 'group_id': source['group_id'], 'split': slot['split'], 'task': 'ocr_recognition',
                   'field': slot['field'], 'slot_id': slot['id'], 'line_index': line['line_index'], 'historical_split': None,
                   'name_is_identifier_placeholder': slot.get('name_is_identifier_placeholder', False)}
            rows.append(row); images[row['image']] = checked(root, line['image']).read_bytes()
            replay.append({'id': row['id'], 'source_box_xyxy': line['source_box_xyxy'], 'rotation_cw_degrees': line['rotation_cw_degrees'],
                           'original_image': line['image'], 'provenance': proof})
    validate_routing(rows)
    rows.sort(key=lambda r: (SPLITS.index(r['split']), r['id']))
    require(review['context_count'] == len(contexts['sources']) and review['accepted_line_count'] == sum(len(s['lines']) for s in slots)
            and review['field_slot_count'] == len(slots) and review['unresolved_slot_count'] == sum(s['decision'] == 'unresolved' for s in slots), 'Visual review counts differ')
    content = read(inside(root, protocol['review']['path']).parent/'catalog.json')
    groups = content['relationships']['groups']
    touched = {r['group_id'] for r in rows}
    commitments = [g for g in groups if g['id'] in touched]
    require(all(len(g['declared_splits']) == 1 and not g['cross_split'] for g in commitments), 'Current OCR group crosses directory split')
    by_hash = defaultdict(list)
    for row in rows: by_hash[row['image_sha256']].append(row['id'])
    pending = [r['input'] for r in prepared_rows if r['declared_split'] in ('validation', 'test') and r['input'] not in context_by_source]
    registry = {'schema_version': 'current-ocr-splits/1.0', 'task': 'ocr_recognition', 'split_authority': 'current_user_directory_layout',
                'assignments': rows, 'group_commitments': commitments,
                'splits': {s: [r['id'] for r in rows if r['split'] == s] for s in SPLITS},
                'historical_control_rows_now_train': sum(r['historical_split'] in ('validation', 'test') for r in rows),
                'independent_evaluation': False}
    labels = {'schema_version': 'current-ocr-labels/1.0', 'historical_metadata_snapshot': historical_labels,
              'current_reading': reading, 'current_visual_review': review, 'new_field_ledger': slots,
              'pending_control_sources': pending, 'training_eligible_for_task': 'ocr_recognition_only',
              'human_verified': False, 'physical_geometry_approved': False, 'native_target_approved': False,
              'semantic_family_approved': False, 'independent_evaluation': False}
    report = {'schema_version': 'current-ocr-release/1.0', 'task': 'P08.9c', 'config': config_pin,
              'sample_count': len(rows), 'task_split_counts': {s: sum(r['split'] == s for r in rows) for s in SPLITS},
              'source_counts_by_split': {s: len({r['source_path'] for r in rows if r['split'] == s}) for s in SPLITS},
              'selected_control_sources': len(context_by_source), 'field_slot_count': len(slots),
              'field_decision_counts': dict(Counter(s['decision'] for s in slots)), 'pending_control_source_count': len(pending),
              'placeholder_name_slots': sum(s.get('name_is_identifier_placeholder', False) for s in slots),
              'historical_texts_images_preserved': 121, 'historical_control_rows_now_train': registry['historical_control_rows_now_train'],
              'duplicate_image_groups': [ids for ids in by_hash.values() if len(ids) > 1],
              'pixel_crops_replayed': len(replay), 'contexts_replayed': len(evidence), 'training_started': False, 'OCR_run': False,
              'independent_evaluation': False, 'physical_geometry_approved': False,
              'limitations': ['Initial fixed subset of dedicated control folders; 165 other sources remain pending.',
                              'Two faint fields of one selected test source remain unresolved with null targets; source was not replaced.',
                              'All selected validation name cells are identifier placeholders, not actual part names.',
                              'One multi-line cell creates several recognition lines; line count is not independent source count.',
                              'Historical split/label metadata is preserved as history; current assignments govern the new release.',
                              'Current quarantine, native exclusions, weak family hints and unknown geometry are not cleared by OCR text acceptance.']}
    files = {'labels.json': encoded(labels), 'task-splits.json': encoded(registry), 'report.json': encoded(report),
             'pixel-evidence.json': encoded({'contexts': evidence, 'crops': replay}),
             'source-dispositions.json': encoded(prepared_rows), 'relationships.json': encoded(content['relationships']), **images}
    for split in SPLITS:
        files[split+'.jsonl'] = b''.join(encoded(r).replace(b'\n', b'')+b'\n' for r in rows if r['split'] == split)
    page = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR: текущие выборки</title>',
            '<style>body{font:17px system-ui;max-width:1200px;margin:auto}img{max-width:100%}section{margin:2em 0}</style>',
            '<h1>OCR: новая структура данных</h1><p>121 прежняя строка в train. Контроль — фиксированное подмножество новых каталогов. Независимость семейств и физическая геометрия не утверждены.</p>',
            '<p><a href="report.json">Сводка</a> · <a href="labels.json">Все метки и неопределённости</a> · <a href="source-dispositions.json">Карантин источников</a></p>']
    for split in SPLITS:
        page.append('<h2>'+split+'</h2>')
        for row in rows:
            if row['split'] == split:
                page.append('<section><p>'+escape(row['id']+' / '+row['field'])+'</p><b>'+escape(row['text'])+'</b><br><img loading="lazy" src="'+escape(row['image'], quote=True)+'"></section>')
    files['index.html'] = (''.join(page)+'</html>').encode('utf-8')
    return files, report, rows


def release_current_ocr(workspace, config_pin, output):
    root = Path(workspace).resolve()
    target = inside(root, output)
    require(target.parent == root/'artifacts' and not target.exists(), 'Choose a new direct artifacts directory')
    files, report, _ = _documents(root, config_pin)
    target.mkdir()
    for name, data in files.items():
        path = inside(target, name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    manifest = {'schema_version': 'current-ocr-manifest/1.0', 'config': config_pin,
                'files': [pin(target, name) for name in sorted(files)]}
    (target/'manifest.json').write_bytes(encoded(manifest))
    return report


def verify_current_ocr(workspace, manifest_pin):
    root = Path(workspace).resolve()
    path, manifest = _package(root, manifest_pin)
    require(set(manifest) == {'schema_version', 'config', 'files'} and manifest['schema_version'] == 'current-ocr-manifest/1.0', 'Unknown current OCR manifest')
    files, report, rows = _documents(root, manifest['config'])
    require(manifest['files'] == [pin(path.parent, name) for name in sorted(files)]
            and all(inside(path.parent, name).read_bytes() == value for name, value in files.items()), 'Current OCR package differs from replay')
    return report, rows, path.parent


def load_current_ocr(workspace, manifest_pin, *, split, task='ocr_recognition'):
    require(task == 'ocr_recognition' and split in SPLITS, 'Explicit OCR recognition task and split required')
    report, rows, base = verify_current_ocr(workspace, manifest_pin)
    return {'task': task, 'split': split, 'records': [{**r, 'image_path': str(inside(base, r['image']))} for r in rows if r['split'] == split],
            'report': report, 'training_started': False, 'independent_evaluation': False}
