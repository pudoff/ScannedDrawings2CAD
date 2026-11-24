"""Build and authenticate the first source-only OCR development release.

Four reviewed recognition targets inherit train-only lineage. Their release
does not approve CAD, geometry, OCR detection boxes, or a physical scale.
The consumer requires a trusted manifest hash and replays the release recipe.
"""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
import shutil
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .clear_pairs import contained, digest
from .label_release_v1 import checked_pin
from .label_schema_v1 import Pin, read_json_v1
from .ocr_label_schema_v1 import (OcrLabelPackageV1, OcrSampleV1, make_ocr_transform_v1,
                                  parse_printed_components_v1, read_ocr_labels_v1)
from .ocr_lineage_v1 import derive_source_lineage_v1, validate_ocr_assignments_v1
from .raster_regions import read_prepared_region
from .training_target_policy import screen_training_target


REVIEW_DIR = 'artifacts/rib-ocr-reading-v1/'
KEY_PATHS = {
    'source': 'dataset/clear_sum_dataset/vse_png/МТСП.24.08.012 Ребро.png',
    'native': 'dataset/clear_sum_dataset/vse_dxf/МТСП.24.08.012_Ребро.dxf',
    'preparation': 'artifacts/clear-raster-prepared-v1/source-0031-097db6a4e770/preparation.json',
    'review': REVIEW_DIR+'review.json', 'review_manifest': REVIEW_DIR+'manifest.json',
    'visual_check': REVIEW_DIR+'visual-check.json',
    'parent_selection': 'artifacts/rib-independent-review-v1/holdout-selection.json',
    'parent_fields': 'artifacts/rib-mini-review-v1/field-review.json',
    'parent_manifest': 'artifacts/rib-mini-review-v1/manifest.json',
    'parent_report': 'artifacts/rib-mini-review-v1/report.json',
    'parent_final': 'artifacts/rib-mini-final-v1.json',
    'old_labels': 'artifacts/labels-corpus-v1/customer-rebro.json',
    'catalog': 'artifacts/family-split-v2/families.json',
    'split_policy': 'artifacts/family-split-v2/splits.json',
    'exclusions': 'artifacts/resource-disposition-v1/excluded-targets.json',
}
CODE_FILES = ['src/cad_recovery/'+name+'.py' for name in (
    'rib_ocr_release_v1', 'ocr_label_schema_v1', 'ocr_lineage_v1',
    'label_schema_v1', 'label_release_v1', 'clear_pairs', 'family_catalog_v2',
    'raster_regions', 'training_target_policy')]+['scripts/release_rib_ocr_v1.py']
RECIPE = [
    ('rib-1B2-length-angle', '1B2', 0, '20×45°', [1840, 3410, 2300, 3570], 0, 'chamfer-left-label'),
    ('rib-1B2-count', '1B2', 1, '2 фаски', [1840, 3630, 2375, 3810], 0, 'chamfer-left-label'),
    ('rib-1C5-nominal', '1C5', 0, '160', [1760, 2210, 1910, 2460], 90, 'inner-height-label'),
    ('rib-1EB-nominal', '1EB', 0, '100', [2030, 2330, 2185, 2590], 90, 'hole-center-spacing-label'),
]
NORMALIZATION = ('Engineering notation: the visible crossing glyph is represented by U+00D7 ×; '
                 'the raster does not identify its original font character code. The visible degree ring is °. '
                 'Cyrillic lettering and line separation are retained. A semicolon in a parent summary is not printed text.')
GLYPH_REVIEW = ('The crossing glyph is transcribed as Unicode multiplication sign U+00D7 by engineering notation convention. '
                'The raster cannot establish the original font character code; no Latin/Cyrillic letter identity is claimed.')
TEXT_REVIEW = ('Visible digits and Cyrillic letters retained; one ordinary U+0020 space between count and word where present.')


class ReleaseConfigV1(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    schema_version: Literal['rib-ocr-release-config/1.0']
    recipe: Literal['rib-three-annotations-four-lines-v1']
    task: Literal['ocr_recognition']
    pins: dict[str, Pin] = Field(min_length=15, max_length=200)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def _rgb(path, size):
    with Image.open(path) as image:
        require(image.format == 'PNG' and list(image.size) == size and image.mode == 'RGB',
                'Expected the exact bounded RGB PNG dimensions')
        require(size[0]*size[1] <= 4_194_304, 'Image exceeds OCR pixel budget')
        return np.asarray(image).copy()


def _verify_manifest(workspace, manifest_pin, by_path=None):
    manifest_path = checked_pin(workspace, manifest_pin)
    manifest = read_json_v1(manifest_path)
    rows = manifest['files']
    if isinstance(rows, dict):
        rows = [{'path': path, 'sha256': sha} for path, sha in rows.items()]
    require(len({row['path'] for row in rows}) == len(rows), 'Repeated manifest paths')
    inventory = {p.relative_to(manifest_path.parent).as_posix()
                 for p in manifest_path.parent.rglob('*') if p.is_file()}
    require(inventory == {r['path'] for r in rows} | {manifest_path.name}, 'Manifest inventory differs')
    for row in rows:
        Pin.model_validate(row)
        path = contained(manifest_path.parent, row['path'])
        pin = {'path': path.relative_to(workspace).as_posix(), 'sha256': row['sha256']}
        checked_pin(workspace, pin)
        if by_path is not None:
            require(by_path.get(pin['path']) == pin, 'Manifest file is absent from pinned inputs')
    return manifest


def verify_ocr_inputs_v1(workspace, config_pin):
    """Authenticate byte lineage, explicit readings and every released pixel."""
    workspace = Path(workspace).resolve()
    config_path = checked_pin(workspace, config_pin)
    config = ReleaseConfigV1.model_validate(read_json_v1(config_path))
    pins = {key: value.model_dump() for key, value in config.pins.items()}
    for key, path in KEY_PATHS.items():
        require(key in pins and pins[key]['path'] == path, 'Missing or replaced OCR input: '+key)
    by_path = {pin['path']: pin for pin in pins.values()}
    require(len(by_path) == len(pins) and all(path in by_path for path in CODE_FILES),
            'Code pins are missing or input paths are repeated')
    for pin in pins.values():
        checked_pin(workspace, pin)
    values = {key: read_json_v1(workspace/pins[key]['path']) for key in KEY_PATHS if key not in ('source', 'native')}
    _verify_manifest(workspace, pins['review_manifest'], by_path)
    _verify_manifest(workspace, pins['parent_manifest'], by_path)
    review, visual, selection = (values[key] for key in ('review', 'visual_check', 'parent_selection'))
    require(review['schema_version'] == 'rib-ocr-reading/1.0' and
            review['task_scope'] == 'reviewed_line_transcriptions_and_parsed_printed_notation_only' and
            all(review[key] is False for key in ('training_assignment_approved_here', 'physical_geometry_approved',
                                                 'physical_scale_confirmed', 'native_target_approved')),
            'Reading review exceeds its transcription-only scope')
    require(review['builder'] == by_path.get(REVIEW_DIR+'build-review.py'), 'Reading builder pin differs')
    parent = values['parent_final']
    require(parent['report'] == pins['parent_report'] and parent['manifest'] == pins['parent_manifest'],
            'Parent final proof does not bind the source review package')
    for key, parent_key in [('parent_selection', 'parent_selection'), ('parent_field_review', 'parent_fields'),
                            ('parent_final', 'parent_final'), ('preparation', 'preparation')]:
        require(review[key] == pins[parent_key], 'OCR reading refers to different prior evidence')
    require(review['source'] == pins['source'] and review['source_size_px'] == [4986, 7042] and
            review['pair_id'] == 'pair-0031' and review['view_id'] == 'main-profile' and
            review['reviewer'] == 'assistant' and review['human_verified'] is False,
            'OCR source or review scope changed')
    require(values['preparation']['source_sha256'] == pins['source']['sha256'] and
            values['parent_fields']['source'] == pins['source'] and
            selection['source']['sha256'] == pins['source']['sha256'], 'Mixed source evidence')
    require(visual['review'] == pins['review'] and visual['reviewer'] == 'assistant' and
            visual['human_verified'] is False and visual['physical_geometry_approved'] is False and
            visual['physical_scale_confirmed'] is False and visual['training_assignment_approved_here'] is False and
            visual['original_three_parent_crops_visually_inspected'] is True and
            visual['all_four_final_oriented_samples_visually_inspected'] is True, 'Visual review is incomplete or exceeds OCR scope')
    samples, annotations = review['samples'], review['annotations']
    require([s['id'] for s in samples] == [r[0] for r in RECIPE] and
            [a['id'] for a in annotations] == ['1B2', '1C5', '1EB'], 'OCR release recipe inventory differs')
    require([s['id'] for s in visual['sample_checks']] == [s['id'] for s in samples], 'Visual review sample inventory differs')
    regions = {r['id']: r for r in selection['evidence_regions']}
    require(len(regions) == len(selection['evidence_regions']), 'Duplicate parent regions')
    fields = values['parent_fields']['fields']
    for annotation in annotations:
        readings = [f for f in fields if f['subject_id'] == annotation['id'] and f['field_path'] == 'printed_text']
        require(len(readings) == 1 and readings[0]['decision'] == 'reviewed_observation' and
                readings[0]['value'] == annotation['source_reading'] and annotation['physical_unit'] is None,
                'OCR annotation conflicts with the reviewed source reading')
        expected_rows = [r for r in RECIPE if r[1] == annotation['id']]
        require(annotation['transcription_lines'] == [r[3] for r in expected_rows] and
                annotation['sample_ids'] == [r[0] for r in expected_rows], 'Annotation lines differ from release recipe')
        components = parse_printed_components_v1(annotation['transcription_lines'])
        expected = {'printed_numeric_value': components['nominal_value']}
        if components['chamfer_count'] is not None:
            expected.update(chamfer_angle_deg=components['chamfer_angle_deg'], count=components['chamfer_count'])
        require(annotation['printed_fields'] == expected, 'Parsed printed notation differs from visible text')
    prep_dir = (workspace/pins['preparation']['path']).parent
    parents, replay = {}, []
    for sample, recipe, check in zip(samples, RECIPE, visual['sample_checks']):
        identifier, annotation, line_index, text, box, rotation, region_id = recipe
        require(sample['annotation_id'] == annotation and sample['text'] == text and
                sample['source_box_xyxy'] == box and sample['rotation_cw_degrees'] == rotation and
                sample['parent_region_id'] == region_id, 'OCR sample differs from reviewed recipe')
        region = regions[region_id]
        region_pin = {key: region[key] for key in ('path', 'sha256')}
        require(sample['parent_region_pin'] == region_pin and sample['parent_region_box_xyxy'] == region['source_box_px'],
                'OCR parent crop binding differs')
        for pin in (region_pin, sample['raw_crop_pin'], sample['image_pin']):
            require(by_path.get(pin['path']) == pin, 'Unpinned sample artifact')
        require(sample['review_status'] == 'accepted_for_ocr_transcription' and sample['reviewer'] == 'assistant' and
                sample['human_verified'] is False and sample['physical_geometry_approved'] is False,
                'Missing text review or unsupported physical approval')
        require(sample['coordinate_convention'] == 'original_pixel_centers_x_right_y_down_crop_boxes_half_open' and
                sample['normalization'] == (GLYPH_REVIEW if identifier == 'rib-1B2-length-angle' else TEXT_REVIEW),
                'Reviewed character normalization or coordinate convention differs')
        transform = make_ocr_transform_v1(box, rotation)
        require(all(sample[key] == val for key, val in transform.items()), 'OCR pixel transform differs')
        if region_id not in parents:
            parent_rgb, _ = read_prepared_region(prep_dir, region['source_box_px'])
            pbox = region['source_box_px']
            require(np.array_equal(parent_rgb, _rgb(workspace/region_pin['path'], [pbox[2]-pbox[0], pbox[3]-pbox[1]])),
                    'Parent crop differs from prepared source pixels')
            parents[region_id] = parent_rgb
        raw, provenance = read_prepared_region(prep_dir, box)
        raw_size = [box[2]-box[0], box[3]-box[1]]
        require(sample['raw_size_px'] == raw_size, 'Declared raw OCR size differs')
        require(np.array_equal(raw, _rgb(workspace/sample['raw_crop_pin']['path'], raw_size)), 'Raw OCR crop differs from source')
        oriented = np.rot90(raw, k=-(rotation//90)).copy()
        require(np.array_equal(oriented, _rgb(workspace/sample['image_pin']['path'], transform['size_px'])),
                'OCR orientation must be an exact pixel permutation')
        # Match the pinned visual-review builder's diagnostic ink threshold.
        ys, xs = np.where(np.any(oriented < 200, axis=2))
        require(len(xs) > 0, 'Released OCR crop is blank')
        ink_box = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
        width, height = transform['size_px']
        margins = [ink_box[0], ink_box[1], width-ink_box[2], height-ink_box[3]]
        require(sample['ink_box_image_xyxy'] == ink_box, 'Declared OCR ink bounds differ')
        require(check['image'] == sample['image_pin'] and all(check[k] is True for k in (
            'full_text_visible', 'text_matches_reviewed_transcription', 'no_arrow_or_dimension_carrier_inside_sample',
            'rotation_preserves_pixel_values')) and min(margins) >= 1 and
            margins == check['white_margins_px_ltrb'] == sample['white_margins_px_ltrb'], 'Text visual review or margins differ')
        replay.append({'sample_id': identifier, 'source_box_xyxy': box, 'size_px': transform['size_px'],
                       'raw_pixel_replay': True, 'orientation_pixel_replay': True, 'white_margins_px_ltrb': margins,
                       'diagnostic_ink_threshold_exclusive': 200,
                       'preparation_sha256': provenance['preparation_report_sha256']})
    labels = values['old_labels']
    require(labels['source'] == pins['source'] and labels['native_target'] == pins['native'] and
            labels['lineage']['catalog'] == pins['catalog'] and labels['annotation_approved'] is False and
            labels['training_eligible'] is False and labels['lineage']['split'] is None,
            'Old label snapshot was changed or promoted')
    lineage = derive_source_lineage_v1(values['catalog'], values['split_policy'],
                pins['source']['path'].removeprefix('dataset/'), pins['source']['sha256'], exclusions=values['exclusions'])
    require(lineage['train_only'] is True and lineage['historical_reserved_split'] == 'train' and
            lineage['allowed_splits'] == ['train'] and lineage['group_id'] == labels['lineage']['group_id'],
            'Rib lineage restrictions changed; requires a separately reviewed release')
    native_gate = screen_training_target(workspace/pins['native']['path'], values['exclusions'])
    require(native_gate['target_sha256'] == pins['native']['sha256'] and native_gate['training_eligible'] is False,
            'Native target gate changed')
    return {'pins': pins, 'values': values, 'lineage': lineage, 'native_gate': native_gate,
            'pixel_replay': replay, 'parent_regions_replayed': len(parents)}


def build_ocr_documents_v1(verified, config_pin):
    """Assemble from verify_ocr_inputs_v1; this pure helper authenticates no files."""
    pins, values, lineage = (verified[key] for key in ('pins', 'values', 'lineage'))
    review = values['review']
    samples = []
    for row, recipe in zip(review['samples'], RECIPE):
        keys = set(OcrSampleV1.model_fields)-{'line_index'}
        samples.append({**{key: row[key] for key in keys}, 'line_index': recipe[2]})
    annotations = [{'id': a['id'], 'dimension_id': a['id'], 'transcription_lines': a['transcription_lines'],
                    'sample_ids': a['sample_ids'], 'printed_components': parse_printed_components_v1(a['transcription_lines']),
                    'normalization_policy': 'engineering_symbols_v1', 'normalization_notes': NORMALIZATION}
                   for a in review['annotations']]
    package = OcrLabelPackageV1.model_validate({
        'id': 'rib-ocr-development-v1', 'source': pins['source'], 'source_size_px': review['source_size_px'],
        'review': pins['review'], 'dependencies': sorted(pins.values(), key=lambda p: p['path']),
        'lineage': {**{key: lineage[key] for key in ('group_id', 'family_ids', 'train_only', 'historical_reserved_split')},
                    'split': 'train', 'catalog': pins['catalog'], 'split_policy': pins['split_policy']},
        'annotations': annotations, 'samples': samples,
        'limitations': ['Four assistant-reviewed text lines from one development source; no human/expert verification.',
            'Only OCR transcription targets are released; crop rectangles are not detection ground truth.',
            'The cross glyph uses explicit engineering-symbol normalization; original font code is unknown.',
            'Parsed printed numbers, angle and count are metadata, not physical size, scale or geometry targets.',
            'Whole drawing/CAD labels remain unapproved; no training or augmentation has run.',
            'Validation/test are empty; no benchmark accuracy or model quality is established.']}).model_dump(mode='json')
    assignments = [{'id': s['id'], 'task': 'ocr_recognition', 'source_path': lineage['source_path'],
                    'source_sha256': lineage['source_sha256'], 'split': 'train', 'image_sha256': s['image_pin']['sha256']}
                   for s in samples]
    routing = validate_ocr_assignments_v1(values['catalog'], values['split_policy'], assignments, exclusions=values['exclusions'])
    train = [{'id': s['id'], 'image': 'images/'+s['id']+'.png', 'image_sha256': s['image_pin']['sha256'],
              'text': s['text'], 'source_sha256': pins['source']['sha256'], 'group_id': lineage['group_id'],
              'split': 'train', 'task': 'ocr_recognition'} for s in samples]
    splits = {'schema_version': 'ocr-task-splits/1.0', 'task': 'ocr_recognition',
              'allocation_scope': 'released_text_samples_only_with_whole_source_lineage_reservation',
              'source_sha256': pins['source']['sha256'], 'group_id': lineage['group_id'],
              'train_only': True, 'historical_reserved_split': 'train',
              'splits': {'train': [s['id'] for s in samples], 'validation': [], 'test': []},
              'global_policy': pins['split_policy'], 'global_policy_modified': False}
    report = {'schema_version': 'rib-ocr-release/1.0', 'config': config_pin, 'status': 'released_for_development_ocr_only',
              'task': 'ocr_recognition', 'annotation_count': len(annotations), 'sample_count': len(samples),
              'source_count': 1, 'group_count': 1, 'task_split_counts': routing['counts_by_split'],
              'ocr_training_eligible': True, 'training_started': False, 'augmentation_created': False,
              'pair_approved': False, 'physical_geometry_approved': False, 'physical_scale_confirmed': False,
              'native_target_gate': verified['native_gate'], 'old_labels_unchanged': pins['old_labels'],
              'global_split_policy_unchanged': pins['split_policy'], 'lineage': lineage,
              'replay': verified['pixel_replay'], 'parent_regions_replayed': verified['parent_regions_replayed'],
              'limitations': package['limitations']}
    return {'labels.json': package, 'routing.json': routing, 'task-splits.json': splits,
            'report.json': report, 'schema.json': OcrLabelPackageV1.model_json_schema()}, train


def _jsonl(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n' for row in rows)


def run_rib_ocr_release_v1(workspace, config_pin, output_dir):
    workspace = Path(workspace).resolve()
    output = contained(workspace, output_dir)
    artifacts = (workspace/'artifacts').resolve()
    require(artifacts.is_relative_to(workspace) and output.is_relative_to(artifacts) and output != artifacts,
            'Use a new directory under workspace/artifacts')
    require(not output.exists(), 'Never overwrite a released OCR package')
    verified = verify_ocr_inputs_v1(workspace, config_pin)
    for pin in verified['pins'].values():
        parent = (workspace/pin['path']).resolve().parent
        require(not (parent.is_relative_to(artifacts) and parent != artifacts and output.is_relative_to(parent)),
                'Output cannot extend a frozen input directory')
    documents, train = build_ocr_documents_v1(verified, config_pin)
    (output/'images').mkdir(parents=True)
    for sample, record in zip(documents['labels.json']['samples'], train):
        shutil.copyfile(workspace/sample['image_pin']['path'], output/record['image'])
        require(digest(output/record['image']) == record['image_sha256'], 'Copied OCR image differs')
    for name, value in documents.items():
        write_json(output/name, value)
    for name, rows in [('train', train), ('validation', []), ('test', [])]:
        (output/(name+'.jsonl')).write_text(_jsonl(rows), encoding='utf-8', newline='\n')
    body = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR: Ребро</title>',
            '<style>body{font:18px system-ui;max-width:1000px;margin:2em auto}img{max-width:100%;border:1px solid #ccc}section{margin:2em 0}</style>',
            '<h1>Четыре строки для распознавания текста</h1><p>Три обозначения «Ребра», один источник. '
            'Выпуск разрешает только OCR в train. Validation/test пусты; обучение не запускалось.</p>',
            '<p>Надписи повёрнуты без интерполяции. Знак × нормализован по инженерной нотации. '
            'Физические размеры, координаты и вся пара PNG/DXF этим выпуском не утверждаются.</p>',
            '<p><a href="labels.json">Метки и происхождение</a> · <a href="train.jsonl">Строки train</a> · '
            '<a href="report.json">Проверка</a> · <a href="task-splits.json">Распределение</a></p>']
    for row in train:
        body.append('<section><p><b>'+escape(row['text'])+'</b> — '+escape(row['id'])+'</p><img src="'+
                    escape(row['image'], quote=True)+'" alt="'+escape(row['text'], quote=True)+'"></section>')
    (output/'index.html').write_text(''.join(body)+'</html>\n', encoding='utf-8')
    for pin in [config_pin, *verified['pins'].values()]:
        checked_pin(workspace, pin)
    write_json(output/'manifest.json', {'schema_version': 'ocr-release-manifest/1.0', 'task': 'ocr_recognition',
        'config': config_pin, 'files': [{'path': p.relative_to(output).as_posix(), 'sha256': digest(p)}
            for p in sorted(output.rglob('*')) if p.is_file()]})
    return documents['report.json']


def load_rib_ocr_training_v1(workspace, manifest_pin, *, task='ocr_recognition', split='train'):
    """Return only authenticated recognition targets, never general CAD labels."""
    require(task == 'ocr_recognition' and split == 'train', 'This release supports development OCR train only')
    workspace = Path(workspace).resolve()
    manifest_path = checked_pin(workspace, manifest_pin)
    manifest = _verify_manifest(workspace, manifest_pin)
    require(set(manifest) == {'schema_version', 'task', 'config', 'files'} and
            manifest['schema_version'] == 'ocr-release-manifest/1.0' and manifest['task'] == task,
            'Expected a task-scoped OCR release manifest')
    verified = verify_ocr_inputs_v1(workspace, manifest['config'])
    documents, train = build_ocr_documents_v1(verified, manifest['config'])
    package_dir = manifest_path.parent
    package = read_ocr_labels_v1(package_dir/'labels.json')
    require(package.model_dump(mode='json') == documents['labels.json'], 'Labels differ from the authenticated OCR recipe')
    for name, value in documents.items():
        require(read_json_v1(package_dir/name) == value, 'Released document differs from the authenticated recipe: '+name)
    for name, rows in [('train', train), ('validation', []), ('test', [])]:
        require((package_dir/(name+'.jsonl')).read_text(encoding='utf-8') == _jsonl(rows),
                'Recognition targets or split allocation differ from the release')
    records = []
    for row in train:
        path = contained(package_dir, row['image'])
        require(digest(path) == row['image_sha256'], 'Training image bytes changed')
        records.append({**row, 'image_path': str(path)})
    # Recheck the manifest after all dependent reads; this is a bounded snapshot.
    for pin in [manifest_pin, manifest['config'], *verified['pins'].values()]:
        checked_pin(workspace, pin)
    return {'task': task, 'split': split, 'records': records, 'count': len(records),
            'training_started': False, 'physical_geometry_approved': False}
