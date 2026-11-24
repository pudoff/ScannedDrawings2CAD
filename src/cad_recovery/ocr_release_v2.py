"""Source-authenticated OCR release for 25 reviewed drawing annotations.

V2 merges an unchanged four-line parent release with explicit per-line reviews
of two sources. Printed notation is separate from physical drawing geometry.
"""
from __future__ import annotations

from collections import defaultdict
from html import escape
import json
from pathlib import Path
import shutil
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .clear_pairs import contained, digest
from .label_release_v1 import checked_pin
from .label_schema_v1 import Pin, read_json_v1
from .ocr_label_schema_v1 import OcrSampleV1, make_ocr_transform_v1
from .ocr_label_schema_v2 import OcrLabelPackageV2, parse_printed_components_v2, read_ocr_labels_v2
from .ocr_lineage_v1 import derive_source_lineage_v1, validate_ocr_assignments_v1
from .raster_regions import read_prepared_region
from .rib_ocr_release_v1 import _rgb, _verify_manifest, _jsonl, write_json, require, load_rib_ocr_training_v1


SOURCES = ('palec', 'rebro')
KEY_PATHS = {
    'parent_final': 'artifacts/rib-ocr-final-v1.json',
    'parent_manifest': 'artifacts/rib-ocr-release-v1/manifest.json',
    'parent_labels': 'artifacts/rib-ocr-release-v1/labels.json',
    'field_report': 'artifacts/dimension-field-review-v1/report.json',
    'raster_evidence': 'artifacts/dimension-field-raster-evidence-v1/raster-evidence.json',
    'raster_seal': 'artifacts/dimension-field-raster-evidence-v1/snapshot-sha256.json',
    'catalog': 'artifacts/family-split-v2/families.json',
    'split_policy': 'artifacts/family-split-v2/splits.json',
    'exclusions': 'artifacts/resource-disposition-v1/excluded-targets.json',
    'source_palec': 'dataset/clear_sum_dataset/vse_png/МТСП.24.03.003 Палец .png',
    'source_rebro': 'dataset/clear_sum_dataset/vse_png/МТСП.24.08.012 Ребро.png',
    'preparation_palec': 'artifacts/clear-raster-prepared-v1/source-0028-8a6669e23a98/preparation.json',
    'preparation_rebro': 'artifacts/clear-raster-prepared-v1/source-0031-097db6a4e770/preparation.json',
    **{kind+'_'+source: f'artifacts/ocr-reading-{source}-v2/'+filename for source in SOURCES
       for kind, filename in [('review', 'review.json'), ('visual', 'visual-check.json'), ('manifest', 'manifest.json')]},
}
CODE_FILES = ['src/cad_recovery/'+name+'.py' for name in (
    'ocr_release_v2', 'ocr_label_schema_v2', 'ocr_label_schema_v1', 'ocr_lineage_v1',
    'rib_ocr_release_v1', 'label_schema_v1', 'label_release_v1', 'raster_regions',
    'clear_pairs', 'family_catalog_v2')]+['scripts/release_ocr_v2.py']
EXPECTED = {'palec': ['130', '145', '160', '183', '197', '1B2', '1D1', '1E4', '1FF', '218', '2E0', '2F4'],
            'rebro': ['EE', '143', '15E', '171', '184', '197', '1D8', '1FE', '20D', '222']}
REVIEW_SCOPE = 'reviewed_line_transcriptions_and_parsed_printed_notation_only'
CONVENTION = 'original_pixel_centers_x_right_y_down_crop_boxes_half_open'


class ReleaseConfigV2(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    schema_version: Literal['ocr-release-config/2.0']
    recipe: Literal['palec-rebro-25-annotations-with-parent-v1']
    task: Literal['ocr_recognition']
    pins: dict[str, Pin] = Field(min_length=19, max_length=500)


def verify_ocr_inputs_v2(workspace, config_pin):
    workspace = Path(workspace).resolve()
    config = ReleaseConfigV2.model_validate(read_json_v1(checked_pin(workspace, config_pin)))
    pins = {key: value.model_dump() for key, value in config.pins.items()}
    for key, path in KEY_PATHS.items():
        require(key in pins and pins[key]['path'] == path, 'Missing or replaced V2 input: '+key)
    by_path = {p['path']: p for p in pins.values()}
    require(len(by_path) == len(pins) and all(p in by_path for p in CODE_FILES), 'Missing code pins or repeated input paths')
    for pin in pins.values():
        checked_pin(workspace, pin)
    values = {key: read_json_v1(workspace/pins[key]['path']) for key in KEY_PATHS if not key.startswith('source_')}
    require(values['parent_final']['manifest'] == pins['parent_manifest'], 'Parent final proof identifies a different release')
    parent_loaded = load_rib_ocr_training_v1(workspace, pins['parent_manifest'])
    require(parent_loaded['count'] == 4 and len(values['parent_labels']['samples']) == 4, 'Parent OCR inventory changed')
    require(values['raster_seal']['raster-evidence.json'] == pins['raster_evidence']['sha256'], 'Raster evidence seal differs')
    regions = {(r['source_id'], r['id']): r for r in values['raster_evidence']['regions']}
    require(len(regions) == len(values['raster_evidence']['regions']), 'Duplicate parent raster regions')
    lineages, replay, accepted, unresolved = {}, [], [], []
    for source in SOURCES:
        review, visual = values['review_'+source], values['visual_'+source]
        _verify_manifest(workspace, pins['manifest_'+source], by_path)
        source_pin, prep_pin = pins['source_'+source], pins['preparation_'+source]
        require(review['schema_version'] == 'ocr-reading/2.0' and review['source_id'] == source and
                review['source'] == source_pin and review['preparation'] == prep_pin and
                review['reviewer'] == 'assistant' and review['human_verified'] is False and review['task_scope'] == REVIEW_SCOPE,
                'Source review identity or task scope differs')
        require(all(review[k] is False for k in ('training_assignment_approved_here', 'physical_geometry_approved',
                                                 'physical_scale_confirmed', 'native_target_approved')), 'Review contains unsupported approval')
        for key, pin_key in [('parent_fields', 'field_report'), ('parent_raster_evidence', 'raster_evidence'), ('parent_raster_seal', 'raster_seal')]:
            require(review[key] == pins[pin_key], 'Review parent binding differs')
        require(by_path.get(review['builder']['path']) == review['builder'], 'Unpinned review builder')
        size = [9946, 7042] if source == 'palec' else [4986, 7042]
        require(review['source_size_px'] == size and values['preparation_'+source]['source_sha256'] == source_pin['sha256'],
                'Source preparation or extent differs')
        pairs = [p for p in values['field_report']['pairs'] if p['source_id'] == source]
        require(len(pairs) == 1 and pairs[0]['input_sha256'] == source_pin['sha256'] and
                'dataset/'+pairs[0]['input_path'] == source_pin['path'] and review['pair_id'] == pairs[0]['pair_id'],
                'Source pair identity is ambiguous or changed')
        previous = {o['identity']['handle']: o['source_reading'] for o in pairs[0]['objects']}
        require(len(previous) == len(pairs[0]['objects']), 'Duplicate source dimension handles')
        annotations, samples = review['annotations'], review['samples']
        require([a['dimension_id'] for a in annotations] == EXPECTED[source] and
                len({a['id'] for a in annotations}) == len(annotations), 'Review annotation coverage differs')
        sample_map = {s['id']: s for s in samples}
        require(len(sample_map) == len(samples), 'Duplicate reviewed sample IDs')
        parent_review_key = 'all_parent_label_crops_visually_inspected' if source == 'palec' else 'all_parent_crops_visually_inspected'
        parent_count_key = 'parent_label_count' if source == 'palec' else 'parent_crop_count'
        require(visual['schema_version'] == 'ocr-visual-check/2.0' and visual['review'] == pins['review_'+source] and
                visual['source_id'] == source and visual['reviewer'] == 'assistant' and visual['human_verified'] is False and
                visual[parent_review_key] is True and visual[parent_count_key] == len(annotations) and
                visual['sample_count'] == len(samples) and
                visual['all_final_oriented_samples_visually_inspected'] is True, 'Visual review incomplete or mismatched')
        require(all(visual[k] is False for k in ('physical_geometry_approved', 'physical_scale_confirmed',
                                                'training_assignment_approved_here')), 'Visual review contains unsupported approval')
        checks = {c['id']: c for c in visual['sample_checks']}
        require(len(checks) == len(visual['sample_checks']) and set(checks) == set(sample_map), 'Visual sample inventory differs')
        for context in visual.get('additional_contexts_visually_inspected', []):
            require(by_path.get(context['path']) == context, 'Unpinned visual context')
        seen, parent_arrays = set(), {}
        for annotation in annotations:
            handle = annotation['dimension_id']
            require(all(annotation.get(k, False) is False for k in ('physical_geometry_approved', 'cad_coordinates_used',
                        'printed_components_are_physical_values')), 'Annotation contains unsupported physical claims')
            require(annotation['id'] == source+'-'+handle and annotation['source_reading'] == previous[handle]['text'] and
                    annotation['physical_unit'] is None and isinstance(annotation['normalization_notes'], str) and
                    bool(annotation['normalization_notes'].strip()), 'Annotation identity, reading or normalization missing')
            if annotation['decision'] == 'unresolved':
                require(annotation['sample_ids'] == [] and annotation['transcription_lines'] == [] and bool(annotation['reason']),
                        'Unresolved text must not release sample targets')
                unresolved.append({'id': annotation['id'], 'source_id': source, 'dimension_id': handle,
                    'source_reading': annotation['source_reading'], 'reason': annotation['reason'],
                    'decision': 'unresolved', 'review': pins['review_'+source]})
                continue
            require(annotation['decision'] == 'accepted_for_ocr_transcription' and
                    '; '.join(annotation['transcription_lines']) == annotation['source_reading'] and
                    len(annotation['sample_ids']) == len(annotation['transcription_lines']),
                    'Accepted annotation differs from prior reviewed reading or line inventory')
            for line_index, (sid, text) in enumerate(zip(annotation['sample_ids'], annotation['transcription_lines'])):
                require(sid in sample_map and sid not in seen, 'Missing or reused line sample')
                seen.add(sid)
                s = sample_map[sid]
                require(s['annotation_id'] == annotation['id'] and s['line_index'] == line_index and s['text'] == text and
                        s['normalization'] == annotation['normalization_notes'] and s['coordinate_convention'] == CONVENTION,
                        'Sample identity, line order or normalization differs')
                sample = OcrSampleV1.model_validate({k: s[k] for k in OcrSampleV1.model_fields})
                region = regions.get((source, s['parent_region_id']))
                require(region is not None and region['source_box_px'] == previous[handle]['label_source_box_px'],
                        'Sample does not belong to the dimension label region')
                region_pin = {'path': 'artifacts/dimension-field-raster-evidence-v1/'+region['path'], 'sha256': region['png_sha256']}
                require(s['parent_region_pin'] == region_pin and s['parent_region_box_xyxy'] == region['source_box_px'] and
                        region['provenance']['source_sha256'] == source_pin['sha256'] and
                        region['provenance']['preparation_report_sha256'] == prep_pin['sha256'], 'Region source/preparation binding differs')
                for pin in (region_pin, s['raw_crop_pin'], s['image_pin']):
                    require(by_path.get(pin['path']) == pin, 'Unpinned line image')
                prep_dir = (workspace/prep_pin['path']).parent
                if s['parent_region_id'] not in parent_arrays:
                    box = region['source_box_px']
                    rgb, _ = read_prepared_region(prep_dir, box)
                    require(np.array_equal(rgb, _rgb(workspace/region_pin['path'], [box[2]-box[0], box[3]-box[1]])),
                            'Parent region pixels differ')
                    parent_arrays[s['parent_region_id']] = True
                box = s['source_box_xyxy']
                raw, provenance = read_prepared_region(prep_dir, box)
                raw_size = [box[2]-box[0], box[3]-box[1]]
                require(s['raw_size_px'] == raw_size and np.array_equal(raw, _rgb(workspace/s['raw_crop_pin']['path'], raw_size)),
                        'Raw line pixels or extent differ')
                oriented = np.rot90(raw, -(s['rotation_cw_degrees']//90)).copy()
                require(np.array_equal(oriented, _rgb(workspace/s['image_pin']['path'], s['size_px'])), 'Orientation is not exact pixel permutation')
                check = checks[sid]
                require(check['image'] == s['image_pin'] and all(check[k] is True for k in (
                    'full_text_visible', 'text_matches_reviewed_transcription', 'no_arrow_or_dimension_carrier_inside_sample',
                    'rotation_preserves_pixel_values')), 'Line visual decision is incomplete')
                require(s['diagnostic_ink_threshold_exclusive'] == 200, 'Diagnostic pixel convention changed')
                yy, xx = np.where(np.any(oriented < 200, axis=2))
                require(len(xx) > 0, 'Released line is blank')
                ink = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
                margins = [ink[0], ink[1], s['size_px'][0]-ink[2], s['size_px'][1]-ink[3]]
                require(s['ink_box_image_xyxy'] == ink and min(margins) > 0 and
                        s['white_margins_px_ltrb'] == check['white_margins_px_ltrb'] == margins,
                        'Ink extent or source margins differ')
                replay.append({'source_id': source, 'sample_id': sid, 'raw_pixel_replay': True, 'rotation_pixel_replay': True,
                               'white_margins_px_ltrb': margins, 'diagnostic_ink_threshold_exclusive': 200})
                accepted.append({**sample.model_dump(mode='json'), 'source_id': source})
        require(seen == set(sample_map), 'A reviewed line is outside the declared annotation scope')
        lineage = derive_source_lineage_v1(values['catalog'], values['split_policy'], source_pin['path'].removeprefix('dataset/'),
                                          source_pin['sha256'], exclusions=values['exclusions'])
        require(lineage['train_only'] is True and lineage['historical_reserved_split'] == 'train' and lineage['allowed_splits'] == ['train'],
                'Reviewed source train-only restrictions changed')
        lineages[source] = lineage
    return {'pins': pins, 'values': values, 'lineages': lineages, 'new_samples': accepted,
            'unresolved': unresolved, 'replay': replay, 'parent_loaded': parent_loaded}


def build_ocr_documents_v2(verified, config_pin):
    pins, values, lineages = (verified[k] for k in ('pins', 'values', 'lineages'))
    old = values['parent_labels']
    inherited = [s['id'] for s in old['samples']]
    samples = [{**s, 'source_id': 'rebro'} for s in old['samples']]+verified['new_samples']
    annotations = []
    for a in old['annotations']:
        annotations.append({**{k: a[k] for k in ('id', 'dimension_id', 'transcription_lines', 'sample_ids', 'normalization_notes')},
            'source_id': 'rebro', 'printed_components': parse_printed_components_v2(a['transcription_lines']),
            'normalization_policy': 'engineering_symbols_v2', 'review': old['review']})
    sources = []
    for source in SOURCES:
        lineage = lineages[source]
        sources.append({'id': source, 'source': pins['source_'+source], 'source_size_px': values['review_'+source]['source_size_px'],
            'preparation': pins['preparation_'+source], 'review': pins['review_'+source],
            'lineage': {**{k: lineage[k] for k in ('group_id', 'family_ids', 'train_only', 'historical_reserved_split')},
                       'catalog': pins['catalog'], 'split_policy': pins['split_policy'], 'split': 'train'}})
        for a in values['review_'+source]['annotations']:
            if a['decision'] != 'accepted_for_ocr_transcription':
                continue
            annotations.append({**{k: a[k] for k in ('id', 'dimension_id', 'transcription_lines', 'sample_ids', 'normalization_notes')},
                'source_id': source, 'printed_components': parse_printed_components_v2(a['transcription_lines']),
                'normalization_policy': 'engineering_symbols_v2', 'review': pins['review_'+source]})
    package = OcrLabelPackageV2.model_validate({'id': 'palec-rebro-ocr-development-v2', 'parent_release': pins['parent_manifest'],
        'inherited_sample_ids': inherited, 'source_records': sources, 'dependencies': sorted(pins.values(), key=lambda p: p['path']),
        'annotations': annotations, 'samples': samples, 'unresolved_annotations': verified['unresolved'],
        'limitations': ['Assistant-reviewed transcription targets from two previously inspected development sources; no independent validation/test.',
            'The original four recognition samples retain every V1 field and image byte; only their explicit source_id is added.',
            'Engineering glyphs use declared normalization; original font character codes are unknown.',
            'Printed fit/thread designators do not establish numeric tolerances, thread pitch, physical units or recovered geometry.',
            'Palec filename/title designation conflict and broken-view physical-length limitations remain unchanged.',
            'Source crop boxes are recognition regions, not precise detection polygons. No CAD target or whole-pair approval.',
            'Frozen V1 releases/global split policies remain unchanged. No training or augmentation has run.']}).model_dump(mode='json')
    for previous, current in zip(old['samples'], package['samples']):
        require({k: v for k, v in current.items() if k != 'source_id'} == previous, 'Inherited V1 sample changed')
    assignments = [{'id': s['id'], 'task': 'ocr_recognition', 'source_path': lineages[s['source_id']]['source_path'],
                    'source_sha256': lineages[s['source_id']]['source_sha256'], 'image_sha256': s['image_pin']['sha256'], 'split': 'train'}
                   for s in package['samples']]
    routing = validate_ocr_assignments_v1(values['catalog'], values['split_policy'], assignments, exclusions=values['exclusions'])
    content = defaultdict(list)
    for s in package['samples']:
        content[s['image_pin']['sha256']].append(s)
    require(all(len({s['text'] for s in rows}) == 1 for rows in content.values()), 'Identical image bytes have conflicting OCR targets')
    train = [{'id': s['id'], 'source_id': s['source_id'], 'image': 'images/'+s['id']+'.png',
              'image_sha256': s['image_pin']['sha256'], 'text': s['text'],
              'source_sha256': lineages[s['source_id']]['source_sha256'], 'group_id': lineages[s['source_id']]['group_id'],
              'split': 'train', 'task': 'ocr_recognition'} for s in package['samples']]
    registry = {'schema_version': 'ocr-task-splits/2.0', 'task': 'ocr_recognition', 'parent_release': pins['parent_manifest'],
        'complete_registry_scope': 'parent_four_samples_plus_all_accepted_current_review_lines',
        'inherited_sample_ids': inherited, 'source_lineages': list(lineages.values()),
        'splits': {'train': [r['id'] for r in train], 'validation': [], 'test': []},
        'global_split_policy': pins['split_policy'], 'global_split_policy_modified': False}
    report = {'schema_version': 'ocr-release/2.0', 'config': config_pin, 'status': 'released_for_development_ocr_only',
        'annotation_count': len(package['annotations']), 'reviewed_annotation_count': 25,
        'new_annotations_accepted': len(package['annotations'])-len(old['annotations']),
        'unresolved_annotation_count': len(package['unresolved_annotations']), 'sample_count': len(train),
        'new_sample_count': len(verified['new_samples']), 'inherited_sample_count': len(inherited),
        'source_count': len(sources), 'group_count': len({r['group_id'] for r in train}),
        'task_split_counts': routing['counts_by_split'], 'unique_image_hashes': len(content),
        'same_group_duplicate_image_occurrences': [{'sha256': sha, 'sample_ids': [s['id'] for s in rows]}
                                                 for sha, rows in sorted(content.items()) if len(rows) > 1],
        'inherited_samples_unchanged': True, 'ocr_training_eligible': True, 'training_started': False,
        'augmentation_created': False, 'physical_geometry_approved': False, 'physical_scale_confirmed': False,
        'pair_approved': False, 'native_target_approved': False, 'global_split_policy_modified': False,
        'lineages': lineages, 'replay': verified['replay'], 'limitations': package['limitations']}
    return {'labels.json': package, 'routing.json': routing, 'task-splits.json': registry,
            'report.json': report, 'schema.json': OcrLabelPackageV2.model_json_schema()}, train


def run_ocr_release_v2(workspace, config_pin, output_dir):
    workspace = Path(workspace).resolve()
    output = contained(workspace, output_dir)
    artifacts = (workspace/'artifacts').resolve()
    require(artifacts.is_relative_to(workspace) and output != artifacts and output.is_relative_to(artifacts), 'Use a new artifacts directory')
    require(not output.exists(), 'Never overwrite an OCR release')
    verified = verify_ocr_inputs_v2(workspace, config_pin)
    for pin in verified['pins'].values():
        parent = (workspace/pin['path']).resolve().parent
        require(not(parent.is_relative_to(artifacts) and parent != artifacts and output.is_relative_to(parent)), 'Output cannot extend frozen inputs')
    documents, train = build_ocr_documents_v2(verified, config_pin)
    (output/'images').mkdir(parents=True)
    for s, row in zip(documents['labels.json']['samples'], train):
        shutil.copyfile(workspace/s['image_pin']['path'], output/row['image'])
        require(digest(output/row['image']) == row['image_sha256'], 'Copied image differs')
    for name, value in documents.items():
        write_json(output/name, value)
    for split, rows in [('train', train), ('validation', []), ('test', [])]:
        (output/(split+'.jsonl')).write_text(_jsonl(rows), encoding='utf-8', newline='\n')
    body = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR двух деталей</title>',
        '<style>body{font:17px system-ui;max-width:1100px;margin:2em auto}section{margin:2em 0}img{max-width:100%;border:1px solid #aaa}</style>',
        '<h1>OCR: Палец и Ребро</h1><p>Общий выпуск включает прежние четыре строки. Только train разработки; обучение не запускалось. '
        'Физические допуски, размеры и CAD-геометрия не утверждаются.</p>',
        '<p><a href="labels.json">Метки</a> · <a href="train.jsonl">Строки train</a> · '
        '<a href="task-splits.json">Общий реестр</a> · <a href="report.json">Отчёт</a></p>']
    for row in train:
        body.append('<section><p>'+escape(row['source_id']+' / '+row['id'])+' — <b>'+escape(row['text'])+
                    '</b></p><img src="'+escape(row['image'], quote=True)+'" alt="'+escape(row['text'], quote=True)+'"></section>')
    (output/'index.html').write_text(''.join(body)+'</html>\n', encoding='utf-8')
    for pin in [config_pin, *verified['pins'].values()]:
        checked_pin(workspace, pin)
    write_json(output/'manifest.json', {'schema_version': 'ocr-release-manifest/2.0', 'task': 'ocr_recognition',
        'config': config_pin, 'files': [{'path': p.relative_to(output).as_posix(), 'sha256': digest(p)}
                                     for p in sorted(output.rglob('*')) if p.is_file()]})
    return documents['report.json']


def load_ocr_training_v2(workspace, manifest_pin, *, task='ocr_recognition', split='train'):
    require(task == 'ocr_recognition' and split == 'train', 'Only development OCR train is released')
    workspace = Path(workspace).resolve()
    manifest_path = checked_pin(workspace, manifest_pin)
    manifest = _verify_manifest(workspace, manifest_pin)
    require(set(manifest) == {'schema_version', 'task', 'config', 'files'} and
            manifest['schema_version'] == 'ocr-release-manifest/2.0' and manifest['task'] == task, 'Expected OCR V2 manifest')
    verified = verify_ocr_inputs_v2(workspace, manifest['config'])
    documents, train = build_ocr_documents_v2(verified, manifest['config'])
    directory = manifest_path.parent
    package = read_ocr_labels_v2(directory/'labels.json')
    require(package.model_dump(mode='json') == documents['labels.json'], 'Labels differ from authenticated V2 recipe')
    for name, value in documents.items():
        require(read_json_v1(directory/name) == value, 'Released document differs from V2 recipe: '+name)
    for name, rows in [('train', train), ('validation', []), ('test', [])]:
        require((directory/(name+'.jsonl')).read_text(encoding='utf-8') == _jsonl(rows), 'OCR targets or combined split registry changed')
    records = []
    for row in train:
        path = contained(directory, row['image'])
        require(digest(path) == row['image_sha256'], 'Released OCR image changed')
        records.append({**row, 'image_path': str(path)})
    for pin in [manifest_pin, manifest['config'], *verified['pins'].values()]:
        checked_pin(workspace, pin)
    return {'task': task, 'split': split, 'records': records, 'count': len(records), 'training_started': False,
            'physical_geometry_approved': False}
