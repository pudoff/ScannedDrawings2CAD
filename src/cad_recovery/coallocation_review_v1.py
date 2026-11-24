"""Replay source-only relationship reviews and publish an additive routing overlay.

The frozen catalog remains the base. This packet authenticates reviewed pixels
and conservative constraints, not the truth of assistant-authored transcriptions
or semantic-family independence. No training or split assignment is performed.
"""
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .clear_pairs import contained, digest
from .coallocation_policy_v1 import build_coallocation_v1, validate_assignments_v1
from .label_release_v1 import checked_pin
from .ocr_release_v2 import load_ocr_training_v2
from .png_stream import PNGStripReader
from .raster_regions import read_prepared_region

PROTOCOL_PIN = {'path': 'configs/reviews/coallocation-protocol-v1.json',
                'sha256': '567fb878db23a744829fb2b00f111c3efa8315b8c7f7a0e45a708385b5b38f2d'}
OCR_PIN = {'path': 'artifacts/ocr-release-v2/manifest.json',
           'sha256': '48139783b64609ba0dfb127c4235e28cb63622106802b9c28a8d9253eed116ef'}
READINGS = ['artifacts/coallocation-'+kind+'-reading-v1/review.json' for kind in ('numeric', 'project')]
DEPENDENCIES = {'parent_final', 'parent_manifest', 'catalog', 'policy', 'exclusions',
                'prepared_catalog', 'dataset_manifest', 'old_intake', 'new_intake'}
CODE = ['src/cad_recovery/'+name+'.py' for name in (
    'coallocation_review_v1', 'coallocation_policy_v1', 'ocr_lineage_v1', 'family_catalog_v2',
    'ocr_control_review_v1', 'ocr_release_v2', 'raster_regions', 'png_stream', 'clear_pairs',
    'label_release_v1')]+['scripts/build_coallocation_review_v1.py']
FALSE_FLAGS = ('annotation_approved', 'training_eligible', 'semantic_family_approved', 'independent_evaluation')
HISTORICAL_SOURCE = 'clear_sum_dataset/vse_png/МТСП.24.03.001_Лист.png'
SOURCE_FIELDS = {'input_path', 'input_sha256', 'preparation', 'source_size_px', 'title_observation',
                 'related_designations', 'observations', 'regions', 'decision', 'split', *FALSE_FLAGS}
SOURCE_EXTRAS = {'explicit_variants', 'overview', 'overview_to_source', 'source_pin', 'visual_checks',
                 'interpretation', 'historical_train_only_retained'}
REVIEW_FIELDS = {'schema_version', 'protocol', 'reviewer', 'human_verified', 'ocr_run', 'native_cad_opened',
                 'reviewed_at_utc', 'sources', 'conclusions'}
REVIEW_EXTRAS = {'scope', 'verification', 'region_provenance', 'scope_summary', 'limitations',
                 'training_started', 'transcription_limits', 'evidence_manifest', 'supplement_manifest',
                 'table_views_manifest', 'historical_comparisons'}
MAX_REGION_PIXELS = 16_777_216
MAX_SOURCE_ROI_PIXELS = 33_554_432


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def pin(root, path):
    return {'path': path, 'sha256': digest(contained(root, path))}


def _base(workspace):
    root = Path(workspace).resolve()
    protocol = read(checked_pin(root, PROTOCOL_PIN))
    require(protocol['schema_version'] == 'coallocation-review-protocol/1.0' and protocol['task'] == 'P14.1b.5'
            and protocol['source_only'] is True and protocol['OCR_run'] is False
            and protocol['training_started'] is False and protocol['labels_approved'] is False
            and protocol['no_quality_based_replacement'] is True, 'Changed coallocation protocol')
    require(set(protocol['dependencies']) == DEPENDENCIES, 'Changed protocol dependencies')
    values = {key: read(checked_pin(root, value)) for key, value in protocol['dependencies'].items()}
    manifest = values['dataset_manifest']['files']
    files = {row['path']: row['sha256'] for row in manifest}
    require(len(files) == len(manifest) == 1622 and
            files == {row['path']: row['sha256'] for row in values['catalog']['files']}, 'Dataset/catalog differ')
    prepared = values['prepared_catalog']['entries']
    require(len(prepared) == len({row['input_path'] for row in prepared}) == 510, 'Changed prepared inventory')
    paths = [row['input_path'] for row in prepared]
    numeric = sorted(path for path in paths if re.match(r'^(1539588|1579170|346866|626278|964414)[._]', Path(path).name))
    project = sorted(path for path in paths if Path(path).name.startswith('260620-'))
    require(len(numeric) == 32 and len(project) == 78 and protocol['numeric_source_paths'] == numeric
            and protocol['project_source_paths'] == project, 'Fixed source scope differs')
    # This is a routing-only validation of the frozen context, not file admission.
    build_coallocation_v1(values['catalog'], values['policy'], values['exclusions'], [])
    return protocol, values


def prepare_coallocation_config_v1(workspace, output_path):
    root = Path(workspace).resolve()
    output = contained(root, output_path)
    require(output.parent == root/'configs/reviews' and not output.exists(), 'Use a new review config')
    _base(root)
    config = {'schema_version': 'coallocation-review-config/1.0', 'task': 'P14.1b.5',
              'protocol': PROTOCOL_PIN, 'previous_ocr_release': OCR_PIN,
              'readings': [pin(root, path) for path in READINGS], 'code': [pin(root, path) for path in CODE]}
    checked_pin(root, OCR_PIN)
    output.write_bytes(encoded(config))
    return pin(root, output.relative_to(root).as_posix())


def _checked_source(root, row, prepared, files):
    require(isinstance(row, dict) and SOURCE_FIELDS <= set(row) <= SOURCE_FIELDS | SOURCE_EXTRAS,
            'Unexpected source review fields')
    require(row.get('historical_train_only_retained', True) is True, 'Historical train constraint cannot be removed')
    path = row['input_path']
    require(path in prepared and path in files, 'Unknown reviewed source')
    binding = prepared[path]
    require(row['input_sha256'] == binding['input_sha256'] == files[path]['sha256']
            and row['preparation'] == binding['preparation']
            and row['source_size_px'] == binding['source_size_px'], 'Review source/preparation identity differs')
    require(row['decision'] == 'coallocation_review_only' and row['split'] is None
            and all(row[key] is False for key in FALSE_FLAGS), 'Review cannot approve families, splits or labels')
    title = row['title_observation']
    require(isinstance(title, dict) and set(title) == {'designation', 'name', 'uncertainty'}
            and all(title[key] is None or isinstance(title[key], str) for key in ('designation', 'name'))
            and isinstance(title['uncertainty'], str) and title['uncertainty'].strip(), 'Invalid title observation')
    require(isinstance(row['observations'], list) and row['observations']
            and all(isinstance(v, str) and v.strip() for v in row['observations']), 'Missing review observations')
    require(isinstance(row['related_designations'], list), 'Invalid printed reference observations')
    for reference in row['related_designations']:
        require(isinstance(reference, dict) and set(reference) == {'text', 'field_label', 'uncertainty'}
                and all(isinstance(v, str) and v.strip() for v in reference.values()), 'Invalid reference observation')
    checked_pin(root, {'path': 'dataset/'+path, 'sha256': row['input_sha256']})
    prep = read(checked_pin(root, row['preparation']))
    require(prep['source_sha256'] == row['input_sha256'] and prep['source_size_px'] == row['source_size_px']
            and 'rgba_conversion' not in prep and 'container_repair' not in prep,
            'Expected original RGB preparation for this review scope')
    regions = row['regions']
    require(isinstance(regions, list) and 1 <= len(regions) <= 8, 'Expected bounded source region inventory')
    kinds, budget = set(), 0
    for region in regions:
        require(isinstance(region, dict) and set(region) == {'kind', 'box_source_px', 'image'}, 'Unexpected source region fields')
        kind, box = region['kind'], region['box_source_px']
        require(isinstance(kind, str) and re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', kind) and kind not in kinds, 'Duplicate or invalid region identity')
        kinds.add(kind)
        require(isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box), 'Expected integer half-open ROI')
        x0, y0, x1, y1 = box
        width, height = row['source_size_px']
        area = (x1-x0)*(y1-y0)
        require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height and area <= MAX_REGION_PIXELS, 'ROI outside source or pixel budget')
        budget += area
        checked_pin(root, region['image'])
    require(budget <= MAX_SOURCE_ROI_PIXELS, 'Per-source review pixel budget exceeded')


def _verify_inputs(workspace, config_pin):
    root = Path(workspace).resolve()
    config = read(checked_pin(root, config_pin))
    require(set(config) == {'schema_version', 'task', 'protocol', 'previous_ocr_release', 'readings', 'code'}
            and config['schema_version'] == 'coallocation-review-config/1.0' and config['task'] == 'P14.1b.5'
            and config['protocol'] == PROTOCOL_PIN and config['previous_ocr_release'] == OCR_PIN, 'Invalid review config')
    require(len(config['code']) == len(CODE) and {p['path'] for p in config['code']} == set(CODE), 'Missing review code pins')
    for record in config['code']:
        checked_pin(root, record)
    checked_pin(root, OCR_PIN)
    protocol, values = _base(root)
    require(len(config['readings']) == len(READINGS) and {p['path'] for p in config['readings']} == set(READINGS), 'Missing reading reviews')
    prepared = {row['input_path']: row for row in values['prepared_catalog']['entries']}
    files = {row['path']: row for row in values['catalog']['files']}
    reviews, authored, historical, constraints = [], {}, [], []
    for record in sorted(config['readings'], key=lambda p: p['path']):
        review = read(checked_pin(root, record))
        require(isinstance(review, dict) and REVIEW_FIELDS <= set(review) <= REVIEW_FIELDS | REVIEW_EXTRAS,
                'Unexpected visual review fields')
        require(review.get('training_started', False) is False
                and review['schema_version'] == 'coallocation-reading/1.0' and review['protocol'] == PROTOCOL_PIN
                and review['reviewer'] == 'assistant' and review['human_verified'] is False
                and review['ocr_run'] is False and review['native_cad_opened'] is False, 'Unexpected visual review claims')
        require(datetime.fromisoformat(review['reviewed_at_utc']) >= datetime.fromisoformat(protocol['created_at_utc']), 'Review precedes protocol')
        scope = protocol['numeric_source_paths' if record['path'] == READINGS[0] else 'project_source_paths']
        require(len(review['sources']) == len(scope) and {row['input_path'] for row in review['sources']} == set(scope), 'All fixed-scope sources must remain reviewed')
        for row in review['sources']:
            _checked_source(root, row, prepared, files)
            require(row['input_path'] not in authored, 'Duplicate reviewed source')
            authored[row['input_path']] = row
        comparisons = review.get('historical_comparisons', [])
        require(isinstance(comparisons, list) and len(comparisons) <= 1, 'Unexpected historical comparison inventory')
        for row in comparisons:
            _checked_source(root, row, prepared, files)
            require(row['input_path'] == HISTORICAL_SOURCE and row['input_path'] not in authored
                    and files[row['input_path']]['train_only'] is True
                    and isinstance(row.get('interpretation'), str) and row['interpretation'].strip(), 'Invalid historical-train comparison')
            historical.append(row)
        covered = set()
        for conclusion in review['conclusions']:
            require(set(conclusion) == {'id', 'member_paths', 'basis', 'reason'}, 'Unexpected review conclusion fields')
            require(set(conclusion['member_paths']) <= set(scope), 'Conclusion contains unreviewed source')
            covered.update(conclusion['member_paths'])
            constraints.append({**conclusion, 'evidence': [record]})
        require(covered == set(scope), 'Every source needs an explicit reviewed constraint or uncertainty hold')
        reviews.append({'reading': record, 'review': review})
    require(len(historical) == 1, 'Retain the historical-train comparison')
    overlay = build_coallocation_v1(values['catalog'], values['policy'], values['exclusions'], constraints)
    return {'config': config, 'protocol': protocol, 'values': values, 'reviews': reviews,
            'sources': [authored[path] for path in sorted(authored)], 'historical': historical, 'overlay': overlay}


def _previous_routing(root, values, overlay):
    previous = load_ocr_training_v2(root, OCR_PIN)
    # The consumer above replays and authenticates labels.json against its recipe.
    labels = read(checked_pin(root, OCR_PIN).parent/'labels.json')
    sources = {row['id']: row['source'] for row in labels['source_records']}
    require(len(sources) == len(labels['source_records']), 'Duplicate prior OCR source identity')
    registry = []
    for row in previous['records']:
        require(row['source_id'] in sources and row['split'] == 'train', 'Previous OCR source or split differs')
        source = sources[row['source_id']]
        require(source['path'].startswith('dataset/') and source['sha256'] == row['source_sha256'], 'Previous OCR source binding differs')
        checked_pin(root, source)
        registry.append({'id': row['id'], 'source_path': source['path'].removeprefix('dataset/'),
                         'source_sha256': row['source_sha256'], 'split': row['split'], 'image_sha256': row['image_sha256']})
    require(previous['count'] == len(registry) == 32, 'Previous OCR train registry differs')
    return validate_assignments_v1(values['catalog'], values['policy'], values['exclusions'], overlay, registry)


def source_regions_v1(source, source_sha, boxes):
    """Replay up to eight bounded ROIs without allocating the full drawing."""
    require(isinstance(boxes, dict) and 1 <= len(boxes) <= 8, 'Expected bounded ROI mapping')
    reader = PNGStripReader(Path(source), expected_sha256=source_sha)
    budget = 0
    for box in boxes.values():
        require(isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box), 'Expected integer half-open ROI')
        x0, y0, x1, y1 = box
        area = (x1-x0)*(y1-y0)
        require(0 <= x0 < x1 <= reader.width and 0 <= y0 < y1 <= reader.height
                and area <= MAX_REGION_PIXELS, 'ROI outside source or pixel budget')
        budget += area
    require(budget <= MAX_SOURCE_ROI_PIXELS, 'Per-source review pixel budget exceeded')
    images = {key: np.empty((box[3]-box[1], box[2]-box[0], 3), dtype=np.uint8) for key, box in boxes.items()}
    for y, strip in reader.iter_strips():
        for key, (x0, y0, x1, y1) in boxes.items():
            start, end = max(y, y0), min(y+len(strip), y1)
            if start < end:
                images[key][start-y0:end-y0] = strip[start-y:end-y, x0:x1]
    require(reader.validated, 'Original PNG stream incomplete')
    return images


def _documents(workspace, config_pin, verified, progress=None):
    root = Path(workspace).resolve()
    evidence = []
    rows = verified['sources']+verified['historical']
    for index, row in enumerate(rows, 1):
        boxes = {r['kind']: r['box_source_px'] for r in row['regions']}
        raw = source_regions_v1(contained(root, 'dataset/'+row['input_path']), row['input_sha256'], boxes)
        prep_path = checked_pin(root, row['preparation'])
        regions = []
        for region in row['regions']:
            kind, box = region['kind'], region['box_source_px']
            pixels, provenance = read_prepared_region(prep_path.parent, box, max_pixels=MAX_REGION_PIXELS)
            require(np.array_equal(raw[kind], pixels), 'Prepared pixels differ from original PNG')
            with Image.open(checked_pin(root, region['image'])) as image:
                require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (box[2]-box[0], box[3]-box[1])
                        and getattr(image, 'n_frames', 1) == 1, 'Reviewed image dimensions or mode differ')
                require(np.array_equal(np.asarray(image), pixels), 'Reviewed image differs from original ROI')
            regions.append({**region, 'tile_provenance': provenance, 'original_png_stream_replay_equal': True,
                            'source_original_rehashed': True, 'transcription_approved': False})
        evidence.append({'source': {'path': 'dataset/'+row['input_path'], 'sha256': row['input_sha256']},
                         'preparation': row['preparation'], 'source_size_px': row['source_size_px'], 'regions': regions})
        if progress:
            progress(index, len(rows), row['input_path'])
    routing = _previous_routing(root, verified['values'], verified['overlay'])
    overlay = verified['overlay']
    report = {'schema_version': 'coallocation-review/1.0', 'task': 'P14.1b.5', 'config': config_pin,
              'status': 'additive_coallocation_complete_semantic_families_and_labels_unapproved',
              'summary': overlay['summary'], 'reviewed_source_count': len(verified['sources']),
              'historical_comparison_count': len(verified['historical']), 'source_files_rehashed': len(rows),
              'region_count': sum(len(row['regions']) for row in rows), 'previous_ocr_train_count': routing['assignment_count'],
              'source_pixel_replay': 'bounded_original_RGB_PNG_strips_equal_prepared_and_viewed_regions',
              'routing_scope': overlay['routing_scope'], 'global_split_policy_modified': False,
              'source_modified': False, 'OCR_run': False, 'training_started': False, 'split': None,
              **{key: False for key in FALSE_FLAGS},
              'limitations': ['Assistant-authored visual observations are not human-confirmed labels.',
                  'Coallocation unions complete old groups; it neither identifies semantic families nor proves independence.',
                  'Different codes or failure to find a connection do not prove independence from historical train.',
                  'Only the previous complete 32-record OCR registry is checked for routing compatibility here.',
                  'Future releases must supply the full existing-plus-proposed registry and pass separate label admission.']}
    documents = {'overlay.json': overlay, 'constraints.json': overlay['constraints'], 'evidence.json': evidence,
                 'readings.json': verified['reviews'], 'previous-ocr-routing.json': routing, 'report.json': report}
    html = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Совместное распределение чертежей</title>',
            '<style>body{font:17px system-ui;max-width:1250px;margin:2em auto}img{max-width:100%}section{margin:3em 0}</style>',
            '<h1>Связи чертежей и совместное распределение</h1><p>Ограничения дополняют прежние группы. Семейства, метки и независимые выборки не утверждены.</p>',
            '<p><a href="overlay.json">Дополнительные группы</a> · <a href="constraints.json">Основания</a> · <a href="report.json">Отчёт</a></p>']
    for row in rows:
        html.append('<section><h2>'+escape(Path(row['input_path']).name)+'</h2><p>'+escape(str(row['title_observation']))+'</p>')
        html.append('<p>'+escape(' '.join(row['observations']))+'</p>')
        for region in row['regions']:
            html.append('<p>'+escape(region['kind'])+'</p><img loading="lazy" src="../../'+escape(region['image']['path'], quote=True)+'" alt="Точная область исходного PNG">')
        html.append('</section>')
    return {**{name: encoded(value) for name, value in documents.items()},
            'index.html': (''.join(html)+'</html>\n').encode('utf-8')}, report


def run_coallocation_review_v1(workspace, config_pin, output_dir, *, progress=None):
    root = Path(workspace).resolve()
    output = contained(root, output_dir)
    require(output.parent == root/'artifacts' and not output.exists(), 'Use a new direct artifacts directory')
    verified = _verify_inputs(root, config_pin)
    files, report = _documents(root, config_pin, verified, progress)
    _verify_inputs(root, config_pin)
    output.mkdir()
    for name, data in files.items():
        (output/name).write_bytes(data)
    manifest = {'schema_version': 'coallocation-review-manifest/1.0', 'config': config_pin,
                'files': [pin(output, name) for name in sorted(files)]}
    (output/'manifest.json').write_bytes(encoded(manifest))
    return report


def verify_coallocation_review_v1(workspace, manifest_pin, *, progress=None):
    root = Path(workspace).resolve()
    path = checked_pin(root, manifest_pin)
    manifest = read(path)
    require(set(manifest) == {'schema_version', 'config', 'files'} and
            manifest['schema_version'] == 'coallocation-review-manifest/1.0', 'Wrong review manifest')
    files, report = _documents(root, manifest['config'], _verify_inputs(root, manifest['config']), progress)
    require(len(manifest['files']) == len(files) and {p['path'] for p in manifest['files']} == set(files), 'Review inventory changed')
    require({p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file()} == set(files)|{'manifest.json'}, 'Unlisted review file')
    for record in manifest['files']:
        artifact = checked_pin(path.parent, record)
        require(artifact.read_bytes() == files[record['path']], 'Review differs from authenticated inputs')
    _verify_inputs(root, manifest['config'])
    return report
