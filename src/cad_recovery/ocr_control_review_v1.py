"""Source-only control review packets, with pending families and no split release."""
from collections import defaultdict
from datetime import datetime
from html import escape
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .clear_pairs import contained, digest
from .label_release_v1 import checked_pin
from .ocr_control_selection_v1 import PROTOCOL_V1, build_control_queue_v1
from .png_stream import PNGStripReader
from .raster_regions import read_prepared_region

PROTOCOL_PIN = {'path': 'configs/reviews/ocr-control-protocol-v1.json',
                'sha256': '68f9f48bc999bc6134a8873f6bc7f8ec534c0d1ed58a11afc84c02e0c098c64f'}
SELECTION_PIN = {'path': 'artifacts/ocr-control-selection-v1.json',
                 'sha256': '81a60df2e1e5037d475236fae6e0c7445517d1ee1f071839d7a88e900e6c4374'}
PARENT_PIN = {'path': 'artifacts/ocr-baseline-final-v1.json',
              'sha256': '98f32a7107451b87db65fea6ea124770a6c13c290d30d3482e9554fda5863880'}
DEPENDENCIES = {'old_intake', 'new_intake', 'catalog', 'policy', 'handoff', 'exclusions',
                'prepared_catalog', 'dataset_manifest', 'parent_final'}
READINGS = ['artifacts/ocr-control-reading-'+letter+'-v1/review.json' for letter in ('a', 'b')]
CODE = ['src/cad_recovery/'+name+'.py' for name in (
    'ocr_control_review_v1', 'ocr_control_selection_v1', 'ocr_lineage_v1', 'family_catalog_v2',
    'raster_regions', 'png_stream', 'clear_pairs', 'label_release_v1')]+['scripts/build_ocr_control_review_v1.py']


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
    require(protocol['schema_version'] == 'ocr-control-protocol/1.0' and protocol['task'] == 'P14.1b.4'
            and protocol['protocol'] == PROTOCOL_V1 and protocol['OCR_run'] is False
            and protocol['training_started'] is False and protocol['split_assignment'] is None, 'Changed selection protocol')
    require(set(protocol['dependencies']) == DEPENDENCIES and protocol['dependencies']['parent_final'] == PARENT_PIN,
            'Unexpected protocol inputs')
    values = {k: read(checked_pin(root, p)) for k, p in protocol['dependencies'].items()}
    sources = values['dataset_manifest']['files']
    expected = {f['path']: f['sha256'] for f in sources}
    require(len(expected) == len(sources) == 1622 and
            expected == {f['path']: f['sha256'] for f in values['catalog']['files']}, 'Dataset inventory differs')
    for path, sha in expected.items():
        checked_pin(root, {'path': 'dataset/'+path, 'sha256': sha})
    queue = build_control_queue_v1(*(values[k] for k in ('old_intake', 'new_intake', 'catalog', 'policy', 'handoff', 'exclusions')),
                                   protocol=protocol['protocol'])
    require(protocol['expected_counts'] == {'old_sources': 127, 'combined_sources': 510, 'new_sources': 383, 'selected_sources': 6}
            and queue['summary']['new_source_count'] == 383 and queue['summary']['selected_count'] == 6,
            'Queue scope changed')
    selection = read(checked_pin(root, SELECTION_PIN))
    require(selection['schema_version'] == 'ocr-control-selection/1.0' and selection['protocol'] == PROTOCOL_PIN
            and selection['queue'] == queue and selection['images_inspected_before_selection'] is False
            and selection['OCR_run'] is False, 'Selection differs from preregistered recipe')
    require(datetime.fromisoformat(protocol['created_before_selection_utc']) <= datetime.fromisoformat(selection['selected_at_utc']),
            'Selection precedes protocol')
    return values, queue, selection


def prepare_control_review_config_v1(workspace, output_path):
    root = Path(workspace).resolve()
    output = contained(root, output_path)
    require(output.parent == root/'configs/reviews' and not output.exists(), 'Use a new review config')
    _base(root)
    config = {'schema_version': 'ocr-control-review-config/1.0', 'task': 'P14.1b.4',
              'protocol': PROTOCOL_PIN, 'selection': SELECTION_PIN,
              'readings': [pin(root, path) for path in READINGS], 'code': [pin(root, path) for path in CODE]}
    output.write_bytes(encoded(config))
    return pin(root, output.relative_to(root).as_posix())


def _verify_inputs(workspace, config_pin):
    root = Path(workspace).resolve()
    config = read(checked_pin(root, config_pin))
    require(set(config) == {'schema_version', 'task', 'protocol', 'selection', 'readings', 'code'}
            and config['schema_version'] == 'ocr-control-review-config/1.0' and config['task'] == 'P14.1b.4'
            and config['protocol'] == PROTOCOL_PIN and config['selection'] == SELECTION_PIN, 'Invalid review config')
    require(len(config['code']) == len(CODE) and {p['path'] for p in config['code']} == set(CODE), 'Missing review code pins')
    for p in config['code']:
        checked_pin(root, p)
    values, queue, selection = _base(root)
    require(len(config['readings']) == len(READINGS) and {p['path'] for p in config['readings']} == set(READINGS), 'Missing reading reviews')
    authored = {}
    for p in config['readings']:
        review = read(checked_pin(root, p))
        require(set(review) == {'schema_version', 'selection', 'reviewer', 'human_verified', 'ocr_run',
                                'native_cad_opened', 'reviewed_at_utc', 'sources'}, 'Unexpected visual review fields')
        require(review['schema_version'] == 'ocr-control-reading/1.0' and review['selection'] == SELECTION_PIN
                and review['reviewer'] == 'assistant' and review['human_verified'] is False
                and review['ocr_run'] is False and review['native_cad_opened'] is False, 'Unexpected visual review claims')
        require(datetime.fromisoformat(review['reviewed_at_utc']) >= datetime.fromisoformat(selection['selected_at_utc']),
                'Review precedes selection')
        for row in review['sources']:
            require(set(row) == {'id', 'input_path', 'source_size_px', 'title_observation', 'related_designations',
                                 'family_observations', 'title_box_source_px', 'context_box_source_px',
                                 'proposed_annotation_roles', 'decision', 'independent_evaluation', 'split',
                                 'annotation_approved', 'training_eligible', 'visual_checks', 'regions'},
                    'Unexpected authored source fields')
            title = row['title_observation']
            require(isinstance(title, dict) and set(title) == {'designation', 'name', 'uncertainty'} and
                    all(title[k] is None or isinstance(title[k], str) for k in ('designation', 'name')) and
                    isinstance(title['uncertainty'], str) and title['uncertainty'].strip(), 'Invalid title observation')
            require(isinstance(row['proposed_annotation_roles'], list) and row['proposed_annotation_roles'] and
                    all(isinstance(v, str) and v.strip() for v in row['proposed_annotation_roles']), 'Missing annotation context roles')
            require(row['id'] not in authored and row['id'] in queue['selected_ids'], 'Repeated or unselected source review')
            require(row['decision'] == 'pending_family_review' and row['split'] is None
                    and all(row[k] is False for k in ('independent_evaluation', 'annotation_approved', 'training_eligible')),
                    'Visual review cannot approve independent data or labels')
            require(row['visual_checks'] == {'overview_seen': True, 'title_seen': True, 'context_seen': True}, 'Incomplete visual review')
            require(set(row['regions']) == {'title', 'context'}, 'Expected title and OCR context images')
            for region_pin in row['regions'].values():
                checked_pin(root, region_pin)
            require(isinstance(row['family_observations'], list) and row['family_observations']
                    and all(isinstance(x, str) and x.strip() for x in row['family_observations']), 'Missing family uncertainty')
            require(isinstance(row['related_designations'], list), 'Missing related designation inventory')
            for relation in row['related_designations']:
                require(set(relation) == {'text', 'field_label', 'uncertainty'}
                        and all(isinstance(v, str) and v.strip() for v in relation.values()), 'Invalid related designation observation')
            authored[row['id']] = {**row, 'reading_pin': p}
    require(set(authored) == set(queue['selected_ids']), 'All six selected sources must retain review outcomes')
    prepared = {p['input_path']: p for p in values['prepared_catalog']['entries']}
    require(len(prepared) == len(values['prepared_catalog']['entries']) == 510, 'Prepared catalog inventory changed')
    for row in queue['entries']:
        if row['id'] not in authored:
            continue
        annotation = authored[row['id']]
        binding = prepared[row['input_path']]
        require(annotation['input_path'] == row['input_path'] and binding['input_sha256'] == row['input_sha256']
                and binding['preparation'] == row['preparation'], 'Review source/preparation identity differs')
        prep = read(checked_pin(root, row['preparation']))
        require(prep['source_sha256'] == row['input_sha256'] and prep['source_size_px'] == annotation['source_size_px']
                and binding['source_size_px'] == annotation['source_size_px'], 'Preparation source differs')
    return config, queue, authored


def source_regions_v1(source, source_sha, boxes):
    """Independently replay small regions using bounded original PNG strips."""
    reader = PNGStripReader(Path(source), expected_sha256=source_sha)
    images = {}
    for key, box in boxes.items():
        require(isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box), 'Expected integer half-open ROI')
        x0, y0, x1, y1 = box
        require(0 <= x0 < x1 <= reader.width and 0 <= y0 < y1 <= reader.height
                and (x1-x0)*(y1-y0) <= 4_194_304, 'ROI outside source or pixel budget')
        images[key] = np.empty((y1-y0, x1-x0, 3), dtype=np.uint8)
    for y, strip in reader.iter_strips():
        for key, (x0, y0, x1, y1) in boxes.items():
            start, end = max(y, y0), min(y+len(strip), y1)
            if start < end:
                images[key][start-y0:end-y0] = strip[start-y:end-y, x0:x1]
    require(reader.validated, 'Original PNG stream incomplete')
    return images


def _documents(workspace, config_pin, verified):
    root = Path(workspace).resolve()
    config, queue, authored = verified
    rows = {r['id']: r for r in queue['entries']}
    images, evidence = {}, []
    for identifier in queue['selected_ids']:
        row, review = rows[identifier], authored[identifier]
        boxes = {key: review[key+'_box_source_px'] for key in ('title', 'context')}
        raw = source_regions_v1(root/'dataset'/row['input_path'], row['input_sha256'], boxes)
        prep_path = checked_pin(root, row['preparation'])
        prep = read(prep_path)
        overview = contained(prep_path.parent, prep['overview']['path'])
        require(digest(overview) == prep['overview']['sha256'], 'Overview hash differs')
        images[identifier+'/overview.png'] = overview.read_bytes()
        regions = {}
        for kind, box in boxes.items():
            pixels, provenance = read_prepared_region(prep_path.parent, box)
            require(np.array_equal(raw[kind], pixels), 'Preparation pixels differ from original PNG')
            with Image.open(checked_pin(root, review['regions'][kind])) as viewed:
                require(viewed.mode == 'RGB' and np.array_equal(np.array(viewed), pixels), 'Reviewed image differs from declared ROI')
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format='PNG')
            path = identifier+'/'+kind+'.png'
            images[path] = stream.getvalue()
            regions[kind] = {'path': path, 'source_box_px': box, 'size_px': [pixels.shape[1], pixels.shape[0]],
                             'reviewed_image': review['regions'][kind], 'tile_provenance': provenance,
                             'original_png_stream_replay_equal': True, 'source_original_rehashed': True,
                             'physical_scale_confirmed': False, 'transcription_approved': False}
        evidence.append({'id': identifier, 'source': {'path': 'dataset/'+row['input_path'], 'sha256': row['input_sha256']},
                         'preparation': row['preparation'], 'overview': {'path': overview.relative_to(root).as_posix(), 'sha256': prep['overview']['sha256']},
                         'source_size_px': review['source_size_px'], 'regions': regions})
    by_reference = defaultdict(set)
    for identifier, review in authored.items():
        for relation in review['related_designations']:
            by_reference[relation['text']].add(identifier)
    links = [{'observed_designation': text, 'source_ids': sorted(ids),
              'status': 'shared_printed_reference_requires_joint_family_review', 'family_approved': False}
             for text, ids in sorted(by_reference.items()) if len(ids) > 1]
    reviews = {'schema_version': 'ocr-control-visual-review/1.0', 'selection': SELECTION_PIN,
               'sources': [authored[x] for x in queue['selected_ids']], 'shared_reference_observations': links,
               'human_verified': False, 'annotation_approved': False, 'independent_evaluation': False, 'split': None}
    report = {'schema_version': 'ocr-control-review/1.0', 'task': 'P14.1b.4', 'config': config_pin,
              'status': 'queue_and_six_visual_packets_complete_pending_family_review', 'summary': queue['summary'],
              'source_files_rehashed': 1622, 'reviewed_source_count': len(evidence), 'region_count': 2*len(evidence),
              'source_pixel_replay': 'bounded_original_RGB_PNG_strips_equal_prepared_regions',
              'shared_reference_observations': links, 'OCR_run': False, 'training_started': False,
              'annotation_approved': False, 'independent_evaluation': False, 'global_split_policy_modified': False,
              'source_modified': False, 'native_CAD_used_for_labels': False, 'split': None,
              'limitations': queue['limitations']+['Printed title/reference observations remain assistant-authored and unconfirmed.',
                  'Candidate regions are context for subsequent line annotation, not released OCR labels.',
                  'Different filename strata and coallocation IDs do not establish independent semantic families.']}
    files = {'queue.json': encoded(queue), 'review.json': encoded(reviews), 'evidence.json': encoded(evidence),
             'report.json': encoded(report), **images}
    html = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Очередь OCR-контроля</title>',
            '<style>body{font:17px system-ui;max-width:1250px;margin:2em auto}img{max-width:100%}section{margin:3em 0}</style>',
            '<h1>Шесть источников для дальнейшей проверки OCR</h1><p>Все остаются pending. Независимые выборки и метки не выпущены.</p>',
            '<p><a href="queue.json">Полная очередь</a> · <a href="review.json">Наблюдения</a> · <a href="report.json">Отчёт</a></p>']
    for identifier in queue['selected_ids']:
        row, review = rows[identifier], authored[identifier]
        html.append('<section><h2>'+escape(Path(row['input_path']).name)+'</h2><p>'+escape(str(review['title_observation']))+
                    '</p><p>'+escape(' '.join(review['family_observations']))+'</p>')
        for kind, title in [('overview', 'Обзор'), ('title', 'Штамп и контекст'), ('context', 'Область будущей разметки')]:
            html.append('<h3>'+title+'</h3><img src="'+identifier+'/'+kind+'.png" alt="'+title+'">')
        html.append('</section>')
    files['index.html'] = (''.join(html)+'</html>\n').encode('utf-8')
    return files, report


def run_control_review_v1(workspace, config_pin, output_dir):
    root = Path(workspace).resolve()
    output = contained(root, output_dir)
    require(output.parent == root/'artifacts' and not output.exists(), 'Use a new direct artifacts directory')
    verified = _verify_inputs(root, config_pin)
    files, report = _documents(root, config_pin, verified)
    _verify_inputs(root, config_pin)
    output.mkdir()
    for name, data in files.items():
        path = contained(output, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {'schema_version': 'ocr-control-review-manifest/1.0', 'config': config_pin,
                'files': [pin(output, name) for name in sorted(files)]}
    (output/'manifest.json').write_bytes(encoded(manifest))
    return report


def verify_control_review_v1(workspace, manifest_pin):
    root = Path(workspace).resolve()
    path = checked_pin(root, manifest_pin)
    manifest = read(path)
    require(set(manifest) == {'schema_version', 'config', 'files'} and
            manifest['schema_version'] == 'ocr-control-review-manifest/1.0', 'Wrong review manifest')
    files, report = _documents(root, manifest['config'], _verify_inputs(root, manifest['config']))
    require(len(manifest['files']) == len(files) and {p['path'] for p in manifest['files']} == set(files), 'Review inventory changed')
    require({p.relative_to(path.parent).as_posix() for p in path.parent.rglob('*') if p.is_file()} == set(files)|{'manifest.json'},
            'Unlisted review file')
    for record in manifest['files']:
        artifact = checked_pin(path.parent, record)
        require(artifact.read_bytes() == files[record['path']], 'Review differs from authenticated inputs')
    _verify_inputs(root, manifest['config'])
    return report
