"""Frozen, offline OCR development diagnostic; never trains or changes targets."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from PIL import Image

from .clear_pairs import contained, digest
from .label_release_v1 import checked_pin
from .ocr_engine import LocalOCREngine, DEFAULT_MODEL_MANIFEST, _preflight
from .ocr_release_v2 import load_ocr_training_v2
from .ocr_baseline_metrics_v1 import aggregate_scores_v1

RELEASE = {'path': 'artifacts/ocr-release-v2/manifest.json',
           'sha256': '48139783b64609ba0dfb127c4235e28cb63622106802b9c28a8d9253eed116ef'}
PARENT = {'path': 'artifacts/ocr-final-v2.json',
          'sha256': '17c6ceb98b3a567031c1b889fa389445655d7b3973afef91cd5b2de18835e603'}
CODE_PATHS = ['src/cad_recovery/ocr_baseline_v1.py', 'src/cad_recovery/ocr_baseline_metrics_v1.py',
              'src/cad_recovery/ocr_engine.py', 'scripts/evaluate_ocr_baseline_v1.py',
              'src/cad_recovery/data/ocr-models-v1.json', 'requirements.lock', 'pyproject.toml']
PROTOCOL = {
    'scope': 'development_train_single_line_crops_det_cls_rec', 'sample_count': 32,
    'sample_order': 'id_ascending', 'detector_order': 'min_x_then_min_y_then_backend_index',
    'join': 'one_ASCII_space_between_unmodified_raw_strings', 'confidence_filter': False,
    'empty_prediction': '', 'error_prediction': '', 'error_policy': 'record_and_score_full_deletion',
    'raw_metric': 'Unicode_codepoint_micro_CER_and_exact_line_match',
    'secondary_normalization': 'NFC_whitespace_collapse_trim_diameter_variants_dash_variants',
    'character_map': {'Ø': '⌀', '∅': '⌀', '-': '–', '−': '–'},
    'warmup_calls': 0, 'engine_instances': 1, 'first_cold_call_included': True,
    'timing': 'perf_counter_init_decode_recognize_separate',
    'training': False, 'parameter_selection': False, 'independent_evaluation': False,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8', newline='\n')


def _pin(root, path):
    path = Path(path).resolve()
    return {'path': path.relative_to(root).as_posix(), 'sha256': digest(path)}


def _inventory(root, records):
    return [{**{k: row[k] for k in ('id', 'source_id', 'source_sha256', 'group_id', 'image_sha256', 'text')},
             'image_path': Path(row['image_path']).relative_to(root).as_posix()}
            for row in sorted(records, key=lambda r: r['id'])]


def prepare_config_v1(workspace, output_path):
    """Seal inputs and comparison policy before any call to recognize()."""
    root = Path(workspace).resolve()
    output = contained(root, output_path)
    _require(output.parent == root/'configs/reviews' and not output.exists(), 'Use a new configs/reviews file')
    checked_pin(root, PARENT)
    records = load_ocr_training_v2(root, RELEASE)['records']
    model_dir = root/'artifacts/models/ppocr-v5'
    _, paths, _ = _preflight(model_dir)
    import rapidocr
    defaults = Path(rapidocr.__file__).parent/'config.yaml'
    pins = [_pin(root, root/p) for p in CODE_PATHS]
    pins += [_pin(root, path) for path in [defaults, *paths.values()]]
    config = {'schema_version': 'ocr-baseline-config/1.0', 'task': 'P16.1a',
              'created_before_inference_utc': datetime.now(timezone.utc).isoformat(),
              'release': RELEASE, 'parent_final': PARENT, 'protocol': PROTOCOL,
              'model_dir': model_dir.relative_to(root).as_posix(),
              'pins': sorted(pins, key=lambda p: p['path']), 'inventory': _inventory(root, records)}
    _require(len(records) == 32, 'Expected all 32 released OCR lines')
    output.parent.mkdir(parents=True, exist_ok=True)
    _write(output, config)
    return _pin(root, output)


def _verify_inputs(workspace, config_pin):
    root = Path(workspace).resolve()
    config = _read(checked_pin(root, config_pin))
    _require(set(config) == {'schema_version', 'task', 'created_before_inference_utc', 'release',
                             'parent_final', 'protocol', 'model_dir', 'pins', 'inventory'}, 'Unexpected config fields')
    _require(config['schema_version'] == 'ocr-baseline-config/1.0' and config['task'] == 'P16.1a', 'Wrong experiment')
    _require(config['protocol'] == PROTOCOL and config['release'] == RELEASE and config['parent_final'] == PARENT,
             'Experiment policy or frozen release changed')
    created = datetime.fromisoformat(config['created_before_inference_utc'])
    _require(created.tzinfo is not None, 'Protocol timestamp must include timezone')
    checked_pin(root, PARENT)
    pins = config['pins']
    _require(len({p['path'] for p in pins}) == len(pins), 'Duplicate input pins')
    for pin in pins:
        checked_pin(root, pin)
    model_dir = contained(root, config['model_dir'])
    _require(model_dir == root/'artifacts/models/ppocr-v5', 'Wrong model directory')
    _, paths, _ = _preflight(model_dir)
    import rapidocr
    required = {root/p for p in CODE_PATHS}
    required.update([Path(rapidocr.__file__).parent/'config.yaml', *paths.values()])
    _require({(root/p['path']).resolve() for p in pins} == {p.resolve() for p in required}, 'Incomplete input/code pins')
    _require((root/'src/cad_recovery/data/ocr-models-v1.json').resolve() == DEFAULT_MODEL_MANIFEST.resolve(),
             'Unexpected installed model manifest')
    records = load_ocr_training_v2(root, config['release'])['records']
    _require(len(records) == 32 and _inventory(root, records) == config['inventory'], 'Experiment inventory changed')
    return config, sorted(records, key=lambda r: r['id']), model_dir


def combine_detections_v1(detections):
    """Reading order for pre-oriented single-line crops, independent of truth."""
    _require(isinstance(detections, list), 'Detections must be a list')
    keys = []
    for i, item in enumerate(detections):
        _require(isinstance(item, dict) and set(item) == {'raw', 'confidence', 'polygon_px'}, 'Invalid detection fields')
        _require(isinstance(item['raw'], str), 'Raw text must be a string')
        score = item['confidence']
        _require(isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score)
                 and 0 <= score <= 1, 'Invalid confidence')
        try:
            polygon = np.asarray(item['polygon_px'], dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid polygon') from exc
        _require(polygon.shape == (4, 2) and np.isfinite(polygon).all(), 'Invalid polygon')
        keys.append((float(polygon[:, 0].min()), float(polygon[:, 1].min()), i))
    order = [key[2] for key in sorted(keys)]
    return {'prediction': ' '.join(detections[i]['raw'] for i in order), 'order': order}


def _make_engine(model_dir):
    return LocalOCREngine(model_dir)


def _effective_engine_settings(engine):
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(engine._backend.cfg, resolve=True, enum_to_str=True)
    model_root = str(Path(engine._backend.cfg.Global.model_root_dir))

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, Path):
            value = str(value)
        return value.replace(model_root, '<model_dir>') if isinstance(value, str) else value
    settings = portable(cfg)
    sessions = {}
    for role in ('det', 'cls', 'rec'):
        session = getattr(engine._backend, 'text_'+role).session.session
        _require(session.get_providers() == ['CPUExecutionProvider'], 'Expected CPU-only OCR')
        options = session.get_session_options()
        sessions[role] = {'providers': session.get_providers(), 'intra_op_threads': options.intra_op_num_threads,
                          'inter_op_threads': options.inter_op_num_threads}
    return {'configuration': settings, 'sessions': sessions}


def _score(predictions):
    return aggregate_scores_v1([{k: row[k] for k in ('id', 'source_id', 'reference', 'prediction', 'status')}
                                for row in predictions])


def _validate_engine_evidence(root, config, provenance, settings, runtime=None):
    manifest, _, models = _preflight(contained(root, config['model_dir']))
    expected = {'engine': 'RapidOCR', 'engine_version': manifest['rapidocr_version'],
                'onnxruntime_version': manifest['onnxruntime_version'], 'provider': 'CPUExecutionProvider',
                'model_manifest_id': manifest['id'], 'model_manifest_sha256': digest(DEFAULT_MODEL_MANIFEST),
                'models': models, 'offline': True, 'automatic_downloads': 'blocked',
                'recognition_score_threshold': 0.0, 'inter_op_threads': 1}
    _require(all(provenance.get(k) == v for k, v in expected.items()), 'Engine provenance differs from pinned models')
    threads = provenance.get('intra_op_threads')
    _require(type(threads) is int and 1 <= threads <= 4, 'Unexpected CPU budget')
    session = {'providers': ['CPUExecutionProvider'], 'intra_op_threads': threads, 'inter_op_threads': 1}
    _require(settings['sessions'] == {role: session for role in ('det', 'cls', 'rec')}, 'Runtime sessions differ from provenance')
    global_cfg = settings['configuration']['Global']
    _require(all(global_cfg[k] is True for k in ('use_det', 'use_cls', 'use_rec')) and global_cfg['text_score'] == 0.0,
             'Expected complete unfiltered OCR pipeline')
    if runtime is not None:
        _require(runtime['packages']['rapidocr'] == manifest['rapidocr_version'] and
                 runtime['packages']['onnxruntime'] == manifest['onnxruntime_version'], 'Runtime package versions changed')


def _report(config_pin, scores, provenance, settings):
    return {'schema_version': 'ocr-baseline-report/1.0', 'config': config_pin, 'task': 'P16.1a',
            'scope': PROTOCOL['scope'], 'sample_count': scores['sample_count'],
            'status_counts': scores['status_counts'], 'raw': scores['raw'], 'normalized': scores['normalized'],
            'by_source': scores['by_source'], 'engine': provenance, 'effective_settings': settings,
            'split': 'train', 'training_started': False, 'parameter_selection': False,
            'independent_evaluation': False, 'source_modified': False,
            'limitations': ['Two already inspected development sources from one train-only lineage.',
                           'Detector, classifier and recognizer run on pre-oriented line crops, not full pages.',
                           'No physical geometry, CAD or generalization accuracy is established.',
                           'Empty/error outputs stay in denominators; confidence is not a correctness probability.',
                           'Timing varies across runs; the first cold inference is included.']}


def run_baseline_v1(workspace, config_pin, output_dir):
    root = Path(workspace).resolve()
    output = contained(root, output_dir)
    artifacts = root/'artifacts'
    _require(output.parent == artifacts and not output.exists(), 'Use a new direct artifacts directory; never overwrite frozen inputs')
    config, rows, model_dir = _verify_inputs(root, config_pin)
    for pin in [config_pin, config['release'], config['parent_final'], *config['pins']]:
        parent = (root/pin['path']).resolve().parent
        _require(not(parent != artifacts and parent.is_relative_to(artifacts) and output.is_relative_to(parent)),
                 'Output cannot extend frozen inputs')
    start_utc = datetime.now(timezone.utc).isoformat()
    _require(datetime.fromisoformat(config['created_before_inference_utc']) <= datetime.fromisoformat(start_utc),
             'Protocol must precede inference')
    started = time.perf_counter()
    engine = _make_engine(model_dir)
    init_seconds = time.perf_counter()-started
    settings, provenance = _effective_engine_settings(engine), engine.provenance
    _validate_engine_evidence(root, config, provenance, settings)
    predictions, timings = [], []
    for row in rows:
        started = time.perf_counter()
        decode_seconds = recognize_seconds = 0.0
        detections, combined, error = [], {'prediction': '', 'order': []}, None
        stage = 'decode'
        try:
            before = time.perf_counter()
            with Image.open(row['image_path']) as image:
                rgb = np.array(image.convert('RGB'))
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            decode_seconds = time.perf_counter()-before
            stage = 'recognize'
            before = time.perf_counter()
            try:
                detections = engine.recognize(bgr)
            finally:
                recognize_seconds = time.perf_counter()-before
            stage = 'combine'
            combined = combine_detections_v1(detections)
        except Exception as exc:
            if stage == 'decode':
                decode_seconds = time.perf_counter()-started
            error = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc)}
            detections = []
        status = 'error' if error else ('ok' if combined['prediction'] else 'empty')
        predictions.append({'id': row['id'], 'source_id': row['source_id'], 'reference': row['text'],
                            'image': 'images/'+row['id']+'.png', 'image_sha256': row['image_sha256'],
                            'prediction': combined['prediction'], 'status': status, 'error': error,
                            'detections': detections, 'order': combined['order']})
        timings.append({'id': row['id'], 'decode_seconds': decode_seconds, 'recognize_seconds': recognize_seconds,
                        'sample_wall_seconds': time.perf_counter()-started})
    # Recheck after inference so an altered input never receives a finished manifest.
    _verify_inputs(root, config_pin)
    scores = _score(predictions)
    report = _report(config_pin, scores, provenance, settings)
    runtime = {'schema_version': 'ocr-baseline-runtime/1.0', 'started_utc': start_utc,
               'initialization_seconds': init_seconds, 'rows': timings,
               'decode_total_seconds': sum(r['decode_seconds'] for r in timings),
               'recognize_total_seconds': sum(r['recognize_seconds'] for r in timings),
               'hardware': {'platform': platform.platform(), 'processor': platform.processor(),
                            'machine': platform.machine(), 'logical_cpus': os.cpu_count()},
               'python': platform.python_version(),
               'packages': {name: importlib.metadata.version(name) for name in
                            ('rapidocr', 'onnxruntime', 'numpy', 'Pillow', 'opencv-python')},
               'cold_first_sample_id': rows[0]['id'], 'warmup_calls': 0}
    (output/'images').mkdir(parents=True)
    for row in rows:
        target = output/'images'/(row['id']+'.png')
        shutil.copyfile(row['image_path'], target)
        _require(digest(target) == row['image_sha256'], 'Copied image changed')
    for name, obj in [('predictions.json', predictions), ('metrics.json', scores), ('report.json', report), ('runtime.json', runtime)]:
        _write(output/name, obj)
    html = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Диагностика OCR</title>',
            '<style>body{font:17px system-ui;max-width:1100px;margin:2em auto}section{margin:2em 0}img{max-width:100%}pre{white-space:pre-wrap}</style>',
            '<h1>Диагностика OCR: 32 строки разработки</h1><p>Только development/train; обучение не проводилось.</p>',
            '<p><a href="report.json">Метрики</a> · <a href="runtime.json">Время</a></p>']
    for row in predictions:
        html.append('<section><h2>'+escape(row['id'])+'</h2><img src="'+escape(row['image'], quote=True)+
                    '" alt="Фрагмент строки"><pre>Эталон: '+escape(row['reference'])+'\nOCR: '+
                    escape(row['prediction'])+'\nСтатус: '+row['status']+'</pre></section>')
    (output/'index.html').write_text(''.join(html)+'</html>\n', encoding='utf-8', newline='\n')
    _write(output/'manifest.json', {'schema_version': 'ocr-baseline-manifest/1.0', 'config': config_pin,
                                   'files': [{'path': p.relative_to(output).as_posix(), 'sha256': digest(p)}
                                             for p in sorted(output.rglob('*')) if p.is_file()]})
    return report


def verify_baseline_v1(workspace, manifest_pin):
    """Authenticate inventory and recompute every metric without rerunning OCR."""
    root = Path(workspace).resolve()
    manifest_path = checked_pin(root, manifest_pin)
    directory = manifest_path.parent
    manifest = _read(manifest_path)
    _require(set(manifest) == {'schema_version', 'config', 'files'} and
             manifest['schema_version'] == 'ocr-baseline-manifest/1.0', 'Wrong baseline manifest')
    config, rows, _ = _verify_inputs(root, manifest['config'])
    expected = {'predictions.json', 'metrics.json', 'report.json', 'runtime.json', 'index.html'}
    expected.update('images/'+r['id']+'.png' for r in rows)
    _require(len(manifest['files']) == len(expected) and {p['path'] for p in manifest['files']} == expected,
             'Baseline manifest inventory changed')
    _require({p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()} == expected | {'manifest.json'},
             'Unexpected baseline files')
    for pin in manifest['files']:
        _require(digest(contained(directory, pin['path'])) == pin['sha256'], 'Baseline file changed')
    predictions = _read(directory/'predictions.json')
    _require(isinstance(predictions, list) and len(predictions) == len(rows), 'Missing predictions')
    for record, row in zip(predictions, rows, strict=True):
        _require(set(record) == {'id', 'source_id', 'reference', 'image', 'image_sha256', 'prediction', 'status',
                                'error', 'detections', 'order'}, 'Unexpected prediction fields')
        for key, value in {'id': row['id'], 'source_id': row['source_id'], 'reference': row['text'],
                           'image_sha256': row['image_sha256'], 'image': 'images/'+row['id']+'.png'}.items():
            _require(record[key] == value, 'Prediction identity or reference changed')
        _require(digest(directory/record['image']) == row['image_sha256'], 'Source image differs')
        if record['status'] == 'error':
            _require(isinstance(record['error'], dict) and set(record['error']) == {'stage', 'type', 'message'}
                     and record['error']['stage'] in ('decode', 'recognize', 'combine')
                     and all(isinstance(v, str) for v in record['error'].values())
                     and record['detections'] == [] and record['order'] == [] and record['prediction'] == '', 'Invalid error record')
        else:
            combined = combine_detections_v1(record['detections'])
            _require(record['error'] is None and record['prediction'] == combined['prediction'] and record['order'] == combined['order']
                     and record['status'] == ('ok' if combined['prediction'] else 'empty'), 'Detection assembly changed')
    scores = _score(predictions)
    _require(_read(directory/'metrics.json') == scores, 'Metrics differ from predictions')
    report = _read(directory/'report.json')
    _require(report == _report(manifest['config'], scores, report['engine'], report['effective_settings']), 'Report metrics or scope changed')
    runtime = _read(directory/'runtime.json')
    _validate_engine_evidence(root, config, report['engine'], report['effective_settings'], runtime)
    _require([r['id'] for r in runtime['rows']] == [r['id'] for r in rows], 'Timing inventory changed')
    for value in [runtime['initialization_seconds'], *[r[k] for r in runtime['rows'] for k in
                                                     ('decode_seconds', 'recognize_seconds', 'sample_wall_seconds')]]:
        _require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid timing')
    _require(runtime['decode_total_seconds'] == sum(r['decode_seconds'] for r in runtime['rows'])
             and runtime['recognize_total_seconds'] == sum(r['recognize_seconds'] for r in runtime['rows'])
             and runtime['warmup_calls'] == 0 and runtime['cold_first_sample_id'] == rows[0]['id'], 'Timing totals changed')
    _require(datetime.fromisoformat(runtime['started_utc']) >= datetime.fromisoformat(config['created_before_inference_utc']),
             'Inference predates protocol')
    return report
