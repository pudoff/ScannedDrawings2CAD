"""Frozen 90-line development OCR diagnostic with inherited-source regression.

No training, tuning or target repair. Verification authenticates saved evidence
and recomputes scores without inference; a separate real repeat tests execution.
"""
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image

from .clear_pairs import contained, digest
from .family_catalog_v2 import stable_digest
from .label_release_v1 import checked_pin
from .ocr_engine import DEFAULT_MODEL_MANIFEST, _preflight
from .ocr_release_v3 import load_ocr_training_v3
from .ocr_baseline_metrics_v2 import build_strata_v2, score_strata_v2
from .ocr_baseline_v1 import (PROTOCOL as PREVIOUS_PROTOCOL, combine_detections_v1, verify_baseline_v1,
                              _make_engine, _effective_engine_settings, _validate_engine_evidence)

RELEASE = {'path': 'artifacts/ocr-release-v3/manifest.json',
           'sha256': '40459498a211b577e6cdfbca9d8772b4c9b04a9768b2d9d842395995abb87fdd'}
PARENT = {'path': 'artifacts/ocr-final-v3.json',
          'sha256': 'e0f6e9f50d3fd3afb2ee6def7bb308cb15b1587bbfd5655b46d274584caa1d9f'}
PREVIOUS = {'path': 'artifacts/ocr-baseline-v1/manifest.json',
            'sha256': '7784f6fda8be48aaa5946b510b34d45378668589ed6c60e9b6fdff3ffa30beb3'}
CODE_PATHS = ['src/cad_recovery/'+name+'.py' for name in (
    'ocr_baseline_v2', 'ocr_baseline_metrics_v2', 'ocr_baseline_v1', 'ocr_baseline_metrics_v1',
    'ocr_engine', 'ocr_release_v3')]+['scripts/evaluate_ocr_baseline_v2.py',
    'src/cad_recovery/data/ocr-models-v1.json', 'requirements.lock', 'pyproject.toml']
PROTOCOL = {**deepcopy(PREVIOUS_PROTOCOL), 'sample_count': 90,
    'scope': 'development_train_90_single_line_crops_det_cls_rec',
    'cohorts': {'inherited32': 32, 'new58': 58},
    'strata': 'release_metadata_only_frozen_before_inference',
    'previous_comparison': 'same_32_references_images_predictions_status_errors_detections_and_order',
    'settings': 'exact_previous_engine_provenance_effective_configuration_and_package_versions'}
PACKAGES = ('rapidocr', 'onnxruntime', 'numpy', 'Pillow', 'opencv-python')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def same(left, right):
    return stable_digest(left) == stable_digest(right)


def pin(root, path):
    return {'path': path, 'sha256': digest(contained(root, path))}


def _environment():
    return {'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in PACKAGES}}


def _inventory(root, rows):
    return [{**{k: row[k] for k in ('id', 'source_id', 'source_sha256', 'group_id', 'image_sha256', 'text')},
             'image_path': Path(row['image_path']).relative_to(root).as_posix()} for row in sorted(rows, key=lambda r: r['id'])]


def _bundle(root):
    checked_pin(root, PARENT)
    require(RELEASE in read(root/PARENT['path'])['manifests'], 'Parent proof identifies a different OCR release')
    rows = load_ocr_training_v3(root, RELEASE)['records']
    require(len(rows) == 90, 'Expected all 90 released lines')
    labels = read(checked_pin(root, RELEASE).parent/'labels.json')
    old_report = verify_baseline_v1(root, PREVIOUS)
    previous_dir = checked_pin(root, PREVIOUS).parent
    previous = read(previous_dir/'predictions.json')
    old_runtime = read(previous_dir/'runtime.json')
    strata = build_strata_v2(labels, rows)
    require({key: sum(row['cohort'] == key for row in strata) for key in ('inherited32', 'new58')} == PROTOCOL['cohorts'], 'Cohort scope changed')
    return sorted(rows, key=lambda row: row['id']), strata, previous, old_report, old_runtime


def _required_pins(root):
    model_dir = root/'artifacts/models/ppocr-v5'
    _, paths, _ = _preflight(model_dir)
    import rapidocr
    require(DEFAULT_MODEL_MANIFEST.resolve() == (root/'src/cad_recovery/data/ocr-models-v1.json').resolve(), 'Unexpected installed model manifest')
    all_paths = [root/path for path in CODE_PATHS]+[Path(rapidocr.__file__).parent/'config.yaml', *paths.values()]
    return sorted([pin(root, path.resolve().relative_to(root).as_posix()) for path in all_paths], key=lambda p: p['path'])


def prepare_config_v2(workspace, output_path):
    """Seal metadata, runtime settings and comparisons before any recognize call."""
    root = Path(workspace).resolve(); output = contained(root, output_path)
    require(output.parent == root/'configs/reviews' and not output.exists(), 'Use a new configs/reviews file')
    rows, strata, _, old_report, old_runtime = _bundle(root)
    environment = {key: old_runtime[key] for key in ('python', 'packages')}
    require(same(_environment(), environment), 'Runtime versions differ from the previous experiment')
    config = {'schema_version': 'ocr-baseline-config/2.0', 'task': 'P16.1b',
        'created_before_inference_utc': datetime.now(timezone.utc).isoformat(),
        'release': RELEASE, 'parent_final': PARENT, 'previous_baseline': PREVIOUS, 'protocol': PROTOCOL,
        'model_dir': 'artifacts/models/ppocr-v5', 'pins': _required_pins(root), 'inventory': _inventory(root, rows),
        'strata': strata, 'expected_engine': old_report['engine'], 'expected_settings': old_report['effective_settings'],
        'expected_environment': environment}
    output.write_bytes(encoded(config))
    return pin(root, output.relative_to(root).as_posix())


def _verify_inputs(workspace, config_pin):
    root = Path(workspace).resolve(); config = read(checked_pin(root, config_pin))
    require(set(config) == {'schema_version', 'task', 'created_before_inference_utc', 'release', 'parent_final',
        'previous_baseline', 'protocol', 'model_dir', 'pins', 'inventory', 'strata', 'expected_engine',
        'expected_settings', 'expected_environment'}, 'Unexpected baseline config fields')
    require(config['schema_version'] == 'ocr-baseline-config/2.0' and config['task'] == 'P16.1b'
        and same(config['protocol'], PROTOCOL) and config['release'] == RELEASE and config['parent_final'] == PARENT
        and config['previous_baseline'] == PREVIOUS and config['model_dir'] == 'artifacts/models/ppocr-v5', 'Changed experiment scope')
    require(datetime.fromisoformat(config['created_before_inference_utc']).tzinfo is not None, 'Protocol timestamp requires timezone')
    require(config['pins'] == _required_pins(root), 'Code/model/default pins changed or incomplete')
    rows, strata, previous, old_report, old_runtime = _bundle(root)
    require(same(config['inventory'], _inventory(root, rows)) and same(config['strata'], strata), 'Experiment inventory or strata changed')
    require(same(config['expected_engine'], old_report['engine']) and same(config['expected_settings'], old_report['effective_settings']), 'Previous engine settings changed')
    require(same(config['expected_environment'], {key: old_runtime[key] for key in ('python', 'packages')})
            and same(_environment(), config['expected_environment']), 'Runtime environment changed')
    return config, rows, contained(root, config['model_dir']), previous


def _check_engine(root, config, provenance, settings):
    _validate_engine_evidence(root, config, provenance, settings)
    require(same(provenance, config['expected_engine']) and same(settings, config['expected_settings']), 'Engine settings differ from frozen previous run')


def _report(config_pin, scores, provenance, settings):
    overall = scores['overall']
    return {'schema_version': 'ocr-baseline-report/2.0', 'task': 'P16.1b', 'config': config_pin,
        'scope': PROTOCOL['scope'], 'sample_count': overall['sample_count'], 'status_counts': overall['status_counts'],
        'raw': overall['raw'], 'normalized': overall['normalized'], 'by_source': overall['by_source'],
        **{key: scores[key] for key in ('by_cohort', 'by_role_group', 'by_role', 'regression')},
        'engine': provenance, 'effective_settings': settings, 'split': 'train',
        'training_started': False, 'parameter_selection': False, 'independent_evaluation': False, 'source_modified': False,
        'limitations': ['Eight already inspected development sources; coallocation does not establish semantic independence.',
            'The unchanged detector/classifier/recognizer runs on pre-oriented line crops, not full pages.',
            'Inherited and new cohorts and text roles are fixed from release metadata before inference.',
            'Empty/error outputs remain in every denominator; confidence is not a correctness probability.',
            'Raw and normalized comparisons use the frozen V1 policy; no target-based repairs are applied.',
            'Global ID order changes the cold first sample; timing is not a speed regression of the old 32 lines.',
            'Verification checks saved evidence without inference; authentic execution is anchored by trusted manifests.']}


def _html(predictions, strata):
    metadata = {row['id']: row for row in strata}
    body = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Диагностика OCR: 90 строк</title>',
        '<style>body{font:17px system-ui;max-width:1150px;margin:2em auto}section{margin:2em 0}img{max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>OCR: прежние 32 и новые 58 строк</h1><p>Данные разработки train. Обучение и подбор параметров не выполнялись.</p>',
        '<p><a href="report.json">Метрики</a> · <a href="regression.json">Прежние строки</a> · <a href="runtime.json">Время</a></p>']
    for row in predictions:
        meta = metadata[row['id']]
        body.append('<section><h2>'+escape(row['id'])+'</h2><p>'+escape(meta['cohort']+' / '+meta['role'])+
            '</p><img loading="lazy" src="'+escape(row['image'], quote=True)+'" alt="Исходная строка"><pre>Эталон: '+
            escape(row['reference'])+'\nOCR: '+escape(row['prediction'])+'\nСтатус: '+escape(row['status'])+'</pre></section>')
    return (''.join(body)+'</html>\n').encode('utf-8')


def _documents(config_pin, config, predictions, previous, provenance, settings):
    scores = score_strata_v2(predictions, config['strata'], previous)
    report = _report(config_pin, scores, provenance, settings)
    documents = {'predictions.json': predictions, 'metrics.json': scores, 'report.json': report,
                 'strata.json': config['strata'], 'regression.json': scores['regression']}
    return {**{name: encoded(value) for name, value in documents.items()}, 'index.html': _html(predictions, config['strata'])}, report


def run_baseline_v2(workspace, config_pin, output_dir):
    root = Path(workspace).resolve(); output = contained(root, output_dir)
    require(output.parent == root/'artifacts' and not output.exists(), 'Use a new direct artifacts directory')
    config, rows, model_dir, previous = _verify_inputs(root, config_pin)
    start_utc = datetime.now(timezone.utc).isoformat()
    require(datetime.fromisoformat(start_utc) >= datetime.fromisoformat(config['created_before_inference_utc']), 'Protocol must precede inference')
    start = time.perf_counter(); engine = _make_engine(model_dir); init_seconds = time.perf_counter()-start
    settings, provenance = _effective_engine_settings(engine), engine.provenance
    _check_engine(root, config, provenance, settings)
    predictions, timings = [], []
    for row in rows:
        started = time.perf_counter(); decode_seconds = recognize_seconds = 0.0
        detections, combined, error, stage = [], {'prediction': '', 'order': []}, None, 'decode'
        try:
            with Image.open(row['image_path']) as image:
                rgb = np.array(image.convert('RGB'))
            bgr = np.ascontiguousarray(rgb[:, :, ::-1]); decode_seconds = time.perf_counter()-started
            stage = 'recognize'; before = time.perf_counter()
            try:
                detections = engine.recognize(bgr)
            finally:
                recognize_seconds = time.perf_counter()-before
            stage = 'combine'; combined = combine_detections_v1(detections)
        except Exception as exc:
            if stage == 'decode':
                decode_seconds = time.perf_counter()-started
            error = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc)}
            detections, combined = [], {'prediction': '', 'order': []}
        status = 'error' if error else ('ok' if combined['prediction'] else 'empty')
        predictions.append({'id': row['id'], 'source_id': row['source_id'], 'reference': row['text'],
            'image': 'images/'+row['id']+'.png', 'image_sha256': row['image_sha256'], 'prediction': combined['prediction'],
            'status': status, 'error': error, 'detections': detections, 'order': combined['order']})
        timings.append({'id': row['id'], 'decode_seconds': decode_seconds, 'recognize_seconds': recognize_seconds,
                        'sample_wall_seconds': time.perf_counter()-started})
    _verify_inputs(root, config_pin)
    files, report = _documents(config_pin, config, predictions, previous, provenance, settings)
    runtime = {'schema_version': 'ocr-baseline-runtime/2.0', 'started_utc': start_utc,
        'initialization_seconds': init_seconds, 'rows': timings, 'decode_total_seconds': sum(r['decode_seconds'] for r in timings),
        'recognize_total_seconds': sum(r['recognize_seconds'] for r in timings),
        'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'machine': platform.machine(), 'logical_cpus': os.cpu_count()},
        **_environment(), 'cold_first_sample_id': rows[0]['id'], 'warmup_calls': 0}
    files['runtime.json'] = encoded(runtime)
    for row in rows:
        path = Path(row['image_path'])
        require(digest(path) == row['image_sha256'], 'Source line image changed before publishing')
        files['images/'+row['id']+'.png'] = path.read_bytes()
    output.mkdir()
    for name, data in files.items():
        path = contained(output, name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    manifest = {'schema_version': 'ocr-baseline-manifest/2.0', 'config': config_pin,
                'files': [pin(output, name) for name in sorted(files)]}
    (output/'manifest.json').write_bytes(encoded(manifest))
    return report


def _predictions(rows, predictions, directory):
    require(isinstance(predictions, list) and len(predictions) == len(rows), 'Missing predictions')
    for record, row in zip(predictions, rows, strict=True):
        require(set(record) == {'id', 'source_id', 'reference', 'image', 'image_sha256', 'prediction', 'status', 'error', 'detections', 'order'}, 'Unexpected prediction fields')
        for key, value in {'id': row['id'], 'source_id': row['source_id'], 'reference': row['text'],
                           'image_sha256': row['image_sha256'], 'image': 'images/'+row['id']+'.png'}.items():
            require(record[key] == value, 'Prediction identity or target differs')
        require(digest(contained(directory, record['image'])) == row['image_sha256'], 'Source image differs')
        if record['status'] == 'error':
            require(isinstance(record['error'], dict) and set(record['error']) == {'stage', 'type', 'message'}
                and record['error']['stage'] in ('decode', 'recognize', 'combine') and all(isinstance(v, str) for v in record['error'].values())
                and record['detections'] == [] and record['order'] == [] and record['prediction'] == '', 'Invalid error record')
        else:
            combined = combine_detections_v1(record['detections'])
            require(record['error'] is None and same(record['order'], combined['order']) and record['prediction'] == combined['prediction']
                and record['status'] == ('ok' if combined['prediction'] else 'empty'), 'Detection assembly differs')


def _runtime(config, rows, runtime):
    require(set(runtime) == {'schema_version', 'started_utc', 'initialization_seconds', 'rows', 'decode_total_seconds',
        'recognize_total_seconds', 'hardware', 'python', 'packages', 'cold_first_sample_id', 'warmup_calls'}
        and runtime['schema_version'] == 'ocr-baseline-runtime/2.0', 'Unexpected runtime scope')
    require(same({key: runtime[key] for key in ('python', 'packages')}, config['expected_environment']), 'Runtime package evidence differs')
    require(set(runtime['hardware']) == {'platform', 'processor', 'machine', 'logical_cpus'} and
        all(isinstance(runtime['hardware'][key], str) for key in ('platform', 'processor', 'machine')) and
        (runtime['hardware']['logical_cpus'] is None or type(runtime['hardware']['logical_cpus']) is int and runtime['hardware']['logical_cpus'] > 0), 'Invalid hardware evidence')
    require(isinstance(runtime['rows'], list) and all(set(row) == {'id', 'decode_seconds', 'recognize_seconds', 'sample_wall_seconds'}
        for row in runtime['rows']) and [row['id'] for row in runtime['rows']] == [row['id'] for row in rows], 'Timing inventory differs')
    for value in [runtime['initialization_seconds'], runtime['decode_total_seconds'], runtime['recognize_total_seconds'],
                  *[row[key] for row in runtime['rows'] for key in ('decode_seconds', 'recognize_seconds', 'sample_wall_seconds')]]:
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid timing')
    require(all(row['sample_wall_seconds'] >= row['decode_seconds']+row['recognize_seconds'] for row in runtime['rows']), 'Timing parts exceed sample duration')
    require(runtime['decode_total_seconds'] == sum(row['decode_seconds'] for row in runtime['rows']) and
        runtime['recognize_total_seconds'] == sum(row['recognize_seconds'] for row in runtime['rows']) and
        type(runtime['warmup_calls']) is int and runtime['warmup_calls'] == 0 and runtime['cold_first_sample_id'] == rows[0]['id'], 'Timing totals or cold call changed')
    started = datetime.fromisoformat(runtime['started_utc'])
    require(started.tzinfo is not None and started >= datetime.fromisoformat(config['created_before_inference_utc']), 'Inference predates protocol')


def verify_baseline_v2(workspace, manifest_pin):
    """Recompute all diagnostics without OCR; require a trusted manifest pin."""
    root = Path(workspace).resolve(); manifest_path = checked_pin(root, manifest_pin); directory = manifest_path.parent
    manifest = read(manifest_path)
    require(set(manifest) == {'schema_version', 'config', 'files'} and manifest['schema_version'] == 'ocr-baseline-manifest/2.0', 'Wrong baseline manifest')
    config, rows, _, previous = _verify_inputs(root, manifest['config'])
    expected = {'predictions.json', 'metrics.json', 'report.json', 'runtime.json', 'index.html', 'strata.json', 'regression.json'}
    expected.update('images/'+row['id']+'.png' for row in rows)
    require(len(manifest['files']) == len(expected) and {p['path'] for p in manifest['files']} == expected, 'Baseline manifest inventory differs')
    require({p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()} == expected|{'manifest.json'}, 'Unlisted baseline files')
    for record in manifest['files']:
        checked_pin(directory, record)
    predictions = read(directory/'predictions.json'); _predictions(rows, predictions, directory)
    report = read(directory/'report.json'); _check_engine(root, config, report['engine'], report['effective_settings'])
    files, expected_report = _documents(manifest['config'], config, predictions, previous, report['engine'], report['effective_settings'])
    for name, data in files.items():
        require((directory/name).read_bytes() == data, 'Diagnostic output differs from saved predictions and frozen metadata')
    _runtime(config, rows, read(directory/'runtime.json'))
    _verify_inputs(root, manifest['config'])
    return expected_report
