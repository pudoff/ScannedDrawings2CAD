"""Fixed existing OCR on the current directory control release, without tuning."""
from copy import deepcopy
from html import escape
import os
from pathlib import Path
import platform
import time

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .family_catalog_v2 import require, stable_digest
from .ocr_baseline_v1 import _make_engine, _effective_engine_settings
from .ocr_baseline_v2 import _environment, _check_engine, _predictions
from .ocr_baseline_v3 import (_required_pins as older_pins, PROTOCOL as OLDER_PROTOCOL,
                              _one, _memory, _runtime as older_runtime, now, timestamp)
from .ocr_baseline_metrics_v4 import build_metadata_v4, score_control_v4, SPLITS
from .ocr_current_release_v1 import verify_current_ocr, CODE as RELEASE_CODE
from .dataset_preparation_v1 import CODE as PREPARATION_CODE
from .ocr_release_v3 import _package

PARENT = {'path': 'artifacts/current-layout-final-v1.json', 'sha256': '718397808b34704e635c60065f65e2dabe8a0f23066c024120580a840f84d51c'}
RELEASE = {'path': 'artifacts/ocr-current-release-v1/manifest.json', 'sha256': 'faae846b483d94e5a6e4ab1c8cc77270f72598334ce64fae35e695e556f7928a'}
PREVIOUS = {'path': 'artifacts/ocr-baseline-v3/manifest.json', 'sha256': '64a98d68bf23bca0d1e93825ec5f0cccc4c22037c06da98b3678c04a609f9fae'}
CODE = ['src/cad_recovery/ocr_baseline_v4.py', 'src/cad_recovery/ocr_baseline_metrics_v4.py', 'scripts/evaluate_ocr_baseline_v4.py']
PROTOCOL = {**deepcopy(OLDER_PROTOCOL), 'scope': 'initial_current_directory_title_control', 'sample_count': 35,
            'split_counts': {'validation': 16, 'test': 19},
            'coverage': '32 selected field slots, 2 unresolved; 165 pending sources; 10 identifier-placeholder name cells',
            'strata': 'split, field role, literal kind, source, group, physical-line context and exact full-field diagnostic',
            'full_field': 'all physical lines must match; unresolved slots have no target',
            'training': False, 'parameter_selection': False}


def same(a, b): return stable_digest(a) == stable_digest(b)


def _required_pins(root):
    values = {p['path']: p for p in older_pins(root)}
    for name in sorted(set(CODE+RELEASE_CODE+PREPARATION_CODE)):
        values[name] = pin(root, name)
    return [values[name] for name in sorted(values)]


def _inventory(root, rows):
    return [{**{k: r[k] for k in ('id', 'source_id', 'source_sha256', 'group_id', 'image_sha256', 'text', 'split')},
             'image_path': Path(r['image_path']).relative_to(root).as_posix()} for r in rows]


def _bundle(root):
    parent = read(checked(root, PARENT))
    require(RELEASE in parent['manifests'], 'Current OCR parent differs')
    report, all_rows, base = verify_current_ocr(root, RELEASE)
    require(report['task_split_counts'] == {'train': 121, 'validation': 16, 'test': 19}, 'Current release counts changed')
    rows = [{**r, 'image_path': str(base/r['image'])} for split in SPLITS
            for r in sorted(all_rows, key=lambda r: r['id']) if r['split'] == split]
    metadata = build_metadata_v4(read(base/'labels.json'), read(base/'task-splits.json'), read(base/'source-dispositions.json'), rows)
    require(metadata['field_slot_count'] == 32 and metadata['decision_counts'] == {'accepted_for_ocr_transcription': 30, 'unresolved': 2}, 'Changed control coverage')
    previous, _ = _package(root, PREVIOUS)
    old_report, old_runtime = read(previous.parent/'report.json'), read(previous.parent/'runtime.json')
    return rows, metadata, old_report, {k: old_runtime[k] for k in ('python', 'packages')}


def _order(rows):
    require(len(rows) == PROTOCOL['sample_count'] and {s: sum(r['split'] == s for r in rows) for s in SPLITS} == PROTOCOL['split_counts'], 'Incomplete baseline inventory')
    require([r['id'] for r in rows] == [r['id'] for r in sorted(rows, key=lambda r: (SPLITS.index(r['split']), r['id']))], 'Changed validation/test order')


def prepare_config_v4(workspace, output):
    root = Path(workspace).resolve(); path = inside(root, output)
    require(path.parent == root/'configs/reviews' and not path.exists(), 'Choose a new experiment config')
    rows, metadata, previous, environment = _bundle(root); _order(rows)
    require(same(environment, _environment()), 'Environment changed from previous baseline')
    config = {'schema_version': 'ocr-baseline-config/4.0', 'task': 'P16.1d', 'created_before_inference_utc': now(),
              'parent_final': PARENT, 'release': RELEASE, 'previous_baseline': PREVIOUS, 'protocol': PROTOCOL,
              'model_dir': 'artifacts/models/ppocr-v5', 'pins': _required_pins(root), 'inventory': _inventory(root, rows),
              'metadata': metadata, 'expected_engine': previous['engine'], 'expected_settings': previous['effective_settings'],
              'expected_environment': environment}
    path.write_bytes(encoded(config)); return pin(root, output)


def _verify_inputs(root, config_pin):
    config = read(checked(root, config_pin))
    require(set(config) == {'schema_version', 'task', 'created_before_inference_utc', 'parent_final', 'release', 'previous_baseline',
                           'protocol', 'model_dir', 'pins', 'inventory', 'metadata', 'expected_engine', 'expected_settings', 'expected_environment'}, 'Unexpected experiment config')
    require(config['schema_version'] == 'ocr-baseline-config/4.0' and config['task'] == 'P16.1d'
            and config['parent_final'] == PARENT and config['release'] == RELEASE and config['previous_baseline'] == PREVIOUS
            and config['model_dir'] == 'artifacts/models/ppocr-v5' and same(config['protocol'], PROTOCOL), 'Changed experiment scope')
    timestamp(config['created_before_inference_utc'])
    require(config['pins'] == _required_pins(root), 'Changed code, models or defaults')
    rows, metadata, previous, environment = _bundle(root); _order(rows)
    require(same(config['inventory'], _inventory(root, rows)) and same(config['metadata'], metadata), 'Changed control targets or metadata')
    require(same(config['expected_engine'], previous['engine']) and same(config['expected_settings'], previous['effective_settings'])
            and same(config['expected_environment'], environment) and same(_environment(), environment), 'Changed previous engine settings or environment')
    return config, rows, inside(root, config['model_dir'])


def _gate(root, config_pin, config, engine):
    checked(root, config_pin)
    require(config['pins'] == _required_pins(root) and same(_environment(), config['expected_environment']), 'Inputs changed before test')
    for row in config['inventory']:
        checked(root, {'path': row['image_path'], 'sha256': row['image_sha256']})
    settings = _effective_engine_settings(engine)
    _check_engine(root, config, engine.provenance, settings)
    return {'checked_utc': now(), 'config': config_pin, 'settings_content_sha256': stable_digest(settings),
            'validation_sample_count': config['protocol']['split_counts']['validation'], 'configuration_changed': False}


def _documents(config_pin, config, predictions, provenance, settings):
    metrics = score_control_v4(predictions, config['metadata'])
    report = {'schema_version': 'ocr-baseline-report/4.0', 'task': 'P16.1d', 'config': config_pin,
              'by_split': {split: {key: value for key, value in part.items()} for split, part in metrics['by_split'].items()},
              'coverage_by_split': metrics['coverage_by_split'], 'coverage_by_source': metrics['coverage_by_source'],
              'duplicates': metrics['duplicates'], 'combined_validation_test_score': None,
              'engine': provenance, 'effective_settings': settings, 'training_started': False,
              'parameter_selection': False, 'independent_evaluation': False,
              'limitations': ['Only the initial fixed title-control subset is scored; 165 sources and two uncertain fields are unscored coverage.',
                              'Validation contains identifiers and repeated identifier placeholders, not real names or dimensions.',
                              'Primary metrics retain all physical lines; duplicates and full fields have separate diagnostics.',
                              'Conservative groups are not approved semantic families; global independence is unproven.',
                              'Full det/cls/rec on line crops, not whole-sheet detection, vector PDF or CAD accuracy.',
                              'Validation starts cold, test uses the same warm engine; no settings or weights change.',
                              'Peak process RAM includes release verification, not isolated model or GPU memory.',
                              'Evidence replay performs no OCR and relies on trusted manifests for execution provenance.']}
    strata = {r['id']: r for r in config['metadata']['strata']}
    page = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR: текущий baseline</title>',
            '<style>body{font:17px system-ui;max-width:1200px;margin:auto}img{max-width:100%}section{margin:2em 0}td,th{padding:.4em;border:1px solid #aaa}table{border-collapse:collapse}</style>',
            '<h1>OCR: актуальные validation/test</h1><p>Первый фиксированный выпуск. Validation содержит идентификаторы и их повторы в поле имени. Обучения и подбора нет; независимость семейств не подтверждена.</p>',
            '<p><a href="report.json">Отчёт</a> · <a href="metrics.json">Метрики</a> · <a href="metadata.json">Полнота и роли</a> · <a href="runtime.json">Время и RAM</a></p>',
            '<table><tr><th>Split</th><th>Строк</th><th>Raw exact</th><th>Raw CER</th><th>Normalized exact</th><th>Normalized CER</th></tr>']
    for split, part in metrics['by_split'].items():
        raw, norm = part['overall']['raw'], part['overall']['normalized']
        page.append(f'<tr><td>{split}</td><td>{raw["sample_count"]}</td><td>{raw["exact_matches"]}</td><td>{raw["distance"]}/{raw["reference_characters"]} = {raw["cer"]:.2%}</td><td>{norm["exact_matches"]}</td><td>{norm["cer"]:.2%}</td></tr>')
    page.append('</table><label>Выборка <select id="split" onchange="filterRows()"><option value="">Все</option><option>validation</option><option>test</option></select></label> <label>Тип текста <select id="kind" onchange="filterRows()"><option value="">Все</option><option>drawing_id</option><option>name_identifier_placeholder</option><option>name_literal</option></select></label>')
    for row in predictions:
        kind = strata[row['id']]['text_kind']
        page.append('<section data-split="'+escape(row['split'], quote=True)+'" data-kind="'+escape(kind, quote=True)+'"><h2>'+escape(row['id'])+'</h2><p>'+escape(row['split']+' / '+kind)+'</p><img loading="lazy" src="'+escape(row['image'], quote=True)+'"><pre>Эталон: '+escape(row['reference'])+'\nOCR: '+escape(row['prediction'])+'\nСтатус: '+escape(row['status'])+'</pre></section>')
    page.append('<script>function filterRows(){const s=document.getElementById("split").value,k=document.getElementById("kind").value;document.querySelectorAll("section[data-split]").forEach(e=>e.hidden=!!((s&&e.dataset.split!==s)||(k&&e.dataset.kind!==k)));}</script></html>')
    return {'predictions.json': encoded(predictions), 'metadata.json': encoded(config['metadata']), 'metrics.json': encoded(metrics),
            'report.json': encoded(report), 'index.html': ''.join(page).encode('utf-8')}, report


def _runtime(config, rows, runtime):
    require(runtime['schema_version'] == 'ocr-baseline-runtime/4.0', 'Unknown current runtime')
    projected = {**runtime, 'schema_version': 'ocr-baseline-runtime/3.0'}
    older_runtime(config, rows, projected)


def run_baseline_v4(workspace, config_pin, output):
    root = Path(workspace).resolve(); target = inside(root, output)
    require(target.parent == root/'artifacts' and not target.exists(), 'Choose a new direct artifacts directory')
    config, rows, model_dir = _verify_inputs(root, config_pin)
    started = now(); require(timestamp(started) >= timestamp(config['created_before_inference_utc']), 'Inference predates config')
    memory = {'before_engine': _memory()}; tick = time.perf_counter(); engine = _make_engine(model_dir); init = time.perf_counter()-tick
    provenance, settings = engine.provenance, _effective_engine_settings(engine)
    _check_engine(root, config, provenance, settings); memory['after_engine'] = _memory()
    predictions, timings, phases = [], [], {}; gate = None
    for split in SPLITS:
        if split == 'test': gate = _gate(root, config_pin, config, engine)
        phase_rows = [r for r in rows if r['split'] == split]; begin = now()
        for row in phase_rows:
            prediction, timing = _one(engine, row); predictions.append(prediction); timings.append(timing)
        phases[split] = {'started_utc': begin, 'finished_utc': now(), 'sample_ids': [r['id'] for r in phase_rows]}
        memory['after_'+split] = _memory()
    _check_engine(root, config, engine.provenance, _effective_engine_settings(engine)); finished = now()
    _verify_inputs(root, config_pin)
    files, report = _documents(config_pin, config, predictions, provenance, settings)
    runtime = {'schema_version': 'ocr-baseline-runtime/4.0', 'started_utc': started, 'finished_utc': finished,
               'initialization_seconds': init, 'rows': timings, 'decode_total_seconds': sum(r['decode_seconds'] for r in timings),
               'recognize_total_seconds': sum(r['recognize_seconds'] for r in timings), 'phases': phases, 'test_gate': gate, 'memory': memory,
               'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'machine': platform.machine(), 'logical_cpus': os.cpu_count()},
               **_environment(), 'cold_first_sample_id': rows[0]['id'], 'warmup_calls': 0}
    _runtime(config, rows, runtime); files['runtime.json'] = encoded(runtime)
    for row in rows:
        path = checked(root, {'path': Path(row['image_path']).relative_to(root).as_posix(), 'sha256': row['image_sha256']})
        files['images/'+row['id']+'.png'] = path.read_bytes()
    target.mkdir()
    for name, data in files.items():
        path = inside(target, name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    (target/'manifest.json').write_bytes(encoded({'schema_version': 'ocr-baseline-manifest/4.0', 'config': config_pin,
                                               'files': [pin(target, n) for n in sorted(files)]}))
    return report


def verify_baseline_v4(workspace, manifest_pin):
    root = Path(workspace).resolve(); path, manifest = _package(root, manifest_pin); base = path.parent
    require(set(manifest) == {'schema_version', 'config', 'files'} and manifest['schema_version'] == 'ocr-baseline-manifest/4.0', 'Unknown baseline package')
    config, rows, _ = _verify_inputs(root, manifest['config']); predictions = read(base/'predictions.json')
    require(isinstance(predictions, list) and len(predictions) == len(rows)
            and all(r['split'] == p['split'] for r, p in zip(rows, predictions, strict=True)), 'Prediction inventory changed')
    _predictions(rows, [{k: v for k, v in r.items() if k != 'split'} for r in predictions], base)
    report = read(base/'report.json'); _check_engine(root, config, report['engine'], report['effective_settings'])
    files, expected = _documents(manifest['config'], config, predictions, report['engine'], report['effective_settings'])
    runtime = read(base/'runtime.json'); _runtime(config, rows, runtime)
    require(runtime['test_gate']['config'] == manifest['config'], 'Test gate used another config')
    files['runtime.json'] = encoded(runtime)
    for row in rows: files['images/'+row['id']+'.png'] = Path(row['image_path']).read_bytes()
    require(manifest['files'] == [pin(base, n) for n in sorted(files)]
            and all(inside(base, n).read_bytes() == value for n, value in files.items()), 'Saved baseline differs from evidence')
    _verify_inputs(root, manifest['config']); return expected
