"""Preregistered comparison of fixed input derivatives; no producer/model changes."""
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import struct
import sys
import zlib

import ezdxf
import numpy as np
from PIL import Image

from .bounded_preprocessing_metrics_v1 import (ARMS, aggregate, compare_candidate, derivative,
    derivative_box, original_frame, recognition_rows, scoped_text, variant_metadata)
from .dataset_layout_v1 import checked, digest, encoded, inside, pin, read
from .family_catalog_v2 import require
from .hard_raster_metrics_v1 import score_saved
from .hard_raster_regression_v1 import png
from .hard_raster_runs_v1 import (check_completion, environment, execute, producer_command, text_scores)
from .raster_quality_review_v1 import COMPARISON, turn, verify as verify_quality
from .source_coallocation_v1 import package

PARENT = {'path': 'artifacts/raster-quality-v1/manifest.json',
          'sha256': '4a9ee7e152b111478e25bad63321cc5e391cd68f5220b17c801886da9a27818d'}
BASELINE = {'path': 'artifacts/hard-raster-runs-final-v2/manifest.json',
            'sha256': '43c0c58a0de2751ff216182c2b6590afb17c17540869883bd1d9f4c5ce57b03b'}
CODE = ('src/cad_recovery/bounded_preprocessing_metrics_v1.py',
        'src/cad_recovery/bounded_preprocessing_v1.py', 'scripts/compare_bounded_preprocessing_v1.py', 'requirements.lock')
PROFILES = ('cli_analyze_dimensions', 'cli_convert', 'worker_baseline', 'worker_geometry')
POLICY = {'task': 'P15.2c.1', 'arms': list(ARMS), 'profiles': list(PROFILES),
    'order': 'arm then case ID then profile; recognition at end of each arm',
    'original': 'reuse sealed P15.2b observations, timings and pixels; no new baseline inference',
    'rec_only_input': 'crop the transformed full context; apply the same reviewed right-angle rotation',
    'refusal_recognition': 'a refused full context also refuses its rec-only fields; no oracle-crop fallback',
    'selection_scope': 'declared validation only; photo and train are descriptive',
    'tie_order': 'highest validation raw exact, then lower CER, then arm ID',
    'closed_test_inference': False, 'new_training': False, 'physical_geometry_approved': False,
    'deployment_changed': False, 'independent_evaluation': False}


def image_header_format(path):
    """Inspect container metadata without decoding huge rasters or changing Pillow guards."""
    with path.open('rb') as stream: header = stream.read(33)
    if header.startswith(b'\x89PNG\r\n\x1a\n'):
        require(len(header) == 33 and header[8:16] == b'\x00\x00\x00\rIHDR', 'Invalid PNG header')
        require(zlib.crc32(header[12:29]) & 0xffffffff == struct.unpack('>I', header[29:33])[0], 'PNG IHDR CRC mismatch')
        width, height = struct.unpack('>II', header[16:24]); require(width > 0 and height > 0, 'Empty PNG dimensions')
        return 'PNG'
    return 'JPEG' if header.startswith(b'\xff\xd8\xff') else 'other_or_invalid_header'


def source_screen(root):
    inventory_pin = pin(root, 'artifacts/dataset-layout-v1/inventory.json')
    inventory = read(checked(root, inventory_pin))['files']
    dataset = root/'dataset'
    actual = {p.relative_to(dataset).as_posix() for p in dataset.rglob('*') if p.is_file()}
    require(actual == {r['path'] for r in inventory}, 'Dataset inventory changed; freeze a new source review')
    # Header/hash inspection only, including test metadata; no test pixels/targets examined.
    formats = Counter(); image_hashes = {}
    for row in inventory:
        if Path(row['path']).suffix.lower() != '.png': continue
        path = checked(dataset, {'path': row['path'], 'sha256': row['sha256']})
        formats[image_header_format(path)] += 1
        image_hashes.setdefault(row['sha256'], []).append(row['path'])
    history_pin = pin(root, 'artifacts/dataset-history-v1/report.json')
    records = [r for r in read(checked(root, history_pin))['records'] if r['historical_path'].lower().endswith(('.jpg', '.jpeg'))]
    historical = [{'path': r['historical_path'], 'sha256': r['sha256'],
        'original_present_now': (dataset/r['historical_path']).is_file(),
        'current_identical_image_bytes': image_hashes.get(r['sha256'], []),
        'new_inference_allowed': False} for r in records]
    decoded = 'artifacts/links-aged-bolt-v4/page-001-source.png'
    old_report = 'artifacts/links-aged-bolt-v4/report.json'
    old = read(root/old_report)
    matched = [r['path'] for r in historical if r['sha256'] == old['source_sha256']]
    return {'schema_version': 'bounded-preprocessing-source-screen/1.0',
        'inventory': inventory_pin, 'history': history_pin, 'dataset_files': len(inventory),
        'dataset_extensions': dict(Counter(Path(n).suffix.lower() for n in actual)),
        'image_header_formats': dict(formats), 'historical_jpg': historical,
        'decoded_legacy_candidate': {'image': pin(root, decoded), 'old_report': pin(root, old_report),
            'historical_source_paths': matched, 'original_jpeg_sha256': old['source_sha256'],
            'status': 'deferred_original_pixels_and_current_closed_group_mapping_unresolved',
            'photo_authenticity': 'unknown_real_or_generated_degradation', 'new_inference_allowed': False},
        'excluded_reference_diagram': {'image': pin(root, 'Example.png'), 'status': 'manually_viewed_non_drawing_reference'},
        'added_confirmed_real_scans': 0, 'all_png_visually_classified_as_clean': False,
        'test_pixels_or_ocr_targets_used': False,
        'coverage_limit': 'PNG headers are not proof of a clean CAD export or a real scan; no additional confirmed real scans admitted.'}


def parents(root):
    verify_quality(root, PARENT)
    base, manifest = package(root, BASELINE, 'hard-raster-runs-manifest/1.0')
    config = read(checked(root, manifest['config']))
    return base, config


def freeze(root, output):
    _, baseline = parents(root)
    config = {'schema_version': 'bounded-preprocessing-config/1.0', 'task': 'P15.2c.1',
        'created_before_variant_inference_utc': datetime.now(timezone.utc).isoformat(),
        'parent': PARENT, 'baseline': BASELINE, 'contract': COMPARISON, 'policy': POLICY,
        'code': [pin(root, p) for p in CODE], 'environment': environment(),
        'cases': baseline['cases'], 'producer_protocol': baseline['producer_protocol'],
        'source_screen': source_screen(root), 'closed_ocr_tests_executed': False, 'training_started': False}
    path = inside(root, output); require(not path.exists(), 'Choose a new comparison config')
    path.write_bytes(encoded(config)); return pin(root, output)


def authenticate(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'bounded-preprocessing-config/1.0' and config['task'] == 'P15.2c.1', 'Unknown comparison config')
    require(config['parent'] == PARENT and config['baseline'] == BASELINE and config['contract'] == COMPARISON
            and config['policy'] == POLICY, 'Changed bounded comparison policy')
    require(config['code'] == [pin(root, p) for p in CODE] and config['environment'] == environment(), 'Comparison code/environment changed')
    require(config['closed_ocr_tests_executed'] is False and config['training_started'] is False, 'No closed tests or training here')
    require(config['source_screen'] == source_screen(root), 'Source-screen evidence changed')
    base, baseline = parents(root)
    require(config['cases'] == baseline['cases'] and config['producer_protocol'] == baseline['producer_protocol'], 'Changed cases/labels/producers')
    return config, base


def task_ledger(config):
    return [{'arm': arm, 'case_id': c['id'], 'profile': profile,
        'status': 'reused_baseline' if arm == 'original' else variant_metadata(c['size_px'], arm)['status']}
        for arm in ARMS for c in config['cases'] for profile in PROFILES]


def allowed_inventory(config):
    """The fixed single-page producers cannot publish training indices or extra runs."""
    names = {'config.json', 'source-screen.json', 'jpeg-references.json', 'tasks.json', 'report.json', 'index.html', 'manifest.json'}
    for arm in ARMS:
        if arm != 'original':
            names.add('recognition-'+arm+'-request.json')
            names.update('recognition/'+arm+'/'+n for n in ('process.json', 'result.json', 'stdout.txt', 'stderr.txt'))
        for case in config['cases']:
            cid = case['id']; names.add('derivatives/'+arm+'/'+cid+'.json')
            if variant_metadata(case['size_px'], arm)['status'] != 'ready': continue
            names.add('inputs/'+arm+'/'+cid+'.png')
            names.update('queries/'+arm+'/'+cid+'--'+f['id']+'.png' for f in case['texts'] if f['state'] == 'readable')
            if arm == 'original': continue
            for profile in PROFILES:
                directory = 'runs/'+arm+'/'+cid+'/'+profile+'/'
                names.update(directory+n for n in ('process.json', 'stdout.txt', 'stderr.txt'))
                files = ['report.json', 'page-001-source.png', 'page-001-report.json', 'page-001-graph.json']
                if profile != 'cli_analyze_dimensions': files.append('page-001.dxf')
                if profile in ('cli_analyze_dimensions', 'worker_geometry'):
                    files += ['page-001-'+n for n in ('processed.png', 'binary.png', 'preprocessing.json', 'regions.png',
                        'roi-content.png', 'roi-title-block-candidate.png', 'geometry.json', 'geometry-preview.png')]
                if profile == 'cli_analyze_dimensions':
                    files += ['page-001-'+n for n in ('ocr.json', 'ocr-preview.png', 'arrows.json', 'arrows-preview.png',
                        'dimension-associations.json', 'dimension-associations-preview.png')]
                names.update(directory+'result/'+n for n in files)
                if profile.startswith('worker'):
                    names.update('requests/'+arm+'/'+cid+'/'+profile+'/'+n for n in ('request.json', 'status.json'))
    return names


def arm_observations(config, target, baseline, arm):
    old = read(baseline/'report.json'); old_tasks = {(t['case_id'], t['profile']): t for t in old['tasks']}
    tasks = []
    for case in config['cases']:
        meta = variant_metadata(case['size_px'], arm)
        for profile in PROFILES:
            if arm == 'original':
                tasks.append({**old_tasks[(case['id'], profile)], 'status': 'reused_baseline', 'execution_source_manifest': BASELINE})
                continue
            result = target/'runs'/arm/case['id']/profile/'result'
            runtime = read(result.parent/'process.json') if meta['status'] == 'ready' else None
            failed = runtime is None or runtime['failure'] is not None
            observation = read(result/'page-001-ocr.json') if not failed and profile == 'cli_analyze_dimensions' else None
            mapped = original_frame(observation, meta)
            graph = read(result/'page-001-graph.json') if not failed else None
            geom = read(result/'page-001-geometry.json') if not failed and (result/'page-001-geometry.json').exists() else None
            counts = dict(Counter(e.dxftype() for e in ezdxf.readfile(result/'page-001.dxf').modelspace())) if not failed and profile != 'cli_analyze_dimensions' else None
            tasks.append({'case_id': case['id'], 'profile': profile,
                'status': meta['status'] if meta['status'] != 'ready' else ('process_failed' if failed else 'ok'),
                'runtime': runtime, 'text': text_scores(case, mapped) if profile == 'cli_analyze_dimensions' else None,
                'graph_counts': {k: len(graph[k]) for k in ('primitives', 'dimensions')} if graph else None,
                'dxf_modelspace_counts': counts, 'geometry_candidates': len(geom['candidates']) if geom else None,
                'circle_search_incomplete': geom['analysis']['hough_circle_limit_reached'] if geom else None,
                'ocr_coverage': observation['coverage'] if observation else None,
                'title_fields': observation['title_block']['fields'] if observation else None})
    if arm == 'original': rec, rec_runtime = old['rec_only_rows'], old['recognition_runtime']
    else:
        folder = target/'recognition'/arm; rec_runtime = read(folder/'process.json')
        outputs = read(folder/'result.json')['rows'] if rec_runtime['failure'] is None else []
        statuses = {c['id']: ('pixel_budget_refused' if variant_metadata(c['size_px'], arm)['status'] != 'ready'
                            else ('process_failed' if rec_runtime['failure'] else 'ok')) for c in config['cases']}
        rec = recognition_rows(config['cases'], outputs, statuses)
    strata = {}
    for stratum in sorted({c['stratum'] for c in config['cases']}):
        ids = {c['id'] for c in config['cases'] if c['stratum'] == stratum}
        strata[stratum] = {'det_rec': aggregate(scoped_text(tasks, ids)['rows']),
                          'rec_only': aggregate([r for r in rec if r['case_id'] in ids], recognition=True)}
    photo_metrics = None
    photo = next(t for t in tasks if t['case_id'] == 'jpeg' and t['profile'] == 'cli_analyze_dimensions')
    if arm == 'original': photo_metrics = old['jpeg_pixel_metrics']
    elif photo['status'] == 'ok':
        folder = target/'runs'/arm/'jpeg/cli_analyze_dimensions/result'
        ocr = original_frame(read(folder/'page-001-ocr.json'), variant_metadata([435, 600], arm))
        photo_metrics = score_saved(read(target/'jpeg-references.json'), ocr, read(folder/'page-001-geometry.json'),
            read(folder/'page-001-arrows.json'), read(folder/'page-001-dimension-associations.json'), read(folder/'page-001-graph.json'), {})['summary']
        photo_metrics['dxf_modelspace_counts'] = None
    return {'tasks': tasks, 'rec_only_rows': rec, 'recognition_runtime': rec_runtime, 'strata': strata, 'jpeg_pixel_metrics': photo_metrics}


def collect(config, target, baseline):
    require(read(target/'tasks.json') == task_ledger(config), 'Lost/replaced assigned comparison tasks')
    arms = {arm: arm_observations(config, target, baseline, arm) for arm in ARMS}
    validation = {c['id'] for c in config['cases'] if c['split'] == 'validation'}
    decisions = {}
    for arm in ARMS[1:]:
        part = [t for t in arms[arm]['tasks'] if t['case_id'] in validation]
        complete = all(t['status'] == 'ok' for t in part)
        budgets = all(t['runtime'] is not None and t['runtime']['budget_pass'] for t in part)
        budgets = budgets and arms[arm]['recognition_runtime']['budget_pass']
        decisions[arm] = compare_candidate(scoped_text(arms['original']['tasks'], validation),
            scoped_text(arms[arm]['tasks'], validation), complete, budgets)
    eligible = [arm for arm in ARMS[1:] if decisions[arm]['eligible_for_followup']]
    chosen = min(eligible, key=lambda arm: (-decisions[arm]['candidate']['raw_exact'], decisions[arm]['candidate']['cer'], arm)) if eligible else 'original'
    return {'schema_version': 'bounded-preprocessing-report/1.0', 'arms': arms, 'validation_decisions': decisions,
        'selected_for_followup': chosen, 'deployment_changed': False, 'policy': POLICY,
        'new_profile_processes': sum(t['status'] not in ('pixel_budget_refused', 'reused_baseline') for a in arms.values() for t in a['tasks']),
        'pixel_refused_profile_tasks': sum(t['status'] == 'pixel_budget_refused' for a in arms.values() for t in a['tasks']),
        'added_confirmed_real_scans': 0, 'full_P15_2c_complete': False,
        'closed_ocr_tests_executed': False, 'training_started': False, 'physical_accuracy': None}


def report_html(report):
    cards = ['<!doctype html><meta charset="utf-8"><title>P15.2c.1: bounded preprocessing</title>',
        '<style>body{font:16px system-ui;margin:2rem}pre{white-space:pre-wrap}img{max-width:100%;max-height:140px}section{border-top:1px solid #aaa}</style>',
        '<h1>Парное сравнение предобработки</h1><p><a href="report.json">Report</a> · <a href="source-screen.json">Source screen</a></p>',
        '<p>Кандидат для дальнейшей проверки: '+escape(report['selected_for_followup'])+'. Рабочий конвейер не изменён; физическая точность не утверждается.</p>']
    for arm, result in report['arms'].items():
        cards.append('<h2>'+escape(arm)+'</h2><pre>'+escape(str(result['strata']))+'</pre>')
    originals = {(r['case_id'], r['id']): r for r in report['arms']['original']['rec_only_rows']}
    for key, row in originals.items():
        ident = key[0]+'--'+key[1]; cards.append('<section><h3>'+escape(ident+' | '+row['reference'])+'</h3>')
        for arm, result in report['arms'].items():
            observed = next(r for r in result['rec_only_rows'] if (r['case_id'], r['id']) == key)
            cards.append('<p>'+escape(arm+' | rec-only '+observed['status']+' | '+repr(observed['prediction']))+'</p>')
            if observed['status'] != 'pixel_budget_refused': cards.append('<img src="queries/'+arm+'/'+ident+'.png">')
        cards.append('</section>')
    return ''.join(cards).encode('utf-8')


def run(root, config_pin, output):
    config, baseline = authenticate(root, config_pin)
    target = inside(root, output); require(target.parent == root/'artifacts' and not target.exists(), 'Choose new experiment output')
    target.mkdir(); (target/'config.json').write_bytes(encoded(config))
    (target/'source-screen.json').write_bytes(encoded(config['source_screen']))
    (target/'jpeg-references.json').write_bytes((root/'configs/reviews/hard-raster-references-v1.json').read_bytes())
    for arm in ARMS:
        queries = []
        for case in config['cases']:
            with Image.open(baseline/'inputs'/(case['id']+'.png')) as image: native = image.convert('RGB')
            array, meta = derivative(np.array(native), arm)
            meta['source'] = pin(root, (baseline/'inputs'/(case['id']+'.png')).relative_to(root).as_posix())
            metadata = target/'derivatives'/arm/(case['id']+'.json'); metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata.write_bytes(encoded(meta))
            if array is None: continue
            image = Image.fromarray(array); source = target/'inputs'/arm/(case['id']+'.png')
            source.parent.mkdir(parents=True, exist_ok=True); source.write_bytes(png(image))
            for field in case['texts']:
                if field['state'] != 'readable': continue
                ident = case['id']+'--'+field['id']; query = turn(image.crop(derivative_box(field['box'], arm)), field['rotation_cw'])
                path = target/'queries'/arm/(ident+'.png'); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(png(query))
                queries.append({'id': ident, 'image': str(path), 'sha256': digest(path)})
            if arm == 'original': continue
            for profile in PROFILES:
                folder = target/'runs'/arm/case['id']/profile; result = folder/'result'
                request = target/'requests'/arm/case['id']/profile/'request.json'
                command = producer_command(root, source, result, profile, config['producer_protocol'], request)
                seconds = COMPARISON['budgets']['worker_seconds' if profile.startswith('worker') else 'analysis_seconds']
                runtime = execute(root, command, folder, seconds, COMPARISON['budgets']['memory_MiB'])
                runtime = check_completion(runtime, result, profile, request)
                (folder/'process.json').write_bytes(encoded(runtime))
                print(arm, case['id'], profile, runtime['failure'] or 'ok', round(runtime['walltime_seconds'], 2), flush=True)
        if arm != 'original':
            request = target/('recognition-'+arm+'-request.json')
            request.write_bytes(encoded({'queries': queries, 'models': str(root/'artifacts/models/ppocr-v5'), 'output': str(target/'recognition'/arm/'result.json')}))
            runtime = execute(root, [sys.executable, str(root/'scripts/hard_raster_recognition_child_v1.py'), '--request', str(request)],
                target/'recognition'/arm, COMPARISON['budgets']['analysis_seconds'], COMPARISON['budgets']['memory_MiB'])
            (target/'recognition'/arm/'process.json').write_bytes(encoded(runtime))
    (target/'tasks.json').write_bytes(encoded(task_ledger(config)))
    authenticate(root, config_pin)
    report = collect(config, target, baseline); (target/'report.json').write_bytes(encoded(report))
    (target/'index.html').write_bytes(report_html(report))
    manifest = {'schema_version': 'bounded-preprocessing-manifest/1.0', 'config': config_pin,
        'files': [pin(target, p.relative_to(target).as_posix()) for p in sorted(target.rglob('*')) if p.is_file()]}
    (target/'manifest.json').write_bytes(encoded(manifest)); return pin(root, output+'/manifest.json')


def verify(root, manifest_pin):
    target, manifest = package(root, manifest_pin, 'bounded-preprocessing-manifest/1.0')
    require({p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
            == {r['path'] for r in manifest['files']}|{'manifest.json'}, 'Comparison inventory changed')
    config, baseline = authenticate(root, manifest['config'])
    require({p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}.issubset(allowed_inventory(config)),
            'Unexpected comparison artifact outside fixed producers')
    require(read(target/'config.json') == config and read(target/'source-screen.json') == config['source_screen'], 'Changed copied config/screen')
    require((target/'jpeg-references.json').read_bytes() == (root/'configs/reviews/hard-raster-references-v1.json').read_bytes(), 'Changed JPEG references')
    for arm in ARMS:
        for case in config['cases']:
            with Image.open(baseline/'inputs'/(case['id']+'.png')) as im: native = np.array(im.convert('RGB'))
            array, meta = derivative(native, arm)
            meta['source'] = pin(root, (baseline/'inputs'/(case['id']+'.png')).relative_to(root).as_posix())
            require(read(target/'derivatives'/arm/(case['id']+'.json')) == meta, 'Changed derivative provenance')
            source = target/'inputs'/arm/(case['id']+'.png')
            if array is None:
                require(not source.exists(), 'Refused context acquired replacement pixels'); continue
            image = Image.fromarray(array)
            require(source.read_bytes() == png(image), 'Derivative differs from frozen pixel operation')
            for field in case['texts']:
                if field['state'] != 'readable': continue
                expected = turn(image.crop(derivative_box(field['box'], arm)), field['rotation_cw'])
                require((target/'queries'/arm/(case['id']+'--'+field['id']+'.png')).read_bytes() == png(expected), 'Changed derivative query')
    replay = collect(config, target, baseline)
    require(read(target/'report.json') == replay and (target/'index.html').read_bytes() == report_html(replay), 'Changed paired metrics/decision/report')
    return {'manifest': manifest_pin, 'selected_for_followup': replay['selected_for_followup'],
        'new_profile_processes': replay['new_profile_processes'], 'pixel_refused_profile_tasks': replay['pixel_refused_profile_tasks'],
        'models_reexecuted': False, 'closed_ocr_tests_executed': False}
