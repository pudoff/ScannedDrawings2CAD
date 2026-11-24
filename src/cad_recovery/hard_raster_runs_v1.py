"""Frozen development runs, closed-source screening and replayable diagnostics.

Producer code/models remain unchanged. Cropped CAD rasters and a viewed photo
are distinct strata; none of these observations approve physical CAD targets.
"""
from collections import Counter
from datetime import datetime, timezone
from html import escape
import importlib.metadata
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

from PIL import Image

from .dataset_layout_v1 import checked, digest, encoded, inside, pin, read
from .family_catalog_v2 import require
from .hard_raster_metrics_v1 import match_texts, score_saved, valid_box
from .hard_raster_regression_v1 import context as pixel_context, png, validate_protocol
from .ocr_baseline_metrics_v1 import normalize_text_v1, score_text_v1
from .source_coallocation_v1 import package

CODE = ('src/cad_recovery/hard_raster_runs_v1.py', 'scripts/run_hard_raster_regression_v1.py',
        'scripts/hard_raster_recognition_child_v1.py', 'src/cad_recovery/updated_raster_batch.py',
        'src/cad_recovery/__main__.py')
QUALIFICATION_ATTEMPT = {'path': 'artifacts/hard-raster-runs-v1/manifest.json',
                        'sha256': '404950b6c141740529f62fec81653830a25a3b17fe4f544325b6c087e220599b'}
MEMORY_ATTEMPT = {'path': 'artifacts/hard-raster-runs-final-v1/manifest.json',
                  'sha256': '0d1b08ba2acaac4b1f39cd1fcaae7f5495aaa3cab0dff046c3aa37b7bce0271d'}
TREE_ATTEMPT = {'path': 'artifacts/hard-raster-runs-tree-v1/manifest.json',
                'sha256': '1a0508a9ca6f0bd3ad54f0fd5f6450e371ffed31c557471fa6d80537ee38bf27'}
PARENT = {'path': 'artifacts/hard-raster-regression-v1/manifest.json',
          'sha256': '91427860234d622b9a1dbfbc484bb0ced0caa5b4e98a66122fde85b147c0af70'}
REGISTRIES = ('artifacts/ocr-control-release-v2/task-splits.json',
              'artifacts/ocr-current-release-v2/task-splits.json')
FRESH = 'artifacts/ocr-fresh-holdout-v1/test.jsonl'
POLICY = {'order': 'case ID then profile ID; one fresh child per profile; recognition last',
          'profiles': ['cli_analyze_dimensions', 'cli_convert', 'worker_baseline', 'worker_geometry'],
          'new_training': False, 'parameter_search': False, 'closed_test_inference': False,
          'whole_large_sheets_executed': False, 'independent_evaluation': False,
          'physical_geometry_approved': False, 'training_eligible': False,
          'recognition': 'local PP-OCRv5 text_rec only; default preprocessing; no detector or classifier; one upright physical line per call',
          'runtime': 'whole child tree via Windows job plus OS parent-PID snapshot; conservative sum of observed per-PID OS lifetime working-set peaks; 50 ms polling',
          'failures': 'retain every assigned task; missing output is a scored deletion, never replace source',
          'partial_scope': 'assigned text lines only; no global precision or physical accuracy'}


def environment():
    return {n: importlib.metadata.version(n) for n in
            ('rapidocr', 'onnxruntime', 'numpy', 'pillow', 'opencv-python', 'ezdxf', 'pydantic')}


def screen(cases, groups, registries, fresh):
    """Metadata only. Never inspect closed target strings or execute their models."""
    membership = {p: g for g in groups for p in g['members']}
    closed_paths, closed_hashes = set(), set()
    for registry in registries:
        for a in registry['assignments']:
            if a['split'] == 'test':
                closed_paths.add(a['source_path']); closed_hashes.add(a['source_sha256'])
        for g in registry.get('group_commitments', []):
            if g.get('split') == 'test': closed_paths.update(g['member_paths'])
    for a in fresh:
        closed_paths.add(a['source_path']); closed_hashes.add(a['source_sha256'])
    closed_groups = {membership[p]['id'] for p in closed_paths if p in membership}
    result = []
    for c in cases:
        require(c['declared_split'] in ('train', 'validation'), 'No test inputs allowed')
        require(c['input'] in membership, 'Unknown case ancestry')
        group = membership[c['input']]
        reasons = []
        if c['input'] in closed_paths: reasons.append('closed_source_path')
        if c['source_sha256'] in closed_hashes: reasons.append('closed_source_bytes')
        if group['id'] in closed_groups: reasons.append('closed_group')
        if set(group['members']) & closed_paths: reasons.append('closed_group_member')
        historical = {}
        for i, registry in enumerate(registries):
            historical[str(i)] = dict(Counter(a['split'] for a in registry['assignments']
                if a['source_path'] in group['members']))
        result.append({'id': c['id'], 'group_id': group['id'], 'blocked': bool(reasons),
                       'reasons': reasons, 'historical_ocr_assignments': historical,
                       'independent_evaluation': False})
    return result


def load_inputs(root):
    _, manifest = package(root, PARENT, 'hard-raster-regression-manifest/1.0')
    _, refs, _, _, protocol, prospective = pixel_context(root, manifest['config'])
    base, _ = package(root, prospective['constraint_manifest'], 'role-admission-manifest/1.0')
    registries = [read(root/p) for p in REGISTRIES]
    fresh = [__import__('json').loads(s) for s in (root/FRESH).read_text('utf-8').splitlines() if s]
    boundary = screen(prospective['cases'], read(base/'groups.json')['groups'], registries, fresh)
    return refs, protocol, prospective, boundary


def validate_case(case):
    require(case['split'] in ('train', 'validation', 'viewed_development'), 'Invalid case split')
    require(case['physical_geometry_approved'] is False and case['training_eligible'] is False, 'No physical/training approval')
    size = case['size_px']
    require(len(size) == 2 and all(type(x) is int and x > 0 for x in size)
            and size[0]*size[1] <= 25_000_000, 'Case exceeds frozen pixel budget')
    ids = [r['id'] for r in case['texts']]
    require(len(ids) == len(set(ids)) and ids, 'Missing/duplicate fields')
    for row in case['texts']:
        require(valid_box(row['box'], size) and row['state'] in ('readable', 'partial', 'unknown'), 'Invalid field')
        require((row['state'] == 'readable' and isinstance(row['transcription'], str) and row['transcription'].strip())
                or (row['state'] != 'readable' and row['transcription'] is None), 'Unknown target must stay unknown')
        require(row['rotation_cw'] in (0, 90, 180, 270) and row['pixel_tolerance'] == 2, 'Changed orientation/tolerance')


def extract(root, case):
    with Image.open(checked(root, case['source'])) as original:
        # Fixed contexts were selected before inference; full large sheets are not processed.
        image = original.crop(case['source_box']).convert('RGB')
    require(list(image.size) == case['size_px'], 'Case dimensions changed')
    return image


def authenticate(root, config_pin):
    config = read(checked(root, config_pin))
    require(config['schema_version'] == 'hard-raster-runs-config/1.0' and config['task'] == 'P15.2b', 'Unknown experiment')
    require(config['policy'] == POLICY and config['code'] == [pin(root, p) for p in CODE], 'Changed run policy/code')
    require(config['environment'] == environment(), 'Changed inference environment')
    for record in config['inputs']: checked(root, record)
    annotations = read(checked(root, config['annotations']))
    require(config['cases'] == annotations['cases'], 'Changed reviewed annotations')
    refs, protocol, prospective, boundary = load_inputs(root)
    require(config['boundary'] == boundary, 'Closed boundary changed')
    validate_protocol(root, protocol)
    require(config['producer_protocol'] == protocol and config['parent'] == PARENT, 'Changed producer profiles')
    prior, _ = package(root, config['reuse_manifest'], 'hard-raster-runs-manifest/1.0')
    old = read(prior/'config.json')
    require(old['cases'] == config['cases'] and old['annotations'] == config['annotations']
            and old['producer_protocol'] == config['producer_protocol'] and old['environment'] == config['environment'],
            'Reuse changed pixels, labels, models or producer settings')
    require(config['rerun_tasks'] == [['jpeg', 'cli_analyze_dimensions']], 'Only missing JPEG interpreter memory is repeated')
    expected = ['jpeg'] + [x['id'] for c in prospective['cases'] for x in c['contexts']]
    require([c['id'] for c in config['cases']] == sorted(expected), 'Assigned contexts lost/replaced')
    require(config['source_ledger'] == prospective['cases'], 'Lost assigned source ledger')
    context_source = {x['id']: (s, x) for s in prospective['cases'] for x in s['contexts']}
    blocked = {r['id'] for r in boundary if r['blocked']}
    for case in config['cases']:
        validate_case(case)
        if case['id'] == 'jpeg':
            require(case['source'] == refs['source'] and case['texts'] == refs['texts']
                    and case['split'] == 'viewed_development' and case['source_box'] == [0, 0, *refs['source_size_px']], 'Viewed JPEG replaced')
        else:
            s, ctx = context_source[case['id']]
            require(s['id'] not in blocked and case['source_id'] == s['id']
                    and case['source_box'] == ctx['source_box'] and case['split'] == s['declared_split']
                    and case['source'] == {'path': 'dataset/'+s['input'], 'sha256': s['source_sha256']}, 'Unscreened/replaced case')
        extract(root, case)
    return config


def freeze(root, annotation_pin, output):
    refs, protocol, prospective, boundary = load_inputs(root)
    annotations = read(checked(root, annotation_pin))
    require(annotations['human_verified'] is False and annotations['training_eligible'] is False, 'Assistant review only')
    cases = annotations['cases']
    sources = [PARENT, annotation_pin, pin(root, FRESH), *[pin(root, p) for p in REGISTRIES],
               pin(root, 'artifacts/role-admission-labels-v1-final/groups.json'),
               pin(root, 'artifacts/ocr-first-training-test-once-v1.json')]
    # These manifests preserve byte identity without running their producers or loading targets for scoring.
    sources += [pin(root, 'artifacts/'+n+'/manifest.json') for n in
                ('ocr-control-release-v2', 'ocr-current-release-v2', 'ocr-fresh-holdout-v1', 'ocr-first-training-test-v1')]
    sources.append(QUALIFICATION_ATTEMPT)
    sources.append(MEMORY_ATTEMPT)
    sources.append(TREE_ATTEMPT)
    config = {'schema_version': 'hard-raster-runs-config/1.0', 'task': 'P15.2b',
              'created_before_inference_utc': datetime.now(timezone.utc).isoformat(),
              'parent': PARENT, 'policy': POLICY, 'producer_protocol': protocol,
              'code': [pin(root, p) for p in CODE], 'inputs': sources,
              'annotations': annotation_pin, 'environment': environment(), 'boundary': boundary,
              'reuse_manifest': TREE_ATTEMPT, 'rerun_tasks': [['jpeg', 'cli_analyze_dimensions']],
              'reuse_reason': '39 profile tasks and recognition have interpreter RSS; repeat only JPEG analysis missed during job assignment race. No target-dependent selection.',
              'previous_attempt': {'manifest': QUALIFICATION_ATTEMPT, 'status': 'invalid_measurement_harness',
                  'errors': ['CLI module did not invoke main', 'worker status file shared across requests'],
                  'rec_only_calls_already_executed': 31, 'labels_or_model_settings_changed': False},
              'memory_attempt': {'manifest': MEMORY_ATTEMPT, 'status': 'RSS_measured_launcher_only',
                  'text_observations_valid_but_runtime_budget_not_qualified': True,
                  'rec_only_calls_already_executed': 31, 'labels_or_model_settings_changed': False},
              'source_ledger': prospective['cases'], 'cases': cases}
    path = inside(root, output); require(not path.exists(), 'Choose a new config')
    path.write_bytes(encoded(config))
    record = pin(root, output); authenticate(root, record)
    return record


def process_peaks(process, guard):
    """Observe every PID in the guarded Windows job, including venv descendants.

    Summed lifetime peaks are a conservative upper bound for observed PIDs,
    not an exact simultaneous RSS. Short-lived unsampled PIDs remain a limit.
    """
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD),
                *[(n, ctypes.c_size_t) for n in ('peak', 'working', 'qpp', 'qp', 'qnp', 'qn', 'page', 'peakpage')]]
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel.QueryInformationJobObject.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        class Pids(ctypes.Structure):
            _fields_ = [('assigned', wintypes.DWORD), ('count', wintypes.DWORD), ('ids', ctypes.c_size_t*1024)]
        pids = Pids()
        if not kernel.QueryInformationJobObject(guard.handle, 3, ctypes.byref(pids), ctypes.sizeof(pids), None):
            raise ctypes.WinError(ctypes.get_last_error())
        require(pids.count < 1024, 'Process tree exceeds measurement bound')
        # A venv child can start before assignment of its launcher to the job.
        # Recover these already-running descendants by OS parent PID, not names.
        class Entry(ctypes.Structure):
            _fields_ = [('size', wintypes.DWORD), ('usage', wintypes.DWORD), ('pid', wintypes.DWORD),
                ('heap', ctypes.c_size_t), ('module', wintypes.DWORD), ('threads', wintypes.DWORD),
                ('parent', wintypes.DWORD), ('priority', wintypes.LONG), ('flags', wintypes.DWORD),
                ('name', wintypes.WCHAR*260)]
        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for method in ('Process32FirstW', 'Process32NextW'):
            getattr(kernel, method).argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
            getattr(kernel, method).restype = wintypes.BOOL
        snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
        if snapshot == wintypes.HANDLE(-1).value: raise ctypes.WinError(ctypes.get_last_error())
        parents = {}
        try:
            entry = Entry(); entry.size = ctypes.sizeof(entry)
            ok = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
            while ok:
                parents[int(entry.pid)] = int(entry.parent)
                ok = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        finally: kernel.CloseHandle(snapshot)
        descendants = {process.pid}
        while True:
            expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
            if expanded == descendants: break
            descendants = expanded
        all_pids = descendants | set(pids.ids[:pids.count])
        result = {}
        handles = [(process.pid, process._handle, False)]
        for pid in all_pids:
            if pid == process.pid: continue
            handle = kernel.OpenProcess(0x410, False, int(pid))
            if handle: handles.append((int(pid), handle, True))
        for pid, handle, close in handles:
            try:
                counters = Counters(); counters.cb = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(wintypes.HANDLE(handle), ctypes.byref(counters), counters.cb):
                    result[pid] = int(counters.peak)
            finally:
                if close: kernel.CloseHandle(handle)
        return result
    status = Path(f'/proc/{process.pid}/status')
    try:
        return {process.pid: int(next(s.split()[1] for s in status.read_text().splitlines() if s.startswith('VmHWM:')))*1024}
    except (FileNotFoundError, StopIteration): return {}


def execute(root, command, folder, seconds, memory_MiB):
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); peak = None; failure = None; pid_peaks = {}
    env = {**os.environ, 'XDG_CACHE_HOME': str(root/'artifacts/.cache'), 'PYTHONIOENCODING': 'utf-8'}
    with (folder/'stdout.txt').open('wb') as stdout, (folder/'stderr.txt').open('wb') as stderr:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        from .service_v1.process import ChildGuard
        guard = None
        try:
            guard = ChildGuard(process)
            while True:
                observed = process_peaks(process, guard)
                for pid, value in observed.items(): pid_peaks[pid] = max(pid_peaks.get(pid, 0), value)
                if pid_peaks: peak = sum(pid_peaks.values())
                finished = process.poll() is not None
                if peak is not None and peak > memory_MiB*1024**2: failure = 'memory_budget_exceeded'
                if time.perf_counter()-started > seconds: failure = failure or 'timeout'
                if failure or finished: break
                time.sleep(.05)
            if failure and process.poll() is None: process.kill()
            process.wait()
        finally:
            if guard: guard.close()
            if process.poll() is None: process.kill(); process.wait()
    return {'command': [str(x) for x in command], 'exit_code': process.returncode,
            'walltime_seconds': time.perf_counter()-started, 'peak_rss_MiB': peak/1024**2 if peak is not None else None,
            'failure': failure or ('process_failed' if process.returncode else None),
            'budget_seconds': seconds, 'budget_rss_MiB': memory_MiB,
            'memory_method': 'sum of observed per-PID OS lifetime working-set peaks in guarded process tree; conservative upper bound, 50 ms polling; not simultaneous RSS',
            'observed_process_count': len(pid_peaks),
            'memory_sampling_limit': 'Short-lived processes exiting between polls might be missed; OS parent-PID snapshots also cover descendants started before job assignment.',
            'budget_pass': failure is None and process.returncode == 0 and peak is not None
                and (os.name != 'nt' or len(pid_peaks) >= 2)}


def producer_command(root, source, result, profile, protocol, request):
    """Use the real CLI entry point and an isolated worker attempt directory."""
    if profile.startswith('cli_'):
        command = [sys.executable, '-m', 'cad_recovery', 'analyze-dimensions' if profile.endswith('dimensions') else 'convert',
            str(source), '--output', str(result), '--render-dpi', '300' if profile.endswith('dimensions') else '120']
        if profile.endswith('dimensions'): command += ['--ocr-models', str(root/'artifacts/models/ppocr-v5'), '--rotation-deg', '0']
        return command
    options = {k: protocol['profiles'][profile][k] for k in ('profile', 'render_dpi', 'image_dpi', 'rotation_deg')}
    request.parent.mkdir(parents=True, exist_ok=False)
    request.write_bytes(encoded({'source': str(source), 'input_sha256': digest(source), 'output': str(result), 'options': options}))
    return [sys.executable, str(root/'scripts/service_pipeline_v1.py'), '--request', str(request)]


def check_completion(runtime, result, profile, request):
    required = ['report.json', 'page-001-graph.json']
    if profile == 'cli_analyze_dimensions': required += ['page-001-ocr.json', 'page-001-geometry.json', 'page-001-dimension-associations.json']
    else: required.append('page-001.dxf')
    if profile == 'worker_geometry': required.append('page-001-geometry.json')
    missing = [n for n in required if not (result/n).is_file()]
    if profile.startswith('worker') and (not (request.parent/'status.json').is_file() or not read(request.parent/'status.json').get('ok')):
        missing.append('worker_success_status')
    if missing:
        runtime['missing_artifacts'] = missing
        runtime['failure'] = runtime['failure'] or 'missing_expected_artifacts'
        runtime['budget_pass'] = False
    return runtime


def text_scores(case, observation):
    lines = observation['lines'] if observation else []
    assigned, duplicates = match_texts(case['texts'], lines)
    lookup = {r['id']: r for r in lines}
    notation = {r['source_line_id']: r['parsed']['kind'] for r in observation['notations']} if observation else {}
    rows = []
    for field in case['texts']:
        if field['state'] != 'readable': continue
        line = lookup.get(assigned.get(field['id'])); prediction = line['raw'] if line else ''
        raw = score_text_v1(field['transcription'], prediction)
        norm = score_text_v1(normalize_text_v1(field['transcription']), normalize_text_v1(prediction))
        rows.append({'id': field['id'], 'reference': field['transcription'], 'prediction': prediction,
            'detected': line is not None, 'raw_exact': raw['exact_match'], 'normalized_exact': norm['exact_match'],
            'distance': raw['distance'], 'characters': raw['reference_characters'],
            'quantity_candidate_kind': notation.get(line['id']) if line else None,
            'reference_quantity_kind': field['quantity_kind']})
    for row in rows:
        kind = row['quantity_candidate_kind']
        row['quantity_candidate_kind'] = 'linear' if kind == 'nominal' else kind
        row['quantity_kind_correct'] = row['quantity_candidate_kind'] == row['reference_quantity_kind'] if row['reference_quantity_kind'] else None
    symbols = [{'field_id': r['id'], 'glyph': glyph,
                'recognized': normalize_text_v1(glyph) in normalize_text_v1(r['prediction'])}
               for r in rows for glyph in ('°', 'Ø', '□', '*') if glyph in r['reference']]
    return {'rows': rows, 'symbols': symbols, 'duplicates': duplicates, 'unscored_predictions': sorted(set(lookup)-set(assigned.values())-set(duplicates)),
            'global_precision': None, 'physical_accuracy': None}


def summarize(rows):
    chars = sum(r['characters'] for r in rows)
    return {'fields': len(rows), 'detected': sum(r.get('detected', False) for r in rows),
            'raw_exact': sum(r['raw_exact'] for r in rows), 'normalized_exact': sum(r['normalized_exact'] for r in rows),
            'distance': sum(r['distance'] for r in rows), 'characters': chars,
            'cer': sum(r['distance'] for r in rows)/chars if chars else None}


def collect(root, config, target):
    tasks = read(target/'tasks.json'); expected = [(c['id'], p) for c in config['cases'] for p in POLICY['profiles']]
    require([(t['case_id'], t['profile']) for t in tasks] == expected, 'Lost assigned task')
    cases = {c['id']: c for c in config['cases']}; scores = []
    import ezdxf
    for task in tasks:
        folder = target/'runs'/task['case_id']/task['profile']; result = folder/'result'
        runtime = read(folder/'process.json')
        observation = read(result/'page-001-ocr.json') if (result/'page-001-ocr.json').exists() and not runtime['failure'] else None
        score = text_scores(cases[task['case_id']], observation) if task['profile'] == 'cli_analyze_dimensions' else None
        graph = read(result/'page-001-graph.json') if (result/'page-001-graph.json').exists() else None
        geometry = read(result/'page-001-geometry.json') if (result/'page-001-geometry.json').exists() else None
        counts = dict(Counter(e.dxftype() for e in ezdxf.readfile(result/'page-001.dxf').modelspace())) if (result/'page-001.dxf').exists() else None
        row = {**task, 'runtime': runtime, 'text': score,
            'graph_counts': {k: len(graph[k]) for k in ('primitives', 'dimensions')} if graph else None,
            'dxf_modelspace_counts': counts, 'ocr_coverage': observation['coverage'] if observation else None,
            'geometry_candidates': len(geometry['candidates']) if geometry else None,
            'circle_search_incomplete': geometry['analysis']['hough_circle_limit_reached'] if geometry else None,
            'title_fields': observation['title_block']['fields'] if observation else None}
        scores.append(row)
    recfile = target/'recognition/result.json'
    rec_runtime = read(target/'recognition/process.json')
    rec = read(recfile) if recfile.exists() and not rec_runtime['failure'] else {'rows': []}
    expected_rec = {c['id']+'--'+f['id'] for c in config['cases'] for f in c['texts'] if f['state'] == 'readable'}
    require(len({r['id'] for r in rec['rows']}) == len(rec['rows'])
            and {r['id'] for r in rec['rows']}.issubset(expected_rec), 'Unexpected/duplicate recognition output')
    if not rec_runtime['failure']: require({r['id'] for r in rec['rows']} == expected_rec, 'Recognition lost assigned lines')
    recmap = {r['id']: r for r in rec['rows']}; recrows = []
    for case in config['cases']:
        for field in case['texts']:
            if field['state'] != 'readable': continue
            ident = case['id']+'--'+field['id']; item = recmap.get(ident)
            prediction = item['prediction'] if item else ''
            metric = score_text_v1(field['transcription'], prediction)
            normalized = score_text_v1(normalize_text_v1(field['transcription']), normalize_text_v1(prediction))
            recrows.append({'case_id': case['id'], 'id': field['id'], 'reference': field['transcription'],
                'prediction': prediction, 'status': item['status'] if item else 'process_failed',
                'raw_exact': metric['exact_match'], 'normalized_exact': normalized['exact_match'],
                'distance': metric['distance'], 'characters': metric['reference_characters']})
    strata = {}
    for stratum in ('viewed_photo', 'clean_CAD_context'):
        ids = {c['id'] for c in config['cases'] if c['stratum'] == stratum}
        strata[stratum] = {'case_count': len(ids),
            'det_rec': summarize([r for s in scores if s['case_id'] in ids and s['text'] for r in s['text']['rows']]),
            'rec_only': summarize([r for r in recrows if r['case_id'] in ids])}
    photo = target/'runs/jpeg/cli_analyze_dimensions/result'
    photo_metrics = None
    if (photo/'page-001-ocr.json').exists() and not next(t for t in scores if t['case_id'] == 'jpeg' and t['profile'] == 'cli_analyze_dimensions')['runtime']['failure']:
        ref = read(root/'configs/reviews/hard-raster-references-v1.json')
        observation = read(photo/'page-001-ocr.json')
        photo_metrics = score_saved(ref, observation, read(photo/'page-001-geometry.json'),
            read(photo/'page-001-arrows.json'), read(photo/'page-001-dimension-associations.json'), read(photo/'page-001-graph.json'), {})['summary']
        # CLI analysis has no export. DXF counts belong to separate convert/worker tasks.
        photo_metrics['dxf_modelspace_counts'] = None
    return {'schema_version': 'hard-raster-runs-report/1.0', 'tasks': scores, 'rec_only_rows': recrows,
        'strata': strata, 'boundary': config['boundary'], 'policy': POLICY,
        'recognition_runtime': rec_runtime, 'jpeg_pixel_metrics': photo_metrics,
        'manual_review_required': True, 'closed_ocr_tests_executed': False,
        'new_train_labels': 0, 'whole_large_sheets_executed': False,
        'physical_accuracy': None, 'family_independence_approved': False}


def run(root, config_pin, output):
    config = authenticate(root, config_pin); target = inside(root, output)
    require(target.parent == root/'artifacts' and not target.exists(), 'Choose new artifacts folder'); target.mkdir()
    (target/'config.json').write_bytes(encoded(config)); tasks = []; queries = []
    prior, _ = package(root, config['reuse_manifest'], 'hard-raster-runs-manifest/1.0')
    for case in config['cases']:
        image = extract(root, case); (target/'inputs').mkdir(exist_ok=True)
        source = target/'inputs'/(case['id']+'.png'); source.write_bytes(png(image))
        for field in case['texts']:
            if field['state'] != 'readable': continue
            q = image.crop(field['box'])
            if field['rotation_cw']: q = q.transpose({90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[field['rotation_cw']])
            ident = case['id']+'--'+field['id']; path = target/'queries'/(ident+'.png'); path.parent.mkdir(exist_ok=True); path.write_bytes(png(q))
            queries.append({'id': ident, 'image': str(path), 'sha256': digest(path)})
        for profile in POLICY['profiles']:
            folder = target/'runs'/case['id']/profile; result = folder/'result'
            if [case['id'], profile] not in config['rerun_tasks']:
                shutil.copytree(prior/'runs'/case['id']/profile, folder)
                tasks.append({'case_id': case['id'], 'profile': profile, 'execution_source_manifest': config['reuse_manifest']})
                continue
            request = target/'requests'/case['id']/profile/'request.json'
            cmd = producer_command(root, source, result, profile, config['producer_protocol'], request)
            budget = config['producer_protocol']['budgets']; seconds = budget['worker_timeout_seconds'] if profile.startswith('worker') else budget['analysis_walltime_ceiling_seconds']
            runtime = execute(root, cmd, folder, seconds, budget['cpu_rss_ceiling_MiB'])
            runtime = check_completion(runtime, result, profile, request)
            (folder/'process.json').write_bytes(encoded(runtime)); tasks.append({'case_id': case['id'], 'profile': profile})
            print(case['id'], profile, runtime['failure'] or 'ok', round(runtime['walltime_seconds'], 2), flush=True)
    (target/'tasks.json').write_bytes(encoded(tasks))
    shutil.copytree(prior/'recognition', target/'recognition')
    (target/'recognition-provenance.json').write_bytes(encoded({'execution_source_manifest': config['reuse_manifest'], 'new_recognition_calls': 0}))
    authenticate(root, config_pin)
    report = collect(root, config, target); (target/'report.json').write_bytes(encoded(report))
    cards = ['<!doctype html><meta charset="utf-8"><title>P15.2b development runs</title><h1>P15.2b</h1>',
             '<p>Viewed development only; no closed test or physical accuracy. <a href="report.json">Report</a></p>']
    for stratum, value in report['strata'].items(): cards.append('<h2>'+escape(stratum)+'</h2><pre>'+escape(str(value))+'</pre>')
    for row in report['rec_only_rows']:
        ident = row['case_id']+'--'+row['id']; cards.append('<p>'+escape(ident+' | '+row['reference']+' | rec-only: '+row['prediction'])+'</p><img style="max-width:100%;max-height:120px" src="queries/'+ident+'.png">')
    (target/'index.html').write_text(''.join(cards), encoding='utf-8')
    manifest = {'schema_version': 'hard-raster-runs-manifest/1.0', 'config': config_pin,
                'files': [pin(target, p.relative_to(target).as_posix()) for p in sorted(target.rglob('*')) if p.is_file()]}
    (target/'manifest.json').write_bytes(encoded(manifest)); return pin(root, output+'/manifest.json')


def verify(root, manifest_pin):
    base, manifest = package(root, manifest_pin, 'hard-raster-runs-manifest/1.0')
    require({p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}
            == {r['path'] for r in manifest['files']} | {'manifest.json'}, 'Changed run inventory')
    config = authenticate(root, manifest['config'])
    require(read(base/'config.json') == config, 'Changed copied config')
    prior, _ = package(root, config['reuse_manifest'], 'hard-raster-runs-manifest/1.0')
    reused = [prior/'recognition'] + [prior/'runs'/c['id']/p for c in config['cases'] for p in POLICY['profiles']
              if [c['id'], p] not in config['rerun_tasks']]
    for folder in reused:
        destination = base/folder.relative_to(prior)
        require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
                == {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}, 'Changed reused inventory')
        for source in folder.rglob('*'):
            if source.is_file(): require(digest(source) == digest(destination/source.relative_to(folder)), 'Changed reused observation')
    for case in config['cases']:
        image = extract(root, case)
        require((base/'inputs'/(case['id']+'.png')).read_bytes() == png(image), 'Changed context pixels')
        for field in case['texts']:
            if field['state'] != 'readable': continue
            q = image.crop(field['box'])
            if field['rotation_cw']: q = q.transpose({90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}[field['rotation_cw']])
            require((base/'queries'/(case['id']+'--'+field['id']+'.png')).read_bytes() == png(q), 'Changed query pixels')
    replay = collect(root, config, base)
    require(read(base/'report.json') == replay, 'Changed saved-observation score')
    return {'manifest': manifest_pin, 'tasks': len(replay['tasks']), 'strata': replay['strata'],
            'models_reexecuted': False, 'closed_ocr_tests_executed': False}
