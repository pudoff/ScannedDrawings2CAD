"""Freeze/run bounded OCR comparison workers; evaluation is a separate step."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .dataset_layout_v1 import checked, encoded, inside, pin, read
from .architecture_data_readiness_v1 import Boundary, Package
from .hard_raster_runs_v1 import process_peaks
from .service_v1.process import ChildGuard
from .ocr_model_comparison_protocol_v1 import (
    ADAPTER, ARMS, BACKEND, POLICY, require, validate_request, verify,
)

PREVIOUS = {'path': 'artifacts/ocr-model-comparison-protocol-release-v1.json',
            'sha256': '657aa8ea36e62dbb5e097c8d3f2d36012247ef2adaed7d3eea7a474849b80e39'}
PROTOCOL = {'path': 'configs/reviews/ocr-model-comparison-v1.json',
            'sha256': '5b3bedc77a7182a25c53ecb66ee2e1d179dbb83fc2bfd0ee14e0e567ad14743f'}
CODE = ['src/cad_recovery/ocr_model_comparison_runtime_v1.py',
        'src/cad_recovery/ocr_model_comparison_execution_v1.py',
        'src/cad_recovery/ocr_model_comparison_evaluation_v1.py',
        'scripts/ocr_model_comparison_worker_v1.py', 'scripts/run_ocr_model_comparison_v1.py',
        'src/cad_recovery/ocr_engine.py', 'src/cad_recovery/ocr_baseline_metrics_v1.py',
        'src/cad_recovery/angular_text_components_v1.py', 'src/cad_recovery/hard_raster_runs_v1.py',
        'src/cad_recovery/service_v1/process.py', 'docs/ocr-model-comparison-execution-v1.md']


def preserve(root):
    value = read(checked(root, PREVIOUS))
    pins = value['prior_protected_pins'] | value['files'] | {PREVIOUS['path']: PREVIOUS['sha256']}
    for name, sha in pins.items():
        checked(root, {'path': name, 'sha256': sha})
    return pins


def regression_inputs(root, folder):
    """Four prior contexts/seven components, ancestry checked before pixels."""
    adapter = read(checked(root, ADAPTER))
    parent = Package(root, adapter['parents']['readiness'])
    metadata = read(checked(root, parent.manifest['config']))['metadata']
    groups, sources, one, two, fresh = [read(checked(root, p)) for p in metadata]
    boundary = Boundary(groups['groups'], sources, [one, two],
        [{'source_path': s['input'], 'source_sha256': s['input_sha256']} for s in fresh['selected_sources']])
    release = Package(root, {'path': 'artifacts/angular-components-inputs-v1/manifest.json',
                            'sha256': '9b1080084e4cdbae3275ed8e0273b91b1f1f1469293d159d03a2a0ec372b603b'})
    cases = release.json('cases.json')
    from .angular_text_components_v1 import validate_case
    require(len(cases) == 4 and sum(len(c['components']) for c in cases) == 7,
            'Changed component regression scope')
    for case in cases:
        validate_case(case)
        if case['split'] == 'train':
            require(not boundary.check(case['source'], 'train', case['id'])['blocked'], 'Closed regression source')
        else:
            require(case['split'] == 'development' and case['source']['sha256'] ==
                    'd3baaca90795c9ccb85e3bcfa71245bb2182996c74132bc410e3ce2653e4ca14', 'Unknown viewed development')
    # All four contexts are screened before the first image read/copy.
    jobs, links = [], []
    def emit(record, size, case_id, component_id):
        identifier = f'sample-{314+len(jobs):04d}'
        path = folder/'images'/(identifier+'.png')
        shutil.copyfile(checked(root, record), path)
        request = {'id': identifier, 'image': pin(root, path.relative_to(root).as_posix()), 'size_px': size}
        validate_request(request)
        jobs.append(request)
        links.append({'id': identifier, 'context_id': case_id, 'component_id': component_id})
    for case in cases:
        emit(case['image'], case['size_px'], case['id'], None)
        for component in case['components']:
            box = component['box_px']
            emit(component['image'], [box[2]-box[0], box[3]-box[1]], case['id'], component['id'])
    return jobs, links, cases, release.record


def freeze(root, destination, folder):
    root, destination, folder = root.resolve(), destination.resolve(), folder.resolve()
    require(destination.is_relative_to(root/'configs') and folder.is_relative_to(root/'artifacts') and
            not destination.exists() and not folder.exists(), 'Choose new workspace execution paths')
    previous = preserve(root)
    verify(root, PROTOCOL)
    protocol = read(checked(root, PROTOCOL))
    folder.mkdir(parents=True)
    (folder/'images').mkdir()
    extra, links, cases, component_release = regression_inputs(root, folder)
    requests = read(checked(root, protocol['data']['requests']))
    assignments = deepcopy(protocol['assignments'])
    for link in links:
        for arm in ('R0', 'R1') if link['component_id'] else ('D0', 'D1'):
            assignments[arm].append(link['id'])
    require({arm: len(ids) for arm, ids in assignments.items()} == {'R0': 320, 'R1': 320, 'D0': 173, 'D1': 173},
            'Unexpected execution assignment count')
    for name, payload in [('requests.json', requests+extra), ('regression-links.json', links), ('regression-cases.json', cases)]:
        (folder/name).write_bytes(encoded(payload))
    qualification = read(checked(root, protocol['data']['model-qualification']))
    contract = {'schema_version': 'ocr-model-comparison-worker/1.0', 'arms': ARMS, 'backend': BACKEND,
        'models': {name: value['model'] for name, value in qualification['models'].items()},
        'runtime_limits': {k: POLICY[k] for k in ('limits', 'precision', 'intra_op_threads', 'inter_op_threads', 'warm_repeats_per_case')},
        'code': [pin(root, path) for path in CODE]}
    (folder/'worker-contract.json').write_bytes(encoded(contract))
    value = {'schema_version': 'ocr-model-comparison-execution/1.0', 'previous': PREVIOUS,
        'protocol': PROTOCOL, 'previous_pins': previous, 'code': contract['code'],
        'worker_contract': pin(root, (folder/'worker-contract.json').relative_to(root).as_posix()),
        'data': {name: pin(root, (folder/(name+'.json')).relative_to(root).as_posix())
                 for name in ('requests', 'regression-links', 'regression-cases')},
        'component_release': component_release, 'assignments': assignments,
        'regression_scope': '4 prior viewed contexts/7 given components, 3 known train role edges; oracle crop regression only',
        'new_quality_calls_before_freeze': 0, 'independent_evaluation': False,
        'physical_geometry_approved': False, 'production_admitted': False}
    destination.write_bytes(encoded(value))
    return pin(root, destination.relative_to(root).as_posix())


def load(root, record):
    config = read(checked(root, record))
    require(config['schema_version'] == 'ocr-model-comparison-execution/1.0' and
            config['protocol'] == PROTOCOL and config['previous'] == PREVIOUS and
            config['code'] == [pin(root, p) for p in CODE], 'Execution contract/code changed')
    for value in [config['worker_contract'], *config['data'].values()]:
        checked(root, value)
    contract = read(checked(root, config['worker_contract']))
    require(contract['code'] == config['code'], 'Worker implementation differs')
    for model in contract['models'].values():
        checked(root, model)
    requests = read(checked(root, config['data']['requests']))
    for request in requests:
        validate_request(request)
        checked(root, request['image'])
    require(len(requests) == len({r['id'] for r in requests}) == 324, 'Lost execution requests')
    require({a: len(ids) for a, ids in config['assignments'].items()} == {'R0': 320, 'R1': 320, 'D0': 173, 'D1': 173}, 'Lost arm assignments')
    return config, contract, requests


def execute_worker(root, config, job, result_folder, *, qualification=False):
    """Hard process-tree memory/time limits plus init/per-case progress deadlines."""
    result_folder.mkdir(parents=True, exist_ok=False)
    request_path = result_folder/'request.json'
    request_path.write_bytes(encoded(job))
    request = pin(root, request_path.relative_to(root).as_posix())
    contract = config['worker_contract']
    command = [sys.executable, str(root/'scripts/ocr_model_comparison_worker_v1.py'),
        '--contract', contract['path'], '--contract-sha256', contract['sha256'],
        '--request', request['path'], '--request-sha256', request['sha256']]
    if qualification:
        command.append('--qualify')
    started = time.perf_counter()
    peaks, failure = {}, None
    limits = POLICY['limits']
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'XDG_CACHE_HOME': str(root/'artifacts/.cache'),
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    with (result_folder/'stdout.txt').open('wb') as stdout, (result_folder/'stderr.txt').open('wb') as stderr:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        guard = None
        try:
            guard = ChildGuard(process)
            while True:
                for pid, memory in process_peaks(process, guard).items():
                    peaks[pid] = max(peaks.get(pid, 0), memory)
                elapsed = time.perf_counter()-started
                if sum(peaks.values()) > limits['worker_memory_MiB']*1024**2:
                    failure = 'process_tree_memory_exceeded'
                if elapsed > limits['worker_seconds']:
                    failure = failure or 'worker_timeout'
                progress_path = root/job['output']/'progress.json'
                if progress_path.exists():
                    state = read(progress_path)
                    allowed = limits['cold_init_seconds'] if state['phase'] == 'init' else limits['case_seconds']
                    if state['phase'] != 'done' and time.perf_counter()-state['started'] > allowed:
                        failure = failure or state['phase']+'_timeout'
                elif elapsed > limits['cold_init_seconds']:
                    failure = failure or 'startup_timeout'
                if failure or process.poll() is not None:
                    break
                time.sleep(.05)
            if failure and process.poll() is None:
                process.kill()
            process.wait()
        finally:
            if guard:
                guard.close()
            if process.poll() is None:
                process.kill()
                process.wait()
    result = {'exit_code': process.returncode, 'failure': failure or ('process_failed' if process.returncode else None),
        'wall_seconds': time.perf_counter()-started, 'tree_peak_MiB': sum(peaks.values())/1024**2,
        'observed_processes': len(peaks), 'memory_method': POLICY['memory'],
        'budget_pass': failure is None and process.returncode == 0 and bool(peaks) and (os.name != 'nt' or len(peaks) >= 2),
        'command': command, 'request': request}
    (result_folder/'runtime.json').write_bytes(encoded(result))
    return result


def run(root, record, output, *, qualification=False, admission_pin=None):
    root, output = root.resolve(), output.resolve()
    config, _, requests = load(root, record)
    require(output.is_relative_to(root/'artifacts') and not output.exists(), 'Use new artifact output')
    output.mkdir(parents=True)
    if qualification:
        return execute_worker(root, config, {'output': (output/'result').relative_to(root).as_posix()},
                              output/'process', qualification=True)
    require(admission_pin is not None, 'Synthetic admission pin is required')
    admission = read(checked(root, admission_pin))
    require(admission['budget_pass'] and admission['customer_calls'] == 0 and
            admission['execution_contract'] == record, 'No matching synthetic qualification')
    by_id = {r['id']: r for r in requests}
    runtimes = []
    for repeat in range(POLICY['repeats']):
        for arm in ARMS:
            folder = output/f'repeat-{repeat+1}'/arm
            job = {'arm': arm, 'requests': [by_id[i] for i in config['assignments'][arm]],
                   'output': (folder/'result').relative_to(root).as_posix()}
            runtime = execute_worker(root, config, job, folder/'process')
            runtimes.append({'repeat': repeat+1, 'arm': arm, **runtime})
            (output/'runtimes.json').write_bytes(encoded(runtimes))
    return runtimes
