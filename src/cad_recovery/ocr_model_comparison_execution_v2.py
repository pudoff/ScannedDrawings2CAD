"""Transport-only correction after recorded Windows progress-file races."""
from pathlib import Path
import os
import subprocess
import sys
import time

from .dataset_layout_v1 import checked, encoded, pin, read
from .ocr_model_comparison_protocol_v1 import POLICY, ARMS, require
from .ocr_model_comparison_execution_v1 import load as original_load, preserve, process_peaks, ChildGuard
from .ocr_model_comparison_transport_v2 import read_events

PARENT = {'path': 'configs/reviews/ocr-model-comparison-execution-v1.json',
          'sha256': '8a1c162838eb446dc15461527937dd782a961147ccb4bef94de6753e57d7d1ae'}
CODE = ['src/cad_recovery/ocr_model_comparison_transport_v2.py',
        'src/cad_recovery/ocr_model_comparison_execution_v2.py', 'scripts/ocr_model_comparison_worker_v2.py',
        'scripts/run_ocr_model_comparison_v2.py', 'docs/ocr-model-comparison-transport-v2.md']


def freeze(root, destination, folder):
    destination, folder = destination.resolve(), folder.resolve()
    require(destination.is_relative_to(root/'configs') and folder.is_relative_to(root/'artifacts') and
            not destination.exists() and not folder.exists(), 'Choose fresh v2 paths')
    parent, contract, _ = original_load(root, PARENT)
    attempt = root/'artifacts/ocr-model-comparison-run-v1'
    require((attempt/'report.json').exists(), 'Record the completed failed attempt first')
    events = [event for path in sorted(attempt.glob('repeat-*/*/result/events.jsonl')) for event in read_events(path)]
    calls = sum(e['event'] == 'start' for e in events)
    folder.mkdir(parents=True)
    contract['code'] += [pin(root, name) for name in CODE]
    (folder/'worker-contract.json').write_bytes(encoded(contract))
    value = {**parent, 'schema_version': 'ocr-model-comparison-execution/2.0', 'parent_execution': PARENT,
        'code': contract['code'], 'worker_contract': pin(root, (folder/'worker-contract.json').relative_to(root).as_posix()),
        'prior_attempt_pins': {p.relative_to(root).as_posix(): pin(root, p.relative_to(root).as_posix())['sha256']
                               for p in sorted(attempt.rglob('*')) if p.is_file()},
        'prior_attempt_image_calls': calls, 'new_quality_calls_before_freeze': calls,
        'correction': 'progress transport only; unique immutable JSON state files avoid reader-held rename target',
        'retuned_models_parameters_labels_or_gates': False}
    destination.write_bytes(encoded(value))
    return pin(root, destination.relative_to(root).as_posix())


def load(root, record):
    value = read(checked(root, record))
    parent, old_contract, requests = original_load(root, PARENT)
    require(value['schema_version'] == 'ocr-model-comparison-execution/2.0' and value['parent_execution'] == PARENT and
        value['code'] == parent['code']+[pin(root, name) for name in CODE], 'Changed v2 transport')
    for key in ('protocol', 'assignments', 'data', 'regression_scope', 'previous', 'previous_pins'):
        require(value[key] == parent[key], 'Transport fix changed comparison factors: '+key)
    for name, sha in value['prior_attempt_pins'].items():
        checked(root, {'path': name, 'sha256': sha})
    contract = read(checked(root, value['worker_contract']))
    require(contract == {**old_contract, 'code': value['code']}, 'Transport fix changed models or backend')
    return value, contract, requests


def execute_worker(root, config, job, folder, *, qualification=False):
    folder.mkdir(parents=True, exist_ok=False)
    request_path = folder/'request.json'
    request_path.write_bytes(encoded(job))
    request = pin(root, request_path.relative_to(root).as_posix())
    contract = config['worker_contract']
    command = [sys.executable, str(root/'scripts/ocr_model_comparison_worker_v2.py'),
        '--contract', contract['path'], '--contract-sha256', contract['sha256'],
        '--request', request['path'], '--request-sha256', request['sha256']]
    if qualification: command.append('--qualify')
    started, peaks, failure = time.perf_counter(), {}, None
    limits = POLICY['limits']
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'XDG_CACHE_HOME': str(root/'artifacts/.cache'),
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'}
    with (folder/'stdout.txt').open('wb') as stdout, (folder/'stderr.txt').open('wb') as stderr:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stdout, stderr=stderr)
        guard = None
        try:
            guard = ChildGuard(process)
            while True:
                for pid, memory in process_peaks(process, guard).items(): peaks[pid] = max(peaks.get(pid, 0), memory)
                elapsed = time.perf_counter()-started
                if sum(peaks.values()) > limits['worker_memory_MiB']*1024**2: failure = 'process_tree_memory_exceeded'
                if elapsed > limits['worker_seconds']: failure = failure or 'worker_timeout'
                paths = list((root/job['output']).glob('progress-*.json'))
                if paths:
                    state = read(max(paths))
                    deadline = limits['cold_init_seconds'] if state['phase'] == 'init' else limits['case_seconds']
                    if state['phase'] != 'done' and time.perf_counter()-state['started'] > deadline:
                        failure = failure or state['phase']+'_timeout'
                elif elapsed > limits['cold_init_seconds']: failure = failure or 'startup_timeout'
                if failure or process.poll() is not None: break
                time.sleep(.05)
            if failure and process.poll() is None: process.kill()
            process.wait()
        finally:
            if guard: guard.close()
            if process.poll() is None: process.kill(); process.wait()
    result = {'exit_code': process.returncode, 'failure': failure or ('process_failed' if process.returncode else None),
        'wall_seconds': time.perf_counter()-started, 'tree_peak_MiB': sum(peaks.values())/1024**2,
        'observed_processes': len(peaks), 'memory_method': POLICY['memory'],
        'budget_pass': failure is None and process.returncode == 0 and bool(peaks) and (os.name != 'nt' or len(peaks) >= 2),
        'command': command, 'request': request}
    (folder/'runtime.json').write_bytes(encoded(result))
    return result


def run(root, record, output, *, qualification=False, admission_pin=None):
    root, output = root.resolve(), output.resolve()
    config, _, requests = load(root, record)
    require(output.is_relative_to(root/'artifacts') and not output.exists(), 'Use new output path')
    output.mkdir(parents=True)
    if qualification:
        return execute_worker(root, config, {'output': (output/'result').relative_to(root).as_posix()}, output/'process', qualification=True)
    require(admission_pin is not None, 'Synthetic admission required')
    admission = read(checked(root, admission_pin))
    require(admission['budget_pass'] and admission['customer_calls'] == 0 and admission['execution_contract'] == record,
            'No matching qualification')
    by_id = {r['id']: r for r in requests}
    runtimes = []
    for repeat in (1, 2):
        for arm in ARMS:
            folder = output/f'repeat-{repeat}'/arm
            job = {'arm': arm, 'requests': [by_id[i] for i in config['assignments'][arm]],
                   'output': (folder/'result').relative_to(root).as_posix()}
            runtime = execute_worker(root, config, job, folder/'process')
            runtimes.append({'repeat': repeat, 'arm': arm, **runtime})
            (output/'runtimes.json').write_bytes(encoded(runtimes))
    return runtimes
