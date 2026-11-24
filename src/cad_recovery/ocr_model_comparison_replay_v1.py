"""Independent artifact checks around pure metric replay; no inference APIs."""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

from .dataset_layout_v1 import checked, encoded, pin, read
from .ocr_model_comparison_execution_v2 import load, preserve
from .ocr_model_comparison_evaluation_v1 import build_report, align_predictions
from .ocr_model_comparison_protocol_v1 import ARMS
from .ocr_model_comparison_transport_v2 import read_events

EXECUTION = {'path': 'configs/reviews/ocr-model-comparison-execution-v2.json',
             'sha256': '43c2bc2ed93e08a71ee0ca8fba6b7e082be65f0068fced1d808b7fd4b6f997ad'}
RUN = 'artifacts/ocr-model-comparison-run-v2'
QUALIFICATION = 'artifacts/ocr-model-comparison-qualification-v2'


def event_proof(ids, rows, events):
    aligned = align_predictions(ids, rows)
    expected = [(identifier, call, event) for identifier in ids for call in range(4) for event in ('start', 'complete')]
    actual = [(e['id'], e['call'], e['event']) for e in events]
    if actual != expected:
        raise ValueError('Missing/extra/reordered inference events')
    for identifier in ids:
        row = aligned[identifier]
        hashes = [e['output_sha256'] for e in events if e['id'] == identifier and e['event'] == 'complete']
        digest = hashlib.sha256(encoded({k: row[k] for k in ('raw', 'confidence', 'observations')})).hexdigest()
        if hashes[0] != digest or row['warm_outputs_identical'] != (len(set(hashes)) == 1):
            raise ValueError('Raw result or warm-repeat evidence differs from event log')
    return len(events)//2


def verify(root):
    config, contract, requests = load(root, EXECUTION)
    previous = preserve(root)
    output = root/RUN
    expected = build_report(root, config, output)
    if (output/'report.json').read_bytes() != encoded(expected):
        raise ValueError('Recomputed metrics/decision differ')
    starts = 0
    prior_equal = 0
    manifests = []
    runtimes = read(output/'runtimes.json')
    if len(runtimes) != 8 or any(not r['budget_pass'] for r in runtimes):
        raise ValueError('Final run is not a complete guarded eight-worker experiment')
    by_id = {r['id']: r for r in requests}
    for repeat in (1, 2):
        for arm in ARMS:
            folder = output/f'repeat-{repeat}'/arm
            job = read(folder/'process/request.json')
            if job != {'arm': arm, 'requests': [by_id[i] for i in config['assignments'][arm]],
                       'output': (folder/'result').relative_to(root).as_posix()}:
                raise ValueError('Worker received other inputs or extra metadata')
            rows = read(folder/'result/results.json')
            starts += event_proof(config['assignments'][arm], rows, read_events(folder/'result/events.jsonl'))
            engine = read(folder/'result/engine.json')
            definition = ARMS[arm]
            role_models = {'det': definition['detector'] or 'mobile', 'rec': definition['recognizer'], 'cls': 'baseline_cls'}
            if engine['models'] != {role: contract['models'][name] for role, name in role_models.items()}:
                raise ValueError('Executed model identities differ from assigned arm')
            prior = root/f'artifacts/ocr-model-comparison-run-v1/repeat-{repeat}/{arm}/result/results.json'
            if prior.exists():
                current = {r['id']: r for r in rows}
                for old in read(prior):
                    if current[old['id']] != old:
                        raise ValueError('Transport-only fix changed an already observed output')
                    prior_equal += 1
            manifests.append(pin(root, (folder/'result/results.json').relative_to(root).as_posix()))
    if starts != 7888:
        raise ValueError('Unexpected total image-level inference count')
    admission = read(root/QUALIFICATION/'admission.json')
    runtime = read(checked(root, admission['runtime']))
    qualification = read(checked(root, admission['result']))
    if admission['execution_contract'] != EXECUTION or not runtime['budget_pass'] or admission['customer_calls'] != 0:
        raise ValueError('Qualification evidence differs')
    for record in qualification['tensor_forwards']:
        with checked(root, record['output']).open('rb') as stream:
            array = np.load(stream, allow_pickle=False)
        if list(array.shape) != record['output_shape'] or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError('Synthetic forward proof corrupted')
    return {'schema_version': 'ocr-model-comparison-replay/1.0', 'execution': EXECUTION,
            'worker_count': 8, 'complete_image_calls': starts, 'event_count': starts*2,
            'prior_attempt_image_calls': config['prior_attempt_image_calls'],
            'prior_completed_outputs_unchanged': prior_equal, 'previous_pins_preserved': len(previous),
            'case_results': manifests, 'metric_replay_equal': True, 'new_model_calls': 0,
            'decisions': {k: v['decision'] for k, v in expected['comparisons'].items()}}
