"""Offline, single-factor OCR adapter; no labels or source groups in requests."""
from copy import deepcopy
import hashlib
import math
import time

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, encoded
from .ocr_engine import LocalOCREngine, _initialize_rapidocr, _offline_backend_scope
from .ocr_model_comparison_protocol_v1 import ARMS, BACKEND, POLICY, require, validate_request


class ComparisonEngine(LocalOCREngine):
    """Reuse baseline implementation; only the explicitly assigned weights vary."""
    def __init__(self, root, contract, arm):
        require(arm in ARMS and contract['arms'] == ARMS and contract['backend'] == BACKEND,
                'Changed model comparison arm/backend')
        require(contract['runtime_limits']['precision'] == 'FP32', 'FP32 required')
        models = contract['models']
        definition = ARMS[arm]
        paths = {role: checked(root, models[name]) for role, name in {
            'det': definition['detector'] or 'mobile', 'rec': definition['recognizer'], 'cls': 'baseline_cls'}.items()}
        with _offline_backend_scope():
            self._backend = _initialize_rapidocr(paths, POLICY['intra_op_threads'])
        self._provenance = {'arm': arm, 'definition': definition,
            'models': {role: {'path': path.relative_to(root).as_posix(),
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for role, path in paths.items()},
            'provider': 'CPUExecutionProvider', 'precision': 'FP32',
            'intra_op_threads': POLICY['intra_op_threads'], 'inter_op_threads': 1,
            'constructor': 'unchanged baseline _initialize_rapidocr, including its language/model-type hints',
            'dictionary': 'embedded in selected recognizer; no external dictionary', 'automatic_downloads': 'blocked'}
        self.arm = arm

    def observe(self, pixels):
        if ARMS[self.arm]['task'] == 'roi_pipeline':
            observations = self.recognize(pixels)
            return {'raw': ' '.join(o['raw'] for o in observations), 'confidence': None,
                    'observations': observations}
        from rapidocr.ch_ppocr_rec.typings import TextRecInput
        with _offline_backend_scope():
            result = self._backend.text_rec(TextRecInput(img=pixels.copy(), return_word_box=False))
        require(result.txts is not None and len(result.txts) == len(result.scores) == 1,
                'Rec-only must return one literal')
        score = float(result.scores[0])
        require(isinstance(result.txts[0], str) and math.isfinite(score) and 0 <= score <= 1,
                'Invalid rec-only output')
        return {'raw': result.txts[0], 'confidence': score, 'observations': []}


def canonical_observation(value):
    require(set(value) == {'raw', 'confidence', 'observations'} and isinstance(value['raw'], str), 'Invalid OCR envelope')
    require(len(value['raw']) <= 4096, 'OCR text exceeds budget')
    if value['confidence'] is not None:
        require(type(value['confidence']) in (int, float) and math.isfinite(value['confidence']) and
                0 <= value['confidence'] <= 1, 'Invalid confidence')
    for row in value['observations']:
        require(set(row) == {'raw', 'confidence', 'polygon_px'} and isinstance(row['raw'], str), 'Unexpected observation fields')
        points = np.asarray(row['polygon_px'], dtype=float)
        require(points.shape == (4, 2) and np.isfinite(points).all() and math.isfinite(row['confidence']) and
                0 <= row['confidence'] <= 1, 'Invalid OCR polygon/confidence')
    encoded(value)  # Reject NaN, infinity and non-JSON backend values.
    return deepcopy(value)


def validate_job(job, contract):
    require(set(job) == {'arm', 'requests', 'output'}, 'Worker cannot receive targets/metadata')
    require(job['arm'] in ARMS and contract['arms'] == ARMS, 'Unknown comparison arm')
    require(len(job['requests']) <= 324 and len({r['id'] for r in job['requests']}) == len(job['requests']),
            'Unbounded or repeated assignments')
    total = 0
    for request in job['requests']:
        validate_request(request)
        total += request['size_px'][0]*request['size_px'][1]
    require(total <= POLICY['limits']['all_case_pixels'], 'Request pixel budget exceeded')


def progress(folder, phase, token):
    temporary = folder/'progress.tmp'
    temporary.write_bytes(encoded({'phase': phase, 'token': token, 'started': time.perf_counter()}))
    temporary.replace(folder/'progress.json')


def run_batch(root, contract, job):
    from .dataset_layout_v1 import inside
    validate_job(job, contract)
    folder = inside(root, job['output'])
    folder.mkdir(parents=True, exist_ok=False)
    progress(folder, 'init', 'init')
    tick = time.perf_counter()
    engine = ComparisonEngine(root, contract, job['arm'])
    init_seconds = time.perf_counter()-tick
    require(init_seconds <= POLICY['limits']['cold_init_seconds'], 'Cold initialization exceeded limit')
    results, timings = [], []
    with (folder/'events.jsonl').open('x', encoding='utf-8') as events:
        for row in job['requests']:
            progress(folder, 'case', row['id'])
            started = time.perf_counter()
            times = []
            values = []
            try:
                with Image.open(checked(root, row['image'])) as original:
                    require(list(original.size) == row['size_px'], 'Image dimensions differ from frozen request')
                    pixels = np.ascontiguousarray(np.asarray(original.convert('RGB'))[:, :, ::-1])
                for iteration in range(1+POLICY['warm_repeats_per_case']):
                    events.write(encoded({'id': row['id'], 'call': iteration, 'event': 'start'}).decode())
                    events.flush()
                    call_started = time.perf_counter()
                    value = canonical_observation(engine.observe(pixels))
                    times.append(time.perf_counter()-call_started)
                    values.append(value)
                    events.write(encoded({'id': row['id'], 'call': iteration, 'event': 'complete',
                        'output_sha256': hashlib.sha256(encoded(value)).hexdigest()}).decode())
                    events.flush()
                status = 'ok' if values[0]['raw'] else 'empty'
                result = {'id': row['id'], 'status': status, **values[0],
                          'warm_outputs_identical': all(v == values[0] for v in values[1:]), 'error': None}
            except Exception as error:
                # No retry and no dropped case. The evaluator scores a deletion.
                first = values[0] if values else {'raw': '', 'confidence': None, 'observations': []}
                status = ('ok' if first['raw'] else 'empty') if values else 'error'
                result = {'id': row['id'], 'status': status, **first, 'warm_outputs_identical': False,
                          'error': type(error).__name__+': '+str(error)}
            elapsed = time.perf_counter()-started
            result['case_budget_pass'] = elapsed <= POLICY['limits']['case_seconds']
            results.append(result)
            timings.append({'id': row['id'], 'first_seconds': times[0] if times else None,
                            'warm_seconds': times[1:], 'all_case_seconds': elapsed})
            (folder/'results.json').write_bytes(encoded(results))
            if not result['case_budget_pass']:
                raise RuntimeError('Per-case time budget exceeded; partial results retained')
    (folder/'timings.json').write_bytes(encoded({'init_seconds': init_seconds, 'rows': timings}))
    (folder/'engine.json').write_bytes(encoded(engine.provenance))
    progress(folder, 'done', 'done')
    return {'cases': len(results), 'errors': sum(r['status'] == 'error' for r in results)}


def qualify(root, contract, folder):
    """Forward synthetic tensors through every pinned ONNX role, twice."""
    import onnxruntime as ort
    from .dataset_layout_v1 import pin
    folder.mkdir(parents=True, exist_ok=False)
    progress(folder, 'init', 'qualification')
    records = []
    for name, record in contract['models'].items():
        options = ort.SessionOptions()
        options.intra_op_num_threads = POLICY['intra_op_threads']
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(checked(root, record)), sess_options=options,
                                       providers=['CPUExecutionProvider'])
        shape = [1, 3, 80, 160] if name == 'baseline_cls' else [1, 3, 48, 320] if 'mobile' in name and name not in ('mobile',) else [1, 3, 320, 320]
        if name == 'server': shape = [1, 3, 320, 320]
        inputs = np.zeros(shape, dtype=np.float32)
        progress(folder, 'case', name)
        outputs = [session.run(None, {session.get_inputs()[0].name: inputs}) for _ in range(2)]
        require(len(outputs[0]) == len(outputs[1]) == 1 and np.isfinite(outputs[0][0]).all(), 'Invalid synthetic forward')
        require(np.array_equal(outputs[0][0], outputs[1][0]), 'Synthetic forward is not repeatable')
        path = folder/(name+'.npy')
        with path.open('xb') as stream: np.save(stream, outputs[0][0], allow_pickle=False)
        records.append({'model': name, 'input_shape': shape, 'output_shape': list(outputs[0][0].shape),
                        'output': pin(root, path.relative_to(root).as_posix()), 'forward_calls': 2,
                        'finite': True, 'repeat_identical': True})
        del session
    # Exercise the actual adapters on generated pixels; no customer images/labels.
    from PIL import ImageDraw, ImageFont
    canvas = Image.new('RGB', (512, 192), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 30), '45° ±2 Втулка', font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 36), fill='black')
    pixels = np.ascontiguousarray(np.asarray(canvas)[:, :, ::-1])
    canvas.save(folder/'synthetic.png')
    adapter = []
    for arm in ARMS:
        progress(folder, 'case', arm)
        engine = ComparisonEngine(root, contract, arm)
        first = canonical_observation(engine.observe(pixels))
        second = canonical_observation(engine.observe(pixels))
        require(first == second, 'Synthetic adapter repeat differs')
        adapter.append({'arm': arm, 'output': first, 'repeat_identical': True, 'calls': 2})
        del engine
    (folder/'qualification.json').write_bytes(encoded({'tensor_forwards': records, 'adapter_smoke': adapter,
        'customer_calls': 0, 'scope': 'synthetic shape/adapter compatibility; not recognition accuracy'}))
    progress(folder, 'done', 'done')
