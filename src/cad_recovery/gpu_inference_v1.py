"""Isolated GPU qualification helpers; production OCR keeps its CPU backend."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import socket

import numpy as np


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def deny_python_network():
    """Reject Python Internet sockets and DNS during the local qualification.

    This is a process-local guard, not a system firewall for native libraries.
    """
    original_socket = socket.socket
    original_dns = socket.getaddrinfo

    class OfflineSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family in (socket.AF_INET, socket.AF_INET6):
                raise RuntimeError('Network is disabled during GPU qualification')
            super().__init__(family, *args, **kwargs)

    def denied_dns(*args, **kwargs):
        raise RuntimeError('DNS is disabled during GPU qualification')

    socket.socket = OfflineSocket
    socket.getaddrinfo = denied_dns
    try:
        yield
    finally:
        socket.socket = original_socket
        socket.getaddrinfo = original_dns


def summarize_profile(events: list[dict]) -> dict:
    """Count actual executed node events, not the requested provider list."""
    counts = Counter()
    operators = defaultdict(Counter)
    duration = Counter()
    for event in events:
        arguments = event.get('args', {})
        provider = arguments.get('provider')
        if event.get('cat') != 'Node' or not provider:
            continue
        counts[provider] += 1
        operators[provider][arguments.get('op_name', 'unknown')] += 1
        duration[provider] += event.get('dur', 0)
    gpu_compute = sum(n for op, n in operators['CUDAExecutionProvider'].items()
                      if any(word in op.lower() for word in ('conv', 'matmul', 'gemm', 'attention')))
    if not counts['CUDAExecutionProvider'] or not gpu_compute:
        raise RuntimeError('No profiled CUDA neural-network operators: refusing CPU-only fallback')
    return {'executed_node_events': dict(counts),
            'operators_by_provider': {key: dict(value) for key, value in operators.items()},
            'node_event_duration_us': dict(duration),
            'cuda_compute_events': gpu_compute,
            'duration_scope': 'ORT node-event durations; not end-to-end GPU kernel timing',
            'all_model_on_cpu': False}


def make_probe_input(image, height: int, width: int) -> np.ndarray:
    """Shape/finite-output probe only; this does not reproduce OCR preprocessing."""
    from PIL import Image
    resized = image.convert('RGB').resize((width, height), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)/np.float32(127.5)-np.float32(1)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def local_models(root: Path) -> list[dict]:
    manifest = json.loads((root/'src/cad_recovery/data/ocr-models-v1.json').read_text('utf-8'))
    if {r['role'] for r in manifest['models']} != {'det', 'rec', 'cls'} or len(manifest['models']) != 3:
        raise ValueError('Expected the three pinned local OCR models')
    records = []
    for row in manifest['models']:
        path = root/'artifacts/models/ppocr-v5'/row['filename']
        if path.is_symlink() or not path.resolve().is_relative_to(root/'artifacts/models/ppocr-v5'):
            raise ValueError('Model path outside the local model directory')
        if sha256(path) != row['sha256']:
            raise ValueError('Changed local OCR model: '+row['filename'])
        records.append({**row, 'path': path.relative_to(root).as_posix()})
    return records
