"""Windows-safe immutable progress files; the OCR implementation is unchanged."""
from contextlib import contextmanager
import json
import time

from .dataset_layout_v1 import encoded
from . import ocr_model_comparison_runtime_v1 as original


def progress(folder, phase, token):
    stamp = time.perf_counter_ns()
    temporary = folder/f'progress-{stamp:020d}.tmp'
    destination = temporary.with_suffix('.json')
    temporary.write_bytes(encoded({'phase': phase, 'token': token, 'started': time.perf_counter()}))
    temporary.rename(destination)


def read_events(path):
    """The v1 writer uses pretty JSON objects separated by whitespace."""
    payload = path.read_text('utf-8')
    decoder = json.JSONDecoder()
    index, records = 0, []
    while index < len(payload):
        if payload[index].isspace():
            index += 1
            continue
        value, index = decoder.raw_decode(payload, index)
        records.append(value)
    return records


@contextmanager
def immutable_progress():
    # Used only in a dedicated, single-task child. The model/inference routines
    # are exactly the pinned v1 implementation; the local progress sink differs.
    previous = original.progress
    original.progress = progress
    try:
        yield
    finally:
        original.progress = previous


def run_batch(root, contract, job):
    with immutable_progress():
        return original.run_batch(root, contract, job)


def qualify(root, contract, folder):
    with immutable_progress():
        return original.qualify(root, contract, folder)
