"""Bounded diagnostic JSON I/O; close readers before decoding their snapshot."""
import json
import math
import os
from pathlib import Path
import time
from uuid import uuid4
from .dataset_layout_v1 import encoded


class TelemetryPersistError(OSError):
    code = 'telemetry_persist_failed'


def read_json(path, *, limit=1024*1024):
    """Enforce the bound during reading, including a concurrently growing file."""
    if type(limit) is not int or not 1 <= limit <= 16*1024*1024:
        raise ValueError('JSON read bound required')
    with open(path, 'rb') as stream:
        data = stream.read(limit+1)
    if len(data) > limit:
        raise ValueError('Diagnostic JSON exceeds read bound')
    def invalid(value):
        raise ValueError('Nonfinite JSON number')
    def finite_float(value):
        result=float(value)
        if not math.isfinite(result):
            raise ValueError('Nonfinite JSON number')
        return result
    result = json.loads(data, parse_constant=invalid, parse_float=finite_float)
    if not isinstance(result, dict):
        raise ValueError('Diagnostic JSON object required')
    return result


def atomic_json(path, value, *, attempts=20, delay=.025, _replace=None, _sleep=None):
    if type(attempts) is not int or not 1 <= attempts <= 40 or not 0 <= delay <= .05:
        raise ValueError('Bounded retry policy required')
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    replace, sleep = _replace or os.replace, _sleep or time.sleep
    data = encoded(value)  # Programming/serialization errors retain their own type.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for index in range(attempts):
            try:
                replace(temporary, path)
                return index
            except PermissionError:
                if index+1 == attempts:
                    raise
                sleep(delay)
    except OSError as error:
        raise TelemetryPersistError('Diagnostic persistence failed') from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass  # Only our unpublished temporary file; old JSON remains valid.


def best_effort(path, value):
    try:
        atomic_json(path, value)
        return True
    except (OSError, ValueError):
        return False
