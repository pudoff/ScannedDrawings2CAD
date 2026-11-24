"""Bounded cooperative diagnostic readback, including persistence failures."""
import math
from .runtime_io_v1 import read_json
from .web_runtime_status_v1 import public as previous_public


def public(value):
    result = previous_public(value)
    if (value.get('first_cause') or {}).get('code') == 'telemetry_persist_failed':
        result['reason'] = 'Не удалось сохранить журнал обработки'
    for key in ('elapsed_seconds', 'peak_memory_MiB'):
        v = result[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            result[key] = None
    return result


def latest(files, relative):
    empty = dict(available=False, phase=None, reason=None, progress_available=False)
    try:
        base = files.path(relative)
        if not base.exists():
            return empty
        folders = list(base.iterdir())
        if len(folders) > 128:
            return empty
        candidates = []
        for folder in folders:
            if not folder.is_dir() or folder.is_symlink() or folder.is_junction():
                continue
            path = folder/'execution-budget-failure.json'
            if not path.exists():
                path = folder/'execution-budget.json'
            path = files.path(path.relative_to(files.root).as_posix())
            if path.exists():
                candidates.append(path)
        if not candidates:
            return empty
        return public(read_json(max(candidates, key=lambda p:p.stat().st_mtime_ns)))
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return empty
