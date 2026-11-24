"""Bounded, atomic phase telemetry outside disposable native-CAD scratch."""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import json
import os
import threading
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    data = (json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)+'\n').encode()
    with temporary.open('wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class Progress:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists(): raise FileExistsError(self.path)
        self.lock = threading.RLock(); self.started = time.monotonic()
        self.stack = []; self.stats = defaultdict(lambda: dict(calls=0, failed=0, inclusive_seconds=0., exclusive_seconds=0.))
        self.completed_windows = 0; self.state = 'running'; self.sequence = 0
        self.last_completed = None; self.error = None
        self.write()

    def snapshot(self):
        return dict(schema_version='runtime-progress/1.0', pid=os.getpid(), state=self.state,
            sequence=self.sequence, elapsed_seconds=time.monotonic()-self.started,
            clock='time.monotonic; parent/child on same host',
            active_phases=[{k:v for k,v in phase.items() if k!='nested_seconds'} for phase in self.stack],
            last_completed=self.last_completed, phases=dict(self.stats),
            completed_windows=self.completed_windows, error=self.error)

    def write(self):
        with self.lock:
            self.sequence += 1
            atomic_json(self.path, self.snapshot())

    @contextmanager
    def phase(self, name, **context):
        with self.lock:
            if len(self.stack)>=16: raise ValueError('Diagnostic phase nesting budget')
            if len(self.stats)>=64 and name not in self.stats: raise ValueError('Diagnostic phase name budget')
            phase = dict(name=name, context=context, started_monotonic=time.monotonic(), nested_seconds=0.)
            self.stack.append(phase); self.write()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            with self.lock:
                elapsed = time.monotonic()-phase['started_monotonic']
                assert self.stack.pop() is phase
                stat = self.stats[name]; stat['calls'] += 1; stat['failed'] += failed
                stat['inclusive_seconds'] += elapsed
                stat['exclusive_seconds'] += max(0., elapsed-phase['nested_seconds'])
                if self.stack: self.stack[-1]['nested_seconds'] += elapsed
                self.last_completed = dict(name=name, context=context, failed=failed, seconds=elapsed)
                self.write()

    def window_completed(self):
        with self.lock: self.completed_windows += 1; self.write()

    def finish(self, state, error=None):
        with self.lock:
            self.state=state; self.error=str(error)[:600] if error else None; self.write()
