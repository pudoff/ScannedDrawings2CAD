"""Bounded Windows sharing retries and exception-safe phase entry.

V1 remains frozen. Computation, budgets and geometry are unaffected; failed
durable persistence still fails explicitly, never masquerades as success.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import time
from uuid import uuid4
from .runtime_progress_v1 import Progress as Previous
from .dataset_layout_v1 import encoded


class TelemetryPersistError(OSError):
    code='telemetry_persist_failed'


def atomic_json(path,value,*,attempts=20,delay=.025,_replace=None,_sleep=None):
    if type(attempts) is not int or not 1<=attempts<=40:raise ValueError('Bounded persistence attempts1..40')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid4().hex[:12]+'.tmp')
    replace=_replace or os.replace;sleep=_sleep or time.sleep;retries=0
    try:
        with temporary.open('xb') as stream:stream.write(encoded(value));stream.flush();os.fsync(stream.fileno())
        for index in range(attempts):
            try:replace(temporary,path);return retries
            except PermissionError as error:
                retries+=1
                if index+1==attempts:raise TelemetryPersistError('Atomic telemetry replacement remained inaccessible') from error
                sleep(delay)
    finally:
        # A failed replace leaves only our own unpublished temp; keep the old
        # complete JSON authoritative. Do not erase another writer's temp.
        if temporary.exists():
            try:temporary.unlink()
            except PermissionError:pass


class Progress(Previous):
    def __init__(self,path):
        self.replace_retries=0;self.secondary_persist_errors=[]
        super().__init__(path)

    def snapshot(self):
        result=super().snapshot()
        result['writer']=dict(version='runtime-progress/2.0',replace_retries=self.replace_retries,
            bounded_attempts=20,delay_seconds=.025,secondary_errors=list(self.secondary_persist_errors))
        return result

    def write(self):
        with self.lock:
            self.sequence+=1
            self.replace_retries+=atomic_json(self.path,self.snapshot())

    @contextmanager
    def phase(self,name,**context):
        with self.lock:
            if len(self.stack)>=16:raise ValueError('Diagnostic phase nesting budget')
            if len(self.stats)>=64 and name not in self.stats:raise ValueError('Diagnostic phase name budget')
            phase=dict(name=name,context=context,started_monotonic=time.monotonic(),nested_seconds=0.)
            self.stack.append(phase)
            try:self.write()
            except BaseException:
                self.stack.pop();raise
        original=None
        try:yield
        except BaseException as error:original=error;raise
        finally:
            with self.lock:
                elapsed=time.monotonic()-phase['started_monotonic']
                assert self.stack.pop() is phase
                stat=self.stats[name];stat['calls']+=1;stat['failed']+=original is not None
                stat['inclusive_seconds']+=elapsed;stat['exclusive_seconds']+=max(0.,elapsed-phase['nested_seconds'])
                if self.stack:self.stack[-1]['nested_seconds']+=elapsed
                self.last_completed=dict(name=name,context=context,failed=original is not None,seconds=elapsed)
                try:self.write()
                except BaseException as persistence:
                    self.secondary_persist_errors.append(type(persistence).__name__)
                    if original is None:raise
