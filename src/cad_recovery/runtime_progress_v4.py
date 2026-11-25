"""Phase stack mutation and validation remain active under Python -O/-OO."""
from contextlib import contextmanager
import time
from .runtime_progress_v3 import Progress as Previous


class PhaseOrderError(RuntimeError):
    code='phase_order_failed'


class Progress(Previous):
    def __init__(self,path):
        self.phase_order_errors=[]
        super().__init__(path)

    def snapshot(self):
        value=super().snapshot()
        value['writer']['version']='runtime-progress/4.0'
        value['writer']['phase_order_errors']=list(self.phase_order_errors)
        return value

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
                if not self.stack or self.stack[-1] is not phase:
                    if len(self.phase_order_errors)<16:self.phase_order_errors.append('phase_order_failed')
                    # Do not pop another owner's phase or mask the computation
                    # error. A successful body with corrupt order always fails.
                    if original is None:raise PhaseOrderError('Diagnostic phase exit order differs')
                else:
                    self.stack.pop()
                    elapsed=time.monotonic()-phase['started_monotonic']
                    stat=self.stats[name];stat['calls']+=1;stat['failed']+=original is not None
                    stat['inclusive_seconds']+=elapsed;stat['exclusive_seconds']+=max(0.,elapsed-phase['nested_seconds'])
                    if self.stack:self.stack[-1]['nested_seconds']+=elapsed
                    self.last_completed=dict(name=name,context=context,failed=original is not None,seconds=elapsed)
                    try:self.write()
                    except BaseException as persistence:
                        self.secondary_persist_errors.append(type(persistence).__name__)
                        if original is None:raise
