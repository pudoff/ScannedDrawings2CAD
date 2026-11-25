"""Durable watchdog whose clock includes initial lease/request/process startup."""
from pathlib import Path
import threading,time
from .durable_attempt_budget_v2 import AttemptBudget as Previous,BudgetError
from .service_v1.process import ChildGuard


class AttemptBudget(Previous):
    def __init__(self,process,folder,files,settings,*,memory_MiB=3072,started=None):
        self.persistence_disabled=False;self.replace_retries=0;self.persistence_error=None
        self.process=process;self.folder=Path(folder);self.files=files;self.settings=settings
        self.memory_MiB=memory_MiB;self.started=time.monotonic() if started is None else started
        self.lock=threading.RLock();self.done=threading.Event();self.peaks={}
        self.first_cause=None;self.secondary=[];self.error=None;self.stopped=False;self.termination_requested=False
        self.outcome='running';self.last_progress=None;self.progress_error=None;self.last_persist=0.
        self.observed_bytes=0;self.observed_files=0;self.rejected_stale_links=0
        self.peak_bytes=0;self.peak_files=0;self.diagnostic_path=self.folder/'execution-budget.json'
        created=False
        try:
            self.guard=ChildGuard(process)
            if self.diagnostic_path.exists():raise FileExistsError(self.diagnostic_path)
            with self.lock:self._sample_locked();self._persist_locked()
            created=True
            self.thread=threading.Thread(target=self._watch,daemon=True,name='whole-attempt-budget')
            self.thread.start();self.check()
        except BaseException:
            if created:
                try:self.stop()
                except Exception:pass
            self.done.set()
            thread=getattr(self,'thread',None)
            if thread is not None:thread.join(timeout=5)
            if hasattr(self,'guard'):self.guard.close()
            if process.poll() is None:process.kill();process.wait(timeout=5)
            raise

    def _measurements_locked(self):
        self.peak_bytes=max(self.peak_bytes,self.observed_bytes)
        self.peak_files=max(self.peak_files,self.observed_files)
        return dict(super()._measurements_locked(),
            deadline_scope='initial attempt-context/lease/request/process startup, compute, seal, cleanup and final publication preparation',
            filesystem_scope='entire attempt folder: output/logs/request/status/manifests/private/cache/TEMP',
            peak_observed_attempt_bytes=self.peak_bytes,peak_observed_attempt_files=self.peak_files,
            peak_scope='observed scans; not a kernel filesystem quota or unobserved transient peak')
