"""First-observed cause and observed memory survive timeout, exit and cancellation."""
from pathlib import Path
import json
import os
import signal
import threading
import time

from .runtime_progress_v1 import atomic_json
from .service_v1.process import ChildGuard
from .process_lineage_budget_v1 import windows_peaks
from .hard_raster_runs_v1 import process_peaks


class AttemptBudget:
    def __init__(self, process, folder, files, settings, *, memory_MiB=3072):
        self.process=process; self.folder=Path(folder); self.files=files; self.settings=settings
        self.memory_MiB=memory_MiB; self.started=time.monotonic(); self.lock=threading.RLock()
        self.guard=ChildGuard(process); self.done=threading.Event(); self.peaks={}
        self.first_cause=None; self.secondary=[]; self.error=None; self.stopped=False; self.termination_requested=False
        self.outcome='running'; self.last_progress=None; self.progress_error=None
        self.last_persist=0.; self.observed_bytes=0; self.observed_files=0; self.rejected_stale_links=0
        self.diagnostic_path=self.folder/'execution-budget.json'
        if self.diagnostic_path.exists():
            self.guard.close(); raise FileExistsError(self.diagnostic_path)
        with self.lock:
            self._sample_locked(); self._persist_locked()
        self.thread=threading.Thread(target=self._watch,daemon=True,name='durable-attempt-budget')
        self.thread.start()

    def _progress_locked(self):
        path=self.folder/'runtime-progress.json'
        if path.exists():
            try:
                if path.stat().st_size>1024**2: raise ValueError('Progress exceeds1MiB')
                value=json.loads(path.read_text(encoding='utf-8'))
                if value.get('schema_version')!='runtime-progress/1.0': raise ValueError('Unknown progress schema')
                self.last_progress=value; self.progress_error=None
            except (OSError,ValueError) as error:
                self.progress_error=str(error)[:300]

    def _sample_locked(self):
        if self.stopped or self.termination_requested:return
        observed,rejected=(windows_peaks(self.process,self.guard) if os.name=='nt' else
                           (process_peaks(self.process,self.guard),0))
        self.rejected_stale_links=max(self.rejected_stale_links,rejected)
        for identity,value in observed.items():self.peaks[identity]=max(self.peaks.get(identity,0),value)

    def _cause_locked(self, code, kind):
        event=dict(code=code,kind=kind,observed_monotonic=time.monotonic(),
                   elapsed_seconds=time.monotonic()-self.started)
        if self.first_cause is None:
            self.first_cause=event; self.error=code; self.outcome='failed'
        elif self.first_cause['code']!=code and not any(e['code']==code for e in self.secondary):
            if len(self.secondary)<16:self.secondary.append(event)

    def fail(self, code, *, kind='worker'):
        with self.lock:
            self._cause_locked(code,kind); self._persist_locked()

    def _kill_locked(self):
        if self.termination_requested:return
        self.termination_requested=True
        self.guard.close()
        if os.name!='nt':
            try:os.killpg(self.process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        if self.process.poll() is None:self.process.kill()

    def _measurements_locked(self):
        self._progress_locked()
        now=time.monotonic()
        active=(self.last_progress or {}).get('active_phases',[])
        return dict(schema_version='execution-budget/2.0',outcome=self.outcome,
            walltime_seconds=now-self.started,peak_rss_MiB=sum(self.peaks.values())/1024**2 if self.peaks else None,
            observed_process_count=len(self.peaks),
            rss_method='sum of observed per-process lifetime peaks;50ms polling;not simultaneous RSS',
            process_identity='Windows PID+creation time and job membership;Linux root VmHWM only',
            lifetime_peaks=[dict(pid=k[0] if isinstance(k,tuple) else k,
                creation_100ns=k[1] if isinstance(k,tuple) else None,peak_bytes=v) for k,v in self.peaks.items()],
            rejected_stale_parent_links=self.rejected_stale_links,
            memory_limit_MiB=self.memory_MiB,timeout_seconds=self.settings.timeout_seconds,
            max_output_bytes=self.settings.max_output_bytes,max_output_files=self.settings.max_output_files,
            observed_output_bytes=self.observed_bytes,observed_output_files=self.observed_files,
            first_cause=self.first_cause,secondary_causes=self.secondary,budget_error=self.error,
            child_exit_code=self.process.poll(),all_owned_children_stop_requested=self.stopped,
            progress=self.last_progress,progress_read_error=self.progress_error,
            active_phase_elapsed=[dict(name=p['name'],context=p['context'],
                elapsed_seconds=max(0.,now-p['started_monotonic'])) for p in active],
            missing_measurement_is_null=True)

    def _persist_locked(self):
        atomic_json(self.diagnostic_path,self._measurements_locked());self.last_persist=time.monotonic()

    def measurements(self):
        with self.lock:return self._measurements_locked()

    def check(self):
        with self.lock:
            if self.error:raise RuntimeError(self.error)
            total,count=self.files.output_size(self.folder)
            self.observed_bytes,self.observed_files=total,count
            code=None
            if time.monotonic()-self.started>self.settings.timeout_seconds:code='processing_timeout'
            elif total>self.settings.max_output_bytes or count>self.settings.max_output_files:code='output_limit'
            elif sum(self.peaks.values())>self.memory_MiB*1024**2:code='memory_limit'
            if code:
                self._cause_locked(code,'budget'); self._persist_locked()  # before termination
                self._kill_locked();raise RuntimeError(self.error)
            if time.monotonic()-self.last_persist>=.5:self._persist_locked()

    def _watch(self):
        try:
            while not self.done.wait(.05):
                with self.lock:self._sample_locked()
                self.check()
        except Exception as error:
            with self.lock:
                if self.error is None:self._cause_locked('budget_monitor_failed','monitor')
                try:self._persist_locked()
                finally:self._kill_locked()

    def stop(self):
        self.done.set()
        if threading.current_thread() is not self.thread:self.thread.join(timeout=5)
        with self.lock:
            if not self.stopped:
                self._sample_locked(); self.stopped=True
                self._persist_locked(); self._kill_locked()
            if self.process.poll() is None:self.process.wait(timeout=5)
            self._persist_locked()

    def finish(self, outcome):
        with self.lock:
            self.outcome=outcome if self.first_cause is None else 'failed'
            self._persist_locked()
