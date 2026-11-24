"""A watchdog stops the child independently of queue/database calls."""
import threading
import time
import os
import signal
from cad_recovery.service_v1.process import ChildGuard
from cad_recovery.hard_raster_runs_v1 import process_peaks

class AttemptBudget:
    def __init__(self,process,folder,files,settings,*,memory_MiB=3072):
        self.process=process;self.folder=folder;self.files=files;self.settings=settings
        self.started=time.monotonic();self.memory_MiB=memory_MiB;self.peaks={};self.error=None
        self.guard=ChildGuard(process);self.done=threading.Event();self.lock=threading.Lock()
        self.thread=threading.Thread(target=self._watch,daemon=True,name='native-attempt-budget');self.thread.start()

    def _kill(self):
        with self.lock:
            self.guard.close()
            if os.name!='nt':
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            if self.process.poll() is None:self.process.kill()

    def check(self):
        if self.error:raise RuntimeError(self.error)
        total,count=self.files.output_size(self.folder)
        if time.monotonic()-self.started>self.settings.timeout_seconds:self.error='processing_timeout'
        elif total>self.settings.max_output_bytes or count>self.settings.max_output_files:self.error='output_limit'
        elif sum(self.peaks.values())>self.memory_MiB*1024**2:self.error='memory_limit'
        if self.error:self._kill();raise RuntimeError(self.error)

    def _watch(self):
        try:
            while not self.done.wait(.05):
                with self.lock:observed=process_peaks(self.process,self.guard)
                for pid,value in observed.items():self.peaks[pid]=max(self.peaks.get(pid,0),value)
                self.check()
        except Exception:
            if self.error is None:self.error='budget_monitor_failed'
            self._kill()

    def stop(self):
        self.done.set();self.thread.join(timeout=5);self._kill()
        if self.process.poll() is None:self.process.wait(timeout=5)

    def measurements(self):
        return {'walltime_seconds':time.monotonic()-self.started,'peak_rss_MiB':sum(self.peaks.values())/1024**2,
                'observed_process_count':len(self.peaks),'rss_method':'sum of observed per-PID lifetime peaks; conservative, 50ms polling',
                'memory_limit_MiB':self.memory_MiB,'budget_error':self.error}
