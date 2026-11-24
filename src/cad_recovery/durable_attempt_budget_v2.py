"""Frozen limits/lineage with failure-safe persistence and guaranteed termination."""
import threading
import time
from .durable_attempt_budget_v1 import AttemptBudget as Previous
from .runtime_io_v1 import atomic_json, read_json, best_effort, TelemetryPersistError


class BudgetError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class AttemptBudget(Previous):
    def __init__(self, *args, **kwargs):
        self.persistence_disabled = False
        self.replace_retries = 0
        self.persistence_error = None
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            # V1 construction can fail after assigning the Windows child job.
            if hasattr(self, 'guard'):
                self.guard.close()
            if hasattr(self, 'process') and self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=5)
            raise

    def _progress_locked(self):
        path = self.folder/'runtime-progress.json'
        if not path.exists():
            return
        try:
            value = read_json(path)
            if value.get('schema_version') != 'runtime-progress/1.0':
                raise ValueError('Unknown progress schema')
            active = value.get('active_phases', [])
            if not isinstance(active, list) or len(active) > 16:
                raise ValueError('Invalid phase count')
            for phase in active:
                if (not isinstance(phase, dict) or not isinstance(phase.get('name'), str)
                        or not isinstance(phase.get('context'), dict)
                        or type(phase.get('started_monotonic')) not in (int, float)):
                    raise ValueError('Invalid phase')
            self.last_progress, self.progress_error = value, None
        except (OSError, ValueError, RecursionError) as error:
            self.progress_error = type(error).__name__

    def _measurements_locked(self):
        result = super()._measurements_locked()
        result['writer'] = dict(version='execution-budget/3.0', replace_retries=self.replace_retries,
            persistence_disabled=self.persistence_disabled, persistence_error=self.persistence_error)
        return result

    def _persist_locked(self):
        if self.persistence_disabled:
            return
        try:
            self.replace_retries += atomic_json(self.diagnostic_path, self._measurements_locked())
            self.last_persist = time.monotonic()
        except TelemetryPersistError:
            self._cause_locked('telemetry_persist_failed', 'telemetry')
            self.persistence_disabled = True
            self.persistence_error = 'telemetry_persist_failed'
            # Keep the last complete main JSON. A separate owned receipt can
            # record the first cause when only its destination is locked.
            best_effort(self.folder/'execution-budget-failure.json', self._measurements_locked())
            raise

    def check(self):
        with self.lock:
            if self.error:
                raise BudgetError(self.error)
            # Check time before a potentially expensive tree scan as well.
            code = None
            if time.monotonic()-self.started > self.settings.timeout_seconds:
                code = 'processing_timeout'
            else:
                try:
                    self.observed_bytes, self.observed_files = self.files.output_size(self.folder)
                except OSError:
                    code = 'budget_monitor_failed'
                if code is None:
                    if time.monotonic()-self.started > self.settings.timeout_seconds:
                        code = 'processing_timeout'
                    elif self.observed_bytes > self.settings.max_output_bytes or self.observed_files > self.settings.max_output_files:
                        code = 'output_limit'
                    elif sum(self.peaks.values()) > self.memory_MiB*1024**2:
                        code = 'memory_limit'
            if code:
                self._cause_locked(code, 'budget')
                try:
                    self._persist_locked()
                finally:
                    self._kill_locked()
                raise BudgetError(self.error)
            if time.monotonic()-self.last_persist >= .5:
                self._persist_locked()

    def _watch(self):
        try:
            while not self.done.wait(.05):
                with self.lock:
                    self._sample_locked()
                self.check()
        except Exception:
            with self.lock:
                if self.first_cause is None:
                    self._cause_locked('budget_monitor_failed', 'monitor')
                try:
                    self._persist_locked()
                except TelemetryPersistError:
                    pass
                finally:
                    self._kill_locked()

    def stop(self):
        self.done.set()
        thread = getattr(self, 'thread', None)
        if thread is not None and threading.current_thread() is not thread:
            thread.join(timeout=5)
        with self.lock:
            if self.stopped:
                return
            try:
                self._sample_locked()
                self.stopped = True
                self._persist_locked()
            finally:
                self.stopped = True
                self._kill_locked()
                if self.process.poll() is None:
                    self.process.wait(timeout=5)
                if self.persistence_disabled:
                    best_effort(self.folder/'execution-budget-failure.json', self._measurements_locked())
