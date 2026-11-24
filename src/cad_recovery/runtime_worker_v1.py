"""Shared supervised child execution; publication belongs to fenced callers."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
from .durable_attempt_budget_v2 import AttemptBudget, BudgetError
from .runtime_io_v1 import atomic_json, best_effort, read_json, TelemetryPersistError


class LeaseLost(RuntimeError):
    code = 'lease_lost_or_cancelled'


def cleanup_private(folder):
    folder = Path(folder)
    paths = [p for p in [folder/'private-native', folder/'private', folder/'.cache',
        *folder.glob('private-table-*')] if p.exists()]
    for path in paths:
        if (not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink()
                or path.is_junction() or any(p.is_symlink() or p.is_junction() for p in path.rglob('*'))):
            raise ValueError('Linked private cache; cleanup refused')
    for path in paths:
        shutil.rmtree(path)


def run_child(request, folder, files, settings, script, heartbeat, *, memory_MiB=3072):
    from .runtime_cache_v1 import attempt_lock
    with attempt_lock(Path(folder)):
        return _run_child(request, folder, files, settings, script, heartbeat, memory_MiB=memory_MiB)


def _run_child(request, folder, files, settings, script, heartbeat, *, memory_MiB=3072):
    """Stop and finalize diagnostics before returning a publishable child result."""
    from .service_profiles_v12.worker import computation_running
    folder = Path(folder)
    process = budget = None
    first_error = None
    atomic_json(folder/'request.json', request)
    try:
        with (folder/'process.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, str(script), '--request', str(folder/'request.json')],
                stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, 'PYTHONIOENCODING':'utf-8', 'XDG_CACHE_HOME':str(folder/'.cache'),
                    'CAD_RUNTIME_PROGRESS':str(folder/'runtime-progress.json')},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), start_new_session=os.name!='nt')
            budget = AttemptBudget(process, folder, files, settings, memory_MiB=memory_MiB)
            next_heartbeat = time.monotonic()
            interval = min(2., getattr(settings, 'lease_seconds', 6)/3)
            def running():
                marker = folder/'status.json'
                if not marker.exists() and (folder/'child-failure.json').exists():
                    marker = folder/'child-failure.json'
                return computation_running(process, budget, marker)
            while running():
                if time.monotonic() >= next_heartbeat:
                    if not heartbeat():
                        raise LeaseLost()
                    next_heartbeat = time.monotonic()+interval
                time.sleep(.05)
        budget.check()
        marker = folder/'status.json'
        if not marker.exists():
            marker = folder/'child-failure.json'
        status = read_json(marker, limit=16384) if marker.exists() else {}
        if process.returncode or status != {'ok': True}:
            raise BudgetError(status.get('code', 'processing_incomplete'))
        budget.stop()
        budget.check()
        budget.finish('completed')
        return budget
    except Exception as error:
        first_error = error
        code = getattr(error, 'code', 'processing_failed')
        if budget is not None:
            try:
                budget.fail(code, kind='lease' if isinstance(error, LeaseLost) else 'worker')
            except TelemetryPersistError:
                pass
            finally:
                try:
                    budget.stop()
                except TelemetryPersistError:
                    pass
            code = budget.first_cause['code']
        elif process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        best_effort(folder/'worker-error.json', dict(type=type(error).__name__, code=code,
            # Raw details remain private; never put them in SQL/API errors.
            message=str(error)[:1000], final_diagnostics_before_publication=True))
        raise BudgetError(code) from error
    finally:
        if budget is not None:
            try:
                budget.stop()
            except TelemetryPersistError:
                if first_error is None:
                    raise
        try:
            cleanup_private(folder)
        except Exception as cleanup_error:
            best_effort(folder/'cleanup-error.json', dict(code='private_cleanup_failed', type=type(cleanup_error).__name__))
            if first_error is None:
                raise


def child_main(request_path, operation):
    """One terminal marker; telemetry failure cannot mask computation failure."""
    from .runtime_progress_v3 import Progress
    path = Path(request_path)
    progress = None
    try:
        progress = Progress(path.parent/'runtime-progress.json')
        request = read_json(path, limit=16*1024*1024)
        operation(request, progress)
        progress.finish('completed')
        atomic_json(path.parent/'status.json', dict(ok=True))
    except BaseException as error:
        if progress is not None:
            try:
                progress.finish('failed', error)
            except BaseException:
                pass
        failure = dict(ok=False, code=getattr(error, 'code', 'processing_failed'))
        best_effort(path.parent/'child-failure.json', failure)
        best_effort(path.parent/'status.json', failure)
        raise
