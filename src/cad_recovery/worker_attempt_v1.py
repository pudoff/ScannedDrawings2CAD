"""One public attempt lifecycle shared by profile and table workers.

Computation diagnostics finish before SQL publication. An uncertain COMMIT is
recorded outside the sealed attempt and never overwritten with a failure state.
"""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .runtime_worker_v1 import run_child as supervised_child, LeaseLost
from .runtime_io_v1 import atomic_json, best_effort
from .worker_contract_v1 import WorkerIdentity, AttemptFiles, AttemptLimits, failure_record


@dataclass(frozen=True)
class AttemptResult:
    state: Literal['published', 'fenced', 'uncertain']
    manifest: dict
    summary: dict


def record_failure(files: AttemptFiles, identity: WorkerIdentity, stage, error, *, code=None):
    """Safe outside-output receipt, including failures before an attempt exists."""
    path = files.root/'worker-events'/identity.request_id/(identity.attempt_id+'.json')
    best_effort(path, failure_record(identity, stage, error, code=code))


def execute_attempt(*, identity: WorkerIdentity, request: dict, folder: Path,
                    files: AttemptFiles, limits: AttemptLimits, script: Path,
                    heartbeat: Callable[[], bool], load_summary: Callable[[], dict],
                    verify: Callable[[], None], publish: Callable[[dict, dict], bool],
                    memory_MiB: int = 3072, child_runner=None) -> AttemptResult:
    stage = 'prepare'
    folder = Path(folder)
    try:
        atomic_json(folder/'attempt-context.json', dict(schema_version='worker-attempt/1.0',
            identity=identity.record(), state='computing'))
        request = {**request, '_worker_identity': identity.record()}
        stage = 'child'
        budget = (child_runner or supervised_child)(request, folder, files, limits, script,
                                                    heartbeat, memory_MiB=memory_MiB)
        stage = 'summary'; summary = load_summary()
        stage = 'verify'; verify()
        stage = 'seal'; manifest = files.seal(folder, summary)
        atomic_json(folder/'attempt-context.json', dict(schema_version='worker-attempt/1.0',
            identity=identity.record(), state='ready_to_publish'))
        budget.check(); budget.finish('completed')
    except Exception as error:
        record_failure(files, identity, stage, error)
        raise
    # No diagnostics are changed inside the sealed attempt after this point.
    try:
        accepted = publish(manifest, summary)
    except Exception as error:
        record_failure(files, identity, 'publish', error, code='publication_uncertain')
        return AttemptResult('uncertain', manifest, summary)
    if not accepted:
        record_failure(files, identity, 'publish', LeaseLost())
        return AttemptResult('fenced', manifest, summary)
    return AttemptResult('published', manifest, summary)
