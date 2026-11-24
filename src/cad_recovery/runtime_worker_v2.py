"""Public correlated child entry point over the qualified bounded runtime."""
from pathlib import Path
from .runtime_worker_v1 import child_main as previous_child_main, cleanup_private
from .runtime_io_v1 import best_effort
from .worker_contract_v1 import WorkerIdentity, failure_record


def child_main(request_path, operation):
    path = Path(request_path)

    def execute(request, progress):
        identity = WorkerIdentity(**request['_worker_identity'])
        try:
            operation(request, progress)
        except BaseException as error:
            best_effort(path.parent/'child-context-error.json', failure_record(identity, 'child', error))
            raise
    return previous_child_main(path, execute)
