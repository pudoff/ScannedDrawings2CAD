"""Terminal markers and correlation with the optimization-safe phase writer."""
from pathlib import Path
import os
from .runtime_progress_v4 import Progress
from .runtime_io_v1 import read_json,atomic_json,best_effort
from .worker_contract_v1 import WorkerIdentity,failure_record
from .service_v1.storage import write_json


def child_main(request_path,operation):
    path=Path(request_path);progress=None
    # Claim once, outside the failure writer. A duplicate invocation must not
    # turn another child's successful status/evidence into a failure marker.
    for name in ('runtime-progress.json','status.json','child-failure.json',
                 'child-context-error.json','sealed-result.json','result-manifest.json'):
        if (path.parent/name).exists():raise FileExistsError(path.parent/name)
    write_json(path.parent/'child-lifecycle-owner.json',dict(schema_version='child-lifecycle-owner/1.0',pid=os.getpid()))
    try:
        progress=Progress(path.parent/'runtime-progress.json')
        request=read_json(path,limit=16*1024*1024)
        identity=WorkerIdentity(**request['_worker_identity'])
        try:operation(request,progress)
        except BaseException as error:
            best_effort(path.parent/'child-context-error.json',failure_record(identity,'child',error))
            raise
        if progress.stack:raise RuntimeError('Operation left active diagnostic phases')
        progress.finish('completed')
        atomic_json(path.parent/'status.json',dict(ok=True))
    except BaseException as error:
        if progress is not None:
            try:progress.finish('failed',error)
            except BaseException:pass
        failure=dict(ok=False,code=getattr(error,'code','processing_failed'))
        best_effort(path.parent/'child-failure.json',failure)
        best_effort(path.parent/'status.json',failure)
        raise
