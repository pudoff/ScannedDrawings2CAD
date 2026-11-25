"""Table worker8: killable seal and atomic parent/deadline publication fence."""
from pathlib import Path
from types import SimpleNamespace
import os
from .service_profiles_v12.queue import Queue
from .service_profiles_v9.storage import Storage
from .web_tables_store_v2 import TableStore
from .web_tables_contract_v1 import binding as first
from .web_tables_contract_v2 import binding as second
from .web_tables_contract_v3 import binding as third
from .web_tables_contract_v4 import binding as fourth
from .web_tables_contract_v5 import binding as fifth
from .runtime_cache_v1 import reap_dead_attempts
from .worker_contract_v1 import identity_for_table,public_failure_code
from .worker_attempt_v1 import record_failure
from .worker_attempt_v2 import execute_attempt
from .table_worker_request_v2 import prepare_table_execution


def process_once(settings,*,store=None,runner=None,queue=None,storage=None):
    repo,files=queue or Queue(settings),storage or Storage(settings)
    store=store or TableStore(repo);row=store.claim()
    if row is None:return False
    identity=identity_for_table(row)
    try:
        document,versions=repo.job(row['job_id']);parent=repo.run(row['job_id'],row['parent_version'])
        identity=identity_for_table(row,parent)
        if max(v['version'] for v in versions)!=row['parent_version']:raise ValueError('Parent version changed')
        prepared=prepare_table_execution(row,parent,document,files=files,store=store,
            factories=(first,second,third,fourth,fifth),models=Path(os.environ.get('CAD_DEMO_OCR_MODELS',
                str(Path(__file__).resolve().parents[2]/'artifacts/models/ppocr-v5'))))
        reap_dead_attempts(files,f"tables-v1/{row['id']}")
        folder=files.path(f"tables-v1/{row['id']}/{row['owner']}");folder.mkdir(parents=True)
        limits=SimpleNamespace(**vars(prepared.limits),lease_seconds=settings.lease_seconds)
        execute_attempt(identity=identity,request={**prepared.request,'folder':str(folder)},folder=folder,
            files=files,limits=limits,memory_MiB=prepared.memory_MiB,
            script=runner or Path(__file__).resolve().parents[2]/'scripts/build_web_tables_v8.py',
            heartbeat=lambda:store.heartbeat(row),
            publish=lambda manifest,summary,deadline:store.finish(row,manifest=manifest,summary=summary,deadline=deadline))
    except Exception as error:
        if not (files.root/'worker-events'/identity.request_id/(identity.attempt_id+'.json')).exists():
            record_failure(files,identity,'prepare',error)
        code=public_failure_code(error,fallback='table_processing_failed')
        if code!='lease_lost_or_cancelled':store.finish(row,error=code)
    finally:
        if hasattr(store,'close_heartbeat'):store.close_heartbeat(row)
    return True
