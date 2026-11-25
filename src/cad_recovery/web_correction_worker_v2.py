"""Correction preview/apply shares supervised compute+seal and fenced SQL."""
from pathlib import Path
from types import SimpleNamespace
import time
from .service_profiles_v21.queue import Queue
from .service_profiles_v9.storage import Storage
from .web_correction_store_v1 import CorrectionStore as HistoricalStore
from .web_correction_store_v2 import CorrectionStore
from .worker_contract_v1 import identity_for_table,public_failure_code
from .worker_attempt_v1 import record_failure
from .worker_attempt_v4 import execute_attempt
from .runtime_cache_v1 import reap_dead_attempts


def process_once(settings,*,store=None,queue=None,storage=None,runner=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    repo.bind_storage()
    if store is None or isinstance(store,HistoricalStore) and not isinstance(store,CorrectionStore):
        store=CorrectionStore(repo)
    row=store.claim()
    if row is None:return False
    started=time.monotonic();identity=identity_for_table(row)
    try:
        document,versions=repo.job(row['job_id']);parent=repo.run(row['job_id'],row['parent_version'])
        identity=identity_for_table(row,parent)
        if (parent['manifest']['sha256']!=row['source_manifest_sha256']
                or max(v['version'] for v in versions)!=row['parent_version']):
            raise ValueError('parent_version_changed')
        run={k:parent[k] for k in ('id','job_id','version','status','options','manifest','summary')}
        run={**run,'id':str(run['id']),'job_id':str(run['job_id']),'input_sha256':document['input_sha256']}
        files.manifest(run)
        reap_dead_attempts(files,f"corrections-v1/{row['id']}")
        folder=files.path(f"corrections-v1/{row['id']}/{row['owner']}");folder.mkdir(parents=True,exist_ok=False)
        limits=SimpleNamespace(timeout_seconds=min(settings.timeout_seconds,60),max_output_bytes=min(settings.max_output_bytes,128*1024**2),
            max_output_files=min(settings.max_output_files,5000),lease_seconds=settings.lease_seconds)
        def publish(manifest,summary,deadline):
            if row['stage']=='preview':return store.finish(row,manifest=manifest,summary=summary,deadline=deadline)
            return bool(store.publish(row,parent,manifest,summary,deadline=deadline))
        execute_attempt(identity=identity,folder=folder,files=files,limits=limits,started=started,memory_MiB=1024,
            request=dict(storage=str(files.root),folder=str(folder),stage=row['stage'],parent=run,
                input_path=document['input_path'],page=row['page_index'],edit=row['request'],
                approval=row['approval'],actor=row['actor'],proposal_id=str(row['id'])),
            script=runner or Path(__file__).resolve().parents[2]/'scripts/build_web_correction_v2.py',
            heartbeat=lambda:store.heartbeat(row),publish=publish)
    except Exception as error:
        if not (files.root/'worker-events'/identity.request_id/(identity.attempt_id+'.json')).exists():
            record_failure(files,identity,'prepare',error)
        code=public_failure_code(error,fallback='correction_operation_failed')
        if code!='lease_lost_or_cancelled':store.finish(row,error=code)
    finally:store.close_heartbeat(row)
    return True
