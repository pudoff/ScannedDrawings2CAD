"""Profile worker16: compute and seal share the supervised child budget."""
from pathlib import Path
from cad_recovery.dataset_layout_v1 import digest
from cad_recovery.profile_contract_v7 import ProcessingOptions,validate_binding,ProfileError
from cad_recovery.service_profiles_v19.queue import Queue
from cad_recovery.service_profiles_v9.storage import Storage
from cad_recovery.runtime_cache_v1 import reap_dead_attempts
from cad_recovery.worker_contract_v1 import identity_for_run,public_failure_code
from cad_recovery.worker_attempt_v1 import record_failure
from cad_recovery.worker_attempt_v2 import execute_attempt


def process_once(settings,queue=None,storage=None,*,runner=None):
    repo,files=queue or Queue(settings),storage or Storage(settings)
    repo.bind_storage()
    run=repo.claim()
    if run is None:return False
    identity=identity_for_run(run)
    try:
        source=files.path(run['input_path'])
        if digest(source)!=run['input_sha256']:raise ProfileError('input_integrity','Uploaded input differs')
        stored=dict(run['options']);binding=stored.pop('_profile_binding',None)
        options=ProcessingOptions.model_validate(stored);validate_binding(options.profile,binding)
        reap_dead_attempts(files,f"attempts/{run['id']}")
        folder=files.attempt(run)
        execute_attempt(identity=identity,folder=folder,files=files,limits=settings,
            request=dict(source=str(source),input_sha256=run['input_sha256'],output=str(folder/'output'),
                         options=options.model_dump(),profile_binding=binding),
            script=runner or Path(__file__).resolve().parents[3]/'scripts/service_profile_diagnostic_v5.py',
            heartbeat=lambda:repo.heartbeat(run),
            publish=lambda manifest,summary,deadline:repo.finish(run,manifest=manifest,summary=summary,deadline=deadline))
    except Exception as error:
        if not (files.root/'worker-events'/identity.request_id/(identity.attempt_id+'.json')).exists():
            record_failure(files,identity,'prepare',error)
        code=public_failure_code(error)
        if code!='lease_lost_or_cancelled':
            repo.finish(run,error=dict(code=code,message='Processing stopped; private diagnostics retained'))
    finally:
        if hasattr(repo,'close_heartbeat'):repo.close_heartbeat(run)
    return True
