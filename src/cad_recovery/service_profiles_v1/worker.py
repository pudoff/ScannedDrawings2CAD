"""Separate bounded child process per leased run; stale attempts cannot publish."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from cad_recovery.service_v1.storage import Storage,write_json,digest
from .queue import Queue
from cad_recovery.profile_contract_v1 import ProcessingOptions,ProfileError,validate_binding
from cad_recovery.service_v1.process import ChildGuard

class LeaseLost(RuntimeError):pass

def process_once(settings,queue=None,storage=None,*,runner=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    if hasattr(repo,'bind_storage'):repo.bind_storage()
    run=repo.claim()
    if run is None:return False
    process=None;guard=None
    try:
        source=files.path(run['input_path'])
        if digest(source)!=run['input_sha256']:raise ValueError('input_integrity')
        stored=dict(run['options']);binding=stored.pop('_profile_binding',None)
        options=ProcessingOptions.model_validate(stored)
        if binding is None:raise ProfileError('profile_contract_changed','Job has no immutable profile binding')
        validate_binding(options.profile,binding);folder=files.attempt(run)
        request={'source':str(source),'input_sha256':run['input_sha256'],'output':str(folder/'output'),
            'options':options.model_dump(),'profile_binding':binding};write_json(folder/'request.json',request)
        script=Path(runner) if runner else Path(__file__).resolve().parents[3]/'scripts/service_profile_pipeline_v1.py'
        env=dict(os.environ);env['XDG_CACHE_HOME']=str(folder/'.cache')
        tick=time.monotonic();heartbeat=tick+settings.lease_seconds/3
        with (folder/'process.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,str(script),'--request',str(folder/'request.json')],
                stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            guard=ChildGuard(process)
            while process.poll() is None:
                now=time.monotonic()
                if now-tick>settings.timeout_seconds:raise TimeoutError('processing_timeout')
                total,count=files.output_size(folder)
                if total>settings.max_output_bytes or count>settings.max_output_files:raise OverflowError('output_limit')
                if now>=heartbeat:
                    if not repo.heartbeat(run):raise LeaseLost('lease_lost')
                    heartbeat=now+settings.lease_seconds/3
                time.sleep(.1)
        if process.returncode!=0:
            result=json.loads((folder/'status.json').read_text('utf-8')) if (folder/'status.json').exists() else {}
            raise RuntimeError(result.get('code','processing_failed'))
        # Fast-finishing children still undergo the complete attempt budget check.
        total,count=files.output_size(folder)
        if total>settings.max_output_bytes or count>settings.max_output_files:raise OverflowError('output_limit')
        report=json.loads((folder/'output/report.json').read_text('utf-8'))
        summary={'status':report['status'],'page_count':report['page_count'],
            'pages':[{'page_index':p['page_index'],'status':p['status'],'issues':p['issues'],
                      **({'demo_export':p['demo_export']} if 'demo_export' in p else {})} for p in report['pages']],
            'limitations':report['limitations'],'processing_profile':report['processing_profile'],
            'processing_options':report['processing_options'],
            'service_limits':{'timeout_seconds':settings.timeout_seconds,'max_output_bytes':settings.max_output_bytes,
                              'max_output_files':settings.max_output_files,'lease_seconds':settings.lease_seconds}}
        manifest=files.seal(folder,summary)
        total,count=files.output_size(folder)
        if total>settings.max_output_bytes or count>settings.max_output_files:raise OverflowError('output_limit')
        # finish() checks token AND unexpired database lease in the same UPDATE.
        repo.finish(run,manifest=manifest,summary=summary)
    except LeaseLost:
        pass # Lost leases never change another owner's row or publish their output.
    except Exception as error:
        # Release native child resources before a potentially slow database write.
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=10)
        if guard is not None:guard.close();guard=None
        codes=('input_integrity','processing_timeout','output_limit','input_invalid','processing_failed',
               'profile_contract_changed','profile_model_missing','profile_model_mismatch','profile_environment_changed',
               'profile_artifact_mismatch','output_exists')
        code=getattr(error,'code',None) or ('output_limit' if isinstance(error,OverflowError) else str(error) if str(error) in codes else 'processing_failed')
        repo.finish(run,error={'code':code,'message':'Processing stopped; inspect local worker attempt logs'})
    finally:
        if process is not None and process.poll() is None:
            process.kill();process.wait(timeout=10)
        if guard is not None:guard.close()
    return True
