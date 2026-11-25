"""Experimental demo worker; retains the previous lease/fencing and independent budgets."""
import os
from pathlib import Path
import subprocess
import sys
import time
from cad_recovery.dataset_layout_v1 import read
from cad_recovery.service_v1.storage import write_json,digest
from cad_recovery.profile_contract_v3 import ProcessingOptions,validate_binding,ProfileError
from .queue import Queue
from cad_recovery.service_profiles_v2.storage import Storage
from cad_recovery.service_profiles_v2.budget import AttemptBudget

class LeaseLost(RuntimeError):pass

def process_once(settings,queue=None,storage=None,*,runner=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    if hasattr(repo,'bind_storage'):repo.bind_storage()
    run=repo.claim()
    if run is None:return False
    process=None;budget=None;folder=None
    try:
        source=files.path(run['input_path'])
        if digest(source)!=run['input_sha256']:raise ProfileError('input_integrity','Uploaded input hash differs')
        stored=dict(run['options']);binding=stored.pop('_profile_binding',None);options=ProcessingOptions.model_validate(stored)
        validate_binding(options.profile,binding);folder=files.attempt(run)
        write_json(folder/'request.json',{'source':str(source),'input_sha256':run['input_sha256'],'output':str(folder/'output'),
                                        'options':options.model_dump(),'profile_binding':binding})
        script=Path(runner) if runner else Path(__file__).resolve().parents[3]/'scripts/service_profile_pipeline_v3.py'
        env={**os.environ,'XDG_CACHE_HOME':str(folder/'.cache'),'PYTHONIOENCODING':'utf-8'}
        with (folder/'process.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,str(script),'--request',str(folder/'request.json')],stdout=log,stderr=subprocess.STDOUT,
                env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
            budget=AttemptBudget(process,folder,files,settings);heartbeat=time.monotonic()+settings.lease_seconds/3
            while process.poll() is None:
                budget.check()
                if time.monotonic()>=heartbeat:
                    if not repo.heartbeat(run):raise LeaseLost('lease_lost')
                    heartbeat=time.monotonic()+settings.lease_seconds/3
                time.sleep(.05)
        budget.check()
        if process.returncode:
            status=read(folder/'status.json') if (folder/'status.json').exists() else {}
            raise ProfileError(status.get('code','processing_failed'),'Child processing failed')
        report=read(folder/'output/report.json')
        summary={'status':report['status'],'page_count':report['page_count'],'processing_profile':report['processing_profile'],
            'processing_options':report['processing_options'],'limitations':report['limitations'],
            'pages':[{k:v for k,v in p.items() if k in ('page_index','status','issues','demo_export','native_analysis')} for p in report['pages']]}
        # Evidence includes counters; log/cache/private files all count in budget.
        write_json(folder/'execution-budget.json',budget.measurements());budget.check()
        manifest=files.seal(folder,summary);budget.check()
        # All descendants are stopped before any publication/database call.
        budget.stop();budget.check()
        repo.finish(run,manifest=manifest,summary=summary)
    except Exception as error:
        if budget:budget.stop()
        elif process is not None and process.poll() is None:process.kill();process.wait(timeout=5)
        if not isinstance(error,LeaseLost):
            if folder is not None:
                write_json(folder/'worker-error.json',{'type':type(error).__name__,'message':str(error)[:1000],
                    'child_exit_code':process.returncode if process is not None else None})
            code=getattr(error,'code',None) or (str(error) if str(error) in ('processing_timeout','output_limit','memory_limit','budget_monitor_failed') else 'processing_failed')
            repo.finish(run,error={'code':code,'message':'Processing stopped; inspect local attempt logs'})
    finally:
        if budget:budget.stop()
    return True
