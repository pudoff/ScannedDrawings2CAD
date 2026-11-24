"""Existing lease/profile envelope with durable first-cause diagnostics."""
from pathlib import Path
import os
import subprocess
import sys
import time

from cad_recovery.dataset_layout_v1 import read
from cad_recovery.service_v1.storage import write_json,digest
from cad_recovery.profile_contract_v6 import ProcessingOptions,validate_binding,ProfileError
from cad_recovery.service_profiles_v11.queue import Queue
from cad_recovery.service_profiles_v9.storage import Storage
from cad_recovery.service_profiles_v11.worker import guarded_children_running,LeaseLost
from cad_recovery.durable_attempt_budget_v1 import AttemptBudget
from cad_recovery.runtime_progress_v1 import atomic_json


def computation_running(process,budget,status):
    budget.check()  # Timeout first, including when Windows launcher already exited.
    try:
        if process.poll() is None:return True
        if process.returncode:return False
        if status.is_file():return False
        if guarded_children_running(budget):return True
        raise ProfileError('processing_incomplete','Child exited without a completion marker')
    except Exception:
        budget.check()  # Preserve a watchdog cause observed during the preceding race.
        raise


def process_once(settings,queue=None,storage=None,*,runner=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    if hasattr(repo,'bind_storage'):repo.bind_storage()
    run=repo.claim()
    if run is None:return False
    process=budget=folder=None
    try:
        source=files.path(run['input_path'])
        if digest(source)!=run['input_sha256']:raise ProfileError('input_integrity','Uploaded input hash differs')
        stored=dict(run['options']);binding=stored.pop('_profile_binding',None)
        options=ProcessingOptions.model_validate(stored);validate_binding(options.profile,binding)
        folder=files.attempt(run)
        write_json(folder/'request.json',dict(source=str(source),input_sha256=run['input_sha256'],
            output=str(folder/'output'),options=options.model_dump(),profile_binding=binding))
        script=Path(runner) if runner else Path(__file__).resolve().parents[3]/'scripts/service_profile_pipeline_v6.py'
        with (folder/'process.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,str(script),'--request',str(folder/'request.json')],
                stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONIOENCODING':'utf-8',
                    'XDG_CACHE_HOME':str(folder/'.cache'),'CAD_RUNTIME_PROGRESS':str(folder/'runtime-progress.json')},
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),start_new_session=os.name!='nt')
            budget=AttemptBudget(process,folder,files,settings)
            heartbeat=time.monotonic()+settings.lease_seconds/3
            while computation_running(process,budget,folder/'status.json'):
                if time.monotonic()>=heartbeat:
                    if not repo.heartbeat(run):raise LeaseLost('lease_lost_or_cancelled')
                    heartbeat=time.monotonic()+settings.lease_seconds/3
                time.sleep(.05)
        budget.check()
        status=read(folder/'status.json') if (folder/'status.json').exists() else {}
        if process.returncode or status!={'ok':True}:
            raise ProfileError(status.get('code','processing_failed'),'Child processing failed')
        budget.stop();budget.check()
        report=read(folder/'output/report.json')
        if report['processing_profile']!=binding:raise ProfileError('profile_contract_changed','Completed profile differs')
        summary=dict(status=report['status'],page_count=report['page_count'],processing_profile=binding,
            processing_options=report['processing_options'],limitations=report['limitations'],
            pages=[{k:v for k,v in p.items() if k in ('page_index','status','issues','demo_export','native_analysis')}
                   for p in report['pages']])
        manifest=files.seal(folder,summary);budget.check()
        repo.finish(run,manifest=manifest,summary=summary);budget.finish('completed')
    except Exception as error:
        code=getattr(error,'code',None) or (str(error) if str(error) in
            ('processing_timeout','output_limit','memory_limit','budget_monitor_failed') else 'processing_failed')
        if isinstance(error,LeaseLost):code='lease_lost_or_cancelled'
        if budget:
            budget.fail(code,kind='lease' if isinstance(error,LeaseLost) else 'worker')
            budget.stop();code=budget.first_cause['code']
        elif process is not None and process.poll() is None:process.kill();process.wait(timeout=5)
        if folder is not None:
            atomic_json(folder/'worker-error.json',dict(type=type(error).__name__,message=str(error)[:1000],
                effective_code=code,child_exit_code=process.returncode if process else None))
            if budget is None:
                atomic_json(folder/'execution-budget.json',dict(schema_version='execution-budget/2.0',
                    outcome='failed_before_monitor',peak_rss_MiB=None,first_cause=dict(code=code,kind='worker'),
                    progress=None,missing_measurement_is_null=True))
        if not isinstance(error,LeaseLost):
            repo.finish(run,error=dict(code=code,message='Processing stopped;durable local diagnostics retained'))
    finally:
        if budget:budget.stop()
    return True
