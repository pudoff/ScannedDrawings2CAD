"""Native CAD Web demo, retained documents and recovery-aware PostgreSQL lifecycle."""
import argparse
from dataclasses import replace
from datetime import datetime
import os
import subprocess
import sys
import time
import urllib.request
from uuid import uuid4

import run_profiles_v8 as previous
from run_demo import (ROOT,RUNTIME,MODELS,DEPS,FLAGS,preflight,available_port,launcher_lock,
    stop_child,atomic_control,stop_requested,read_active,stop_path)
from profile_database_v2 import start_database,stop_owned_database

def component(mode, port):
    from cad_recovery.service_profiles_v2.config import Settings
    settings=replace(Settings.environment(),lease_seconds=90)
    if mode=='web':
        import uvicorn
        from cad_recovery.service_profiles_v10.api import create_web_app
        uvicorn.run(create_web_app(settings),host='127.0.0.1',port=port,access_log=False,
            server_header=False,timeout_keep_alive=5,limit_concurrency=16)
    elif mode=='corrections':
        from cad_recovery.correction_loop_v1 import run
        from cad_recovery.service_profiles_v10.queue import Queue
        from cad_recovery.web_correction_store_v1 import CorrectionStore
        from cad_recovery.web_correction_worker_v1 import process_once
        store=CorrectionStore(Queue(settings))
        run(settings,process=lambda configured:process_once(configured,store=store),
            sleep=lambda seconds:time.sleep(max(2.,seconds)))
    else:
        from cad_recovery.service_profiles_v10.worker import process_once
        from cad_recovery.correction_loop_v1 import run
        run(settings,process=process_once,label='Native CAD demo',
            sleep=lambda seconds:time.sleep(max(2.,seconds)))

initialize_database=previous.initialize_database


def request_stop(timeout=180):
    state=read_active()
    if state is None or state['status']=='stopped':
        print('No active launcher is recorded. No process was signalled.');return
    identifier=state['id'];atomic_control(stop_path(identifier),{'id':identifier,'action':'stop'})
    print('Stop requested. Waiting for service processes and safe PostgreSQL shutdown...',flush=True)
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        current=read_active()
        if current is None or current['id']!=identifier:raise RuntimeError('Launcher identity changed; no other launcher was stopped.')
        if current['status']=='stopped':
            print('Stopped. Documents retained. Database:',current.get('database_stop',{}).get('outcome','see logs'),flush=True);return
        time.sleep(.2)
    raise RuntimeError(f'Shutdown has not completed in {timeout}s. Inspect launcher/postgres logs; no process was forcibly killed.')


def run(args):
    preflight()
    from cad_recovery.profile_contract_v5 import DEFINITIONS,resolve_profile,verify_models,verify_cv_runtime
    for name in DEFINITIONS:verify_models(resolve_profile(name),MODELS)
    verify_cv_runtime()
    if args.check:
        print('OK: all shared profile/code/model/dependency bindings verified. No services started.');return
    with launcher_lock():
        available_port(args.port)
        if args.port==args.db_port:raise RuntimeError('HTTP and database ports must differ.')
        os.environ.update(CAD_SERVICE_DSN=f'postgresql://cad_demo@127.0.0.1:{args.db_port}/cad_recovery_profiles_v1',
            CAD_SERVICE_STORAGE=str(ROOT/'artifacts/profile-service-v1/storage'),CAD_SERVICE_TIMEOUT_SECONDS='900',
            CAD_DEMO_OCR_MODELS=str(MODELS),CAD_SERVICE_DEPS=str(DEPS),PYTHONIOENCODING='utf-8',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
        logs=ROOT/'artifacts/profile-service-v12/logs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8]);logs.mkdir(parents=True)
        identifier=uuid4().hex
        state={'schema_version':'demo-launcher/1.0','id':identifier,'status':'starting','pid':os.getpid(),
            'port':args.port,'db_port':args.db_port,'logs':str(logs.relative_to(ROOT)),'service':'shared-profiles/12.0'}
        pg=web=worker=corrections=None;database_result=None
        with (logs/'postgres.log').open('xb') as pglog,(logs/'web.log').open('xb') as weblog,(logs/'worker.log').open('xb') as workerlog,(logs/'corrections.log').open('xb') as correctionlog:
            try:
                atomic_control(RUNTIME/'active-launcher.json',state)
                print('Starting shared-profile service. Logs:',logs,flush=True)
                pg=start_database(args.db_port,pglog,timeout=args.db_startup_timeout,cancelled=lambda:stop_requested(identifier))
                if stop_requested(identifier):return
                initialize_database(args.db_port)
                command=[sys.executable,str(ROOT/'scripts/run_profiles_v12.py'),'--port',str(args.port),'--component']
                web=subprocess.Popen(command+['web'],cwd=ROOT,stdout=weblog,stderr=subprocess.STDOUT,creationflags=FLAGS)
                worker=subprocess.Popen(command+['worker'],cwd=ROOT,stdout=workerlog,stderr=subprocess.STDOUT,creationflags=FLAGS)
                corrections=subprocess.Popen(command+['corrections'],cwd=ROOT,stdout=correctionlog,stderr=subprocess.STDOUT,creationflags=FLAGS)
                url=f'http://127.0.0.1:{args.port}'
                headers={'Authorization':'Bearer '+os.environ['CAD_SERVICE_TOKEN']} if os.environ.get('CAD_SERVICE_TOKEN') else {}
                deadline=time.monotonic()+40
                while time.monotonic()<deadline:
                    if stop_requested(identifier):return
                    if any(p.poll() is not None for p in (web,worker,corrections)):raise RuntimeError('Service component stopped; inspect logs.')
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url+'/health',headers=headers),timeout=2) as response:
                            if response.status==200:break
                    except OSError:time.sleep(.2)
                else:raise RuntimeError('HTTP service did not become ready.')
                state['status']='running';atomic_control(RUNTIME/'active-launcher.json',state)
                print('Ready:',url,flush=True)
                print('Stop with Ctrl+C or scripts/run_profiles_v12.py --stop.',flush=True)
                if args.open:
                    import webbrowser
                    webbrowser.open(url)
                while not stop_requested(identifier):
                    if any(p.poll() is not None for p in (web,worker,corrections)) or pg is not None and pg.poll() is not None:
                        raise RuntimeError('A service process stopped; inspect logs.')
                    time.sleep(.5)
            except KeyboardInterrupt as error:database_result=getattr(error,'database_shutdown',None)
            except Exception as error:
                database_result=getattr(error,'database_shutdown',None)
                raise
            finally:
                stop_child(corrections);stop_child(worker);stop_child(web)
                state['database_stop']=database_result or stop_owned_database(pg,pglog)
                state['status']='stopped';atomic_control(RUNTIME/'active-launcher.json',state)
                print('Service UI/workers stopped. Profile-service documents retained. Database:',state['database_stop']['outcome'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    actions=p.add_mutually_exclusive_group();actions.add_argument('--check',action='store_true');actions.add_argument('--stop',action='store_true')
    p.add_argument('--port',type=int,default=8000);p.add_argument('--db-port',type=int,default=55445)
    p.add_argument('--db-startup-timeout',type=int,default=300,help='Seconds allowed for PostgreSQL startup/crash recovery (10..1800).')
    p.add_argument('--open',action='store_true');p.add_argument('--component',choices=('web','worker','corrections'),help=argparse.SUPPRESS)
    args=p.parse_args();os.chdir(ROOT)
    if not 1<=args.port<=65535 or not 1<=args.db_port<=65535:p.error('Ports must be in 1..65535')
    if not 10<=args.db_startup_timeout<=1800:p.error('--db-startup-timeout must be in 10..1800')
    try:
        if args.stop:request_stop(timeout=180)
        elif args.component:component(args.component,args.port)
        else:run(args)
    except Exception as error:
        print('Profile service stopped:',str(error) if isinstance(error,(RuntimeError,ValueError)) else type(error).__name__,file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
