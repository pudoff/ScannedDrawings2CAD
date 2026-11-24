"""Run the shared-profile service using the existing local PostgreSQL runtime."""
import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
import time
import urllib.request
from uuid import uuid4

# Reuse the immutable, tested ownership/stop/start primitives; no module patching.
from run_demo import (ROOT,RUNTIME,MODELS,DEPS,FLAGS,preflight,available_port,
    launcher_lock,start_database,stop_owned_database,stop_child,atomic_control,
    stop_requested,request_stop,db_connect)


def initialize_database(port):
    from psycopg import sql
    with db_connect(port) as connection:
        if not connection.execute('SELECT 1 FROM pg_database WHERE datname=%s',('cad_recovery_profiles_v1',)).fetchone():
            connection.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier('cad_recovery_profiles_v1')))
    from cad_recovery.service_demo_v1.config import Settings
    from cad_recovery.service_profiles_v1.queue import Queue
    from cad_recovery.web_v1.reviews import Reviews
    settings=Settings.environment();queue=Queue(settings)
    queue.initialize();queue.bind_storage();Reviews(settings).initialize()


def component(mode,port):
    from cad_recovery.service_demo_v1.config import Settings
    settings=Settings.environment()
    if mode=='web':
        import uvicorn
        from cad_recovery.service_profiles_v1.app import create_web_app
        uvicorn.run(create_web_app(settings),host='127.0.0.1',port=port,access_log=False,
            server_header=False,timeout_keep_alive=5,limit_concurrency=16)
    else:
        from cad_recovery.service_profiles_v1.worker import process_once
        while True:
            if not process_once(settings):time.sleep(.5)


def run(args):
    preflight()
    from cad_recovery.profile_contract_v1 import DEFINITIONS,resolve_profile,verify_models
    for name in DEFINITIONS:verify_models(resolve_profile(name),MODELS)
    if args.check:
        print('OK: all shared profile/code/model/dependency bindings verified. No services started.');return
    with launcher_lock():
        available_port(args.port)
        if args.port==args.db_port:raise RuntimeError('HTTP and database ports must differ.')
        os.environ.update(CAD_SERVICE_DSN=f'postgresql://cad_demo@127.0.0.1:{args.db_port}/cad_recovery_profiles_v1',
            CAD_SERVICE_STORAGE=str(ROOT/'artifacts/profile-service-v1/storage'),
            CAD_SERVICE_TIMEOUT_SECONDS='900',CAD_DEMO_OCR_MODELS=str(MODELS),
            CAD_SERVICE_DEPS=str(DEPS),PYTHONIOENCODING='utf-8')
        logs=ROOT/'artifacts/profile-service-v1/logs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
        logs.mkdir(parents=True)
        identifier=uuid4().hex
        state={'schema_version':'demo-launcher/1.0','id':identifier,'status':'starting','pid':os.getpid(),
            'port':args.port,'db_port':args.db_port,'logs':str(logs.relative_to(ROOT)),'service':'shared-profiles/1.0'}
        pg=web=worker=None
        with (logs/'postgres.log').open('xb') as pglog,(logs/'web.log').open('xb') as weblog,(logs/'worker.log').open('xb') as workerlog:
            try:
                atomic_control(RUNTIME/'active-launcher.json',state)
                print('Starting shared-profile service. Logs:',logs,flush=True)
                pg=start_database(args.db_port,pglog);initialize_database(args.db_port)
                command=[sys.executable,str(ROOT/'scripts/run_profiles_v1.py'),'--port',str(args.port),'--component']
                web=subprocess.Popen(command+['web'],cwd=ROOT,stdout=weblog,stderr=subprocess.STDOUT,creationflags=FLAGS)
                worker=subprocess.Popen(command+['worker'],cwd=ROOT,stdout=workerlog,stderr=subprocess.STDOUT,creationflags=FLAGS)
                url=f'http://127.0.0.1:{args.port}'
                headers={'Authorization':'Bearer '+os.environ['CAD_SERVICE_TOKEN']} if os.environ.get('CAD_SERVICE_TOKEN') else {}
                deadline=time.monotonic()+40
                while time.monotonic()<deadline:
                    if stop_requested(identifier):return
                    if web.poll() is not None or worker.poll() is not None:raise RuntimeError('Service component stopped; inspect logs.')
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url+'/health',headers=headers),timeout=2) as response:
                            if response.status==200:break
                    except OSError:time.sleep(.2)
                else:raise RuntimeError('HTTP service did not become ready.')
                state['status']='running';atomic_control(RUNTIME/'active-launcher.json',state)
                print('Ready:',url,flush=True)
                print('Stop with Ctrl+C or scripts/run_profiles_v1.py --stop. Historical demo data is in its separate database.',flush=True)
                if args.open:
                    import webbrowser
                    webbrowser.open(url)
                while not stop_requested(identifier):
                    if web.poll() is not None or worker.poll() is not None or (pg is not None and pg.poll() is not None):
                        raise RuntimeError('A service process stopped; inspect logs.')
                    time.sleep(.5)
            except KeyboardInterrupt:pass
            finally:
                stop_child(worker);stop_child(web);stop_owned_database(pg,pglog)
                state['status']='stopped';atomic_control(RUNTIME/'active-launcher.json',state)
                print('Owned processes stopped. Profile-service documents retained.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    actions=p.add_mutually_exclusive_group();actions.add_argument('--check',action='store_true');actions.add_argument('--stop',action='store_true')
    p.add_argument('--port',type=int,default=8000);p.add_argument('--db-port',type=int,default=55445)
    p.add_argument('--open',action='store_true');p.add_argument('--component',choices=('web','worker'),help=argparse.SUPPRESS)
    args=p.parse_args();os.chdir(ROOT)
    if not 1<=args.port<=65535 or not 1<=args.db_port<=65535:p.error('Ports must be in 1..65535')
    try:
        if args.stop:request_stop()
        elif args.component:component(args.component,args.port)
        else:run(args)
    except Exception as error:
        print('Profile service stopped:',str(error) if isinstance(error,(RuntimeError,ValueError)) else type(error).__name__,file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
