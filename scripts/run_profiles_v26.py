"""Local V26: compatible PNG input with retained bounded V24 workers/UI/documents."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime
import os
import subprocess
import sys
import time
import urllib.request
from uuid import uuid4

import run_profiles_v13 as previous
from run_demo import ROOT, RUNTIME, MODELS, FLAGS, preflight, available_port, launcher_lock, stop_child, atomic_control, stop_requested
from profile_database_v2 import start_database, stop_owned_database


def component(mode, port):
    from cad_recovery.service_profiles_v2.config import Settings
    settings = replace(Settings.environment(), lease_seconds=90)
    if mode == 'web':
        import uvicorn
        from cad_recovery.service_profiles_v21.api import create_web_app
        uvicorn.run(create_web_app(settings), host='127.0.0.1', port=port, access_log=False,
                    server_header=False, timeout_keep_alive=5, limit_concurrency=16)
    elif mode == 'tables':
        from cad_recovery.correction_loop_v1 import run
        from cad_recovery.web_tables_worker_v9 import process_once
        from cad_recovery.service_profiles_v20.queue import Queue
        from cad_recovery.web_tables_store_v2 import TableStore
        store = TableStore(Queue(settings)); store.initialize()
        run(settings, process=lambda configured: process_once(configured, store=store),
            label='Manual table review', sleep=lambda s: time.sleep(max(2., s)))
    elif mode == 'worker':
        from cad_recovery.service_profiles_v20.worker import process_once
        from cad_recovery.correction_loop_v1 import run
        runner=ROOT/'scripts/service_profile_diagnostic_v6.py'
        run(settings,process=lambda configured:process_once(configured,runner=runner),
            label='Runtime-safe demo',sleep=lambda s:time.sleep(max(2.,s)))
    else: previous.component(mode, port)


def run(args):
    preflight()
    from cad_recovery.profile_contract_v8 import DEFINITIONS, resolve_profile, verify_models, verify_cv_runtime
    from cad_recovery.web_tables_contract_v6 import binding
    for name in DEFINITIONS: verify_models(resolve_profile(name), MODELS)
    verify_cv_runtime(); binding()
    if args.check:
        print('OK: compatible PNG input, retained CV/OCR/export and profile-bound table registration and cumulative text publication verified. No services started.'); return
    with launcher_lock():
        available_port(args.port)
        if args.port == args.db_port: raise ValueError('HTTP and database ports must differ')
        os.environ.update(CAD_SERVICE_DSN=f'postgresql://cad_demo@127.0.0.1:{args.db_port}/cad_recovery_profiles_v1',
            CAD_SERVICE_STORAGE=str(ROOT/'artifacts/profile-service-v1/storage'), CAD_SERVICE_TIMEOUT_SECONDS='900',
            CAD_DEMO_OCR_MODELS=str(MODELS), PYTHONIOENCODING='utf-8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
        logs = ROOT/'artifacts/profile-service-v26/logs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
        logs.mkdir(parents=True)
        identifier = uuid4().hex
        state = dict(schema_version='demo-launcher/1.0', id=identifier, status='starting', pid=os.getpid(),
                     port=args.port, db_port=args.db_port, logs=str(logs.relative_to(ROOT)), service='shared-profiles/26.0')
        processes = []; pg = None; database_result = None
        with ExitStack() as stack:
            pglog = stack.enter_context((logs/'postgres.log').open('xb'))
            try:
                atomic_control(RUNTIME/'active-launcher.json', state)
                print('Starting shared service with manual tables. Logs:', logs, flush=True)
                pg = start_database(args.db_port, pglog, timeout=args.db_startup_timeout, cancelled=lambda: stop_requested(identifier))
                if stop_requested(identifier): return
                previous.initialize_database(args.db_port)
                for mode in ('web', 'worker', 'corrections', 'tables'):
                    log = stack.enter_context((logs/(mode+'.log')).open('xb'))
                    process = subprocess.Popen([sys.executable, str(ROOT/'scripts/run_profiles_v26.py'),
                        '--port', str(args.port), '--component', mode], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)
                    processes.append(process)
                url = f'http://127.0.0.1:{args.port}'
                headers = {'Authorization': 'Bearer '+os.environ['CAD_SERVICE_TOKEN']} if os.environ.get('CAD_SERVICE_TOKEN') else {}
                deadline = time.monotonic()+40
                while time.monotonic() < deadline:
                    if stop_requested(identifier): return
                    if any(p.poll() is not None for p in processes): raise RuntimeError('A service component stopped; inspect logs')
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url+'/health', headers=headers), timeout=2) as response:
                            if response.status == 200: break
                    except OSError: time.sleep(.2)
                else: raise RuntimeError('HTTP service did not become ready')
                state['status'] = 'running'; atomic_control(RUNTIME/'active-launcher.json', state)
                print('Ready:', url, flush=True)
                if args.open:
                    import webbrowser
                    webbrowser.open(url)
                while not stop_requested(identifier):
                    if any(p.poll() is not None for p in processes) or pg is not None and pg.poll() is not None:
                        raise RuntimeError('A service process stopped; inspect logs')
                    time.sleep(.5)
            except KeyboardInterrupt as error: database_result = getattr(error, 'database_shutdown', None)
            except Exception as error:
                database_result = getattr(error, 'database_shutdown', None); raise
            finally:
                for process in reversed(processes): stop_child(process)
                state['database_stop'] = database_result or stop_owned_database(pg, pglog)
                state['status'] = 'stopped'; atomic_control(RUNTIME/'active-launcher.json', state)
                print('Service stopped. Documents retained. Database:', state['database_stop']['outcome'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group(); actions.add_argument('--check', action='store_true'); actions.add_argument('--stop', action='store_true')
    p.add_argument('--port', type=int, default=8000); p.add_argument('--db-port', type=int, default=55445)
    p.add_argument('--db-startup-timeout', type=int, default=300); p.add_argument('--open', action='store_true')
    p.add_argument('--component', choices=('web', 'worker', 'corrections', 'tables'), help=argparse.SUPPRESS)
    a = p.parse_args(); os.chdir(ROOT)
    if not 1 <= a.port <= 65535 or not 1 <= a.db_port <= 65535: p.error('Ports must be in1..65535')
    if not 10 <= a.db_startup_timeout <= 1800: p.error('Startup timeout must be10..1800 seconds')
    if a.component: component(a.component, a.port)
    elif a.stop: previous.request_stop()
    else: run(a)
