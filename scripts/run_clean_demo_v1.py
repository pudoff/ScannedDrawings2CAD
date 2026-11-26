"""Local compact demo with Web V32 DXF display and retained V31 processing."""
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

from cad_recovery.compact_runtime_v1 import activate
activate()
import run_profiles_v13 as previous
from run_demo import ROOT, RUNTIME, MODELS, FLAGS, preflight, available_port, launcher_lock, stop_child, atomic_control, stop_requested
from profile_database_v2 import start_database, stop_owned_database
from cad_recovery.compact_preflight_v1 import preflight


def component(mode, port):
    from cad_recovery.service_profiles_v2.config import Settings
    settings = replace(Settings.environment(), lease_seconds=90)
    if mode == 'web':
        import uvicorn
        from cad_recovery.compact_web_api_v3 import create_web_app
        uvicorn.run(create_web_app(settings), host='127.0.0.1', port=port, access_log=False,
                    server_header=False, timeout_keep_alive=5, limit_concurrency=16)
    elif mode == 'tables':
        from cad_recovery.correction_loop_v1 import run
        from cad_recovery.web_tables_worker_v11 import process_once
        from cad_recovery.service_profiles_v21.queue import Queue
        from cad_recovery.web_tables_store_v2 import TableStore
        store = TableStore(Queue(settings)); store.initialize()
        run(settings, process=lambda configured: process_once(configured, store=store, runner=ROOT/'scripts/compact_child_v1.py'),
            label='Manual table review', sleep=lambda s: time.sleep(max(2., s)))
    elif mode == 'worker':
        from cad_recovery.service_profiles_v22.worker import process_once
        from cad_recovery.correction_loop_v1 import run
        runner=ROOT/'scripts/compact_child_v1.py'
        run(settings,process=lambda configured:process_once(configured,runner=runner),
            label='Runtime-safe demo',sleep=lambda s:time.sleep(max(2.,s)))
    elif mode == 'corrections':
        from cad_recovery.correction_loop_v1 import run
        from cad_recovery.service_profiles_v21.queue import Queue
        from cad_recovery.web_correction_store_v2 import CorrectionStore
        from cad_recovery.web_correction_worker_v2 import process_once
        store=CorrectionStore(Queue(settings));store.initialize()
        run(settings,process=lambda configured:process_once(configured,store=store,runner=ROOT/'scripts/compact_child_v1.py'),
            label='Budgeted engineering corrections',sleep=lambda s:time.sleep(max(2.,s)))
    else: raise ValueError('Unknown service component')


def check_workers():
    from importlib import import_module
    from cad_recovery.worker_attempt_v4 import execute_attempt
    from cad_recovery import worker_attempt_v4,runtime_worker_v3
    from cad_recovery.durable_attempt_budget_v3 import AttemptBudget
    for name in ('cad_recovery.service_profiles_v22.worker','cad_recovery.web_tables_worker_v11','cad_recovery.web_correction_worker_v2'):
        if import_module(name).execute_attempt is not execute_attempt:
            raise ValueError('Worker uses a different attempt engine: '+name)
    if worker_attempt_v4.run_child is not runtime_worker_v3.run_child or runtime_worker_v3.AttemptBudget is not AttemptBudget:
        raise ValueError('Worker runtime budget differs')


def check_uploads():
    from cad_recovery.service_profiles_v24 import api
    from cad_recovery.service_profiles_v10.storage import Storage
    from cad_recovery import upload_io_v1
    if api.Storage is not Storage or Storage.upload.__globals__['upload'] is not upload_io_v1.upload:
        raise ValueError('Web/API upload storage differs from the qualified lifecycle')


def check_phases():
    from cad_recovery import child_lifecycle_v1,sealed_child_v3,sealed_child_v2
    from cad_recovery.runtime_progress_v4 import Progress
    import service_profile_diagnostic_v8 as profile
    import service_profile_diagnostic_v7 as old_profile
    import build_web_tables_v11 as tables
    import build_web_correction_v3 as corrections
    import build_web_correction_v2 as old_corrections
    from cad_recovery.table_build_dispatch_v4 import build_table
    if (child_lifecycle_v1.Progress is not Progress or
            sealed_child_v3.correlated_child is not child_lifecycle_v1.child_main or
            sealed_child_v3.finalize is not sealed_child_v2.finalize):
        raise ValueError('Child phase lifecycle or seal differs')
    if (any(m.child_main is not sealed_child_v3.child_main for m in (profile,tables,corrections)) or
            profile.execute is not old_profile.execute or tables.build_table is not build_table or
            corrections.operation is not old_corrections.operation):
        raise ValueError('Child operation/entry binding differs')


def run(args):
    preflight()
    check_workers()
    check_uploads()
    check_phases()
    from cad_recovery.compact_web_api_v3 import check_assets
    check_assets()
    from cad_recovery.profile_contract_v9 import DEFINITIONS, resolve_profile, verify_models, verify_cv_runtime
    from cad_recovery.web_tables_contract_v7 import binding
    for name in DEFINITIONS: verify_models(resolve_profile(name), MODELS)
    verify_cv_runtime(); binding()
    if args.check:
        print('OK: compact Web V32 legible DXF display, retained workers and CV/OCR/CAD verified. No services started.'); return
    with launcher_lock():
        available_port(args.port)
        if args.port == args.db_port: raise ValueError('HTTP and database ports must differ')
        os.environ.update(CAD_SERVICE_DSN=f'postgresql://cad_demo@127.0.0.1:{args.db_port}/cad_recovery_profiles_v1',
            CAD_SERVICE_STORAGE=str(ROOT/'artifacts/profile-service-v1/storage'), CAD_SERVICE_TIMEOUT_SECONDS='900',
            CAD_DEMO_OCR_MODELS=str(MODELS), PYTHONIOENCODING='utf-8', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
        logs = ROOT/'artifacts/profile-service-clean-v1/logs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
        logs.mkdir(parents=True)
        identifier = uuid4().hex
        state = dict(schema_version='demo-launcher/1.0', id=identifier, status='starting', pid=os.getpid(),
                     port=args.port, db_port=args.db_port, logs=str(logs.relative_to(ROOT)), service='clean-profiles/1.0')
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
                    process = subprocess.Popen([sys.executable, str(ROOT/'scripts/run_clean_demo_v1.py'),
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
