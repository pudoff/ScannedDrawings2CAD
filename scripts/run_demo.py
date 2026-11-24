"""Start the isolated local presentation service; stop only processes we own."""
from __future__ import annotations

import argparse
from datetime import datetime
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT/'artifacts/demo-runtime-v1'
DATA = RUNTIME/'postgres/data'
BIN = ROOT/'artifacts/service-postgres-v1/pgsql/bin'
MODELS = ROOT/'artifacts/models/ppocr-v5'
DEPS = ROOT/'artifacts/service-runtime-v1/deps'
FLAGS = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
sys.path.insert(0, str(DEPS))
sys.path.insert(0, str(ROOT/'src'))


def atomic_control(path, value):
    """Only small launcher-owned JSON files are replaced, inside this workspace."""
    if not path.resolve().is_relative_to(RUNTIME.resolve()):
        raise ValueError('Control file must remain inside the demo runtime.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_active():
    path = RUNTIME/'active-launcher.json'
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(state, dict) or state.get('schema_version') != 'demo-launcher/1.0'
            or not isinstance(state.get('id'), str) or not re.fullmatch(r'[0-9a-f]{32}', state['id'])
            or state.get('status') not in ('starting', 'running', 'stopped')):
        raise RuntimeError('Invalid launcher state; inspect active-launcher.json before restarting.')
    return state


def stop_path(identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r'[0-9a-f]{32}', identifier):
        raise ValueError('Invalid launcher identifier.')
    return RUNTIME/'control'/f'stop-{identifier}.json'


def stop_requested(identifier):
    path = stop_path(identifier)
    try:
        request = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return isinstance(request, dict) and request.get('id') == identifier and request.get('action') == 'stop'


def request_stop(timeout=30):
    state = read_active()
    if state is None or state['status'] == 'stopped':
        print('No active launcher is recorded. No process was signalled.')
        return
    identifier = state['id']
    atomic_control(stop_path(identifier), {'id': identifier, 'action': 'stop'})
    print('Stop requested for the recorded launcher. Waiting for owned-process shutdown...', flush=True)
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        current = read_active()
        if current is None or current['id'] != identifier:
            raise RuntimeError('The launcher identity changed while waiting; no other launcher was stopped.')
        if current['status'] == 'stopped':
            print('Stopped. Documents and results are retained.', flush=True)
            return
        time.sleep(.2)
    raise RuntimeError('The launcher did not acknowledge shutdown in 30 seconds. Inspect its terminal/logs; no process was forcibly killed.')


@contextmanager
def launcher_lock():
    """An OS-owned lock prevents concurrent supervisors from replacing state."""
    import msvcrt
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME/'launcher.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('A demo launcher is already active. Use scripts/run_demo.py --stop first.') from None
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def preflight():
    if os.name != 'nt':
        raise RuntimeError('This launcher uses the supplied Windows PostgreSQL runtime.')
    if not DEPS.is_dir():
        raise RuntimeError('Missing artifacts/service-runtime-v1/deps; see docs/demo-run-guide.md.')
    for name in ('postgres.exe', 'initdb.exe', 'pg_ctl.exe'):
        if not (BIN/name).is_file():
            raise RuntimeError(f'Missing PostgreSQL binary: {BIN/name}')
    os.environ.setdefault('XDG_CACHE_HOME', str(ROOT/'artifacts/.cache'))
    for name in ('fastapi', 'uvicorn', 'psycopg', 'ezdxf', 'cv2', 'PIL', 'pypdf', 'pypdfium2', 'fitz', 'rapidocr', 'onnxruntime'):
        try:
            importlib.import_module(name)
        except Exception as error:
            raise RuntimeError(f'Cannot import {name} ({type(error).__name__}); see docs/demo-run-guide.md.') from None
    from cad_recovery.ocr_engine import _preflight
    _preflight(MODELS.resolve())
    for filename in ('src/cad_recovery/web_demo_v1/assets/index.html',
                     'src/cad_recovery/web_demo_v1/assets/app.js',
                     'scripts/service_demo_pipeline_v1.py'):
        if not (ROOT/filename).is_file():
            raise RuntimeError(f'Missing demo component: {filename}')


def available_port(port):
    try:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    except OSError:
        raise RuntimeError(f'Local port {port} is busy; select another port or stop its service.') from None


def db_connect(port, *, database='postgres'):
    import psycopg
    return psycopg.connect(host='127.0.0.1', port=port, user='cad_demo', dbname=database,
                          connect_timeout=2, autocommit=True)


def verify_database(port):
    with db_connect(port) as connection:
        actual = Path(connection.execute('SHOW data_directory').fetchone()[0]).resolve()
        if actual != DATA.resolve():
            raise RuntimeError('Database port belongs to another data directory; nothing was changed.')
        if connection.execute('SHOW listen_addresses').fetchone()[0] != '127.0.0.1':
            raise RuntimeError('The demo database must listen only on 127.0.0.1.')


def stop_child(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def stop_owned_database(process, log):
    if process is None or process.poll() is not None:
        return
    # pg_ctl must never target a server that replaced our original process.
    pidfile = DATA/'postmaster.pid'
    try:
        owned = int(pidfile.read_text().splitlines()[0]) == process.pid
    except (OSError, ValueError, IndexError):
        owned = False
    if owned:
        try:
            subprocess.run([str(BIN/'pg_ctl.exe'), '-D', str(DATA.resolve()), '-m', 'fast',
                            '-w', '-t', '15', 'stop'], stdout=log, stderr=subprocess.STDOUT,
                           creationflags=FLAGS, timeout=20, check=False)
        except subprocess.TimeoutExpired:
            pass
    stop_child(process)


def start_database(port, log):
    # An existing expected cluster may be reused, but is never owned by this run.
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=.5):
            listening = True
    except OSError:
        listening = False
    if listening:
        try:
            verify_database(port)
        except Exception:
            raise RuntimeError(f'Port {port} is occupied by an unverified database/service; choose --db-port.') from None
        return None

    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not (DATA/'PG_VERSION').exists():
        if DATA.exists() and any(DATA.iterdir()):
            raise RuntimeError('The new cluster directory is nonempty but incomplete; preserve it and inspect the startup log.')
        subprocess.run([str(BIN/'initdb.exe'), '-D', str(DATA.resolve()), '-U', 'cad_demo',
                        '--auth-local=trust', '--auth-host=trust', '-E', 'UTF8', '--locale=C'],
                       stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS, timeout=60, check=True)
    status = subprocess.run([str(BIN/'pg_ctl.exe'), '-D', str(DATA.resolve()), 'status'],
                            stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS, timeout=5)
    if status.returncode == 0:
        raise RuntimeError('The demo cluster is already running on another port; use its original --db-port.')
    available_port(port)
    process = subprocess.Popen([str(BIN/'postgres.exe'), '-D', str(DATA.resolve()),
                                '-h', '127.0.0.1', '-p', str(port)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=FLAGS)
    try:
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('PostgreSQL stopped during startup; inspect postgres.log.')
            try:
                verify_database(port)
                return process
            except Exception:
                time.sleep(.2)
        raise RuntimeError('PostgreSQL did not become ready in 30 seconds; inspect postgres.log.')
    except BaseException:
        stop_owned_database(process, log)
        raise


def initialize_database(port):
    from psycopg import sql
    with db_connect(port) as connection:
        if not connection.execute('SELECT 1 FROM pg_database WHERE datname=%s', ('cad_recovery_demo',)).fetchone():
            connection.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier('cad_recovery_demo')))
    from cad_recovery.service_demo_v1.config import Settings
    from cad_recovery.service_v1.queue import Queue
    from cad_recovery.web_v1.reviews import Reviews
    settings = Settings.environment()
    queue = Queue(settings)
    queue.initialize()
    queue.bind_storage()
    Reviews(settings).initialize()


def component(mode, port):
    from cad_recovery.service_demo_v1.config import Settings
    settings = Settings.environment()
    if mode == 'web':
        import uvicorn
        from cad_recovery.web_demo_v1.app import create_web_app
        uvicorn.run(create_web_app(settings), host='127.0.0.1', port=port,
                    access_log=False, server_header=False, timeout_keep_alive=5, limit_concurrency=16)
    else:
        from cad_recovery.service_demo_v1.worker import process_once
        while True:
            if not process_once(settings):
                time.sleep(.5)


def run_server(args):
    available_port(args.port)
    if args.port == args.db_port:
        raise RuntimeError('--port and --db-port must be different.')
    os.environ.update(CAD_SERVICE_DSN=f'postgresql://cad_demo@127.0.0.1:{args.db_port}/cad_recovery_demo',
                      CAD_SERVICE_STORAGE=str(RUNTIME/'storage'),
                      CAD_SERVICE_TIMEOUT_SECONDS='900',
                      CAD_SERVICE_OCR_MODELS=str(MODELS), CAD_DEMO_OCR_MODELS=str(MODELS),
                      CAD_SERVICE_DEPS=str(DEPS), PYTHONIOENCODING='utf-8')
    logs = RUNTIME/'logs'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
    logs.mkdir(parents=True)
    identifier = uuid4().hex
    state = {'schema_version': 'demo-launcher/1.0', 'id': identifier, 'status': 'starting',
             'pid': os.getpid(), 'port': args.port, 'db_port': args.db_port,
             'logs': str(logs.relative_to(RUNTIME))}
    pg = web = worker = None
    with (logs/'postgres.log').open('xb') as pg_log, (logs/'web.log').open('xb') as web_log, (logs/'worker.log').open('xb') as worker_log:
        try:
            atomic_control(RUNTIME/'active-launcher.json', state)
            print('Starting the isolated local demo. Logs:', logs, flush=True)
            pg = start_database(args.db_port, pg_log)
            initialize_database(args.db_port)
            command = [sys.executable, str(Path(__file__).resolve()), '--port', str(args.port), '--component']
            web = subprocess.Popen(command+['web'], cwd=ROOT, stdout=web_log, stderr=subprocess.STDOUT, creationflags=FLAGS)
            worker = subprocess.Popen(command+['worker'], cwd=ROOT, stdout=worker_log, stderr=subprocess.STDOUT, creationflags=FLAGS)
            url = f'http://127.0.0.1:{args.port}'
            headers = {'Authorization': 'Bearer '+os.environ['CAD_SERVICE_TOKEN']} if os.environ.get('CAD_SERVICE_TOKEN') else {}
            deadline = time.monotonic()+40
            while time.monotonic() < deadline:
                if stop_requested(identifier):
                    print('Stop request received during startup.', flush=True)
                    return
                if web.poll() is not None or worker.poll() is not None:
                    raise RuntimeError('A service component stopped during startup; inspect web.log and worker.log.')
                try:
                    with urllib.request.urlopen(urllib.request.Request(url+'/health', headers=headers), timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    time.sleep(.2)
            else:
                raise RuntimeError('The HTTP service did not become ready in 40 seconds.')
            print(f'Ready: {url}', flush=True)
            state['status'] = 'running'
            atomic_control(RUNTIME/'active-launcher.json', state)
            if os.environ.get('CAD_SERVICE_TOKEN'):
                print('An access key is enabled; enter the existing CAD_SERVICE_TOKEN in the browser.', flush=True)
            print('Keep this terminal open. Stop with Ctrl+C or scripts/run_demo.py --stop from another terminal.', flush=True)
            if args.open:
                import webbrowser
                webbrowser.open(url)
            while True:
                if stop_requested(identifier):
                    print('Stop request received.', flush=True)
                    break
                if web.poll() is not None:
                    raise RuntimeError('Web server stopped; inspect web.log and restart the launcher.')
                if worker.poll() is not None:
                    raise RuntimeError('Worker stopped; inspect worker.log and restart the launcher.')
                if pg is not None and pg.poll() is not None:
                    raise RuntimeError('PostgreSQL stopped; inspect postgres.log and restart the launcher.')
                time.sleep(.5)
        except KeyboardInterrupt:
            print('\nStopping the local demo...', flush=True)
        finally:
            stop_child(worker)
            stop_child(web)
            stop_owned_database(pg, pg_log)
            state['status'] = 'stopped'
            atomic_control(RUNTIME/'active-launcher.json', state)
            print('Owned processes stopped. Documents and logs retained:', logs, flush=True)


def run(args):
    preflight()
    if args.check:
        print('OK: Python dependencies, local OCR hashes, PostgreSQL binaries and demo components.')
        print('No servers started. Launch: .venv/Scripts/python.exe scripts/run_demo.py')
        return
    with launcher_lock():
        run_server(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true', help='Check local runtime and OCR model hashes without starting servers')
    actions.add_argument('--stop', action='store_true', help='Ask the recorded launcher to stop its own processes')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--db-port', type=int, default=55445)
    parser.add_argument('--open', action='store_true', help='Open the local interface in your default browser')
    parser.add_argument('--component', choices=('web', 'worker'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.db_port <= 65535:
        parser.error('Ports must be between 1 and 65535.')
    os.chdir(ROOT)
    try:
        if args.stop:
            request_stop()
        elif args.component:
            component(args.component, args.port)
        else:
            run(args)
    except Exception as error:
        # A database/HTTP exception may carry credentials; do not echo its text.
        message = str(error) if type(error) is RuntimeError else f'{type(error).__name__}; inspect the local startup logs.'
        print('Startup stopped:', message, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
