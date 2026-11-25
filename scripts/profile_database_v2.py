"""Wait for crash recovery and never forcibly kill PostgreSQL on a startup timeout."""
from pathlib import Path
import os
import subprocess
import time

import run_demo as base

ROOT=base.ROOT;DATA=base.DATA;BIN=base.BIN;FLAGS=base.FLAGS


def verify_database(port):
    with base.db_connect(port) as connection:
        # Windows PostgreSQL can return this path in the system ANSI code page.
        raw = connection.execute("SELECT pg_catalog.textsend(current_setting('data_directory'))").fetchone()[0]
        try:
            directory = raw.decode('utf-8')
        except UnicodeDecodeError:
            if os.name != 'nt':
                raise
            directory = raw.decode('mbcs')
        if Path(directory).resolve() != DATA.resolve():
            raise RuntimeError('Database port belongs to another data directory; nothing was changed.')
        if connection.execute('SHOW listen_addresses').fetchone()[0] != '127.0.0.1':
            raise RuntimeError('The demo database must listen only on 127.0.0.1.')


def cluster_identity(port=None):
    try:
        lines=(DATA/'postmaster.pid').read_text().splitlines()
        return int(lines[0]),Path(lines[1]).resolve()==DATA.resolve() and (port is None or int(lines[3])==port)
    except (OSError,ValueError,IndexError):return None,False


def ready(port):
    # SQL connection probes during recovery create a new FATAL log entry each time.
    try:
        result=subprocess.run([str(BIN/'pg_isready.exe'),'-q','-h','127.0.0.1','-p',str(port),'-U','cad_demo','-d','postgres','-t','1'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=FLAGS,timeout=4,check=False)
    except subprocess.TimeoutExpired:return False
    return result.returncode==0


def wait_ready(port,process,*,timeout,cancelled=lambda:False):
    deadline=time.monotonic()+timeout;next_progress=time.monotonic()+10
    while time.monotonic()<deadline:
        if cancelled():raise RuntimeError('Database startup cancelled; waiting for safe shutdown.')
        if process is not None and process.poll() is not None:raise RuntimeError('PostgreSQL exited during startup; inspect postgres.log.')
        if ready(port):
            verify_database(port)
            return
        if time.monotonic()>=next_progress:
            print('PostgreSQL is starting/recovering; waiting for readiness (up to '+str(timeout)+'s).',flush=True)
            next_progress=time.monotonic()+10
        time.sleep(min(1.,max(0.,deadline-time.monotonic())))
    raise RuntimeError(f'PostgreSQL did not become ready in {timeout} seconds; inspect postgres.log. No forced database termination is used.')


def stop_owned_database(process,log,*,timeout=120):
    if process is None:return {'outcome':'reused_not_stopped'}
    if process.poll() is not None:return {'outcome':'already_exited','pid':process.pid}
    pid,matches=cluster_identity()
    if pid!=process.pid or not matches:return {'outcome':'identity_changed_not_signalled','pid':process.pid}
    print('Waiting for PostgreSQL to finish a safe shutdown...',flush=True)
    try:
        result=subprocess.run([str(BIN/'pg_ctl.exe'),'-D',str(DATA.resolve()),'-m','fast','-w','-t',str(timeout),'stop'],
            stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS,timeout=timeout+5,check=False)
        if result.returncode==0:
            return {'outcome':'stopped','pid':process.pid}
    except subprocess.TimeoutExpired:pass
    # A recovery fsync can outlast pg_ctl. Leave the cluster intact for the next run.
    print('PostgreSQL is still completing shutdown/recovery. It was left intact; inspect postgres.log before restarting.',flush=True)
    return {'outcome':'retained_finishing_shutdown','pid':process.pid}


def start_database(port,log,*,timeout=300,cancelled=lambda:False):
    if not 10<=timeout<=1800:raise ValueError('Database startup timeout must be in 10..1800 seconds.')
    DATA.parent.mkdir(parents=True,exist_ok=True)
    if not (DATA/'PG_VERSION').exists():
        if DATA.exists() and any(DATA.iterdir()):raise RuntimeError('Incomplete nonempty cluster preserved; inspect postgres.log.')
        subprocess.run([str(BIN/'initdb.exe'),'-D',str(DATA.resolve()),'-U','cad_demo','--auth-local=trust',
            '--auth-host=trust','-E','UTF8','--locale=C'],stdout=log,stderr=subprocess.STDOUT,
            creationflags=FLAGS,timeout=120,check=True)
    status=subprocess.run([str(BIN/'pg_ctl.exe'),'-D',str(DATA.resolve()),'status'],
        stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS,timeout=5,check=False)
    if status.returncode==0:
        _,matches=cluster_identity(port)
        if not matches:raise RuntimeError('Expected cluster is running on a different port; use its original --db-port. Nothing was signalled.')
        wait_ready(port,None,timeout=timeout,cancelled=cancelled)
        return None
    base.available_port(port)
    process=subprocess.Popen([str(BIN/'postgres.exe'),'-D',str(DATA.resolve()),'-h','127.0.0.1','-p',str(port)],
        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=FLAGS)
    try:
        wait_ready(port,process,timeout=timeout,cancelled=cancelled)
        return process
    except BaseException as error:
        error.database_shutdown=stop_owned_database(process,log)
        raise
