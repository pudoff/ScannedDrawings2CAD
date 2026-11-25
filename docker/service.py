"""Linux entry points; Compose owns PostgreSQL and process supervision."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import time

from cad_recovery.compact_runtime_v1 import activate, verify_environment

activate()
from cad_recovery.service_profiles_v2.config import Settings
from cad_recovery.service_profiles_v21.queue import Queue

ROOT = Path(__file__).resolve().parents[1]


class LocalOperator:
    """Preserve desktop operator permissions behind Docker's localhost port NAT.

    Enable only with a host port bound to 127.0.0.1, as in compose.yaml.
    Forwarded client headers are deliberately ignored by Uvicorn.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            scope = {**scope, 'client': ('127.0.0.1', 0)}
        await self.app(scope, receive, send)


def check():
    verify_environment()
    from run_clean_demo_v1 import check_workers, check_uploads, check_phases
    from cad_recovery.compact_web_api_v2 import check_assets
    from cad_recovery.ocr_engine import _preflight
    check_workers()
    check_uploads()
    check_phases()
    check_assets()
    _preflight(ROOT / 'artifacts/models/ppocr-v5')
    print('OK: Linux runtime, Web V31, workers, models and assets verified.', flush=True)


def main(mode):
    if mode == 'check':
        check()
        return
    settings = replace(Settings.environment(), lease_seconds=90)
    if mode == 'init':
        from cad_recovery.web_v1.reviews import Reviews
        from cad_recovery.web_correction_store_v2 import CorrectionStore
        from cad_recovery.web_tables_store_v2 import TableStore
        queue = Queue(settings)
        queue.initialize()
        queue.bind_storage()
        Reviews(settings).initialize()
        CorrectionStore(queue).initialize()
        TableStore(queue).initialize()
        print('Database and document storage initialized.', flush=True)
        return
    if mode == 'web':
        import uvicorn
        from cad_recovery.compact_web_api_v2 import create_web_app
        app = create_web_app(settings)
        if os.environ.get('CAD_DOCKER_LOCAL_OPERATOR') == '1':
            app = LocalOperator(app)
        uvicorn.run(app, host='0.0.0.0', port=8000, proxy_headers=False,
                    server_header=False, timeout_keep_alive=5, limit_concurrency=16)
        return
    from cad_recovery.correction_loop_v1 import run
    runner = ROOT / 'scripts/compact_child_v1.py'
    if mode == 'worker':
        from cad_recovery.service_profiles_v22.worker import process_once
        process = lambda configured: process_once(configured, runner=runner)
    elif mode == 'corrections':
        from cad_recovery.web_correction_worker_v2 import process_once
        from cad_recovery.web_correction_store_v2 import CorrectionStore
        store = CorrectionStore(Queue(settings))
        process = lambda configured: process_once(configured, store=store, runner=runner)
    else:
        from cad_recovery.web_tables_worker_v11 import process_once
        from cad_recovery.web_tables_store_v2 import TableStore
        store = TableStore(Queue(settings))
        process = lambda configured: process_once(configured, store=store, runner=runner)
    run(settings, process=process, label=mode, sleep=lambda seconds: time.sleep(max(2., seconds)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'init', 'web', 'worker', 'corrections', 'tables'])
    main(parser.parse_args().mode)
