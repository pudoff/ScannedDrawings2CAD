"""Independent heartbeat/cancellation and bounded child; publication is fenced."""
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import shutil
import sys
import time

from .dataset_layout_v1 import read, encoded
from .service_profiles_v11.queue import Queue
from .service_profiles_v9.storage import Storage
from .service_profiles_v2.budget import AttemptBudget
from .web_tables_store_v1 import TableStore
from .web_tables_contract_v2 import binding
from .web_tables_contract_v1 import binding as previous_binding
from .service_profiles_v11.worker import computation_running


def process_once(settings, *, store=None, runner=None):
    queue = Queue(settings); files = Storage(settings); store = store or TableStore(queue)
    row = store.claim()
    if row is None: return False
    folder = files.path(f"tables-v1/{row['id']}/{row['owner']}"); folder.mkdir(parents=True)
    process = budget = None
    try:
        document, versions = queue.job(row['job_id']); parent = queue.run(row['job_id'], row['parent_version'])
        if max(v['version'] for v in versions) != row['parent_version']: raise ValueError('parent_version_changed')
        if parent['manifest']['sha256'] != row['request']['source_manifest_sha256']: raise ValueError('parent_manifest_changed')
        parent = {**parent, 'id': str(parent['id']), 'job_id': str(parent['job_id'])}
        # Only fields used by the child; no datetime/lease tokens or DB secrets.
        parent = {k: parent[k] for k in ('id', 'job_id', 'version', 'status', 'options', 'manifest', 'summary')}
        document = {k: str(document[k]) if k == 'id' else document[k] for k in ('id', 'input_path', 'input_sha256')}
        legacy = row['request']['binding_sha256'] == previous_binding()['sha256']
        selected_binding = previous_binding() if legacy else binding()
        if row['request']['binding_sha256'] != selected_binding['sha256']: raise ValueError('Unsupported table binding')
        request = dict(storage=str(files.root), folder=str(folder), parent=parent, document=document,
            page=row['page_index'], edit=row['request'], table_binding=selected_binding, proposal_id=str(row['id']),
            models=os.environ.get('CAD_DEMO_OCR_MODELS', str(Path(__file__).resolve().parents[2]/'artifacts/models/ppocr-v5')))
        (folder/'request.json').write_bytes(encoded(request))
        script = Path(runner) if runner else Path(__file__).resolve().parents[2]/('scripts/build_web_tables_v1.py' if legacy else 'scripts/build_web_tables_v2.py')
        with (folder/'child.log').open('xb') as log:
            process = subprocess.Popen([sys.executable, str(script), '--request', str(folder/'request.json')],
                stdout=log, stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[2],
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'XDG_CACHE_HOME': str(folder/'.cache')})
            limits = SimpleNamespace(timeout_seconds=900., max_output_bytes=1536*1024**2, max_output_files=12000)
            budget = AttemptBudget(process, folder, files, limits, memory_MiB=6144)
            heartbeat = time.monotonic()
            while computation_running(process, budget, folder/'status.json'):
                budget.check()
                if time.monotonic() >= heartbeat:
                    if not store.heartbeat(row): raise ValueError('lease_lost_or_cancelled')
                    heartbeat = time.monotonic()+2
                time.sleep(.1)
        budget.check(); budget.stop(); budget.check()
        if process.returncode: raise ValueError('table_processing_failed')
        if read(folder/'status.json') != {'ok': True}: raise ValueError('Table producer did not complete')
        summary = read(folder/'completion.json')
        files.manifest(parent)
        manifest = files.seal(folder, summary)
        budget.check()
        store.finish(row, manifest=manifest, summary=summary)
    except Exception as error:
        if budget: budget.stop()
        elif process is not None and process.poll() is None: process.kill(); process.wait(timeout=5)
        (folder/'worker-error.json').write_bytes(encoded(dict(error=str(error)[:1000])))
        store.finish(row, error=str(error)[:200])
    finally:
        if budget: budget.stop()
        # A cancelled/killed child cannot clean its own native scratch files.
        for cache in (folder/'private', folder/'.cache'):
            if cache.exists():
                if (not cache.resolve().is_relative_to(folder.resolve()) or cache.is_symlink()
                        or any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in cache.rglob('*'))):
                    raise ValueError('Linked table cache; cleanup refused')
                shutil.rmtree(cache)
    return True
