"""Fenced text-only extension worker; legacy queued proposals remain supported."""
from pathlib import Path
from types import SimpleNamespace
import os
from .service_profiles_v11.queue import Queue
from .service_profiles_v9.storage import Storage
from .web_tables_store_v1 import TableStore
from .web_tables_contract_v1 import binding as first
from .web_tables_contract_v2 import binding as second
from .web_tables_contract_v3 import binding as third
from .web_tables_contract_v4 import binding as fourth
from .runtime_worker_v1 import run_child
from .runtime_io_v1 import read_json, best_effort
from .runtime_cache_v1 import reap_dead_attempts


def process_once(settings, *, store=None, runner=None):
    queue, files = Queue(settings), Storage(settings); store = store or TableStore(queue)
    row = store.claim()
    if row is None:
        return False
    reap_dead_attempts(files, f"tables-v1/{row['id']}")
    folder = files.path(f"tables-v1/{row['id']}/{row['owner']}"); folder.mkdir(parents=True)
    try:
        document, versions = queue.job(row['job_id']); parent = queue.run(row['job_id'], row['parent_version'])
        if (max(v['version'] for v in versions) != row['parent_version']
                or parent['manifest']['sha256'] != row['request']['source_manifest_sha256']):
            raise ValueError('Parent version changed')
        selected = next((b for factory in (first, second, third, fourth)
                         if (b := factory())['sha256'] == row['request']['binding_sha256']), None)
        if selected is None:
            raise ValueError('Unsupported extension binding')
        request = dict(storage=str(files.root), folder=str(folder),
            parent={k: str(parent[k]) if k in ('id', 'job_id') else parent[k]
                    for k in ('id', 'job_id', 'version', 'status', 'options', 'manifest', 'summary')},
            document={k: str(document[k]) if k == 'id' else document[k]
                      for k in ('id', 'input_path', 'input_sha256')},
            page=row['page_index'], edit=row['request'], table_binding=selected, proposal_id=str(row['id']),
            models=os.environ.get('CAD_DEMO_OCR_MODELS', str(Path(__file__).resolve().parents[2]/'artifacts/models/ppocr-v5')))
        choice = row['request'].get('text_selection')
        if choice:
            if choice.get('source_kind') == 'published':
                if choice['proposal_id'] != str(parent['id']) or choice['preview_sha256'] != parent['manifest']['sha256']:
                    raise ValueError('Current text version changed')
                request.update(text_source_manifest=parent['manifest'], text_source_summary=parent['summary'])
            else:
                source = store.get(row['job_id'], row['parent_version'], row['page_index'], choice['proposal_id'])
                if source['status'] != 'prepared' or source['preview_manifest']['sha256'] != choice['preview_sha256']:
                    raise ValueError('Text source proposal changed')
                request.update(text_source_manifest=source['preview_manifest'], text_source_summary=source['summary'])
        native = selected['sha256'] == fourth()['sha256']
        limits = SimpleNamespace(timeout_seconds=300. if native else 900.,
            max_output_bytes=1536*1024**2, max_output_files=12000)
        budget = run_child(request, folder, files, limits,
            runner or Path(__file__).resolve().parents[2]/'scripts/build_web_tables_v5.py',
            lambda: store.heartbeat(row), memory_MiB=3072 if native else 6144)
        summary = read_json(folder/'completion.json', limit=16*1024*1024)
        # A new version arriving during computation cannot acquire publication.
        _, versions = queue.job(row['job_id'])
        if max(v['version'] for v in versions) != row['parent_version']:
            raise ValueError('Parent version changed during processing')
        files.manifest(parent); manifest = files.seal(folder, summary)
        budget.check(); budget.finish('completed'); store.finish(row, manifest=manifest, summary=summary)
    except Exception as error:
        code = getattr(error, 'code', 'table_processing_failed')
        best_effort(folder/'publication-error.json', dict(code=code, type=type(error).__name__))
        if code != 'lease_lost_or_cancelled':
            store.finish(row, error=code)
    return True
