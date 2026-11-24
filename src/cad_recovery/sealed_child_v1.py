"""Seal artifacts inside the supervised computation process, before success."""
from pathlib import Path
from types import SimpleNamespace
from .runtime_worker_v2 import child_main as correlated_child
from .runtime_io_v1 import read_json, atomic_json
from .service_profiles_v9.storage import Storage
from .profile_contract_v6 import ProfileError


def finalize(request, progress, *, seal=None):
    """A blocked hash/fsync is owned by the same killable process as CV/OCR."""
    table = 'folder' in request
    folder = Path(request['folder']) if table else Path(request['output']).parent
    root = Path(request['_artifact_storage'])
    files = Storage(SimpleNamespace(root=root))
    if files.path(folder.relative_to(root).as_posix()) != folder:
        raise ValueError('Attempt differs from owned storage')
    with progress.phase('artifact_seal'):
        if table:
            summary = read_json(folder/'completion.json', limit=16*1024*1024)
        else:
            report = read_json(folder/'output/report.json', limit=16*1024*1024)
            if report['processing_profile'] != request['profile_binding']:
                raise ProfileError('profile_contract_changed', 'Completed profile differs')
            summary = dict(status=report['status'], page_count=report['page_count'],
                processing_profile=report['processing_profile'], processing_options=report['processing_options'],
                limitations=report['limitations'], pages=[{k: v for k, v in p.items()
                    if k in ('page_index','status','issues','demo_export','native_analysis')} for p in report['pages']])
        manifest = (seal or files.seal)(folder, summary)
        atomic_json(folder/'sealed-result.json', dict(schema_version='sealed-child/1.0',
            identity=request['_worker_identity'], manifest=manifest, summary=summary))


def child_main(path, operation, *, seal=None):
    def execute(request, progress):
        operation(request, progress)
        finalize(request, progress, seal=seal)
    return correlated_child(path, execute)
