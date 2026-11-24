"""Profile worker using shared runtime; no diagnostic writes after publication."""
from pathlib import Path
from cad_recovery.dataset_layout_v1 import read, digest
from cad_recovery.profile_contract_v6 import ProcessingOptions, validate_binding, ProfileError
from cad_recovery.service_profiles_v11.queue import Queue
from cad_recovery.service_profiles_v9.storage import Storage
from cad_recovery.runtime_worker_v1 import run_child
from cad_recovery.runtime_io_v1 import best_effort
from cad_recovery.runtime_cache_v1 import reap_dead_attempts


def process_once(settings, queue=None, storage=None, *, runner=None):
    repo, files = queue or Queue(settings), storage or Storage(settings)
    if hasattr(repo, 'bind_storage'):
        repo.bind_storage()
    run = repo.claim()
    if run is None:
        return False
    folder = None
    try:
        source = files.path(run['input_path'])
        if digest(source) != run['input_sha256']:
            raise ProfileError('input_integrity', 'Uploaded input differs')
        stored = dict(run['options'])
        binding = stored.pop('_profile_binding', None)
        options = ProcessingOptions.model_validate(stored)
        validate_binding(options.profile, binding)
        reap_dead_attempts(files, f"attempts/{run['id']}")
        folder = files.attempt(run)
        request = dict(source=str(source), input_sha256=run['input_sha256'], output=str(folder/'output'),
            options=options.model_dump(), profile_binding=binding)
        script = runner or Path(__file__).resolve().parents[3]/'scripts/service_profile_diagnostic_v2.py'
        budget = run_child(request, folder, files, settings, script, lambda: repo.heartbeat(run))
        report = read(folder/'output/report.json')
        if report['processing_profile'] != binding:
            raise ProfileError('profile_contract_changed', 'Completed profile differs')
        summary = dict(status=report['status'], page_count=report['page_count'], processing_profile=binding,
            processing_options=report['processing_options'], limitations=report['limitations'],
            pages=[{k:v for k,v in p.items() if k in ('page_index','status','issues','demo_export','native_analysis')}
                for p in report['pages']])
        manifest = files.seal(folder, summary)
        budget.check()
        budget.finish('completed')
        repo.finish(run, manifest=manifest, summary=summary)
    except Exception as error:
        code = getattr(error, 'code', 'processing_failed')
        if folder is not None:
            best_effort(folder/'publication-error.json', dict(code=code, type=type(error).__name__))
        if code != 'lease_lost_or_cancelled':
            repo.finish(run, error=dict(code=code, message='Processing stopped; private diagnostics retained'))
    return True
