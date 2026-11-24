"""Read-only, scoped public projection of correlated private worker events."""
from pathlib import Path
from uuid import UUID
from .worker_contract_v1 import WorkerIdentity, PUBLIC_CODES
from .runtime_io_v1 import read_json


def public_failure(value: dict) -> dict:
    identity = WorkerIdentity(**value['identity'])
    code = value.get('code')
    return dict(available=True, request_id=identity.request_id, job_id=identity.job_id,
                version=identity.version, proposal_id=identity.proposal_id, page_index=identity.page_index,
                code=code if isinstance(code, str) and code in PUBLIC_CODES else 'processing_failed')


def latest_failure(files, identity: WorkerIdentity) -> dict:
    folder = files.path(f'worker-events/{identity.request_id}')
    if not folder.exists():
        return dict(available=False)
    try:
        paths = list(folder.glob('*.json'))
        paths = sorted(paths, key=lambda p: p.stat().st_mtime_ns, reverse=True)
    except OSError:
        return dict(available=False)
    if len(paths) > 128:
        return dict(available=False)
    for path in paths:
        try:
            if path.is_symlink() or path.is_junction():
                continue
            value = read_json(path, limit=16384)
            other = WorkerIdentity(**value['identity'])
            if ((other.request_id, other.job_id, other.version, other.proposal_id, other.page_index)
                    == (identity.request_id, identity.job_id, identity.version, identity.proposal_id, identity.page_index)
                    and other.run_id in (None, identity.run_id)):
                return public_failure(value)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return dict(available=False)


def install(api, repo, files, *, table_store=None):
    from fastapi import Path as ApiPath
    from starlette.concurrency import run_in_threadpool
    from .worker_contract_v1 import identity_for_run, identity_for_table

    @api.get('/v1/jobs/{job_id}/versions/{version}/worker-error')
    async def profile_error(job_id: UUID, version: int = ApiPath(ge=1)):
        await run_in_threadpool(repo.job, job_id)
        run = await run_in_threadpool(repo.run, job_id, version)
        if run.get('owner') is None:
            run = dict(run, owner=str(UUID(int=0)))
        value = await run_in_threadpool(latest_failure, files, identity_for_run(run))
        return {**value, 'recovered': run['status'] == 'completed'} if value['available'] else value

    if table_store is not None:
        @api.get('/v1/jobs/{job_id}/versions/{version}/pages/{page}/tables/{identifier}/worker-error')
        async def table_error(job_id: UUID, identifier: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
            await run_in_threadpool(repo.job, job_id)
            parent = await run_in_threadpool(repo.run, job_id, version)
            row = await run_in_threadpool(table_store.get, job_id, version, page, identifier)
            # A prepared/failed SQL row has no current owner. The read-only
            # lookup matches logical identities and never exposes lease tokens.
            if row.get('owner') is None:
                row = dict(row, owner=str(UUID(int=0)))
            value = await run_in_threadpool(latest_failure, files, identity_for_table(row, parent))
            return {**value, 'recovered': row['status'] in ('prepared', 'applied')} if value['available'] else value
    return api
