"""Unchanged V16 UI/routes with cooperative bounded runtime readback."""
from uuid import UUID
from fastapi import Path as ApiPath
from starlette.concurrency import run_in_threadpool
from . import web_tables_api_v3 as previous
from .service_profiles_v11.queue import Queue
from .service_profiles_v9.storage import Storage
from .web_tables_store_v1 import TableStore
from .web_runtime_status_v2 import latest


def install(api, settings, dependencies):
    repo = dependencies.get('queue') or Queue(settings)
    files = dependencies.get('storage') or Storage(settings)
    store = dependencies.get('table_store') or TableStore(repo)
    prefix=previous.PREFIX
    paths={'/v1/jobs/{job_id}/versions/{version}/diagnostics',prefix+'/{identifier}/diagnostics'}
    api.router.routes=[r for r in api.router.routes if getattr(r,'path',None) not in paths]

    @api.get('/v1/jobs/{job_id}/versions/{version}/diagnostics')
    async def diagnostics(job_id:UUID, version:int=ApiPath(ge=1)):
        _, _ = await run_in_threadpool(repo.job, job_id)
        run = await run_in_threadpool(repo.run, job_id, version)
        value = await run_in_threadpool(latest, files, f"attempts/{run['id']}")
        review = run.get('options',{}).get('_table_review')
        if not value['available'] and review:
            row = await run_in_threadpool(store.get, job_id, review['parent_version'], review['page'], UUID(review['id']))
            value = await run_in_threadpool(latest, files, f"tables-v1/{row['id']}")
        if not value.get('reason') and run.get('error',{}):
            if run['error'].get('code') == 'telemetry_persist_failed':
                value={**value,'reason':'Не удалось сохранить журнал обработки'}
        return value

    @api.get(prefix+'/{identifier}/diagnostics')
    async def table_diagnostics(job_id:UUID, identifier:UUID, version:int=ApiPath(ge=1), page:int=ApiPath(ge=0,lt=20)):
        row = await run_in_threadpool(store.get, job_id, version, page, identifier)
        return await run_in_threadpool(latest, files, f"tables-v1/{row['id']}")
    return api


def create_api(settings, **dependencies):
    return install(previous.create_api(settings, **dependencies), settings, dependencies)


def create_web_app(settings, **dependencies):
    web = previous.create_web_app(settings, **dependencies)
    api = next(r.app for r in web.router.routes if getattr(r,'path',None)=='')
    install(api, settings, dependencies)
    return web
