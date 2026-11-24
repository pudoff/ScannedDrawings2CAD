"""Manual table API and opt-in Web extension over all existing service routes."""
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID
import hmac
import json
import os

from fastapi import FastAPI, HTTPException, Request, Path as ApiPath
from fastapi.responses import HTMLResponse, Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import ValidationError

from .service_profiles_v11.api import create_api as base_api
from .service_profiles_v10.api import index_html as old_html, app_javascript as old_js, ASSETS, BRAND
from .service_profiles_v11.queue import Queue
from .service_profiles_v9.storage import Storage
from .web_tables_store_v1 import TableStore, TableConflict
from .web_tables_contract_v2 import Approval, parse_request, binding
from .web_tables_contract_v1 import binding as previous_binding
from .web_tables_context_v2 import describe, validate_edit

ASSET_DIR = Path(__file__).with_name('web_tables_assets_v1')


def create_api(settings, *, table_store=None, queue=None, storage=None, table_accept_key=None, **dependencies):
    repo = queue or Queue(settings); files = storage or Storage(settings)
    api = base_api(settings, queue=repo, storage=files, **dependencies)
    store = table_store or TableStore(repo)
    if table_store is None: store.initialize()
    key = os.environ.get('CAD_TABLE_ACCEPT_TOKEN', '') if table_accept_key is None else table_accept_key
    prefix = '/v1/jobs/{job_id}/versions/{version}/pages/{page}/tables'

    @api.exception_handler(TableConflict)
    async def conflict(request, error): return JSONResponse({'detail': str(error)}, status_code=409)

    def bound(job, version, page):
        document, versions = repo.job(job); parent = repo.run(job, version)
        if str(parent['job_id']) != str(job) or parent['version'] != version: raise ValueError('Version mismatch')
        return document, parent

    def can_apply(request):
        if key: return hmac.compare_digest(request.headers.get('x-table-accept-key', '').encode(), key.encode())
        return not settings.token and request.client is not None and request.client.host in ('127.0.0.1', '::1')

    def public(row):
        return dict(id=str(row['id']), status=row['status'], summary=row['summary'],
                    applied_version=row['applied_version'], error=row['error'],
                    preview_sha256=(row['preview_manifest'] or {}).get('sha256'),
                    regions=row['request']['regions'], reviewer=row['request']['reviewer'],
                    page_kind=row['request']['page_kind'])

    async def body(request):
        if request.headers.get('content-type', '').split(';')[0] != 'application/json': raise HTTPException(415, 'JSON required')
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'): raise HTTPException(403, 'Same-origin request required')
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > 16384: raise HTTPException(413, 'Table request limit exceeded')
        try: return json.loads(data)
        except (ValueError, UnicodeError): raise HTTPException(422, 'Invalid JSON')

    @api.get(prefix)
    async def context(request: Request, job_id: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        document, parent = await run_in_threadpool(bound, job_id, version, page)
        try: value, _ = await run_in_threadpool(describe, files, document, parent, page)
        except (ValueError, LookupError) as error: raise HTTPException(409, str(error))
        rows = await run_in_threadpool(store.list, job_id, version, page)
        return dict(context=value, proposals=[public(r) for r in rows],
                    permissions=dict(can_apply=can_apply(request), requires_accept_key=bool(key)))

    @api.get(prefix+'/image')
    async def image(job_id: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        document, parent = await run_in_threadpool(bound, job_id, version, page)
        _, image_path = await run_in_threadpool(describe, files, document, parent, page)
        if image_path is None: raise HTTPException(404, 'No raster page')
        return FileResponse(image_path, media_type='image/png', headers={'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'})

    @api.post(prefix+'/preview', status_code=202)
    async def prepare(request: Request, job_id: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        try: edit = parse_request(await body(request))
        except (ValidationError, ValueError, TypeError, AttributeError): raise HTTPException(422, 'Проверьте области и имя проверившего.')
        document, parent = await run_in_threadpool(bound, job_id, version, page)
        try:
            value, _ = await run_in_threadpool(describe, files, document, parent, page)
            validate_edit(value, edit)
        except (ValueError, LookupError) as error: raise HTTPException(409, str(error))
        row = await run_in_threadpool(store.create, job_id, version, page, edit)
        return public(row)

    @api.get(prefix+'/{identifier}')
    async def status(job_id: UUID, identifier: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        return public(await run_in_threadpool(store.get, job_id, version, page, identifier))

    @api.post(prefix+'/{identifier}/cancel')
    async def cancel(request: Request, job_id: UUID, identifier: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        await body(request)
        row = await run_in_threadpool(store.get, job_id, version, page, identifier)
        changed = await run_in_threadpool(store.cancel, row)
        if changed is None: raise HTTPException(409, 'Завершённую версию нельзя отменить.')
        return public(changed)

    @api.get(prefix+'/{identifier}/files/{kind}')
    async def artifact(job_id: UUID, identifier: UUID, kind: str, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        row = await run_in_threadpool(store.get, job_id, version, page, identifier)
        if row['status'] not in ('prepared', 'rejected', 'applied') or not row['preview_manifest']:
            raise HTTPException(409, 'Результат ещё не подготовлен.')
        suffix = {'dxf': '.dxf', 'preview': '-dxf.png', 'graph': '-tables.json'}.get(kind)
        if suffix is None: raise HTTPException(404, 'Unknown artifact')
        sealed = dict(status='completed', manifest=row['preview_manifest'])
        _, manifest = await run_in_threadpool(files.manifest, sealed)
        filename = f'page-{page+1:03d}'+suffix
        entry = next((f for f in manifest['files'] if f['filename'] == filename), None)
        if entry is None: raise HTTPException(404, 'Artifact missing')
        path, entry = await run_in_threadpool(files.artifact, sealed, entry['id'])
        return FileResponse(path, filename=filename, media_type=entry['media_type'], headers={'Cache-Control': 'private, no-store'})

    @api.post(prefix+'/{identifier}/apply')
    async def apply(request: Request, job_id: UUID, identifier: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        raw = await body(request)
        if not can_apply(request): raise HTTPException(403, 'Сохранение версии разрешено локальному оператору или по отдельному ключу.')
        try: approval = Approval.model_validate(raw)
        except ValidationError: raise HTTPException(422, 'Подтвердите просмотр точного результата.')
        _, parent = await run_in_threadpool(bound, job_id, version, page)
        row = await run_in_threadpool(store.get, job_id, version, page, identifier)
        if row['request']['binding_sha256'] not in (binding()['sha256'], previous_binding()['sha256']):
            raise HTTPException(409, 'Код табличного режима изменился. Подготовьте новый просмотр.')
        if not row['preview_manifest']: raise HTTPException(409, 'Предпросмотр не готов.')
        sealed = dict(status='completed', manifest=row['preview_manifest'])
        await run_in_threadpool(files.manifest, sealed)
        number = await run_in_threadpool(store.publish, row, parent, approval.model_dump(), row['preview_manifest'], row['summary'])
        return dict(version=number, proposal_id=str(identifier))
    return api


def index_html():
    html = old_html()
    marker = '<button id="tab-history"'
    if html.count(marker) != 1: raise ValueError('Unexpected frozen tabs')
    html = html.replace(marker, '<button id="tab-tables" role="tab" aria-selected="false" aria-controls="view-tables" data-tab="tables" tabindex="-1">Таблицы</button>'+marker)
    marker = '<section id="view-history"'
    if marker not in html: marker = '<div id="view-history"'
    if html.count(marker) != 1: raise ValueError('Unexpected frozen history panel')
    html = html.replace(marker, (ASSET_DIR/'panel.html').read_text(encoding='utf8')+marker)
    return html.replace('</head>', '<link rel="stylesheet" href="/demo/assets/tables.css"></head>')


def app_javascript():
    js = old_js()
    hook = "  const corrections = createCorrections({request,onApplied:async version=>{await changeVersion(version);await loadDocuments();tab('observations');}});"
    if js.count(hook) != 1: raise ValueError('Unexpected frozen corrections hook')
    js = "import {createTables} from './tables.js';\n"+js.replace(hook, hook+"\n  const tables = createTables({request,onApplied:async version=>{await changeVersion(version);await loadDocuments();tab('tables');}});")
    js = js.replace('inspector.reset();corrections.reset();', 'inspector.reset();corrections.reset();tables.reset();')
    hook = '    corrections.context({jobId:state.jobId, version:state.version, page:state.page, key:context(), ready:!!state.bundle});'
    if js.count(hook) != 1: raise ValueError('Unexpected frozen page context hook')
    js = js.replace(hook, hook+'\n    tables.context({jobId:state.jobId, version:state.version, page:state.page, key:context(), ready:!!state.bundle});')
    return js


def create_web_app(settings, **dependencies):
    api = create_api(settings, **dependencies)
    @asynccontextmanager
    async def lifespan(app):
        async with api.router.lifespan_context(api): yield
    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    @web.get('/')
    def index():
        return HTMLResponse(index_html(), headers={'Cache-Control': 'no-store',
            'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"})
    @web.get('/demo/assets/app.js')
    def javascript(): return Response(app_javascript(), media_type='text/javascript', headers={'Cache-Control': 'no-store'})
    @web.get('/demo/assets/tables.js')
    def tables_js(): return Response((ASSET_DIR/'tables.js').read_text(encoding='utf8'), media_type='text/javascript', headers={'Cache-Control': 'no-store'})
    @web.get('/demo/assets/tables.css')
    def tables_css(): return Response((ASSET_DIR/'tables.css').read_text(encoding='utf8'), media_type='text/css')
    web.mount('/demo/assets', StaticFiles(directory=ASSETS)); web.mount('/workspace/assets', StaticFiles(directory=ASSETS))
    web.mount('/ui/assets', StaticFiles(directory=BRAND)); web.mount('/', api)
    return web
