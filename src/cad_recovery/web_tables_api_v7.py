"""Profile-bound Web V26: text-only previews, cumulative choices and fenced versions."""
from pathlib import Path
from uuid import UUID
import hmac
import os
from fastapi import Request, HTTPException, Path as ApiPath, Query
from fastapi.responses import Response, FileResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from . import web_tables_api_v4 as previous
from .web_tables_api_v3 import PREFIX, body, public
from .web_tables_contract_v6 import binding, parse_request, TextSelection, Approval
from .web_tables_context_v6 import describe, validate_edit
from .web_text_publication_v1 import packet, load_state, merge
from .web_text_lineage_v1 import validate_proposal
from .service_profiles_v20.queue import Queue
from .service_profiles_v9.storage import Storage
from .web_tables_store_v1 import TableStore

ASSETS = Path(__file__).with_name('web_table_text_assets_v2')


def supported_bindings():
    from .web_tables_contract_v4 import binding as fourth
    from .web_tables_contract_v5 import binding as fifth
    return {binding()['sha256'],fourth()['sha256'],fifth()['sha256']}


def install(api, settings, dependencies):
    repo = dependencies.get('queue') or Queue(settings)
    files = dependencies.get('storage') or Storage(settings)
    store = dependencies.get('table_store') or TableStore(repo)
    key = dependencies.get('table_accept_key')
    if key is None:
        key = os.environ.get('CAD_TABLE_ACCEPT_TOKEN', '')
    removed = {PREFIX, PREFIX+'/image', PREFIX+'/preview', PREFIX+'/published/text', PREFIX+'/published/text-preview',
               PREFIX+'/{identifier}/text', PREFIX+'/{identifier}/text-preview',
               PREFIX+'/{identifier}/apply', PREFIX+'/{identifier}/cancel'}
    api.router.routes = [r for r in api.router.routes if getattr(r, 'path', None) not in removed]

    def can_apply(request):
        if key:
            return hmac.compare_digest(request.headers.get('x-table-accept-key', '').encode(), key.encode())
        return not settings.token and request.client and request.client.host in ('127.0.0.1', '::1')

    def require_operator(request):
        if not can_apply(request):
            raise HTTPException(403, 'Изменения требуют прав оператора.')

    def bound(job, version, latest=False):
        document, versions = repo.job(job); parent = repo.run(job, version)
        if str(parent['job_id']) != str(job) or parent['version'] != version:
            raise ValueError('Version differs')
        is_latest = max(v['version'] for v in versions) == version
        if latest and not is_latest:
            raise ValueError('Current version changed')
        return document, parent, is_latest

    def source(job, version, page, identifier=None):
        document, parent, latest = bound(job, version)
        if identifier is None:
            path, _ = files.manifest(parent)
            output = path.parent/'output'; manifest = parent['manifest']; identity = parent['id']; row = None
        else:
            row = store.get(job, version, page, identifier)
            if (row['status'] not in ('prepared', 'applied') or not row['preview_manifest']
                    or row['request']['binding_sha256'] not in supported_bindings()):
                raise ValueError('Text-only preview unavailable')
            path, _ = files.manifest(dict(status='completed', manifest=row['preview_manifest']))
            output = path.parent/'output'; manifest = row['preview_manifest']; identity = row['id']
        state = load_state(output, page, job, document['input_sha256'])
        return document, parent, latest, output, manifest, str(identity), state, row

    def get_packet(request, job, version, page, identifier, offset, limit):
        document, parent, latest, output, manifest, identity, state, row = source(job, version, page, identifier)
        value = packet(output, page, offset, limit)
        value.update(source_kind='published' if identifier is None else 'proposal', proposal_id=identity,
                     preview_sha256=manifest['sha256'], read_only=not(latest and can_apply(request)))
        return value

    @api.get(PREFIX)
    async def context(request: Request, job_id: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        try:
            document, parent, latest = await run_in_threadpool(bound, job_id, version)
            value, _ = await run_in_threadpool(describe, files, document, parent, page)
            rows = await run_in_threadpool(store.list, job_id, version, page)
            try:
                published = await run_in_threadpool(get_packet, request, job_id, version, page, None, 0, 40)
            except (ValueError, LookupError, OSError):
                published = None
        except (ValueError, LookupError):
            raise HTTPException(409, 'Версия или регистрация изменились. Обновите документ.')
        return dict(context=value,
            proposals=[public(r) for r in rows if r['request']['binding_sha256'] in supported_bindings()],
            published_text=published, permissions=dict(can_apply=bool(latest and can_apply(request)), requires_accept_key=bool(key)))

    @api.post(PREFIX+'/preview', status_code=202)
    async def prepare(request: Request, job_id: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        require_operator(request)
        try:
            edit = parse_request(await body(request))
            if edit.text_selection:
                raise ValueError('Use text route')
        except (ValueError, ValidationError, TypeError, AttributeError):
            raise HTTPException(422, 'Проверьте области и имя проверившего.')
        try:
            document, parent, _ = await run_in_threadpool(bound, job_id, version, True)
            ctx, _ = await run_in_threadpool(describe, files, document, parent, page); validate_edit(ctx, edit)
        except (ValueError, LookupError):
            raise HTTPException(409, 'Документ изменился. Обновите лист.')
        return public(await run_in_threadpool(store.create, job_id, version, page, edit))

    async def image(job_id: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        try:
            document,parent,_ = await run_in_threadpool(bound,job_id,version)
            _,path = await run_in_threadpool(describe,files,document,parent,page)
        except (ValueError,LookupError):
            raise HTTPException(409,'Версия или регистрация изменились. Обновите документ.')
        if path is None:raise HTTPException(404,'No raster page')
        return FileResponse(path,media_type='image/png',headers={
            'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'})
    # The retained /{identifier} status route must not consume the word image.
    from fastapi.routing import APIRoute
    api.router.routes.insert(0,APIRoute(PREFIX+'/image',image,methods=['GET']))

    # Register the literal published route before the UUID proposal route.
    @api.get(PREFIX+'/published/text')
    async def published(request: Request, job_id: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20),
                        offset: int = Query(default=0, ge=0, le=100000), limit: int = Query(default=40, ge=1, le=100)):
        try:
            return await run_in_threadpool(get_packet, request, job_id, version, page, None, offset, limit)
        except (ValueError, LookupError, OSError):
            raise HTTPException(409, 'В этой версии нет актуального пакета выбора текста.')

    async def choose(request, job_id, version, page, identifier):
        require_operator(request); raw = await body(request); reviewer = raw.pop('reviewer', '')
        try:
            selection = TextSelection.model_validate(raw)
            if raw.get('reviewed') is not True or not isinstance(reviewer, str) or not reviewer.strip():
                raise ValueError('Explicit review required')
            if selection.source_kind != ('published' if identifier is None else 'proposal'):
                raise ValueError('Source kind differs')
        except (ValueError, ValidationError):
            raise HTTPException(422, 'Проверьте выбранные строки, области и имя проверившего.')
        try:
            document, parent, _ = await run_in_threadpool(bound, job_id, version, True)
            _, _, _, output, manifest, identity, state, row = await run_in_threadpool(source, job_id, version, page, identifier)
            value = await run_in_threadpool(packet, output, page, 0, 1)
            if (str(selection.proposal_id) != identity or selection.preview_sha256 != manifest['sha256']
                    or selection.review_sha256 != value['review_sha256']):
                raise ValueError('Text source changed')
            if identifier is not None:
                parent_path,_=await run_in_threadpool(files.manifest,parent)
                prefix=f'page-{page+1:03d}.dxf'
                await run_in_threadpool(validate_proposal,state,parent,document,page,
                    output/prefix,parent_path.parent/'output'/prefix)
            merge(state, selection.model_dump(mode='json'))
            ctx, _ = await run_in_threadpool(describe, files, document, parent, page)
            edit = {k: ctx[k] for k in ('context_sha256', 'source_manifest_sha256', 'source_sha256', 'binding_sha256')}
            edit.update(regions=state['regions'], page_kind=state['page_kind'], reviewer=reviewer,
                        text_selection=selection.model_dump(mode='json'))
            edit = parse_request(edit); validate_edit(ctx, edit)
        except (ValueError, LookupError, OSError):
            raise HTTPException(409, 'Исходный просмотр или версия изменились. Обновите лист.')
        return public(await run_in_threadpool(store.create, job_id, version, page, edit))

    @api.post(PREFIX+'/published/text-preview', status_code=202)
    async def choose_published(request: Request, job_id: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        return await choose(request, job_id, version, page, None)

    @api.get(PREFIX+'/{identifier}/text')
    async def text(request: Request, job_id: UUID, identifier: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20),
                   offset: int = Query(default=0, ge=0, le=100000), limit: int = Query(default=40, ge=1, le=100)):
        try:
            return await run_in_threadpool(get_packet, request, job_id, version, page, identifier, offset, limit)
        except (ValueError, LookupError, OSError):
            raise HTTPException(409, 'Текстовый просмотр изменился или не готов.')

    @api.post(PREFIX+'/{identifier}/text-preview', status_code=202)
    async def choose_proposal(request: Request, job_id: UUID, identifier: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        return await choose(request, job_id, version, page, identifier)

    @api.post(PREFIX+'/{identifier}/apply')
    async def apply(request: Request, job_id: UUID, identifier: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        require_operator(request)
        try:
            approval = Approval.model_validate(await body(request))
        except ValidationError:
            raise HTTPException(422, 'Подтвердите точный просмотр.')
        # Store.publish supplies transactional CAS and idempotence after apply.
        _, parent, _ = await run_in_threadpool(bound, job_id, version)
        row = await run_in_threadpool(store.get, job_id, version, page, identifier)
        if row['request']['binding_sha256'] not in supported_bindings() or not row['preview_manifest']:
            raise HTTPException(409, 'Нужен новый готовый текстовый просмотр.')
        await run_in_threadpool(files.manifest, dict(status='completed', manifest=row['preview_manifest']))
        number = await run_in_threadpool(store.publish, row, parent, approval.model_dump(), row['preview_manifest'], row['summary'])
        return dict(version=number, proposal_id=str(identifier))

    @api.post(PREFIX+'/{identifier}/cancel')
    async def cancel(request: Request, job_id: UUID, identifier: UUID, version: int = ApiPath(ge=1), page: int = ApiPath(ge=0, lt=20)):
        require_operator(request); await body(request)
        row = await run_in_threadpool(store.get, job_id, version, page, identifier)
        changed = await run_in_threadpool(store.cancel, row)
        if changed is None:
            raise HTTPException(409, 'Завершённую версию нельзя отменить.')
        return public(changed)
    return api


def create_api(settings, **dependencies):
    return install(previous.create_api(settings, **dependencies), settings, dependencies)


def create_web_app(settings, **dependencies):
    web = previous.create_web_app(settings, **dependencies)
    api = next(r.app for r in web.router.routes if getattr(r, 'path', None) == '')
    install(api, settings, dependencies)
    web.router.routes = [r for r in web.router.routes if getattr(r, 'path', None) != '/demo/assets/text-review.js']
    # Insert before the static mounts, which otherwise consume this URL.
    async def text_js():
        return Response((ASSETS/'text-review.js').read_text(encoding='utf-8'), media_type='text/javascript',
                        headers={'Cache-Control': 'no-store'})
    from fastapi.routing import APIRoute
    web.router.routes.insert(0, APIRoute('/demo/assets/text-review.js', text_js, methods=['GET']))
    return web
