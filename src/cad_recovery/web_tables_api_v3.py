"""Local opt-in text choices over retained Web/queue routes and immutable previews."""
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID
import hmac
import json
import os
from fastapi import FastAPI,HTTPException,Request,Path as ApiPath,Query
from fastapi.responses import HTMLResponse,Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import ValidationError
from . import web_tables_api_v2 as previous
from .web_tables_contract_v3 import binding,parse_request,Approval,TextSelection
from .web_tables_contract_v1 import binding as first
from .web_tables_contract_v2 import binding as second
from .web_tables_context_v3 import describe,validate_edit
from .web_tables_store_v1 import TableStore
from .service_profiles_v11.queue import Queue
from .service_profiles_v9.storage import Storage
from .web_text_selection_v1 import packet
from .web_runtime_status_v1 import latest

ASSETS=Path(__file__).with_name('web_table_text_assets_v1')
PREFIX='/v1/jobs/{job_id}/versions/{version}/pages/{page}/tables'


async def body(request):
    if request.headers.get('content-type','').split(';')[0]!='application/json':raise HTTPException(415,'JSON required')
    origin=request.headers.get('origin')
    if origin and origin!=str(request.base_url).rstrip('/'):raise HTTPException(403,'Same-origin request required')
    data=bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data)>128*1024:raise HTTPException(413,'Запрос выбора слишком большой.')
    try:
        result=json.loads(data)
        if not isinstance(result,dict):raise ValueError('Object required')
        return result
    except (ValueError,UnicodeError):raise HTTPException(422,'Проверьте запрос.')


def public(row):return dict(id=str(row['id']),status=row['status'],summary=row['summary'],
    applied_version=row['applied_version'],error=row['error'],preview_sha256=(row['preview_manifest'] or {}).get('sha256'),
    regions=row['request']['regions'],reviewer=row['request']['reviewer'],page_kind=row['request']['page_kind'])


def create_api(settings,*,table_store=None,queue=None,storage=None,table_accept_key=None,**dependencies):
    repo=queue or Queue(settings);files=storage or Storage(settings);store=table_store or TableStore(repo)
    if table_store is None:store.initialize()
    key=os.environ.get('CAD_TABLE_ACCEPT_TOKEN','') if table_accept_key is None else table_accept_key
    api=previous.create_api(settings,queue=repo,storage=files,table_store=store,table_accept_key=key,**dependencies)
    removed={(PREFIX,'GET'),(PREFIX+'/preview','POST'),(PREFIX+'/{identifier}/apply','POST')}
    api.router.routes=[r for r in api.router.routes if not any((getattr(r,'path',None),m) in removed for m in getattr(r,'methods',[]))]
    def bound(job,version):
        document,_=repo.job(job);parent=repo.run(job,version)
        if str(parent['job_id'])!=str(job) or parent['version']!=version:raise ValueError('Version differs')
        return document,parent
    def can_apply(request):
        if key:return hmac.compare_digest(request.headers.get('x-table-accept-key','').encode(),key.encode())
        return not settings.token and request.client is not None and request.client.host in ('127.0.0.1','::1')
    def source(job,version,page,identifier):
        row=store.get(job,version,page,identifier)
        if row['status'] not in ('prepared','applied') or not row['preview_manifest']:
            raise ValueError('Preview unavailable')
        path,_=files.manifest(dict(status='completed',manifest=row['preview_manifest']))
        return row,path.parent/'output'

    @api.get(PREFIX)
    async def context(request:Request,job_id:UUID,version:int=ApiPath(ge=1),page:int=ApiPath(ge=0,lt=20)):
        document,parent=await run_in_threadpool(bound,job_id,version)
        try:value,_=await run_in_threadpool(describe,files,document,parent,page)
        except (ValueError,LookupError):raise HTTPException(409,'Версия или регистрация изменились. Обновите документ.')
        rows=await run_in_threadpool(store.list,job_id,version,page)
        path,_=await run_in_threadpool(files.manifest,parent)
        published=await run_in_threadpool(packet,path.parent/'output',page,0,40)
        return dict(context=value,proposals=[public(r) for r in rows],
            published_text={**published,'read_only':True} if published.get('enabled') else None,
            permissions=dict(can_apply=can_apply(request),requires_accept_key=bool(key)))

    @api.post(PREFIX+'/preview',status_code=202)
    async def prepare(request:Request,job_id:UUID,version:int=ApiPath(ge=1),page:int=ApiPath(ge=0,lt=20)):
        try:
            edit=parse_request(await body(request))
            if edit.text_selection is not None:raise ValueError('Use text-choice route')
        except (ValidationError,ValueError,TypeError,AttributeError):raise HTTPException(422,'Проверьте области и имя проверившего.')
        document,parent=await run_in_threadpool(bound,job_id,version)
        try:
            value,_=await run_in_threadpool(describe,files,document,parent,page);validate_edit(value,edit)
        except (ValueError,LookupError):raise HTTPException(409,'Документ изменился. Обновите лист.')
        return public(await run_in_threadpool(store.create,job_id,version,page,edit))

    @api.get(PREFIX+'/published/text')
    async def published_text(job_id:UUID,version:int=ApiPath(ge=1),page:int=ApiPath(ge=0,lt=20),
                             offset:int=Query(default=0,ge=0,le=100000),limit:int=Query(default=40,ge=1,le=100)):
        _,parent=await run_in_threadpool(bound,job_id,version)
        path,_=await run_in_threadpool(files.manifest,parent)
        value=await run_in_threadpool(packet,path.parent/'output',page,offset,limit)
        return {**value,'read_only':True}

    @api.get(PREFIX+'/{identifier}/text')
    async def text(job_id:UUID,identifier:UUID,version:int=ApiPath(ge=1),page:int=ApiPath(ge=0,lt=20),
                   offset:int=Query(default=0,ge=0,le=100000),limit:int=Query(default=40,ge=1,le=100)):
        try:
            row,output=await run_in_threadpool(source,job_id,version,page,identifier)
            value=await run_in_threadpool(packet,output,page,offset,limit)
        except (ValueError,LookupError):raise HTTPException(409,'Текстовый просмотр изменился или не готов.')
        return {**value,'proposal_id':str(identifier),'preview_sha256':row['preview_manifest']['sha256']}

    @api.post(PREFIX+'/{identifier}/text-preview',status_code=202)
    async def choose(request:Request,job_id:UUID,identifier:UUID,version:int=ApiPath(ge=1),page:int=ApiPath(ge=0,lt=20)):
        raw=await body(request);reviewer=raw.pop('reviewer','')
        try:
            selection=TextSelection.model_validate(raw)
            if raw.get('reviewed') is not True or selection.proposal_id!=identifier or not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('Reviewer/source required')
            if not selection.selections and not selection.associations:raise ValueError('No selections')
        except (ValidationError,ValueError):raise HTTPException(422,'Выберите строки, проверьте области и укажите проверившего.')
        try:
            row,output=await run_in_threadpool(source,job_id,version,page,identifier)
            value=await run_in_threadpool(packet,output,page,0,1)
            if not value['enabled'] or selection.review_sha256!=value['review_sha256'] or selection.preview_sha256!=row['preview_manifest']['sha256']:
                raise ValueError('Stale text preview')
            document,parent=await run_in_threadpool(bound,job_id,version)
            ctx,_=await run_in_threadpool(describe,files,document,parent,page)
            edit={k:ctx[k] for k in ('context_sha256','source_manifest_sha256','source_sha256','binding_sha256')}
            edit.update(regions=row['request']['regions'],page_kind=row['request']['page_kind'],reviewer=reviewer,
                        text_selection=selection.model_dump(mode='json'))
            edit=parse_request(edit);validate_edit(ctx,edit)
        except (ValueError,LookupError):raise HTTPException(409,'Исходный просмотр или версия изменились. Обновите лист.')
        return public(await run_in_threadpool(store.create,job_id,version,page,edit))

    @api.post(PREFIX+'/{identifier}/apply')
    async def apply(request:Request,job_id:UUID,identifier:UUID,version:int=ApiPath(ge=1),page:int=ApiPath(ge=0,lt=20)):
        raw=await body(request)
        if not can_apply(request):raise HTTPException(403,'Сохранение версии требует прав оператора.')
        try:approval=Approval.model_validate(raw)
        except ValidationError:raise HTTPException(422,'Подтвердите просмотр результата.')
        _,parent=await run_in_threadpool(bound,job_id,version)
        row=await run_in_threadpool(store.get,job_id,version,page,identifier)
        if row['request']['binding_sha256'] not in [f()['sha256'] for f in (first,second,binding)]:
            raise HTTPException(409,'Режим изменился. Подготовьте новый просмотр.')
        if not row['preview_manifest']:raise HTTPException(409,'Просмотр не готов.')
        await run_in_threadpool(files.manifest,dict(status='completed',manifest=row['preview_manifest']))
        number=await run_in_threadpool(store.publish,row,parent,approval.model_dump(),row['preview_manifest'],row['summary'])
        return dict(version=number,proposal_id=str(identifier))

    @api.get('/v1/jobs/{job_id}/versions/{version}/diagnostics')
    async def diagnostics(job_id:UUID,version:int=ApiPath(ge=1)):
        _,run=await run_in_threadpool(bound,job_id,version)
        value=await run_in_threadpool(latest,files,f"attempts/{run['id']}")
        review=run.get('options',{}).get('_table_review')
        if not value['available'] and review:
            identifier=UUID(review['id'])
            row=await run_in_threadpool(store.get,job_id,review['parent_version'],review['page'],identifier)
            value=await run_in_threadpool(latest,files,f"tables-v1/{row['id']}")
        return value

    @api.get(PREFIX+'/{identifier}/diagnostics')
    async def table_diagnostics(job_id:UUID,identifier:UUID,version:int=ApiPath(ge=1),page:int=ApiPath(ge=0,lt=20)):
        row=await run_in_threadpool(store.get,job_id,version,page,identifier)
        return await run_in_threadpool(latest,files,f"tables-v1/{row['id']}")
    return api


def tables_js():
    js=(previous.ASSET_DIR/'tables.js').read_text(encoding='utf8')
    js="import {createTextReview} from './text-review.js';\n"+js
    marker="  const base=()=>`/v1/jobs/${current.jobId}/versions/${current.version}/pages/${current.page}/tables`;"
    js=js.replace(marker,marker+"\n  const textReview=createTextReview({request,base,onProposal:async row=>{serial++;$('table-reviewed').checked=false;await show(row,epoch);},draw,canvas,reviewer:()=>$('table-reviewer').value});")
    js=js.replace('function reset(){epoch++;','function reset(){textReview.reset();epoch++;')
    js=js.replace('function changed(){serial++;','function changed(){textReview.reset();serial++;')
    js=js.replace("  canvas.onpointerdown=e=>{","  canvas.onpointerdown=e=>{if(textReview.isDrawing()){start=point(e);canvas.setPointerCapture(e.pointerId);return;}")
    js=js.replace("canvas.onpointerup=e=>{if(!start)return;const b=box(start,point(e));start=null;",
        "canvas.onpointerup=e=>{if(!start)return;const b=box(start,point(e));start=null;if(textReview.isDrawing()){textReview.associate(b);draw(b);return;}")
    marker="    if(row.status==='applied'){"
    js=js.replace(marker,"    await textReview.show(row,current);\n"+marker)
    marker="proposal=latest;await show(latest,own);}" 
    assert js.count(marker)==1
    js=js.replace(marker,marker+"else if(value.published_text?.enabled){await textReview.showPublished(value.published_text);}")
    return js


def app_js():
    js="import {createRuntimeStatus} from './text-review.js';\n"+previous.app_javascript()
    marker="  const tables = createTables({request,onApplied:async version=>{await changeVersion(version);await loadDocuments();tab('tables');}});"
    assert js.count(marker)==1
    js=js.replace(marker,marker+'\n  const runtimeStatus=createRuntimeStatus({request});')
    marker='    tables.context({jobId:state.jobId, version:state.version, page:state.page, key:context(), ready:!!state.bundle});'
    assert js.count(marker)==1
    js=js.replace(marker,marker+'\n    runtimeStatus.context({jobId:state.jobId,version:state.version,key:context(),status:run()?.status});')
    js=js.replace('inspector.reset();corrections.reset();tables.reset();','inspector.reset();corrections.reset();tables.reset();runtimeStatus.reset();')
    js=js.replace("if(path.includes('/corrections'))", "if(path.includes('/corrections')||path.includes('/tables'))")
    return js


def create_web_app(settings,**dependencies):
    api=create_api(settings,**dependencies)
    @asynccontextmanager
    async def lifespan(app):
        async with api.router.lifespan_context(api):yield
    web=FastAPI(docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    @web.get('/')
    def index():
        html=previous.index_html().replace('</head>','<link rel="stylesheet" href="/demo/assets/text-review.css"></head>')
        html=html.replace('<div id="quality-message"','<div id="runtime-status" role="status"></div><div id="quality-message"')
        html=html.replace('<div id="table-result"></div>','<div id="table-result"></div><section id="text-review" aria-label="Выбор текста"></section>')
        return HTMLResponse(html,headers={'Cache-Control':'no-store','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"})
    @web.get('/demo/assets/app.js')
    def javascript():return Response(app_js(),media_type='text/javascript',headers={'Cache-Control':'no-store'})
    @web.get('/demo/assets/tables.js')
    def tables():return Response(tables_js(),media_type='text/javascript',headers={'Cache-Control':'no-store'})
    @web.get('/demo/assets/text-review.js')
    def textjs():return Response((ASSETS/'text-review.js').read_text(encoding='utf8'),media_type='text/javascript',headers={'Cache-Control':'no-store'})
    @web.get('/demo/assets/text-review.css')
    def textcss():return Response((ASSETS/'text-review.css').read_text(encoding='utf8'),media_type='text/css')
    @web.get('/demo/assets/tables.css')
    def css():return Response((previous.ASSET_DIR/'tables.css').read_text(encoding='utf8'),media_type='text/css')
    web.mount('/demo/assets',StaticFiles(directory=previous.ASSETS));web.mount('/workspace/assets',StaticFiles(directory=previous.ASSETS))
    web.mount('/ui/assets',StaticFiles(directory=previous.BRAND));web.mount('/',api)
    return web
