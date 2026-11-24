"""Proposal/review/apply API; atomic publication is delegated to the correction worker."""
from contextlib import asynccontextmanager
import hmac
import hashlib
import os
from pathlib import Path
from uuid import UUID
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from cad_recovery.service_profiles_v5.api import create_api as previous_api,BRAND
from cad_recovery.service_profiles_v2.queue import Queue
from cad_recovery.service_profiles_v2.storage import Storage
from cad_recovery.comparison_service_v1 import Comparisons,ComparisonLimit
from cad_recovery.web_correction_contract_v1 import context,validate_edit,published,decode,EditRequest,EditApproval
from cad_recovery.web_correction_store_v1 import CorrectionStore,CorrectionConflict
from cad_recovery.dataset_layout_v1 import encoded

ASSETS=Path(__file__).parent/'assets'


class ReviewedComparisons(Comparisons):
    def _render(self,path,digest,render_box):
        report=decode((path.parent/'report.json').read_bytes())
        page=int(path.stem.split('-')[-1])-1
        # P13.4 model coordinates may span several local views: do not reuse pixel/paper mapping.
        if report['pages'][page].get('demo_export',{}).get('mode')=='reviewed_geometry':render_box=None
        return super()._render(path,digest,render_box)

    def describe(self,run,page):
        value=super().describe(run,page)
        prefix,entries,load=published(self.files,run,page)
        manual=load(prefix+'-manual-graph.json')
        if manual:
            value['manual_corrections']=manual
            for row in value['inspection']['objects']:
                if row['id'] in manual['readings']:row['reviewed_reading']=manual['readings'][row['id']]
        graph=load(prefix+'-reviewed-graph.json')
        if graph and manual and any(h['kind']=='scalar' for h in manual['history']):
            value['verified_geometry_counts']={name:sum(e['status']=='accepted' and e['evidence']['status']=='verified' for e in graph[name])
                for name in ('primitives','dimensions')}
            value['accepted_scope']='reviewed_graph; raster observations remain unaccepted'
        return value


def create_api(settings,queue=None,storage=None,reviews=None,correction_store=None,accept_token=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    api=previous_api(settings,queue=repo,storage=files,reviews=reviews)
    store=correction_store or CorrectionStore(repo)
    if correction_store is None:store.initialize()
    accept_key=os.environ.get('CAD_CORRECTION_ACCEPT_TOKEN','') if accept_token is None else accept_token
    comparisons=ReviewedComparisons(files);prior_lifespan=api.router.lifespan_context
    @asynccontextmanager
    async def lifespan(app):
        try:
            async with prior_lifespan(app):yield
        finally:comparisons.close()
    api.router.lifespan_context=lifespan
    api.router.routes=[r for r in api.router.routes if getattr(r,'path','') not in (
        '/v1/jobs/{job_id}/versions/{version}/pages/{page}/comparison',
        '/v1/jobs/{job_id}/versions/{version}/pages/{page}/comparison/dxf.svg')]
    @api.exception_handler(CorrectionConflict)
    async def conflict(request,error):return JSONResponse({'detail':str(error)},status_code=409)

    def bound(job,version,page):
        if version<1 or not 0<=page<20:raise HTTPException(422,'Invalid version/page')
        document,_=repo.job(job);run=repo.run(job,version)
        if str(document['id'])!=str(job) or str(run['job_id'])!=str(job) or run['version']!=version:raise ValueError('Context mismatch')
        run={**run,'input_sha256':document['input_sha256']};published(files,run,page)
        return run

    def access(request):
        if accept_key:return hmac.compare_digest(request.headers.get('x-correction-accept-key','').encode(),accept_key.encode())
        return not settings.token and request.client is not None and request.client.host in ('127.0.0.1','::1')

    def permissions(request):return {'can_accept':access(request),'requires_accept_key':bool(accept_key),
        'mode':'separate_accept_key' if accept_key else 'local_operator' if not settings.token else 'acceptance_disabled'}

    async def body(request,model):
        if request.headers.get('content-type','').split(';')[0]!='application/json':raise HTTPException(415,'JSON body required')
        origin=request.headers.get('origin')
        if origin and origin!=str(request.base_url).rstrip('/'):raise HTTPException(403,'Same-origin request required')
        data=bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data)>4096:raise HTTPException(413,'Correction body limit')
        try:return model.model_validate(decode(bytes(data)))
        except (ValidationError,ValueError):raise HTTPException(422,'Некорректное исправление или запись проверки.')

    def public(row):
        return {k:row[k] for k in ('id','job_id','parent_version','page_index','stage','status',
            'preview_summary','applied_version','error','created_at','updated_at')}

    @api.get('/v1/jobs/{job_id}/versions/{version}/pages/{page}/corrections')
    def edit_context(request:Request,job_id:UUID,version:int,page:int):
        run=bound(job_id,version,page)
        return {**context(files,run,page),'permissions':permissions(request),
            'proposals':[public(row) for row in store.list(job_id,version,page)]}

    @api.post('/v1/jobs/{job_id}/versions/{version}/pages/{page}/corrections/preview',status_code=202)
    async def preview(request:Request,job_id:UUID,version:int,page:int):
        edit=await body(request,EditRequest);run=await run_in_threadpool(bound,job_id,version,page)
        await run_in_threadpool(validate_edit,files,run,page,edit)
        if edit.review_id:
            from cad_recovery.web_v1.reviews import Reviews
            source=next((r for r in (reviews or Reviews(settings)).list(job_id,version) if str(r['id'])==str(UUID(edit.review_id))),None)
            if source is None or source['page_index']!=page or source['source_manifest_sha256']!=run['manifest']['sha256']:
                raise HTTPException(409,'Исходное замечание относится к другому контексту.')
            title_map={'drawing_id':'document_number','part_name':'name','material':'material','person':'authors'}
            same_target=(title_map.get(source['field'])==edit.target_id if edit.kind=='title_field' else source['target']==edit.target_id)
            try:same_value=(float(source['value'])==edit.new_value if edit.kind=='scalar' else source['value']==edit.new_value)
            except ValueError:same_value=False
            if not same_target or not same_value:raise HTTPException(409,'Предложение не совпадает с исходным замечанием.')
        row=await run_in_threadpool(store.create,job_id,version,page,edit)
        return public(row)

    @api.get('/v1/jobs/{job_id}/versions/{version}/pages/{page}/corrections/{identifier}')
    def proposal(job_id:UUID,version:int,page:int,identifier:UUID):
        bound(job_id,version,page);row=store.get(job_id,version,page,identifier)
        if row['preview_manifest']:files.manifest({'status':'completed','manifest':row['preview_manifest']})
        return public(row)

    @api.get('/v1/jobs/{job_id}/versions/{version}/pages/{page}/corrections/{identifier}/images/{name}')
    def proposal_image(job_id:UUID,version:int,page:int,identifier:UUID,name:str):
        bound(job_id,version,page);row=store.get(job_id,version,page,identifier)
        if name not in ('before','after') or not row['preview_manifest']:raise KeyError(name)
        temp={'status':'completed','manifest':row['preview_manifest']}
        _,manifest=files.manifest(temp);entry=next((e for e in manifest['files'] if e['filename']==name+'.png'),None)
        if not entry:raise KeyError(name)
        path,_=files.artifact(temp,entry['id'])
        return FileResponse(path,media_type='image/png',headers={'Cache-Control':'no-store','X-Content-SHA256':entry['sha256']})

    @api.post('/v1/jobs/{job_id}/versions/{version}/pages/{page}/corrections/{identifier}/accept',status_code=202)
    async def accept(request:Request,job_id:UUID,version:int,page:int,identifier:UUID):
        if not access(request):raise HTTPException(403,'Для принятия требуется право локального оператора или отдельный ключ принятия.')
        approval=await body(request,EditApproval);await run_in_threadpool(bound,job_id,version,page)
        row=await run_in_threadpool(store.get,job_id,version,page,identifier)
        if row['preview_manifest']:
            await run_in_threadpool(files.manifest,{'status':'completed','manifest':row['preview_manifest']})
        row=await run_in_threadpool(store.accept,row,approval,'accept_key_operator' if accept_key else 'local_operator')
        return public(row)

    @api.get('/v1/jobs/{job_id}/versions/{version}/pages/{page}/comparison')
    def comparison(job_id:UUID,version:int,page:int):
        try:data=encoded(comparisons.describe(bound(job_id,version,page),page))
        except ComparisonLimit:raise HTTPException(413,'Comparison resource limit')
        sha=hashlib.sha256(data).hexdigest()
        return Response(data,media_type='application/json',headers={'Cache-Control':'no-store','ETag':'"'+sha+'"',
            'X-Content-SHA256':sha,'X-Content-Type-Options':'nosniff'})

    @api.get('/v1/jobs/{job_id}/versions/{version}/pages/{page}/comparison/dxf.svg')
    def dxf_image(job_id:UUID,version:int,page:int):
        data,value=comparisons.svg(bound(job_id,version,page),page)
        return Response(data,media_type='image/svg+xml',headers={'Cache-Control':'no-store',
            'X-DXF-SHA256':value['dxf']['sha256'],'X-Content-SHA256':hashlib.sha256(data).hexdigest(),'X-Content-Type-Options':'nosniff',
            'Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'"})
    return api


def create_web_app(settings,**dependencies):
    api=create_api(settings,**dependencies)
    @asynccontextmanager
    async def lifespan(app):
        async with api.router.lifespan_context(api):yield
    web=FastAPI(docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    @web.get('/')
    def index():return FileResponse(ASSETS/'index.html',media_type='text/html',headers={'Cache-Control':'no-cache',
        'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"})
    web.mount('/workspace/assets',StaticFiles(directory=ASSETS));web.mount('/ui/assets',StaticFiles(directory=BRAND));web.mount('/',api)
    return web
