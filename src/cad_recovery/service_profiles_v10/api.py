"""Latest experimental upload/version routes composed over the existing service."""
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from cad_recovery.service_profiles_v6.api import create_api as previous_api, BRAND
from cad_recovery.profile_contract_v5 import ProcessingOptions, DEFINITIONS, resolve_profile, NATIVE, REVIEW, CANDIDATE
from .queue import Queue
from cad_recovery.web_v1.reviews import Reviews
from cad_recovery.service_profiles_v8.api import index_html as old_index,app_javascript as old_javascript
from cad_recovery.native_comparison_v1 import NativeComparisons
from cad_recovery.comparison_service_v1 import ComparisonLimit
from cad_recovery.dataset_layout_v1 import encoded
import hashlib

ASSETS = Path(__file__).parents[1]/'service_profiles_v7/assets'


def create_api(settings, **dependencies):
    repo = dependencies.pop('queue', None) or Queue(settings)
    if dependencies.get('reviews') is None:
        class DemoReviews(Reviews):
            def __init__(self, queue):
                self.queue = queue
        reviews = DemoReviews(repo)
        reviews.initialize()
        dependencies['reviews'] = reviews
    from cad_recovery.service_profiles_v9.storage import Storage
    if dependencies.get('storage') is None:dependencies['storage']=Storage(settings)
    api = previous_api(settings, queue=repo, **dependencies)
    replaced = {('/v1/jobs','POST'), ('/v1/jobs/{job_id}/versions','POST'), ('/v1/profiles','GET')}
    api.router.routes = [r for r in api.router.routes if not any(
        (getattr(r,'path',None),m) in replaced for m in getattr(r,'methods',[]))]
    files = dependencies.get('storage') or Storage(settings)

    @api.get('/v1/profiles')
    def profiles():
        return {'schema_version':'processing-profile-list/5.0', 'default':NATIVE,
                'profiles':[resolve_profile(n) for n in DEFINITIONS]}

    @api.post('/v1/jobs', status_code=202)
    async def upload(request:Request, filename:str=Query(min_length=1,max_length=180),
                     profile:str=NATIVE, render_dpi:float=120., image_dpi:float|None=None, rotation_deg:int=0):
        try:
            options = ProcessingOptions(profile=profile,render_dpi=render_dpi,image_dpi=image_dpi,rotation_deg=rotation_deg)
        except ValidationError:
            raise HTTPException(422, 'Invalid processing options')
        if request.headers.get('content-type','').split(';')[0] not in (
                'application/pdf','image/png','image/jpeg','application/octet-stream'):
            raise HTTPException(415, 'Send a PDF/PNG/JPEG raw body')
        # Resolve before storing an upload if the deployment binding is invalid.
        await run_in_threadpool(resolve_profile, options.profile)
        try:
            record = await files.upload(filename, request.stream())
        except OverflowError:
            raise HTTPException(413, 'Upload limit exceeded')
        row = await run_in_threadpool(repo.create, record, options.model_dump())
        return {'job_id':record['id'], 'version':row['version'], 'status':row['status'],
                'status_url':f"/v1/jobs/{record['id']}"}

    @api.post('/v1/jobs/{job_id}/versions', status_code=202)
    async def version(request:Request, job_id:UUID):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>4096:
                raise HTTPException(413, 'Options limit exceeded')
        try:
            options = ProcessingOptions.model_validate_json(bytes(body) or b'{}')
        except ValidationError:
            raise HTTPException(422, 'Invalid processing options')
        row = await run_in_threadpool(repo.new_version, job_id, options.model_dump())
        return {'job_id':str(job_id), 'version':row['version'], 'status':row['status']}
    comparisons = NativeComparisons(files); prior_lifespan = api.router.lifespan_context
    @asynccontextmanager
    async def lifespan(app):
        try:
            async with prior_lifespan(app): yield
        finally: comparisons.close()
    api.router.lifespan_context = lifespan
    paths = ('/v1/jobs/{job_id}/versions/{version}/pages/{page}/comparison',
             '/v1/jobs/{job_id}/versions/{version}/pages/{page}/comparison/dxf.svg')
    api.router.routes = [r for r in api.router.routes if getattr(r,'path',None) not in paths]
    def bound(job,version,page):
        if version<1 or not 0<=page<20: raise HTTPException(422,'Invalid version/page')
        document,_ = repo.job(job); run = repo.run(job,version)
        if str(document['id'])!=str(job) or str(run['job_id'])!=str(job) or run['version']!=version:
            raise ValueError('Context mismatch')
        return {**run,'input_sha256':document['input_sha256']}
    @api.get(paths[0])
    async def comparison(job_id:UUID,version:int,page:int):
        run = await run_in_threadpool(bound,job_id,version,page)
        try: value = await run_in_threadpool(comparisons.describe,run,page)
        except ComparisonLimit: raise HTTPException(413,'Comparison resource limit')
        body = encoded(value); digest = hashlib.sha256(body).hexdigest()
        return Response(body,media_type='application/json',headers={'Cache-Control':'no-store','ETag':'"'+digest+'"',
            'X-Content-SHA256':digest,'X-Content-Type-Options':'nosniff'})
    @api.get(paths[1])
    async def comparison_svg(job_id:UUID,version:int,page:int):
        run = await run_in_threadpool(bound,job_id,version,page)
        body,value = await run_in_threadpool(comparisons.svg,run,page)
        return Response(body,media_type='image/svg+xml',headers={'ETag':value['dxf']['svg_sha256'],
            'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff',
            'X-DXF-SHA256':value['dxf']['sha256'],'X-Content-SHA256':hashlib.sha256(body).hexdigest(),
            'Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'"})
    return api


def index_html():
    html = old_index()
    old = '<option value="experimental_draft" selected>Текущий CV · DXF по слоям</option>'
    if html.count(old)!=1: raise ValueError('Unexpected frozen profile selector')
    html = html.replace('DXF-черновик доступен в режиме просмотра. Native-анализ сохраняет подробные наблюдения; DXF в нём не создаётся.',
        'Для новой сборки выберите «CAD · целые объекты и рамка». Результат можно просмотреть и скачать в DXF.')
    return html.replace(old,'<option value="native_cad_draft" selected>CAD-объекты и текст</option>'
        '<option value="native_cad_layers_draft">CAD · целые объекты и рамка (эксперимент)</option>'
        '<option value="native_cad_review_demo">CAD + размер из примера</option>'
        '<option value="experimental_draft">Текущий CV · DXF по слоям</option>')


def app_javascript():
    source = old_javascript()
    source = source.replace("{experimental_draft:","{native_cad_layers_draft: 'CAD · целые объекты и рамка (эксперимент)', native_cad_draft: 'CAD-объекты и текст', native_cad_review_demo: 'CAD + размер из примера', experimental_draft:")
    hook = '    if (report.input_resized) {'
    if source.count(hook)!=1: raise ValueError('Unexpected frozen quality message')
    source = source.replace(hook,"""    if (report.mode === 'native_annotations_draft' && !report.empty) {
      warning.textContent = `CAD-объектов ${report.geometry_entity_count}. Текст: ${report.visible_text_count} видимых, ${report.review_text_count} для проверки. Размеры: ${report.automatic_dimensions} автоматически, ${report.review_dimensions} из примера. Масштаб не подтверждён.`;
      if (report.review_example_requested && !report.review_example_applied) warning.textContent += ' Пример размера не относится к этому файлу и не применён.';
    }
    if (report.candidate_revision === '5.0') warning.textContent += ` Новый CAD-профиль: подтверждено линий рамки ${report.frame_layer_assignments || 0}.`;
    if (report.input_resized) {""")
    return source


def create_web_app(settings, **dependencies):
    api = create_api(settings, **dependencies)
    @asynccontextmanager
    async def lifespan(app):
        async with api.router.lifespan_context(api):
            yield
    web = FastAPI(docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    @web.get('/')
    def index():
        return HTMLResponse(index_html(), headers={'Cache-Control':'no-store',
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"})
    @web.get('/demo/assets/app.js')
    def javascript():
        return Response(app_javascript(),media_type='text/javascript',headers={'Cache-Control':'no-store'})
    # Relative module imports resolve to the same frozen UI assets.
    web.mount('/demo/assets',StaticFiles(directory=ASSETS))
    web.mount('/workspace/assets',StaticFiles(directory=ASSETS))
    web.mount('/ui/assets',StaticFiles(directory=BRAND))
    web.mount('/',api)
    return web
