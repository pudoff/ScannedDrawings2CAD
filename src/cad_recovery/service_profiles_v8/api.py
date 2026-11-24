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
from cad_recovery.profile_contract_v3 import ProcessingOptions, DEFINITIONS, resolve_profile, EXPERIMENTAL
from .queue import Queue
from cad_recovery.web_v1.reviews import Reviews

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
    api = previous_api(settings, queue=repo, **dependencies)
    replaced = {('/v1/jobs','POST'), ('/v1/jobs/{job_id}/versions','POST'), ('/v1/profiles','GET')}
    api.router.routes = [r for r in api.router.routes if not any(
        (getattr(r,'path',None),m) in replaced for m in getattr(r,'methods',[]))]
    from cad_recovery.service_profiles_v2.storage import Storage
    files = dependencies.get('storage') or Storage(settings)

    @api.get('/v1/profiles')
    def profiles():
        return {'schema_version':'processing-profile-list/3.0', 'default':EXPERIMENTAL,
                'profiles':[resolve_profile(n) for n in DEFINITIONS]}

    @api.post('/v1/jobs', status_code=202)
    async def upload(request:Request, filename:str=Query(min_length=1,max_length=180),
                     profile:str=EXPERIMENTAL, render_dpi:float=120., image_dpi:float|None=None, rotation_deg:int=0):
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
    return api


def index_html():
    html = (ASSETS/'index.html').read_text('utf-8')
    html = html.replace('/workspace/assets/app.js', '/demo/assets/app.js')
    return html.replace('<option value="presentation_draft" selected>',
        '<option value="experimental_draft" selected>Текущий CV · DXF по слоям</option><option value="presentation_draft">')


def app_javascript():
    source = (ASSETS/'app.js').read_text('utf-8')
    source = source.replace("{native_observations: 'Native-анализ", "{experimental_draft: 'Текущий CV · DXF по слоям', native_observations: 'Native-анализ")
    source = source.replace("report.mode === 'raster_ink_draft' ? 'Штриховой черновик скана' : 'Векторный черновик PDF'",
        "report.mode === 'experimental_cv_draft' ? 'Экспериментальный CV · CAD-слои' : report.mode === 'raster_ink_draft' ? 'Штриховой черновик скана' : 'Векторный черновик PDF'")
    source = source.replace("return candidates.find(a => a.filename.startsWith(prefix)",
        "return candidates.find(a => a.filename === `${prefix}-cv-preview.png`) || candidates.find(a => a.filename.startsWith(prefix)")
    source = source.replace("if (report.input_resized) {",
        "if (report.experimental_fallback) warning.textContent += ' Экспериментальный CV отказался от листа; показан предыдущий штриховой профиль.';\n    if (report.input_resized) {")
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
