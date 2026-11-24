"""Authenticated comparison API over immutable processing workers/profiles."""
from pathlib import Path
from uuid import UUID
import hashlib
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,Response
from fastapi.staticfiles import StaticFiles
from cad_recovery.service_profiles_v3.api import create_api as previous_api,BRAND
from cad_recovery.service_profiles_v2.queue import Queue
from cad_recovery.service_profiles_v2.storage import Storage
from cad_recovery.comparison_service_v1 import Comparisons,ComparisonLimit
from cad_recovery.dataset_layout_v1 import encoded

ASSETS=Path(__file__).parent/'assets'


def create_api(settings,queue=None,storage=None,reviews=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    api=previous_api(settings,queue=repo,storage=files,reviews=reviews);comparisons=Comparisons(files)
    api.state.comparisons=comparisons
    previous_lifespan=api.router.lifespan_context
    @asynccontextmanager
    async def lifespan(app):
        try:
            async with previous_lifespan(app):yield
        finally:comparisons.close()
    api.router.lifespan_context=lifespan
    def bound(job_id,version,page):
        if version<1 or page<0:raise HTTPException(422,'Invalid version or page')
        run=repo.run(job_id,version);document,_=repo.job(job_id)
        if str(run['job_id'])!=str(job_id) or run['version']!=version or str(document['id'])!=str(job_id):raise ValueError('Comparison context mismatch')
        return {**run,'input_sha256':document['input_sha256']}
    @api.get('/v1/jobs/{job_id}/versions/{version}/pages/{page}/comparison')
    def comparison(job_id:UUID,version:int,page:int):
        try:value=comparisons.describe(bound(job_id,version,page),page)
        except ComparisonLimit:raise HTTPException(413,'Comparison resource limit')
        body=encoded(value);digest=hashlib.sha256(body).hexdigest()
        return Response(body,media_type='application/json',headers={'Cache-Control':'no-store','ETag':'"'+digest+'"',
            'X-Content-SHA256':digest,'X-Content-Type-Options':'nosniff'})
    @api.get('/v1/jobs/{job_id}/versions/{version}/pages/{page}/comparison/dxf.svg')
    def dxf_image(job_id:UUID,version:int,page:int):
        body,value=comparisons.svg(bound(job_id,version,page),page)
        return Response(body,media_type='image/svg+xml',headers={'Cache-Control':'no-store',
            'X-Content-SHA256':hashlib.sha256(body).hexdigest(),'X-DXF-SHA256':value['dxf']['sha256'],
            'X-Content-Type-Options':'nosniff','Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'"})
    return api


def create_web_app(settings,**dependencies):
    api=create_api(settings,**dependencies)
    @asynccontextmanager
    async def lifespan(app):
        async with api.router.lifespan_context(api):yield
    web=FastAPI(docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    @web.get('/')
    def index():
        return FileResponse(ASSETS/'index.html',media_type='text/html',headers={'Cache-Control':'no-cache',
            'X-Content-Type-Options':'nosniff','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"})
    web.mount('/workspace/assets',StaticFiles(directory=ASSETS));web.mount('/ui/assets',StaticFiles(directory=BRAND))
    web.mount('/',api)
    return web
