"""Public 44LABS shell with authenticated native/v1 profile API."""
from pathlib import Path
from uuid import UUID
from fastapi import FastAPI,Request,HTTPException,Query
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from cad_recovery.service_demo_v1.api import create_app as historical_api
from cad_recovery.profile_contract_v2 import ProcessingOptions,DEFINITIONS,resolve_profile,ProfileError
from cad_recovery.web_v1.reviews import Reviews,ReviewInput
from .queue import Queue
from .storage import Storage

ASSETS=Path(__file__).parent/'assets'
BRAND=Path(__file__).parents[1]/'web_v1/assets'

def create_api(settings,queue=None,storage=None,reviews=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    api=historical_api(settings,queue=repo,storage=files)
    # Compose a new app instance; historical modules/routes remain immutable.
    replaced={('/v1/jobs','POST'),('/v1/jobs/{job_id}/versions','POST')}
    api.router.routes=[r for r in api.router.routes if not any((getattr(r,'path',None),m) in replaced for m in getattr(r,'methods',[]))]
    review_repo=reviews or Reviews(settings)
    if reviews is None:review_repo.initialize()

    @api.exception_handler(ProfileError)
    async def profile_error(request,error):
        return JSONResponse({'detail':str(error),'code':error.code},status_code=409 if error.code=='profile_contract_changed' else 503)

    @api.get('/v1/profiles')
    def profiles():return {'schema_version':'processing-profile-list/2.0','default':'presentation_draft','profiles':[resolve_profile(n) for n in DEFINITIONS]}

    @api.post('/v1/jobs',status_code=202)
    async def upload(request:Request,filename:str=Query(min_length=1,max_length=180),profile:str='presentation_draft',
                     render_dpi:float=120.,image_dpi:float|None=None,rotation_deg:int=0):
        try:options=ProcessingOptions(profile=profile,render_dpi=render_dpi,image_dpi=image_dpi,rotation_deg=rotation_deg)
        except ValidationError:raise HTTPException(422,'Invalid processing options')
        if request.headers.get('content-type','').split(';')[0] not in ('application/pdf','image/png','image/jpeg','application/octet-stream'):
            raise HTTPException(415,'Send a PDF/PNG/JPEG raw body')
        try:record=await files.upload(filename,request.stream())
        except OverflowError:raise HTTPException(413,'Upload limit exceeded')
        row=await run_in_threadpool(repo.create,record,options.model_dump())
        return {'job_id':record['id'],'version':row['version'],'status':row['status'],'status_url':f"/v1/jobs/{record['id']}"}

    @api.post('/v1/jobs/{job_id}/versions',status_code=202)
    async def version(request:Request,job_id:UUID):
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>4096:raise HTTPException(413,'Options limit exceeded')
        try:options=ProcessingOptions.model_validate_json(bytes(body) or b'{}')
        except ValidationError:raise HTTPException(422,'Invalid processing options')
        row=await run_in_threadpool(repo.new_version,job_id,options.model_dump())
        return {'job_id':job_id,'version':row['version'],'status':row['status']}

    @api.get('/v1/jobs')
    def documents(limit:int=Query(default=30,ge=1,le=100),offset:int=Query(default=0,ge=0,le=100000),query:str=Query(default='',max_length=80)):
        return review_repo.documents(limit,offset,query)

    @api.get('/v1/jobs/{job_id}/versions/{version}/reviews')
    def list_reviews(job_id:UUID,version:int):return {'reviews':review_repo.list(job_id,version),'applied_to_automatic_result':False}

    @api.post('/v1/jobs/{job_id}/versions/{version}/reviews',status_code=201)
    async def add_review(request:Request,job_id:UUID,version:int):
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>4096:raise HTTPException(413,'Review limit exceeded')
        try:review=ReviewInput.model_validate_json(bytes(body))
        except ValidationError:raise HTTPException(422,'Invalid review proposal')
        row=await run_in_threadpool(review_repo.add,job_id,version,review)
        return {'review':row,'applied_to_automatic_result':False}
    return api

def create_web_app(settings,**dependencies):
    web=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    @web.get('/')
    def index():
        return FileResponse(ASSETS/'index.html',media_type='text/html',headers={'Cache-Control':'no-cache',
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            'X-Content-Type-Options':'nosniff'})
    web.mount('/native/assets',StaticFiles(directory=ASSETS));web.mount('/ui/assets',StaticFiles(directory=BRAND))
    web.mount('/',create_api(settings,**dependencies));return web
