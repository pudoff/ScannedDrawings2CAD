"""Same-origin UI; public static shell, authenticated API and append-only review."""
from pathlib import Path
from uuid import UUID
from fastapi import FastAPI,Request,HTTPException,Query
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from cad_recovery.service_demo_v1.api import create_app
from cad_recovery.web_v1.reviews import Reviews,ReviewInput
from cad_recovery.profile_contract_v1 import DEFINITIONS,ProfileError,resolve_profile
from .queue import Queue

ASSETS=Path(__file__).parents[1]/'web_demo_v1/assets'
BRAND_ASSETS=Path(__file__).parents[1]/'web_v1/assets'

def create_web_app(settings):
    api=create_app(settings,queue=Queue(settings));reviews=Reviews(settings);reviews.initialize()

    @api.exception_handler(ProfileError)
    async def profile_error(request,error):
        status=409 if error.code=='profile_contract_changed' else 503
        return JSONResponse({'detail':str(error),'code':error.code},status_code=status)

    @api.get('/v1/profiles')
    def profiles():
        return {'schema_version':'processing-profile-list/1.0','default':'presentation_draft',
                'profiles':[resolve_profile(name) for name in DEFINITIONS]}

    @api.get('/v1/jobs')
    def documents(limit:int=Query(default=30,ge=1,le=100),offset:int=Query(default=0,ge=0,le=100000),query:str=Query(default='',max_length=80)):
        return reviews.documents(limit,offset,query)

    @api.get('/v1/jobs/{job_id}/versions/{version}/reviews')
    def list_reviews(job_id:UUID,version:int):
        return {'reviews':reviews.list(job_id,version),'applied_to_automatic_result':False}

    @api.post('/v1/jobs/{job_id}/versions/{version}/reviews',status_code=201,
        openapi_extra={'requestBody':{'required':True,'content':{'application/json':{'schema':ReviewInput.model_json_schema()}}}})
    async def add_review(request:Request,job_id:UUID,version:int):
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>4096:raise HTTPException(413,'Review body limit exceeded')
        try:review=ReviewInput.model_validate_json(bytes(body))
        except ValidationError:raise HTTPException(422,'Invalid review proposal')
        row=await run_in_threadpool(reviews.add,job_id,version,review)
        return {'review':row,'applied_to_automatic_result':False}

    web=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)

    @web.get('/',include_in_schema=False)
    def index():
        return FileResponse(ASSETS/'index.html',media_type='text/html',headers={
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            'Referrer-Policy':'no-referrer','X-Content-Type-Options':'nosniff','Cache-Control':'no-cache'})

    web.mount('/demo/assets',StaticFiles(directory=ASSETS),name='demo-assets')
    web.mount('/ui/assets',StaticFiles(directory=BRAND_ASSETS),name='web-assets')
    web.mount('/',api,name='api')
    return web
