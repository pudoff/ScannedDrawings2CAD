"""Bounded binary uploads; long conversions never occupy an HTTP request."""
import hmac
import json
from uuid import UUID
from fastapi import FastAPI,Request,HTTPException,Query
from fastapi.responses import FileResponse,JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import ValidationError
from .config import ProcessingOptions
from cad_recovery.service_v1.queue import Queue
from cad_recovery.service_v1.storage import Storage

def public_run(row):
    return {k:row[k] for k in ('id','job_id','version','status','attempts','created_at','started_at',
        'finished_at','options','summary','error')}

def create_app(settings,queue=None,storage=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    if hasattr(repo,'bind_storage'):repo.bind_storage()
    app=FastAPI(title='CAD Recovery local presentation API',version='demo-1.0',docs_url=None,redoc_url=None)

    @app.middleware('http')
    async def authentication(request,call_next):
        if settings.token and not hmac.compare_digest(request.headers.get('authorization','').encode(),('Bearer '+settings.token).encode()):
            return JSONResponse({'detail':'Authorization required'},status_code=401)
        if request.method=='POST':
            value=request.headers.get('content-length')
            try:length=int(value) if value is not None else None
            except ValueError:return JSONResponse({'detail':'Invalid Content-Length'},status_code=400)
            if length is not None and length<0:return JSONResponse({'detail':'Invalid Content-Length'},status_code=400)
            limit=settings.max_upload_bytes if request.url.path=='/v1/jobs' else 4096
            if length is not None and length>limit:return JSONResponse({'detail':'Request body limit exceeded'},status_code=413)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing(request,error):return JSONResponse({'detail':'Job, version or artifact not found'},status_code=404)

    @app.exception_handler(LookupError)
    async def pending(request,error):return JSONResponse({'detail':'Artifacts are available only after completion'},status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request,error):return JSONResponse({'detail':'Invalid input or artifact integrity check failed'},status_code=400)

    @app.exception_handler(Exception)
    async def failure(request,error):return JSONResponse({'detail':'Service operation failed; retry status request later'},status_code=503)

    @app.get('/health')
    def health():
        if not repo.ping():raise HTTPException(503,'Database schema is unavailable')
        return {'status':'ok','schema_version':1}

    @app.post('/v1/jobs',status_code=202,openapi_extra={'requestBody':{'required':True,
        'content':{kind:{'schema':{'type':'string','format':'binary'}} for kind in ('application/pdf','image/png','image/jpeg')}}})
    async def upload(request:Request,filename:str=Query(min_length=1,max_length=180),
            profile:str='presentation_draft',render_dpi:float=120.,image_dpi:float|None=None,rotation_deg:int=0):
        try:options=ProcessingOptions(profile=profile,render_dpi=render_dpi,image_dpi=image_dpi,rotation_deg=rotation_deg)
        except ValidationError:raise HTTPException(422,'Invalid processing options')
        if request.headers.get('content-type','').split(';')[0] not in ('application/pdf','image/png','image/jpeg','application/octet-stream'):
            raise HTTPException(415,'Send a PDF, PNG or JPEG as the raw request body')
        try:record=await files.upload(filename,request.stream())
        except OverflowError:raise HTTPException(413,'Upload size limit exceeded')
        # A connection error may arrive after COMMIT; retain the original even
        # when enqueue acknowledgement is uncertain. Orphans are never served.
        run=await run_in_threadpool(repo.create,record,options.model_dump())
        return {'job_id':record['id'],'version':run['version'],'status':run['status'],
            'status_url':f"/v1/jobs/{record['id']}"}

    @app.get('/v1/jobs/{job_id}')
    def status(job_id:UUID):
        doc,runs=repo.job(job_id)
        return {'job_id':doc['id'],'original_name':doc['original_name'],'input_sha256':doc['input_sha256'],
            'size_bytes':doc['size_bytes'],'created_at':doc['created_at'],
            'status':runs[-1]['status'],'latest_version':runs[-1]['version'],'versions':[public_run(r) for r in runs]}

    @app.get('/v1/jobs/{job_id}/versions/{version}')
    def version_status(job_id:UUID,version:int):return public_run(repo.run(job_id,version))

    @app.post('/v1/jobs/{job_id}/versions',status_code=202,openapi_extra={'requestBody':{
        'content':{'application/json':{'schema':ProcessingOptions.model_json_schema()}}}})
    async def version(request:Request,job_id:UUID):
        body=bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body)>4096:raise HTTPException(413,'Options body limit exceeded')
        try:options=ProcessingOptions.model_validate_json(bytes(body) or b'{}')
        except ValidationError:raise HTTPException(422,'Invalid processing options')
        run=await run_in_threadpool(repo.new_version,job_id,options.model_dump())
        return {'job_id':job_id,'version':run['version'],'status':run['status']}

    @app.get('/v1/jobs/{job_id}/versions/{version}/artifacts')
    def artifacts(job_id:UUID,version:int):
        run=repo.run(job_id,version);_,manifest=files.manifest(run)
        return {'job_id':job_id,'version':version,'quality':manifest['summary'],
            'artifacts':[{k:v for k,v in item.items() if k!='path'}|{
                'download_url':f"/v1/jobs/{job_id}/versions/{version}/artifacts/{item['id']}"} for item in manifest['files']]}

    @app.get('/v1/jobs/{job_id}/versions/{version}/artifacts/{artifact_id}')
    def artifact(job_id:UUID,version:int,artifact_id:str):
        run=repo.run(job_id,version);path,item=files.artifact(run,artifact_id)
        return FileResponse(path,media_type=item['media_type'],filename=item['filename'],
            headers={'X-Content-SHA256':item['sha256'],'X-Content-Type-Options':'nosniff'})

    return app
