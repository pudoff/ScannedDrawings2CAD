"""44LABS observation inspector; existing processing/queue bindings are reused."""
from pathlib import Path
from uuid import UUID
import hashlib
from fastapi import FastAPI,HTTPException,Query
from fastapi.responses import FileResponse,JSONResponse,Response
from fastapi.staticfiles import StaticFiles
from cad_recovery.service_profiles_v2.api import create_api as previous_api,BRAND
from cad_recovery.service_profiles_v2.queue import Queue
from cad_recovery.service_profiles_v2.storage import Storage
from cad_recovery.dataset_layout_v1 import encoded
from .inspection import Inspector,InspectionLimit
from cad_recovery.observation_graph_v1 import ObservationLimit

ASSETS=Path(__file__).parent/'assets'


def create_api(settings,queue=None,storage=None,reviews=None):
    repo=queue or Queue(settings);files=storage or Storage(settings)
    api=previous_api(settings,queue=repo,storage=files,reviews=reviews);inspector=Inspector(files)

    @api.get('/v1/jobs/{job_id}/versions/{version}/pages/{page_index}/inspection')
    def inspect(job_id:UUID,version:int,page_index:int):
        if version<1 or page_index<0:raise HTTPException(422,'Invalid version or page')
        run=repo.run(job_id,version);document,_=repo.job(job_id)
        if str(run['job_id'])!=str(job_id) or run['version']!=version or str(document['id'])!=str(job_id):
            raise ValueError('Inspection context mismatch')
        run={**run,'input_sha256':document['input_sha256']}
        try:value=inspector.inspect(run,page_index)
        except (InspectionLimit,ObservationLimit):raise HTTPException(413,'Observation inspection resource limit exceeded')
        body=encoded(value);etag=hashlib.sha256(body).hexdigest()
        return Response(body,media_type='application/json',headers={'ETag':'"'+etag+'"','Cache-Control':'no-store',
            'X-Content-SHA256':etag,'X-Content-Type-Options':'nosniff'})
    return api


def create_web_app(settings,**dependencies):
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    @app.get('/')
    def index():
        return FileResponse(ASSETS/'index.html',media_type='text/html',headers={'Cache-Control':'no-cache',
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            'X-Content-Type-Options':'nosniff'})
    app.mount('/graph/assets',StaticFiles(directory=ASSETS));app.mount('/ui/assets',StaticFiles(directory=BRAND))
    app.mount('/',create_api(settings,**dependencies));return app
