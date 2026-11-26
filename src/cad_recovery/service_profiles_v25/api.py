"""Same processing API; correctly scaled SVG and legible standalone DXF."""
from contextlib import asynccontextmanager
from uuid import UUID
import hashlib,inspect
from fastapi import HTTPException
from fastapi.responses import Response
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from cad_recovery.service_profiles_v24 import api as previous
from cad_recovery.service_profiles_v21.queue import Queue
from cad_recovery.service_profiles_v10.storage import Storage
from cad_recovery.service_profiles_v15.api import dependencies_for
from cad_recovery.comparison_display_v2 import DisplayComparisons,inspector_javascript
from cad_recovery.comparison_service_v1 import ComparisonLimit
from cad_recovery.dataset_layout_v1 import encoded


def check_assets():
    sources=previous.check_assets();sources['inspector.js']=inspector_javascript(sources['inspector.js'])
    return sources


def values_for(settings,dependencies):
    dependencies.setdefault('queue',Queue(settings));dependencies.setdefault('storage',Storage(settings))
    return dependencies_for(settings,dependencies)


def install(api,values):
    comparisons=DisplayComparisons(values['storage']);prior=api.router.lifespan_context;repo=values['queue']
    @asynccontextmanager
    async def lifespan(app):
        try:
            async with prior(app):yield
        finally:comparisons.close()
    api.router.lifespan_context=lifespan
    base='/v1/jobs/{job_id}/versions/{version}/pages/{page}/comparison'
    api.router.routes=[r for r in api.router.routes if getattr(r,'path',None) not in (base,base+'/dxf.svg')]
    def bound(job,version,page):
        if version<1 or not 0<=page<20:raise HTTPException(422,'Invalid version/page')
        document,_=repo.job(job);run=repo.run(job,version)
        if str(document['id'])!=str(job) or str(run['job_id'])!=str(job) or run['version']!=version:raise ValueError('Context mismatch')
        return {**run,'input_sha256':document['input_sha256']}
    @api.get(base)
    async def description(job_id:UUID,version:int,page:int):
        run=await run_in_threadpool(bound,job_id,version,page)
        try:value=await run_in_threadpool(comparisons.describe,run,page)
        except ComparisonLimit:raise HTTPException(413,'Comparison resource limit')
        body=encoded(value);sha=hashlib.sha256(body).hexdigest()
        return Response(body,media_type='application/json',headers={'Cache-Control':'no-store','ETag':'"'+sha+'"','X-Content-SHA256':sha,'X-Content-Type-Options':'nosniff'})
    @api.get(base+'/dxf.svg')
    async def svg(job_id:UUID,version:int,page:int):
        run=await run_in_threadpool(bound,job_id,version,page);body,value=await run_in_threadpool(comparisons.svg,run,page)
        return Response(body,media_type='image/svg+xml',headers={'ETag':value['dxf']['svg_sha256'],
            'Cache-Control':'private, no-store','X-DXF-SHA256':value['dxf']['sha256'],
            'X-Content-SHA256':hashlib.sha256(body).hexdigest(),'Content-Security-Policy':"default-src 'none'; style-src 'unsafe-inline'"})
    return api


def create_api(settings,**dependencies):
    values=values_for(settings,dependencies)
    return install(previous.create_api(settings,**values),values)


def create_web_app(settings,**dependencies):
    values=values_for(settings,dependencies);web=previous.create_web_app(settings,**values)
    install(next(r.app for r in web.routes if getattr(r,'path',None)==''),values)
    path='/demo/assets/inspector.js';old=next(r for r in web.routes if getattr(r,'path',None)==path)
    async def javascript():
        response=old.endpoint()
        if inspect.isawaitable(response):response=await response
        return Response(inspector_javascript(response.body.decode('utf-8')),media_type='text/javascript',headers={'Cache-Control':'no-store'})
    web.router.routes=[r for r in web.routes if r is not old]
    web.router.routes.insert(0,APIRoute(path,javascript,methods=['GET']))
    return web
