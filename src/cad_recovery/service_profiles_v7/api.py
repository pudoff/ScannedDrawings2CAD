"""Compact UI with explicit diagnostic-only handling of missing part geometry."""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from cad_recovery.service_profiles_v6.api import create_api,BRAND

ASSETS=Path(__file__).parent/'assets'


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
