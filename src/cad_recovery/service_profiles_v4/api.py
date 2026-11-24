"""Compact assets over the unchanged authenticated processing/inspection API."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from cad_recovery.service_profiles_v3.api import create_api,BRAND

ASSETS=Path(__file__).parent/'assets'


def create_web_app(settings,**dependencies):
    app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)
    @app.get('/')
    def index():
        return FileResponse(ASSETS/'index.html',media_type='text/html',headers={
            'Cache-Control':'no-cache','X-Content-Type-Options':'nosniff',
            'Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; font-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'"})
    app.mount('/workspace/assets',StaticFiles(directory=ASSETS))
    app.mount('/ui/assets',StaticFiles(directory=BRAND))
    app.mount('/',create_api(settings,**dependencies))
    return app
