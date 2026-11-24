"""Versioned UI assets; server authorization and V19 worker contracts retained."""
from pathlib import Path
from fastapi.responses import Response
from fastapi.routing import APIRoute
from cad_recovery.service_profiles_v15.api import create_api,create_web_app as previous

ASSETS=Path(__file__).resolve().parents[1]/'web_operator_assets_v1'


def create_web_app(settings,**dependencies):
    web=previous(settings,**dependencies)
    paths={'/demo/assets/tables.js':'tables.js','/demo/assets/operator-key.js':'operator-key.js'}
    web.router.routes=[r for r in web.router.routes if getattr(r,'path',None) not in paths]
    def endpoint(name):
        async def asset():
            return Response((ASSETS/name).read_text(encoding='utf-8'),media_type='text/javascript',
                            headers={'Cache-Control':'no-store'})
        return asset
    for path,name in paths.items():
        web.router.routes.insert(0,APIRoute(path,endpoint(name),methods=['GET']))
    return web
