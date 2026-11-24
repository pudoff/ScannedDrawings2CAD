"""Serve the semantic delta UI and qualified pure helper, without API changes."""
from pathlib import Path
from fastapi.responses import Response
from fastapi.routing import APIRoute
from cad_recovery.service_profiles_v17.api import create_api,create_web_app as previous

ROOT=Path(__file__).resolve().parents[1]


def create_web_app(settings,**dependencies):
    web=previous(settings,**dependencies)
    paths={'/demo/assets/text-review.js':ROOT/'web_table_text_assets_v3/text-review.js',
           '/demo/assets/text-delta.js':ROOT/'web_table_text_protocol_v1/text-delta.js'}
    web.router.routes=[r for r in web.router.routes if getattr(r,'path',None) not in paths]
    def endpoint(path):
        async def asset():
            return Response(path.read_text(encoding='utf-8'),media_type='text/javascript',headers={'Cache-Control':'no-store'})
        return asset
    for name,path in paths.items():web.router.routes.insert(0,APIRoute(name,endpoint(path),methods=['GET']))
    return web
