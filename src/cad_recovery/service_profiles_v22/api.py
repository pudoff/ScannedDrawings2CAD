"""V27 adds coverage UI; routes keep authentication and stored version identity."""
import inspect
from pathlib import Path
from fastapi.responses import Response
from fastapi.routing import APIRoute
from cad_recovery.service_profiles_v21 import api as previous
from cad_recovery.service_profiles_v21.queue import Queue
from cad_recovery.service_profiles_v15.api import dependencies_for
from cad_recovery.profile_contract_v9 import DEFINITIONS,NATIVE,resolve_profile
from cad_recovery.web_tables_api_v8 import install

ASSET=Path(__file__).resolve().parents[1]/'search_completeness_assets_v1/coverage.js'


def upgrade(api,settings,values):
    api.router.routes=[r for r in api.router.routes if not (getattr(r,'path',None)=='/v1/profiles' and 'GET' in getattr(r,'methods',[]))]
    @api.get('/v1/profiles')
    def profiles():return dict(schema_version='processing-profile-list/9.0',default=NATIVE,profiles=[resolve_profile(n) for n in DEFINITIONS])
    install(api,settings,values);return api


def create_api(settings,**dependencies):
    dependencies.setdefault('queue',Queue(settings));values=dependencies_for(settings,dependencies)
    return upgrade(previous.create_api(settings,**values),settings,values)


def augment_javascript(source):
    state_hook='  function renderState() {';page_hook='  function renderPage() {'
    if source.count(state_hook)!=1 or source.count(page_hook)!=1:raise ValueError('Coverage UI hooks changed')
    call="renderSearchCoverage(document, state.bundle?.quality?.pages?.find(p => p.page_index === state.page)?.search_completeness, run()?.status);"
    source="import {renderSearchCoverage} from './search-coverage.js';\n"+source
    source=source.replace(state_hook,state_hook+'\n    '+call).replace(page_hook,page_hook+'\n    '+call)
    return source


def create_web_app(settings,**dependencies):
    dependencies.setdefault('queue',Queue(settings));values=dependencies_for(settings,dependencies)
    web=previous.create_web_app(settings,**values)
    api=next(r.app for r in web.routes if getattr(r,'path',None)=='');upgrade(api,settings,values)
    old=next(r for r in web.routes if getattr(r,'path',None)=='/demo/assets/app.js')
    async def javascript():
        response=old.endpoint()
        if inspect.isawaitable(response):response=await response
        return Response(augment_javascript(response.body.decode('utf-8')),media_type='text/javascript',headers={'Cache-Control':'no-store'})
    async def coverage():return Response(ASSET.read_text(encoding='utf-8'),media_type='text/javascript',headers={'Cache-Control':'no-store'})
    web.router.routes=[r for r in web.routes if r is not old]
    web.router.routes.insert(0,APIRoute('/demo/assets/app.js',javascript,methods=['GET']))
    web.router.routes.insert(0,APIRoute('/demo/assets/search-coverage.js',coverage,methods=['GET']))
    return web
