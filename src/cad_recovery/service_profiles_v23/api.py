"""Guard late UI responses without changing the V27 API or pipeline."""
import inspect
from pathlib import Path
from fastapi.responses import Response
from fastapi.routing import APIRoute
from cad_recovery.service_profiles_v22.api import create_api,create_web_app as previous
from cad_recovery.web_response_scope_v1 import upgrade

STATIC=Path(__file__).resolve().parents[1]/'service_profiles_v7/assets'
NAMES=('app.js','tables.js','text-review.js','corrections.js','inspector.js')


def check_assets():
    """Validate every guarded asset without a database or service startup."""
    from cad_recovery.web_tables_api_v3 import app_js
    from cad_recovery.service_profiles_v22.api import augment_javascript
    root=STATIC.parents[1]
    sources={'app.js':augment_javascript(app_js()),
        'tables.js':(root/'web_operator_assets_v1/tables.js').read_text(encoding='utf-8'),
        'text-review.js':(root/'web_table_text_assets_v3/text-review.js').read_text(encoding='utf-8')}
    sources.update({name:(STATIC/name).read_text(encoding='utf-8') for name in ('corrections.js','inspector.js')})
    return {name:upgrade('/demo/assets/'+name,sources[name]) for name in NAMES}


def create_web_app(settings,**dependencies):
    web=previous(settings,**dependencies)
    for name in NAMES:
        path='/demo/assets/'+name
        rows=[r for r in web.routes if getattr(r,'path',None)==path]
        if len(rows)!=(0 if name in ('corrections.js','inspector.js') else 1):
            raise ValueError('Unexpected upstream UI route: '+path)
        old=rows[0] if rows else None
        def endpoint(path,name,old):
            async def javascript():
                if old is None:source=(STATIC/name).read_text(encoding='utf-8')
                else:
                    response=old.endpoint()
                    if inspect.isawaitable(response):response=await response
                    source=response.body.decode('utf-8')
                return Response(upgrade(path,source),media_type='text/javascript',headers={'Cache-Control':'no-store'})
            return javascript
        web.router.routes=[r for r in web.routes if r is not old]
        web.router.routes.insert(0,APIRoute(path,endpoint(path,name,old),methods=['GET']))
    return web
