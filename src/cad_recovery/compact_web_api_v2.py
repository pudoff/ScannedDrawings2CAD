"""Clean-demo policy over the qualified compact runtime and V31 HTTP service."""
from .compact_runtime_v1 import activate,binding
activate()
from .service_profiles_v24 import api as previous
from .clean_demo_policy_v1 import install,install_web,DEFAULT,EXCLUDED,SCHEMA


def check_assets():
    from .service_profiles_v10.api import index_html
    from .clean_demo_policy_v1 import index_html as transform,javascript
    assets=previous.check_assets();assets['app.js']=javascript(assets['app.js'])
    html=transform(index_html())
    if 'value="'+DEFAULT+'" selected' not in html or 'value="'+EXCLUDED+'"' in html:
        raise ValueError('Clean demo default selector differs')
    return assets


def upgrade(api):
    install(api)
    @api.get('/v1/runtime')
    def runtime():
        return dict(binding(),reprocessing_excluded_profiles=[EXCLUDED],
            clean_demo_default_policy_accepted=True,default_profile=DEFAULT,submission_policy=SCHEMA)
    return api


def create_api(settings,**dependencies):return upgrade(previous.create_api(settings,**dependencies))


def create_web_app(settings,**dependencies):
    web=previous.create_web_app(settings,**dependencies)
    api=next(r.app for r in web.routes if getattr(r,'path',None)=='');upgrade(api)
    return install_web(web)
