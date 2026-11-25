"""Retain V31 HTTP behavior and expose the separately versioned runtime ID."""
from .compact_runtime_v1 import activate,binding
activate()
from .service_profiles_v24 import api as previous


def install(api):
    @api.get('/v1/runtime')
    def runtime():
        return dict(binding(),reprocessing_excluded_profiles=['native_cad_review_demo'],
            clean_demo_default_policy_accepted=False)
    return api


def create_api(settings,**dependencies):return install(previous.create_api(settings,**dependencies))


def create_web_app(settings,**dependencies):
    web=previous.create_web_app(settings,**dependencies)
    api=next(r.app for r in web.routes if getattr(r,'path',None)=='');install(api)
    return web
