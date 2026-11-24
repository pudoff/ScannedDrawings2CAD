"""Advertise compatible input bindings through the retained V24 interface."""
from cad_recovery.service_profiles_v19 import api as previous
from cad_recovery.profile_contract_v8 import DEFINITIONS, NATIVE, resolve_profile
from .queue import Queue


def upgrade(api):
    api.router.routes = [r for r in api.router.routes if not
        (getattr(r,'path',None)=='/v1/profiles' and 'GET' in getattr(r,'methods',[]))]
    @api.get('/v1/profiles')
    def profiles():
        return dict(schema_version='processing-profile-list/8.0',default=NATIVE,
                    profiles=[resolve_profile(n) for n in DEFINITIONS])
    return api


def create_api(settings,**dependencies):
    dependencies.setdefault('queue',Queue(settings))
    return upgrade(previous.create_api(settings,**dependencies))


def create_web_app(settings,**dependencies):
    dependencies.setdefault('queue',Queue(settings))
    web = previous.create_web_app(settings,**dependencies)
    for route in web.routes:
        if getattr(route,'path',None)=='' and hasattr(route,'app'):
            upgrade(route.app)
    return web
