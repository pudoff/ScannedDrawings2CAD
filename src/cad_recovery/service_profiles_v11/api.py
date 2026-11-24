"""Reuse existing upload/UI routes; queue and advertised bindings use inputv2."""
from cad_recovery.service_profiles_v10 import api as previous
from cad_recovery.profile_contract_v6 import DEFINITIONS,resolve_profile,NATIVE
from cad_recovery.raster_stream_v2 import POLICY
from .queue import Queue


def upgrade(api):
    api.router.routes=[r for r in api.router.routes if not (getattr(r,'path',None)=='/v1/profiles' and 'GET' in getattr(r,'methods',[]))]
    @api.get('/v1/profiles')
    def profiles():return dict(schema_version='processing-profile-list/6.0',default=NATIVE,profiles=[resolve_profile(n) for n in DEFINITIONS])
    @api.get('/v1/input-capabilities')
    def capabilities():return dict(policy=POLICY,PNG_native_windows=True,JPEG_native_windows=False,
        PDF='existing bounded page renderer; native image tiling is10.4.3',full_resolution_quality_accepted=False)
    return api


def create_api(settings,**dependencies):
    dependencies.setdefault('queue',Queue(settings))
    return upgrade(previous.create_api(settings,**dependencies))


def create_web_app(settings,**dependencies):
    dependencies.setdefault('queue',Queue(settings));web=previous.create_web_app(settings,**dependencies)
    for route in web.routes:
        if getattr(route,'path',None)=='' and hasattr(route,'app'):upgrade(route.app)
    return web
