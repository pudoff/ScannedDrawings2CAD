"""Keep processing profile8 and the UI; install table registration extension6."""
from cad_recovery.service_profiles_v20 import api as previous
from cad_recovery.service_profiles_v20.queue import Queue
from cad_recovery.service_profiles_v15.api import dependencies_for
from cad_recovery.web_tables_api_v7 import install


def create_api(settings,**dependencies):
    dependencies.setdefault('queue',Queue(settings))
    values=dependencies_for(settings,dependencies)
    return install(previous.create_api(settings,**values),settings,values)


def create_web_app(settings,**dependencies):
    dependencies.setdefault('queue',Queue(settings))
    values=dependencies_for(settings,dependencies)
    web=previous.create_web_app(settings,**values)
    api=next(r.app for r in web.routes if getattr(r,'path',None)=='')
    install(api,settings,values)
    return web
