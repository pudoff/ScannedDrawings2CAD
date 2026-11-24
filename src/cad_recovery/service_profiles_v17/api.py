"""Use the exact-parent text API while preserving V20 credential UI."""
from cad_recovery.service_profiles_v15.api import dependencies_for
from cad_recovery.service_profiles_v16.api import create_web_app as previous_web
from cad_recovery.service_profiles_v15.api import create_api as previous_api
from cad_recovery.web_tables_api_v6 import install


def create_api(settings,**dependencies):
    values=dependencies_for(settings,dependencies)
    return install(previous_api(settings,**values),settings,values)


def create_web_app(settings,**dependencies):
    values=dependencies_for(settings,dependencies);web=previous_web(settings,**values)
    api=next(r.app for r in web.router.routes if getattr(r,'path',None)=='')
    install(api,settings,values)
    return web
