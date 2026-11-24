"""V18 API/UI contracts with bounded queues and read-only worker receipts."""
from cad_recovery.service_profiles_v12.queue import Queue
from cad_recovery.service_profiles_v9.storage import Storage
from cad_recovery.web_tables_store_v2 import TableStore
from cad_recovery.web_tables_api_v5 import create_api as previous_api,create_web_app as previous_web
from cad_recovery.worker_diagnostics_api_v1 import install


def dependencies_for(settings,dependencies):
    values=dict(dependencies)
    repo=values.get('queue') or Queue(settings)
    files=values.get('storage') or Storage(settings)
    store=values.get('table_store') or TableStore(repo)
    if values.get('table_store') is None:store.initialize()
    values.update(queue=repo,storage=files,table_store=store)
    return values


def create_api(settings,**dependencies):
    values=dependencies_for(settings,dependencies)
    return install(previous_api(settings,**values),values['queue'],values['storage'],table_store=values['table_store'])


def create_web_app(settings,**dependencies):
    values=dependencies_for(settings,dependencies);web=previous_web(settings,**values)
    api=next(r.app for r in web.router.routes if getattr(r,'path',None)=='')
    install(api,values['queue'],values['storage'],table_store=values['table_store'])
    return web
