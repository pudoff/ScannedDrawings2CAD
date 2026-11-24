"""Process-local registration adapter; reuse the qualified evidence/TEXT producer."""
from contextlib import ExitStack
from unittest.mock import patch
from . import web_tables_build_v5 as previous
from .web_tables_contract_v6 import binding
from .web_tables_context_v6 import describe
from .web_tables_context_v2 import _sealed


def build(request,*,engine=None,progress=None):
    if request['table_binding'] != binding():
        raise ValueError('Table registration adapter binding changed')
    from . import web_tables_context_v3 as old_context
    from . import web_tables_build_v3 as old_producer
    from .web_tables_contract_v3 import binding as old_binding

    def registered_legacy(*args,**kwargs):
        value,image=describe(*args,**kwargs)
        value.pop('context_sha256',None)
        value.update(schema_version='web-table-context/3.0',binding_sha256=old_binding()['sha256'])
        return _sealed(value),image

    # Only the dedicated, serial table child is patched. The threaded Web API
    # uses its own v7 routes and never changes a shared profile resolver.
    with ExitStack() as stack:
        for module,name,value in ((previous,'binding',binding),(previous,'describe',describe),
                (old_context,'describe',registered_legacy),(old_producer,'describe',registered_legacy)):
            stack.enter_context(patch.object(module,name,value))
        return previous.build(request,engine=engine,progress=progress)
