"""Same page registration; only the explicit extension identity changes."""
from .web_tables_context_v2 import describe as previous,validate_edit,_sealed
from .web_tables_contract_v3 import binding


def describe(*args,**kwargs):
    context,image=previous(*args,**kwargs)
    context.pop('context_sha256',None)
    context['binding_sha256']=binding()['sha256']
    context['schema_version']='web-table-context/3.0'
    return _sealed(context),image
