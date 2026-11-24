"""Retained registration, independently bound TEXT chain producer."""
from .web_tables_context_v4 import describe as previous,validate_edit
from .web_tables_context_v2 import _sealed
from .web_tables_contract_v5 import binding


def describe(*args,**kwargs):
    value,image=previous(*args,**kwargs)
    value.pop('context_sha256',None)
    value.update(schema_version='web-table-context/5.0',binding_sha256=binding()['sha256'])
    return _sealed(value),image
