"""Retained source registration with a separate text-only extension identity."""
from .web_tables_context_v2 import describe as previous, validate_edit, _sealed
from .web_tables_contract_v4 import binding


def describe(*args, **kwargs):
    value, image = previous(*args, **kwargs)
    value.pop('context_sha256', None)
    value.update(schema_version='web-table-context/4.0', binding_sha256=binding()['sha256'])
    return _sealed(value), image
