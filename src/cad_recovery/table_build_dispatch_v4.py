"""Dispatch only explicit v6 requests to the new registration composition."""
from .table_build_dispatch_v3 import build_table as previous


def build_table(request,progress):
    from .web_tables_contract_v7 import binding
    if request['table_binding']==binding():
        from .web_tables_build_v7 import build
        with progress.phase('registered_TEXT_preview' if request['edit'].get('text_selection') else 'registered_table_OCR'):
            return build(request,progress=progress)
    return previous(request,progress)
