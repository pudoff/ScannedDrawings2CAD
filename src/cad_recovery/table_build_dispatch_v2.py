"""Versioned chain producer, preserving all previously registered producers."""
from .table_build_dispatch_v1 import build_table as previous


def build_table(request,progress):
    from .web_tables_contract_v5 import binding
    if request['table_binding']==binding():
        from .web_tables_build_v5 import build
        with progress.phase('text_preview_chain' if request['edit'].get('text_selection') else 'table_OCR_evidence'):
            return build(request,progress=progress)
    return previous(request,progress)
