"""Exact-parent TEXT preview chains; CV/OCR and selection schema retained."""
import hashlib
from .dataset_layout_v1 import digest,encoded
from .web_tables_contract_v4 import ROOT,Approval,TextSelection,Request,parse_request


def binding():
    from .web_tables_contract_v4 import binding as previous
    names=('web_tables_contract_v5.py','web_tables_context_v5.py','web_text_lineage_v1.py',
        'web_tables_build_v5.py','web_tables_api_v6.py','web_tables_worker_v8.py',
        'table_worker_request_v2.py','table_build_dispatch_v2.py')
    body=dict(schema_version='web-table-extension/5.0',previous=previous(),
        mode='exact_parent_cumulative_TEXT_previews',quality_accepted=False,automatic_routing=False,
        code=[dict(path='src/cad_recovery/'+name,sha256=digest(ROOT/'src/cad_recovery'/name)) for name in names]
        +[dict(path='scripts/'+name,sha256=digest(ROOT/'scripts'/name)) for name in ('build_web_tables_v8.py','run_profiles_v21.py')])
    return {**body,'sha256':hashlib.sha256(encoded(body)).hexdigest()}
