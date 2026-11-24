"""Stored processing profile validation for registered manual TEXT previews."""
import hashlib
from .dataset_layout_v1 import digest,encoded
from .web_tables_contract_v5 import ROOT,Approval,TextSelection,Request,parse_request


def binding():
    from .web_tables_contract_v5 import binding as previous
    names=('web_tables_contract_v6.py','web_tables_context_v6.py','web_tables_api_v7.py',
        'web_tables_build_v6.py','web_tables_worker_v9.py','table_worker_request_v3.py',
        'table_build_dispatch_v3.py','service_profiles_v21/__init__.py','service_profiles_v21/api.py',
        'profile_contract_v8.py')
    body=dict(schema_version='web-table-extension/6.0',previous=previous(),
        mode='stored_profile_bound_cumulative_TEXT_previews',supported_parent_profiles=[6,7,8],
        automatic_routing=False,quality_accepted=False,CV_OCR_algorithms_changed=False,
        code=[dict(path='src/cad_recovery/'+n,sha256=digest(ROOT/'src/cad_recovery'/n)) for n in names]
        +[dict(path='scripts/'+n,sha256=digest(ROOT/'scripts'/n)) for n in
          ('build_web_tables_v9.py','run_profiles_v26.py','start_profiles_v26.py')])
    return {**body,'sha256':hashlib.sha256(encoded(body)).hexdigest()}
