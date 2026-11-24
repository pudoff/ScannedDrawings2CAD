"""Opt-in safe-text Web extension; prior bindings remain independently valid."""
import hashlib
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from .dataset_layout_v1 import encoded,digest
from .web_tables_contract_v1 import ROOT, Request as PreviousRequest, Approval

CODE=('web_tables_contract_v3.py','web_tables_context_v3.py','web_tables_build_v3.py',
      'web_tables_worker_v3.py','web_tables_api_v3.py','web_text_selection_v1.py',
      'web_runtime_status_v1.py','web_table_text_assets_v1/text-review.js',
      'web_table_text_assets_v1/text-review.css','table_text_reconcile_v1.py',
      'native_table_export_v2.py','durable_attempt_budget_v1.py','runtime_progress_v1.py',
      'service_profiles_v12/worker.py')


def binding():
    from .web_tables_contract_v2 import binding as previous
    body=dict(schema_version='web-table-extension/3.0',previous=previous(),
        code=[dict(path='src/cad_recovery/'+f,sha256=digest(ROOT/'src/cad_recovery'/f)) for f in CODE]
        +[dict(path='scripts/'+f,sha256=digest(ROOT/'scripts'/f)) for f in
          ('build_web_tables_v3.py','service_profile_diagnostic_v1.py','run_profiles_v16.py')],
        automatic_routing=False,quality_accepted=False,mode='explicit_safe_text_choices')
    return {**body,'sha256':hashlib.sha256(encoded(body)).hexdigest()}


class CellChoice(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    region_id:str=Field(min_length=1,max_length=40)
    cell_id:str=Field(min_length=1,max_length=160)
    origin:Literal['baseline','candidate']
    handles:list[str]=Field(min_length=1,max_length=128)


class Association(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    handle:str=Field(pattern=r'^[0-9A-Fa-f]{1,16}$')
    box_px:list[int]=Field(min_length=4,max_length=4)


class TextSelection(BaseModel):
    model_config=ConfigDict(extra='forbid')
    proposal_id:UUID
    preview_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    review_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    reviewed:Literal[True]
    selections:list[CellChoice]=Field(default_factory=list,max_length=128)
    associations:list[Association]=Field(default_factory=list,max_length=64)


class Request(PreviousRequest):
    model_config=ConfigDict(extra='forbid')
    text_selection:TextSelection|None=None


def parse_request(value):
    for r in value.get('regions',[]):
        if any(type(v) is not int for v in r.get('box_px',[])):raise ValueError('Integer pixels required')
        r['box_px']=tuple(r.get('box_px',[]))
    return Request.model_validate(value)
