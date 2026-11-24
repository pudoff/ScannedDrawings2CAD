"""Explicit, source-bound table requests and a separate immutable extension binding."""
from pathlib import Path
from typing import Literal
import hashlib
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .dataset_layout_v1 import encoded, digest, read

ROOT = Path(__file__).resolve().parents[2]
CODE = ('web_tables_contract_v1.py', 'web_tables_context_v1.py', 'web_tables_store_v1.py',
        'web_tables_build_v1.py', 'web_tables_worker_v1.py', 'web_tables_api_v1.py',
        'native_table_structure_v1.py', 'native_table_export_v1.py', 'native_table_pipeline_v2.py',
        'native_table_pipeline_v1.py', 'native_sheet_source_v1.py',
        'web_tables_assets_v1/panel.html', 'web_tables_assets_v1/tables.js', 'web_tables_assets_v1/tables.css')


def binding():
    release = ROOT/'artifacts/native-table-release-v1.json'
    if read(release)['decision'] != 'limited_manual_candidate': raise ValueError('Table experiment decision changed')
    body = dict(schema_version='web-table-extension/1.0', mode='explicit_manual_review',
        code=[dict(path=f'src/cad_recovery/{f}', sha256=digest(ROOT/'src/cad_recovery'/f)) for f in CODE]
             + [dict(path=f'scripts/{f}', sha256=digest(ROOT/'scripts'/f))
                for f in ('build_web_tables_v1.py', 'run_profiles_v14.py')],
        qualification=dict(path='artifacts/native-table-release-v1.json', sha256=digest(release)),
        automatic_routing=False, quality_accepted=False)
    return {**body, 'sha256': hashlib.sha256(encoded(body)).hexdigest()}


class Region(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,40}$')
    kind: Literal['table', 'title']
    box_px: tuple[int, int, int, int]


class Request(BaseModel):
    model_config = ConfigDict(extra='forbid')
    context_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    binding_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    page_kind: Literal['table_document', 'mixed'] = 'mixed'
    reviewer: str = Field(min_length=1, max_length=120)
    regions: list[Region] = Field(min_length=1, max_length=16)

    @model_validator(mode='after')
    def boxes(self):
        if not self.reviewer.strip(): raise ValueError('Reviewer required')
        seen = set()
        for i, region in enumerate(self.regions):
            l, t, r, b = region.box_px
            if min(l, t) < 0 or l >= r or t >= b or region.id in seen: raise ValueError('Invalid region')
            seen.add(region.id)
            for other in self.regions[:i]:
                x, y, xx, yy = other.box_px
                if l < xx and x < r and t < yy and y < b: raise ValueError('Regions overlap')
        return self


class Approval(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    preview_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reviewed: Literal[True]


def parse_request(value):
    # JSON arrays are expected, but floats/bools must not become pixel ints.
    for r in value.get('regions', []):
        if any(type(v) is not int for v in r.get('box_px', [])): raise ValueError('Integer pixels required')
        r['box_px'] = tuple(r.get('box_px', []))
    return Request.model_validate(value)
