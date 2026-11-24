"""Explicit text-only Web extension; no change to the registered CV/OCR."""
import hashlib
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .dataset_layout_v1 import digest, encoded
from .web_tables_contract_v1 import ROOT, Approval, Request as BaseRequest
from .web_tables_contract_v3 import CellChoice, Association, TextSelection as OldSelection


class CellKey(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    region_id: str = Field(min_length=1, max_length=40)
    cell_id: str = Field(min_length=1, max_length=160)


class TextSelection(OldSelection):
    source_kind: Literal['proposal', 'published'] = 'proposal'
    clear_cells: list[CellKey] = Field(default_factory=list, max_length=128)
    clear_associations: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode='after')
    def distinct_operations(self):
        cells = [(c.region_id, c.cell_id) for c in self.selections]
        clears = [(c.region_id, c.cell_id) for c in self.clear_cells]
        associations = [a.handle for a in self.associations]
        if (len(set(cells)) != len(cells) or len(set(clears)) != len(clears)
                or set(cells) & set(clears) or len(set(associations)) != len(associations)
                or len(set(self.clear_associations)) != len(self.clear_associations)
                or set(associations) & set(self.clear_associations)):
            raise ValueError('Duplicate or conflicting text operations')
        if not (cells or clears or associations or self.clear_associations):
            raise ValueError('An explicit text operation is required')
        return self


class Request(BaseRequest):
    text_selection: TextSelection | None = None


def parse_request(value):
    from copy import deepcopy
    value = deepcopy(value)
    for region in value.get('regions', []):
        if any(type(v) is not int for v in region.get('box_px', [])):
            raise ValueError('Integer pixel coordinates required')
        region['box_px'] = tuple(region.get('box_px', []))
    return Request.model_validate(value)


def binding():
    from .web_tables_contract_v3 import binding as previous
    names = ('web_tables_contract_v4.py', 'web_tables_context_v4.py', 'web_text_publication_v1.py',
             'web_tables_build_v4.py', 'web_tables_worker_v5.py', 'web_tables_api_v5.py',
             'web_table_text_assets_v2/text-review.js', 'table_text_publish_v1.py')
    body = dict(schema_version='web-table-extension/4.0', previous=previous(),
        mode='baseline_geometry_explicit_cumulative_text', automatic_routing=False,
        quality_accepted=False,
        code=[dict(path='src/cad_recovery/'+n, sha256=digest(ROOT/'src/cad_recovery'/n)) for n in names]
        + [dict(path='scripts/'+n, sha256=digest(ROOT/'scripts'/n)) for n in
           ('run_profiles_v18.py', 'build_web_tables_v5.py')])
    return {**body, 'sha256': hashlib.sha256(encoded(body)).hexdigest()}
