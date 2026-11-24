"""Explicit, bounded edit contracts over published artifacts; no inferred scale."""
from copy import deepcopy
import json
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator
from .graph_correction_v1 import canonical_sha256
from .graph_v2 import read_graph_v2
from .service_profiles_v3.inspection import Inspector

MAX_JSON=16*1024**2


def decode(data):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    if len(data)>MAX_JSON:raise ValueError('Correction JSON limit')
    return json.loads(data,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


class EditRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    source_manifest_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    kind:Literal['scalar','ocr_reading','title_field']
    target_id:str=Field(min_length=1,max_length=200)
    view_id:str|None=Field(default=None,max_length=200)
    expected_old_value:str|float|None=None
    new_value:str|float
    evidence_ref:str=Field(min_length=1,max_length=300)
    max_point_displacement_mm:float=Field(default=10.,gt=0,le=1000,allow_inf_nan=False)
    max_radius_change_mm:float=Field(default=10.,gt=0,le=1000,allow_inf_nan=False)
    review_id:str|None=Field(default=None,pattern=r'^[0-9a-f-]{36}$')

    @model_validator(mode='after')
    def meaningful(self):
        if not self.evidence_ref.strip():raise ValueError('A meaningful correction reference is required')
        if self.kind=='scalar':
            if not self.view_id or type(self.new_value)not in (float,int) or type(self.expected_old_value)not in (float,int):
                raise ValueError('Scalar correction requires a view and numeric before/after values')
            if not 0<float(self.new_value)<=1e10 or not 0<float(self.expected_old_value)<=1e10:raise ValueError('Invalid scalar')
        elif not isinstance(self.new_value,str) or not self.new_value.strip() or len(self.new_value)>600:
            raise ValueError('Text correction requires 1..600 visible characters')
        for text in (self.target_id,self.evidence_ref,self.new_value):
            if isinstance(text,str) and any(ord(c)<32 or ord(c)==127 for c in text):raise ValueError('Control characters unsupported')
        if self.new_value==self.expected_old_value:raise ValueError('Correction must change a value')
        return self


class EditApproval(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    proposal_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    branch_reviewed:bool
    derived_dimensions_reviewed:bool
    review_ref:str=Field(min_length=1,max_length=300)

    @model_validator(mode='after')
    def reviewed(self):
        if not self.branch_reviewed or not self.derived_dimensions_reviewed or not self.review_ref.strip():
            raise ValueError('Explicit review of the exact proposal is required')
        if any(ord(c)<32 or ord(c)==127 for c in self.review_ref):raise ValueError('Invalid review reference')
        return self


def published(files,run,page):
    manifest_path,manifest=files.manifest(run)
    if type(page)is not int or not 0<=page<manifest['summary'].get('page_count',0):raise KeyError('Page missing')
    entries={e['filename']:e for e in manifest['files']};prefix=f'page-{page+1:03d}'
    def load(name,default=None):
        entry=entries.get(name)
        if entry is None:return deepcopy(default)
        path,_=files.artifact(run,entry['id']);return decode(path.read_bytes())
    report=load('report.json')
    if report['source_sha256']!=run['input_sha256']:raise ValueError('Source mismatch')
    return prefix,entries,load


def context(files,run,page):
    prefix,entries,load=published(files,run,page)
    result={'source_manifest_sha256':run['manifest']['sha256'],'targets':[],
        'geometry_reason':'Нужны опубликованный DrawingGraph 0.2, подтверждённый масштаб и привязки P13.4.',
        'reading_reason':'Чтение и метаданные меняются отдельно; исходная краска скана сохраняется.',
        'manual_corrections_affect_automatic_metrics':False}
    reviewed=load(prefix+'-reviewed-graph.json')
    if reviewed is not None:
        graph=read_graph_v2(reviewed)
        if graph.source.sha256!=run['input_sha256'] or graph.source.page_index!=page:raise ValueError('Reviewed graph source mismatch')
        if len(graph.primitives)>512 or len(graph.constraints)>128 or len(graph.dimensions)>128:raise ValueError('Reviewed graph budget exceeded')
        bindings=load(prefix+'-correction-bindings.json')
        if bindings is not None:
            result['geometry_reason']='Предпросмотр проверит точные условия P13.4; решение потребует отдельной проверки ветви.'
            views={v.id:v for v in graph.views}
            for c in graph.constraints:
                if (c.kind in ('distance','projected_distance','radius','diameter') and c.status=='accepted'
                        and c.evidence.status=='verified' and views[c.view_id].physical_scale_confirmed):
                    result['targets'].append({'kind':'scalar','id':c.id,'view_id':c.view_id,'value':c.value,'label':f'{c.view_id} / {c.id}',
                        'entity_ids':c.entity_ids,'parent_graph_sha256':canonical_sha256(graph)})
    try:inspection=Inspector(files).inspect(run,page)
    except LookupError as exc:
        if type(exc)is not LookupError:raise
        inspection=None
    manual=load(prefix+'-manual-graph.json',{'readings':{},'title_fields':{}})
    if inspection:
        for row in inspection['objects']:
            if row['kind']=='text':
                value=manual['readings'].get(row['id'],{}).get('value',row['observation']['raw'])
                result['targets'].append({'kind':'ocr_reading','id':row['id'],'value':value,'label':row['id']+' · '+value,
                    'original_raw':row['observation']['raw']})
    graph=load(prefix+'-graph.json',{})
    for name in ('name','document_number','material','authors'):
        value=manual['title_fields'].get(name,{}).get('value',graph.get('title_block',{}).get(name,{}).get('value'))
        result['targets'].append({'kind':'title_field','id':name,'value':value,'label':name})
    result['targets_truncated']=len(result['targets'])>1200
    result['targets']=result['targets'][:1200]
    return result


def validate_edit(files,run,page,edit):
    edit=EditRequest.model_validate(edit)
    if edit.source_manifest_sha256!=run['manifest']['sha256']:raise ValueError('Stale source manifest')
    target=next((t for t in context(files,run,page)['targets'] if (t['kind'],t['id'],t.get('view_id'))==
        (edit.kind,edit.target_id,edit.view_id)),None)
    if target is None:raise ValueError('Unknown or unsupported correction target')
    if target['value']!=edit.expected_old_value:raise ValueError('Old value changed')
    return edit,target
