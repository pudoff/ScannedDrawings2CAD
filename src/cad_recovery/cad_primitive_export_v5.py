"""Reuse the frozen v2 pixel partitioner; add explicit reviewed CENTER patterns."""
import hashlib
import json
from pathlib import Path
import ezdxf
import numpy as np
from .cad_object_assembly_v5 import assemble as primitive_assembly
from .cad_line_completion_v3 import complete
from collections import Counter
from .cad_axis_review_v1 import apply_review
from .cad_object_export_v2 import export_objects as previous,read_provenance,entity_geometry,APPID,RECORD
from .native_annotation_export_v2 import export_annotations as previous_annotations
from .part_admission_v1 import object_sha
from .annotation_style_v1 import render_review_dxf


from .cad_line_export_v3 import _record


def assemble(gray,geometry,*,roles=None,protected=None):
    roles=np.full_like(gray,255) if roles is None else roles
    result=primitive_assembly(gray,geometry,roles=roles,protected=protected)
    before=Counter((m['candidate_id'],n) for o in result['objects'] for m in o['source_members'] for n in range(*m['source_index_interval']))
    result['objects'],evidence=complete(result['objects'],gray,roles)
    after=Counter((m['candidate_id'],n) for o in result['objects'] for m in o['source_members'] for n in range(*m['source_index_interval']))
    if before!=after or any(n!=1 for n in after.values()):raise ValueError('LINE completion changed source edge partition')
    result.update(object_count=len(result['objects']),kind_counts=dict(Counter(o['kind'] for o in result['objects'])),
        multi_candidate_objects=sum(len({m['candidate_id'] for m in o['source_members']})>1 for o in result['objects']),
        line_completion=evidence,search_complete=result['search_complete'] and evidence['search_complete'])
    return result


def export_objects(packet,target,*,protected=None,ocr=None,render=True):
    result=[]
    def assembler(gray,geometry,**kwargs):
        r=assemble(gray,geometry,**kwargs)
        roles=kwargs.get('roles',np.full_like(gray,255))
        r=apply_review(r,gray,roles,geometry,packet.get('confirmed_axis_groups',[]))
        result.append(r);return r
    report=previous(packet,target,protected=protected,ocr=ocr,render=False,assembler=assembler)
    target=Path(target);doc=ezdxf.readfile(target);expected={}
    for obj,row in zip(result[0]['objects'],report['objects']):
        entity=doc.entitydb[row['handle']];data=read_provenance(entity)
        data['line_continuation_evidence']=obj.get('line_continuation_evidence')
        if 'axis_review' in obj:
            pattern=obj['linetype_pattern_px'];name='CENTER_REVIEW_'+object_sha(pattern)[:8]
            if name not in doc.linetypes:doc.linetypes.new(name,dxfattribs={
                'description':'Source-measured reviewed dash-gap-dot-gap','pattern':[sum(map(abs,pattern)),*pattern]})
            layer='AXES_REVIEWED_'+object_sha(pattern)[:8]
            if layer not in doc.layers:doc.layers.new(layer,dxfattribs={'color':7,'lineweight':13,'linetype':name})
            entity.dxf.layer=layer;entity.dxf.linetype='BYLAYER';entity.dxf.ltscale=1
            data.update(axis_review=obj['axis_review'],linetype_pattern_px=pattern)
        _record(entity,data);row.update(data,layer=entity.dxf.layer)
        expected[entity.dxf.handle]=[entity_geometry(entity),data]
    doc.header['$LTSCALE']=1.;doc.saveas(target);again=ezdxf.readfile(target);audit=again.audit()
    if audit.errors or audit.fixes:raise ValueError('Completed LINE DXF requires repair')
    for handle,value in expected.items():
        if object_sha([entity_geometry(again.entitydb[handle]),read_provenance(again.entitydb[handle])])!=object_sha(value):
            raise ValueError('Completed LINE geometry/provenance roundtrip differs')
    report.update(schema_version='cad-primitives-export/5.0',dxf_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    if render:render_review_dxf(target,target.with_name(target.stem+'-preview.png'))
    target.with_name(target.stem+'-objects.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
    return report


def export_annotations(packet,annotations,target,*,protected=None,render=True):
    return previous_annotations(packet,annotations,target,protected=protected,render=render,geometry_export=export_objects)
