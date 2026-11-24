"""Reuse the frozen v2 pixel partitioner; add explicit reviewed CENTER patterns."""
import hashlib
import json
from pathlib import Path
import ezdxf
import numpy as np
from .cad_line_completion_v3 import assemble
from .cad_axis_review_v1 import apply_review
from .cad_object_export_v2 import export_objects as previous,read_provenance,entity_geometry,APPID,RECORD
from .native_annotation_export_v2 import export_annotations as previous_annotations
from .part_admission_v1 import object_sha
from .annotation_style_v1 import render_review_dxf


def _record(entity,data):
    payload=json.dumps(data,ensure_ascii=True,sort_keys=True,separators=(',',':'))
    entity.set_xdata(APPID,[(1000,data['object_id']),(1000,data['source_sha256']),
        (1000,hashlib.sha256(payload.encode('ascii')).hexdigest())])
    record=entity.get_extension_dict()[RECORD];record.clear()
    record.extend([(1,payload[i:i+240]) for i in range(0,len(payload),240)])


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
    report.update(schema_version='cad-line-export/3.0',dxf_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    if render:render_review_dxf(target,target.with_name(target.stem+'-preview.png'))
    target.with_name(target.stem+'-objects.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8')
    return report


def export_annotations(packet,annotations,target,*,protected=None,render=True):
    return previous_annotations(packet,annotations,target,protected=protected,render=render,geometry_export=export_objects)
