"""Keep frozen annotation semantics while using source-partitioned primitives."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import cv2
import ezdxf
import numpy as np
from PIL import Image
from .native_annotation_export_v1 import APPID,STYLE,FONT_FILE,_replace_source,_text,_dimension,snapshot,provenance
from .native_annotation_plan_v1 import compile_annotations
from .cad_object_export_v2 import export_objects
from .part_admission_v1 import object_sha
from .annotation_style_v1 import render_review_dxf

def export_annotations(packet,annotations,target,*,protected=None,render=True,geometry_export=export_objects):
    target = Path(target)
    if target.exists() or target.with_suffix('.json').exists(): raise FileExistsError(target)
    plan = compile_annotations(packet,annotations); shape = packet['gray'].shape
    protection = plan['protection'] | (np.zeros(shape,bool) if protected is None else protected)
    base = geometry_export(packet,target.with_name('geometry-base.dxf'),protected=protection,render=False)
    doc = ezdxf.readfile(target.with_name('geometry-base.dxf')); doc.appids.new(APPID)
    doc.styles.new(STYLE,dxfattribs={'font':FONT_FILE})
    for name in ('TEXT_DRAFT','TEXT_REVIEW','DIMENSIONS_DRAFT'):
        doc.layers.new(name,dxfattribs={'color':7,'lineweight':13})
    doc.layers.get('TEXT_REVIEW').off(); doc.header['$DIMASSOC'] = 1
    _,removed = _replace_source(doc,plan['replacement'],shape)
    records = []; expected = {}
    for text in plan['text']:
        if text['consumed_by_dimension']: continue
        entity,record = _text(doc,text,shape[0],annotations['binding'])
        records.append({**record,'handle':entity.dxf.handle,'type':entity.dxftype()})
        expected[entity.dxf.handle] = [snapshot(entity),record]
    for row in plan['dimensions']:
        entity,record = _dimension(doc,row,shape[0],annotations)
        records.append({**record,'handle':entity.dxf.handle,'type':entity.dxftype(),'visible':True})
        expected[entity.dxf.handle] = [snapshot(entity),record]
    doc.saveas(target); reopened = ezdxf.readfile(target); audit = reopened.audit()
    if audit.errors or audit.fixes: raise ValueError('Annotation DXF requires repair')
    for handle,value in expected.items():
        entity = reopened.entitydb[handle]
        if object_sha([snapshot(entity),provenance(entity)])!=object_sha(value): raise ValueError('Annotation round trip changed')
    geometry_removed = np.array(Image.open(target.with_name('geometry-base-removed.png')))>0
    residual_before = np.array(Image.open(target.with_name('geometry-base-residual.png')))>0
    residual = residual_before & ~removed
    if np.any(geometry_removed & removed) or not np.array_equal(residual_before,residual|removed):
        raise ValueError('Annotation source partition failed')
    Image.fromarray(removed.astype(np.uint8)*255).save(target.with_name(target.stem+'-replaced.png'))
    Image.fromarray(residual.astype(np.uint8)*255).save(target.with_name(target.stem+'-residual.png'))
    if render: render_review_dxf(target,target.with_suffix('.png'))
    report = {'schema':'native-annotations/2.0','status':'draft_requires_review','binding':annotations['binding'],
        'annotations':records,'dimension_rejections':plan['dimension_rejections'],
        'unbound_chain_count':len(annotations.get('chains',[]))-sum(d['evidence_mode']=='automatic_chain_draft' for d in plan['dimensions']),
        'native_text_count':sum(r['type']=='TEXT' for r in records),
        'native_mtext_count':sum(r['type']=='MTEXT' for r in records),
        'visible_text_count':sum(r['type'] in ('TEXT','MTEXT') and r['visible'] for r in records),
        'native_dimension_count':sum(r['type']=='DIMENSION' for r in records),
        'native_leader_count':sum(r['type']=='LEADER' for r in records),
        'consumed_ocr_line_ids':plan['consumed_ocr_ids'],
        'source_pixel_sha256':plan['source_sha256'],'source_pixels':base['source_pixels'],
        'geometry_replaced_pixels':int(geometry_removed.sum()),'annotation_replaced_pixels':int(removed.sum()),
        'residual_pixels':int(residual.sum()),'source_partition_exact':True,
        'source_ink_layer':'SOURCE_INK','annotation_source_layer':'ANNOTATION_SOURCE_REVIEW',
        'counts':dict(Counter(e.dxftype() for e in reopened.modelspace())),
        'geometry':{k:base[k] for k in ('object_count','kind_counts','input_candidate_count')},
        'units':'working_pixels_unitless','font_file':FONT_FILE,'font_installation_verified_in_CAD':False,
        'audit_errors':0,'audit_fixes':0,'roundtrip_annotations_exact':True,
        'dxf_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'manual_CAD_verified':False,'physical_geometry_approved':False,'web_profile_changed':False}
    target.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return report
