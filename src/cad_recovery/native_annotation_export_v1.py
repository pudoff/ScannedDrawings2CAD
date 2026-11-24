"""Native TEXT/MTEXT and unitless DIMENSION layered over source-bound geometry."""
from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import cv2
import ezdxf
import numpy as np
from PIL import Image
from .annotation_style_v1 import render_review_dxf
from .cad_object_export_v1 import export_objects
from .experimental_export_v1 import _solids
from .exporter_v2 import dimension_display_text
from .native_annotation_plan_v1 import compile_annotations
from .part_admission_v1 import object_sha

APPID = 'CADRECOVERY_ANNOTATION'
STYLE = 'CAD_TEXT_DRAFT'
FONT_FILE = 'cambria.ttc'


def attach(entity,value):
    payload = json.dumps(value,ensure_ascii=True,sort_keys=True,separators=(',',':'))
    entity.set_xdata(APPID,[(1000,value['id']),(1000,hashlib.sha256(payload.encode('ascii')).hexdigest())])
    entity.new_extension_dict().add_xrecord(APPID).extend([(1,payload[i:i+240]) for i in range(0,len(payload),240)])


def provenance(entity):
    payload = ''.join(t.value for t in entity.get_extension_dict()[APPID].tags if t.code==1)
    data = json.loads(payload); tags = [t.value for t in entity.get_xdata(APPID) if t.code==1000]
    if tags!=[data['id'],hashlib.sha256(payload.encode('ascii')).hexdigest()]: raise ValueError('Annotation provenance differs')
    return data


def snapshot(entity):
    d = entity.dxf; kind = entity.dxftype(); value = {'type':kind,'layer':d.layer}
    if kind=='TEXT': value.update(raw=d.text,insert=list(d.insert),height=d.height,width=d.width,rotation=float(d.rotation) or 0.,style=d.style)
    elif kind=='MTEXT':
        value.update(raw=entity.text,plain=entity.plain_text(),insert=list(d.insert),height=d.char_height,
            width=d.width,rotation=float(d.get('rotation',0.)) or 0.,style=d.style,attachment=d.attachment_point)
    elif kind=='DIMENSION':
        value.update(kind=d.dimtype&7,measurement=float(entity.get_measurement()),
            raw=d.text,display=dimension_display_text(entity),dimstyle=d.dimstyle,
            definitions={key:list(d.get(key)) for key in ('defpoint','defpoint2','defpoint3','defpoint4') if d.hasattr(key)},
            angle=d.get('angle',0.),dimlfac=entity.override().get('dimlfac',1.))
    elif kind=='LEADER': value.update(vertices=[list(p) for p in entity.vertices],dimstyle=d.dimstyle)
    return value


def escaped_mtext(raw):
    return raw.replace('\\','\\\\').replace('{','\\{').replace('}','\\}').replace('\r\n','\n').replace('\r','\n').replace('\n','\\P')


def _text(doc,plan,height,binding):
    row = plan['line']; p = np.asarray(row['polygon_px'],float)
    baseline = p[1]-p[0]; width = float(np.linalg.norm(baseline))
    if width<1: raise ValueError('Degenerate OCR baseline')
    side = p[3]-p[0]
    text_height = abs(float(baseline[0]*side[1]-baseline[1]*side[0]))/width
    if text_height<1: raise ValueError('Degenerate OCR height')
    rotation = -math.degrees(math.atan2(baseline[1],baseline[0]))
    layer = 'TEXT_DRAFT' if plan['visible'] else 'TEXT_REVIEW'
    attrs = {'layer':layer,'style':STYLE,'rotation':rotation,'color':256,'lineweight':-1}
    point = lambda q:(float(q[0]),height-1-float(q[1]))
    multiline = '\n' in row['raw'] or '\r' in row['raw']
    if multiline:
        count = row['raw'].count('\n')+1
        attrs.update(insert=point(p[0]),char_height=text_height/(count*1.2),width=width,attachment_point=1)
        entity = doc.modelspace().add_mtext(escaped_mtext(row['raw']),dxfattribs=attrs)
    else:
        attrs.update(insert=point(p[3]),height=.8*text_height)
        entity = doc.modelspace().add_text(row['raw'],dxfattribs=attrs)
        from ezdxf.tools.text_size import text_size
        measured = text_size(entity).width
        if measured>0: entity.dxf.width = width/measured
    record = {'id':'ocr-text/'+row['id'],'ocr_line_id':row['id'],'raw':row['raw'],
        'confidence':row.get('confidence'),'recognition_score_calibrated':False,
        'status':row.get('status','automatic_draft'),'visible':plan['visible'],'reason':plan['reason'],
        'polygon_px':row['polygon_px'],'binding':deepcopy(binding),
        'original_ocr_raw':row.get('original_ocr_raw',row['raw']),'text_review':row.get('text_review'),
        'physical_geometry_approved':False,
        'manual_text_confirmation':row.get('text_review',{}).get('evidence_mode')=='user_reviewed_draft'}
    attach(entity,record)
    return entity,record


def _dimension(doc,row,height,annotations):
    point = lambda p:(float(p[0]),height-1-float(p[1]))
    p = np.asarray(row['points_px'],float); placement = point(row['placement_px'])
    heads = {h['id']:h for h in annotations['heads']}
    sizes = [np.linalg.norm(np.asarray(heads[i]['polygon_px'][1:],float).mean(0)-heads[i]['tip_px'])
             for i in row.get('head_ids',[])]
    span = float(np.linalg.norm(p[1]-p[0])); text_height = float(row.get('text_height_px',max(3.,min(16.,span*.08))))
    arrow = float(np.median(sizes)) if sizes else max(2.,text_height*.8)
    style = {'dimlfac':1.,'dimscale':1.,'dimtxt':text_height,'dimasz':arrow,'dimdec':2,
        'dimtxsty':STYLE,'dimlwd':13,'dimlwe':13,'dimpost':'<> px','dimclrd':0,'dimclre':0,'dimclrt':0,
        'dimblk':'','dimsah':0}
    attrs = {'layer':'DIMENSIONS_DRAFT','color':256,'lineweight':-1}
    model = doc.modelspace(); kind = row['kind']
    if kind=='leader':
        entity = model.add_leader([point(q) for q in p],override=style,dxfattribs=attrs)
        native = None
    elif kind in ('linear','aligned'):
        delta = p[1]-p[0]; angle = -math.degrees(math.atan2(delta[1],delta[0]))
        native = model.add_linear_dim(base=placement,p1=point(p[0]),p2=point(p[1]),angle=angle,override=style,dxfattribs=attrs)
        if kind=='aligned': native.dimension.dxf.dimtype = (native.dimension.dxf.dimtype&~7)|1
    elif kind in ('radius','diameter'):
        create = model.add_radius_dim if kind=='radius' else model.add_diameter_dim
        native = create(center=point(p[0]),mpoint=point(p[1]),override=style,dxfattribs=attrs)
        native.user_location_override(placement)
    else:
        style['dimpost'] = '<>'
        native = model.add_angular_dim_3p(base=placement,center=point(p[0]),p1=point(p[2]),p2=point(p[1]),
            override=style,dxfattribs=attrs)
    if native is not None: native.render(); entity = native.dimension
    record = {**deepcopy(row),'id':'dimension/'+row['id'],'binding':deepcopy(annotations['binding']),
        'observed_texts':[{'ocr_line_id':i,'raw':next(r['raw'] for r in annotations['lines'] if r['id']==i)}
                          for i in row.get('ocr_line_ids',[])],
        'computed_display':dimension_display_text(entity) if native is not None else None,
        'native_measurement':float(entity.get_measurement()) if native is not None else None,
        'physical_scale_confirmed':False,'CAD_associative_to_contour':False}
    attach(entity,record); return entity,record


def _replace_source(doc,mask,shape):
    """Rebuild residual rectangles, preserving every unclaimed source pixel."""
    layers = {}; h,w = shape
    for entity in list(doc.modelspace().query('SOLID')):
        if entity.dxf.layer=='SOURCE_INK': continue
        vertices = np.array([getattr(entity.dxf,'vtx'+str(i)) for i in range(4)])[:,:2]
        low,high = vertices.min(0),vertices.max(0)
        left,right = int(round(low[0]+.5)),int(round(high[0]+.5))
        top,bottom = int(round(h-.5-high[1])),int(round(h-.5-low[1]))
        if not (0<=left<right<=w and 0<=top<bottom<=h): raise ValueError('Unexpected source SOLID')
        if entity.dxf.layer not in layers: layers[entity.dxf.layer] = np.zeros(shape,bool)
        layers[entity.dxf.layer][top:bottom,left:right] = True
        doc.modelspace().delete_entity(entity)
    before = np.zeros(shape,bool)
    for layer,value in layers.items(): before |= value; _solids(doc.modelspace(),value & ~mask,h,layer)
    removed = before & mask
    doc.layers.new('ANNOTATION_SOURCE_REVIEW',dxfattribs={'color':1,'plot':0})
    doc.layers.get('ANNOTATION_SOURCE_REVIEW').off(); _solids(doc.modelspace(),removed,h,'ANNOTATION_SOURCE_REVIEW')
    return before,removed


def export_annotations(packet,annotations,target,*,protected=None,render=True):
    target = Path(target)
    if target.exists() or target.with_suffix('.json').exists(): raise FileExistsError(target)
    plan = compile_annotations(packet,annotations); shape = packet['gray'].shape
    protection = plan['protection'] | (np.zeros(shape,bool) if protected is None else protected)
    base = export_objects(packet,target.with_name('geometry-base.dxf'),protected=protection,render=False)
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
    report = {'schema':'native-annotations/1.0','status':'draft_requires_review','binding':annotations['binding'],
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
