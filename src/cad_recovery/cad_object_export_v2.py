"""Source-bound native draft objects, provenance and exclusive residual ink."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np

from .annotation_style_v1 import render_review_dxf
from .cad_object_assembly_v2 import assemble
from .demo_export_v1 import _configure_view, _observations
from .experimental_export_v1 import ROLE_LAYERS, THIN, _solids
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha

APPID = 'CADRECOVERY_OBJECT'
RECORD = 'CADRECOVERY_SOURCE'


def _ownership(gray,result,geometry,protected):
    """Nearest observed source seed, limited by measured ink radius; never bbox."""
    _,raw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    ink = raw>0; seeds = np.zeros_like(gray,bool); seed_owner = np.zeros_like(gray,np.int32)
    # Include withheld sources as competitors, so converted objects cannot steal their ink.
    for c in geometry:
        p = np.asarray(c.get('support_points',c.get('points_px')),np.int32)
        seeds[p[:,1],p[:,0]] = True
    used = np.zeros_like(gray,bool)
    candidates={c['id']:c for c in geometry}
    for i,obj in enumerate(result['objects'],1):
        claimed=[]
        for member in obj['source_members']:
            c=candidates[member['candidate_id']];start=c['source_index_interval'][0]
            low,high=member['source_index_interval']
            claimed.extend(c.get('support_points',c.get('points_px'))[low-start:high-start+1])
        p=np.asarray(claimed,np.int32);yy,xx=p[:,1],p[:,0]
        conflicts = used[yy,xx] & (seed_owner[yy,xx]!=i)
        seed_owner[yy,xx] = np.where(conflicts,-1,i); used[yy,xx] = True
    # A split candidate can be only partly emitted. Keep every unconverted
    # source point competitive, including boundaries of protected subpaths.
    coverage={c['id']:np.zeros(len(c.get('support_points',c.get('points_px'))),bool) for c in geometry}
    lookup={c['id']:c for c in geometry}
    for obj in result['objects']:
        for member in obj['source_members']:
            c=lookup[member['candidate_id']];start=c['source_index_interval'][0]
            low,high=member['source_index_interval'];coverage[c['id']][low-start:high-start+1]=True
    for c in geometry:
        p=np.asarray(c.get('support_points',c.get('points_px')),np.int32)
        q=p[~coverage[c['id']]]
        seed_owner[q[:,1],q[:,0]]=0
    removed = np.zeros_like(gray,bool); ownership = np.zeros_like(gray,np.int32)
    if seeds.any():
        distance,labels = cv2.distanceTransformWithLabels(np.where(seeds,0,255).astype(np.uint8),
            cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        yy,xx = np.nonzero(seeds); labels_at_seed = labels[yy,xx]
        lookup = np.zeros(int(labels.max())+1,np.int32); lookup[labels_at_seed] = seed_owner[yy,xx]
        dt = cv2.distanceTransform(raw,cv2.DIST_L2,5)
        radii = np.zeros(len(lookup),np.float32); radii[labels_at_seed] = dt[yy,xx]
        ownership = lookup[labels]
        removed = ink & (ownership>0) & (distance<=radii[labels]+.75) & ~protected
    ownership[~removed] = 0
    return ink,removed,ownership


def _provenance(entity,obj,source_sha):
    data = {'object_id':obj['id'],'source_sha256':source_sha,'source_members':obj['source_members'],
        'path_ids':obj['path_ids'],'kind':obj['kind'],'role':obj['role'],
        'fit_max_px':obj.get('fit_max_px',0.),'fit_p95_px':obj.get('fit_p95_px',0.),
        'units':'working_pixels_unitless','status':'draft_requires_review',
        'source_junction_links':obj.get('source_junction_links',[]),
        'circle_family_source_sectors':obj.get('circle_family_source_sectors',[]),
        'parameter_sensitivity_native_px':obj.get('parameter_sensitivity_native_px'),
        'parameter_evidence':obj.get('parameter_evidence','shared_source_sectors' if obj.get('circle_family_support') else 'local_source')}
    payload = json.dumps(data,ensure_ascii=True,sort_keys=True,separators=(',',':'))
    entity.set_xdata(APPID,[(1000,obj['id']),(1000,source_sha),
        (1000,hashlib.sha256(payload.encode('ascii')).hexdigest())])
    entity.new_extension_dict().add_xrecord(RECORD).extend([(1,payload[i:i+240]) for i in range(0,len(payload),240)])
    return data


def read_provenance(entity):
    record = entity.get_extension_dict()[RECORD]
    payload = ''.join(tag.value for tag in record.tags if tag.code==1)
    xdata = [tag.value for tag in entity.get_xdata(APPID) if tag.code==1000]
    if hashlib.sha256(payload.encode('ascii')).hexdigest()!=xdata[2]: raise ValueError('DXF provenance changed')
    value = json.loads(payload)
    if [value['object_id'],value['source_sha256']]!=xdata[:2]: raise ValueError('DXF provenance binding differs')
    return value


def entity_geometry(entity):
    kind = entity.dxftype(); value = {'type':kind,'layer':entity.dxf.layer}
    if kind=='LINE': value.update(start=list(entity.dxf.start)[:2],end=list(entity.dxf.end)[:2])
    elif kind=='POINT': value['point'] = list(entity.dxf.location)[:2]
    elif kind in ('ARC','CIRCLE'):
        value.update(center=list(entity.dxf.center)[:2],radius=entity.dxf.radius)
        if kind=='ARC': value.update(start=entity.dxf.start_angle,end=entity.dxf.end_angle)
    elif kind=='LWPOLYLINE': value.update(points=entity.get_points('xyb'),closed=entity.closed)
    return value


def export_objects(packet,target,*,protected=None,ocr=None,render=True,assembler=assemble):
    """Prepare a separate DXF; existing profile and physical exporter are untouched."""
    target = Path(target); stem = target.with_suffix('')
    artifacts = {name:stem.with_name(stem.name+'-'+name+suffix) for name,suffix in
        [('objects','.json'),('ownership','.png'),('removed','.png'),('residual','.png'),('preview','.png')]}
    if target.exists() or any(p.exists() for p in artifacts.values()): raise FileExistsError(target)
    if not target.parent.is_dir(): raise ValueError('Output directory must exist')
    gray,roles,geometry = packet['gray'],packet['roles'],packet['geometry']
    h,w = gray.shape
    protected = np.zeros_like(gray,bool) if protected is None else protected.copy()
    if protected.dtype!=bool or protected.shape!=gray.shape: raise ValueError('Aligned boolean protection required')
    protected |= roles==6
    result = assembler(gray,geometry,roles=roles,protected=protected)
    ink,removed,ownership = _ownership(gray,result,geometry,protected)
    residual = ink & ~removed
    if np.any(removed & protected) or np.any(removed & ~ink) or not np.array_equal(removed|residual,ink):
        raise ValueError('Source ink partition or protection failed')
    doc = ezdxf.new('R2007',setup=True); doc.units = 0; doc.appids.new(APPID)
    names = [*ROLE_LAYERS.values(),'GEOMETRY_DRAFT','THIN_GEOMETRY_DRAFT','SOURCE_INK','UNRESOLVED_INK','ISOLATED_POINTS']
    for name in names:
        doc.layers.new(name,dxfattribs={'color':7,'lineweight':13 if name in
            [ROLE_LAYERS[r] for r in THIN]+['THIN_GEOMETRY_DRAFT'] else 35})
    doc.layers.get('SOURCE_INK').off(); doc.layers.get('SOURCE_INK').dxf.plot = 0
    model = doc.modelspace(); expected = {}; mapping = []
    point = lambda p:(float(p[0]),h-1-float(p[1]))
    dt = cv2.distanceTransform(ink.astype(np.uint8)*255,cv2.DIST_L2,5)
    for index,obj in enumerate(result['objects'],1):
        p = np.asarray(obj['support_points'],np.int32)
        pen = float(np.median(dt[p[:,1],p[:,0]])*2-1)
        layer = ROLE_LAYERS.get(obj['role'],'GEOMETRY_DRAFT' if pen>=2.5 else 'THIN_GEOMETRY_DRAFT')
        attrs = {'layer':layer,'color':256,'lineweight':-1,'linetype':'BYLAYER'}
        kind = obj['kind']
        if kind=='line': entity = model.add_line(*map(point,obj['points_px']),dxfattribs=attrs)
        elif kind=='circle': entity = model.add_circle(point(obj['center_px']),obj['radius_px'],dxfattribs=attrs)
        elif kind=='arc':
            start,sweep = obj['start_angle_deg'],obj['sweep_angle_deg']
            entity = model.add_arc(point(obj['center_px']),obj['radius_px'],-(start+sweep),-start,dxfattribs=attrs)
        elif kind=='point':
            attrs['layer'] = 'ISOLATED_POINTS'; entity = model.add_point(point(obj['points_px'][0]),dxfattribs=attrs)
        else:
            vertices = obj['points_px']
            if obj['closed'] and vertices[0]==vertices[-1]: vertices = vertices[:-1]
            entity = model.add_lwpolyline([point(p) for p in vertices],close=obj['closed'],dxfattribs=attrs)
        provenance = _provenance(entity,obj,result['source_sha256'])
        expected[entity.dxf.handle] = (entity_geometry(entity),provenance)
        mapping.append({**provenance,'handle':entity.dxf.handle,'layer':entity.dxf.layer,
            'source_candidate_count':len(obj['source_members']),
            'replaced_ink_pixels':int(np.count_nonzero(ownership==index))})
    rest = residual.copy()
    for role,layer in ROLE_LAYERS.items():
        selected = rest & (roles==role); _solids(model,selected,h,layer); rest[selected] = False
    _solids(model,rest,h,'UNRESOLVED_INK'); _solids(model,ink,h,'SOURCE_INK')
    _observations(doc,h,None,ocr)
    _configure_view(doc,(-.5,-.5,w-.5,h-.5),pixel_units=True)
    doc.header['$LWDISPLAY'] = True; doc.header['$PDMODE'] = 0
    doc.saveas(target)
    reopened = ezdxf.readfile(target); audit = reopened.audit()
    if audit.errors or audit.fixes: raise ValueError('Native object DXF requires repair')
    for handle,(geometry_before,source_before) in expected.items():
        entity = reopened.entitydb[handle]
        if object_sha(entity_geometry(entity))!=object_sha(geometry_before) or read_provenance(entity)!=source_before:
            raise ValueError('Native object geometry/provenance changed in DXF round trip')
    if reopened.units!=0 or reopened.modelspace().query('IMAGE UNDERLAY DIMENSION HATCH'):
        raise ValueError('Unexpected physical or raster dependency')
    from PIL import Image
    Image.fromarray(removed.astype(np.uint8)*255).save(artifacts['removed'])
    Image.fromarray(residual.astype(np.uint8)*255).save(artifacts['residual'])
    Image.fromarray(ownership.astype(np.uint16)).save(artifacts['ownership'])
    if render: render_review_dxf(target,artifacts['preview'])
    summary = {k:v for k,v in result.items() if k not in ('objects',)}
    summary.update(schema_version='cad-object-export/2.0',dxf=target.name,
        dxf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),objects=mapping,
        modelspace_counts=dict(Counter(e.dxftype() for e in reopened.modelspace())),
        source_pixels=int(ink.sum()),replaced_pixels=int(removed.sum()),residual_pixels=int(residual.sum()),
        protected_pixels_preserved=True,source_partition_exact=True,
        source_reference_initially_hidden=True,audit_errors=0,audit_fixes=0,
        roundtrip_geometry_and_provenance_exact=True,coordinate_system='working_raster_pixel_centres_y_up',
        input_to_export=[[1,0,0],[0,-1,h-1],[0,0,1]],size_px=[w,h],
        native_dimensions=0,cad_gui_verified=False,production_changed=False,
        artifacts={k:p.name for k,p in artifacts.items() if render or k!='preview'})
    artifacts['objects'].write_text(json.dumps(summary,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
    return summary
