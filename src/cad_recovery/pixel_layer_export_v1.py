"""Source-bound reviewed pixel geometry, exact residuals and independent CAD layers.

This is a draft/review handoff. Automatic role proposals are never approval.
No physical calibration, OCR text substitution, DIMENSION or HATCH is inferred.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from PIL import Image

from .demo_export_v1 import merged_runs,_configure_view
from .exporter_v2 import semantic_entity_snapshot,_same_snapshot

APPID='CAD_RECOVERY_PIXEL_V1'
LAYERS={'part_contour':('MAIN',50),'thin':('THIN',13),'dimension_line':('DIMS',13),
    'extension_line':('EXTENSIONS',13),'leader_line':('LEADERS',13),
    'arrowhead':('ARROWHEADS',13),'axis_construction':('AXIS',13),
    'sheet_frame':('FRAME',18),'title_grid':('TITLE_GRID',13),
    'hatching':('HATCH_STROKES',13),'text':('TEXT_INK',13)}
POLICY={'version':'pixel-layer-export/1.0','maximum_pixels':4194304,'maximum_objects':4096,
    'maximum_owned_pixel_visits':2000000,'maximum_roi_pixels':40000000,'maximum_curve_samples':250000,
    'maximum_entities':50000,'maximum_support_error_px':1.5,'shared_contacts':'explicit endpoint pixels only',
    'units':'unitless processed pixels','automatic_semantic_approval':False,'physical_geometry_approved':False}


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def spans_mask(spans,shape):
    if not isinstance(spans,list) or len(spans)>2000000:raise ValueError('Ownership span budget')
    out=np.zeros(shape,bool);visits=0
    for row in spans:
        if len(row)!=3 or any(type(v)!=int for v in row):raise ValueError('Integer ownership spans required')
        y,a,b=row
        if not (0<=y<shape[0] and 0<=a<b<=shape[1]):raise ValueError('Ownership outside source')
        visits+=b-a
        if visits>2000000:raise ValueError('Ownership pixel budget')
        if out[y,a:b].any():raise ValueError('Duplicate ownership spans')
        out[y,a:b]=True
    return out


def primitive_points(c,shape):
    kind=c['kind'];h,w=shape
    if kind in ('line','polyline','arrow_open','arrow_filled'):
        p=np.asarray(c['points_px'],float)
        minimum=2 if kind in ('line','polyline') else 3
        if p.ndim!=2 or p.shape[1]!=2 or not minimum<=len(p)<=8192:raise ValueError('Invalid primitive vertices')
        if kind=='line' and len(p)!=2 or kind.startswith('arrow_') and len(p)!=3:raise ValueError('Invalid primitive vertex count')
        if np.any(np.linalg.norm(np.diff(p,axis=0),axis=1)<1e-7):raise ValueError('Degenerate segment')
        if kind.startswith('arrow_'):
            a,b=p[1]-p[0],p[2]-p[0]
            if abs(float(a[0]*b[1]-a[1]*b[0]))<.5:raise ValueError('Degenerate head')
        if kind=='polyline' and c.get('closed',False) and len(p)<3:raise ValueError('Degenerate closed polyline')
    elif kind in ('circle','arc'):
        center=np.asarray(c['center_px'],float);r=float(c['radius_px'])
        start=float(c.get('start_angle_deg',0));sweep=360. if kind=='circle' else float(c['sweep_angle_deg'])
        if center.shape!=(2,) or not np.isfinite([*center,r,start,sweep]).all() or r<=0 or not 0<sweep<=360:
            raise ValueError('Invalid finite curve')
        if kind=='arc' and sweep>=360:raise ValueError('Full curve requires CIRCLE')
        count=max(8,math.ceil(r*math.radians(sweep)/.25)+1)
        if count>250000:raise ValueError('Curve sample budget')
        a=np.linspace(math.radians(start),math.radians(start+sweep),count)
        p=center+r*np.c_[np.cos(a),np.sin(a)]
    else:raise ValueError('Unsupported observed primitive')
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p[:,0]>=w) or np.any(p[:,1]>=h):raise ValueError('Primitive outside source')
    return p


def footprint(c,p,shape):
    image=np.zeros(shape,np.uint8);q=np.rint(p).astype(np.int32)
    if np.any(q<0) or np.any(q[:,0]>=shape[1]) or np.any(q[:,1]>=shape[0]):raise ValueError('Rounded primitive outside source')
    if c['kind']=='arrow_filled':cv2.fillPoly(image,[q],1)
    elif c['kind']=='arrow_open':
        cv2.line(image,tuple(q[0]),tuple(q[1]),1,1);cv2.line(image,tuple(q[0]),tuple(q[2]),1,1)
    else:cv2.polylines(image,[q],c['kind']=='circle' or c.get('closed',False),1,1)
    return image>0


def region_bijection(a,b,connectivity):
    na,la=cv2.connectedComponents(a.astype(np.uint8),connectivity=connectivity)
    nb,lb=cv2.connectedComponents(b.astype(np.uint8),connectivity=connectivity)
    if na!=nb:return False
    common=a&b;pairs=np.unique(la[common].astype(np.int64)*nb+lb[common])
    return len(pairs)==na-1 and len(np.unique(pairs//nb))==na-1 and len(np.unique(pairs%nb))==nb-1


def same_pixel_partitions(before,after):
    """Anchored ink and background components; counts alone are insufficient."""
    return region_bijection(before,after,8) and region_bijection(
        np.pad(~before,1,constant_values=True),np.pad(~after,1,constant_values=True),4)


def compile_layers(ink,observations,review,source_sha256):
    if ink.dtype!=bool or ink.ndim!=2 or not 0<ink.size<=4194304:raise ValueError('Bounded boolean source required')
    size=[ink.shape[1],ink.shape[0]]
    if (observations['source_sha256']!=source_sha256 or review['source_sha256']!=source_sha256
            or observations['size_px']!=size or review['size_px']!=size
            or observations.get('coordinate_space')!='processed_pixel_centres_xright_ydown'
            or review['observations_sha256']!=object_hash(observations)):
        raise ValueError('Review/observations do not bind this source and coordinate frame')
    if observations.get('schema_version')!='pixel-observations/1.0' or review.get('schema_version')!='pixel-role-review/1.0':
        raise ValueError('Unsupported review contract')
    items=observations['items'];reviews=review['items']
    if len(items)>4096 or len(reviews)>4096 or len({r['id'] for r in items})!=len(items) or len({r['id'] for r in reviews})!=len(reviews):
        raise ValueError('Duplicate IDs or object budget')
    if len({r['observation_id'] for r in reviews})!=len(reviews):raise ValueError('Multiple decisions for one observation')
    by_id={c['id']:c for c in items}
    protected=spans_mask(review.get('protected_spans',[]),ink.shape)
    text=spans_mask(review.get('text_protected_spans',[]),ink.shape)
    if np.any((protected|text)&~ink):raise ValueError('Protection must reference source ink')
    claims=np.zeros(ink.shape,np.uint16);paint_claims=np.zeros_like(claims)
    pending=[];ledger=[];roi_work=0;owned_visits=0;sample_count=0
    for row in sorted(reviews,key=lambda r:r['id']):
        c=by_id.get(row['observation_id']);reason=None
        if c is None:reason='unknown_observation'
        elif row.get('status')!='reviewed_pixel_role' or not row.get('reviewer') or not row.get('review_ref'):reason='role_not_reviewed'
        elif row['observation_sha256']!=object_hash(c):reason='observation_changed'
        elif row.get('role') not in LAYERS:reason='unsupported_role'
        elif c['kind'] not in ('line','polyline','circle','arc','arrow_open','arrow_filled','ink_region'):reason='unsupported_primitive'
        elif (c['kind'].startswith('arrow_'))!=(row['role']=='arrowhead'):reason='role_type_conflict'
        elif (c['kind']=='ink_region')!=(row['role']=='text'):reason='region_is_not_native_geometry_or_text'
        if reason:
            ledger.append({'id':row['id'],'observation_id':row['observation_id'],'status':'needs_review','reason':reason});continue
        if any(not isinstance(v,str) or len(v)>128 or not v.strip() for v in [c['id'],c.get('path_id',c['id']),row['id'],row['review_ref'],row['reviewer']]):
            raise ValueError('Invalid provenance identifier')
        bound=float(row.get('uncertainty_px',1.5))
        if not math.isfinite(bound) or not 0<bound<=1.5:raise ValueError('Unapproved pixel uncertainty bound')
        owned=spans_mask(row['owned_spans'],ink.shape);owned_visits+=int(owned.sum())
        if owned_visits>2000000:raise ValueError('Global ownership budget')
        if np.any(owned&~ink):raise ValueError('Ownership must reference source ink')
        if not owned.any():
            ledger.append({'id':row['id'],'observation_id':c['id'],'status':'needs_review','reason':'empty_ownership'});continue
        p=None if c['kind']=='ink_region' else primitive_points(c,ink.shape)
        sample_count+=0 if p is None else len(p)
        if sample_count>250000:raise ValueError('Global sample budget')
        # Deterministic per-item mask work; wall time and RSS are separate limits.
        roi_work+=ink.size*(2 if p is None else 6)
        if roi_work>40000000:raise ValueError('Raster work budget')
        paint=owned.copy() if p is None else footprint(c,p,ink.shape)
        shared=np.zeros_like(ink)
        contacts=row.get('shared_contacts',[])
        if len(contacts)>2:raise ValueError('Shared contact budget')
        for contact in contacts:
            q=np.asarray(contact)
            if q.shape!=(2,) or not np.isfinite(q).all() or not np.equal(q,np.rint(q)).all():raise ValueError('Integer shared pixel required')
            x,y=map(int,q)
            connectors=[] if p is None else [p[0],p[-1]] if c['kind'] in ('line','polyline') and not c.get('closed') else []
            if c['kind'].startswith('arrow_'):connectors=[p[0],(p[1]+p[2])/2]
            if not (0<=x<ink.shape[1] and 0<=y<ink.shape[0]) or not ink[y,x] or not any(np.linalg.norm(q-a)<=.75 for a in connectors):
                raise ValueError('Shared contact is not an observed endpoint')
            shared[y,x]=True
        if np.any(shared&text):reason='text_cannot_be_shared_contact'
        elif p is not None and np.any((paint|owned)&(protected|text)&~shared):reason='protected_ink_conflict'
        elif p is not None and np.any(paint&ink&~owned&~shared):reason='unowned_ink_conflict'
        error=0.
        if p is not None and reason is None:
            reference=(owned|shared).astype(np.uint8)
            outward=cv2.distanceTransform(1-reference,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            inward=cv2.distanceTransform(1-paint.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            error=max(float(outward[paint].max()),float(inward[owned].max()))
            if error>bound:reason='unsupported_or_mixed_source_ink'
            elif cv2.connectedComponents(reference,connectivity=8)[0]!=2:reason='disconnected_source_support'
        effective=owned&~shared
        if not effective.any():reason=reason or 'no_exclusive_source_ink'
        if p is not None and reason is None:
            yy,xx=np.nonzero(owned|paint|shared)
            l,t=max(0,int(xx.min())-2),max(0,int(yy.min())-2)
            r,b=min(ink.shape[1],int(xx.max())+3),min(ink.shape[0],int(yy.max())+3)
            roi_work+=6*(r-l)*(b-t)
            if roi_work>40000000:raise ValueError('Raster work budget')
            original=ink[t:b,l:r]
            changed=(original&~effective[t:b,l:r])|paint[t:b,l:r]
            if not region_bijection(original,changed,8) or not region_bijection(~original,~changed,4):
                reason='source_pixel_connectivity_changed'
        claims[effective]+=1
        if p is not None:paint_claims[paint&~shared]+=1
        pending.append({'observation':deepcopy(c),'review':deepcopy(row),'owned':owned,'effective':effective,
            'shared':shared,'paint':paint,'reason':reason,'error':error})
    conflicts=claims>1;removed=np.zeros_like(ink);regions={};accepted=[];shared_all=np.zeros_like(ink)
    requested_shared=np.zeros_like(ink);bad_shared=np.zeros_like(ink)
    for p in pending:requested_shared|=p['shared']
    for p in pending:bad_shared|=(p['paint']|p['owned'])&requested_shared&~p['shared']
    for p in pending:
        c,row=p['observation'],p['review'];reason=p['reason']
        if np.any(p['effective']&conflicts):reason='ambiguous_overlapping_ownership'
        elif np.any((p['paint']|p['owned'])&bad_shared):reason='inconsistent_shared_contact_review'
        elif np.any((p['paint']&~p['shared'])&(paint_claims>1)):reason='unreviewed_vector_overlap'
        if reason:
            ledger.append({'id':row['id'],'observation_id':c['id'],'status':'needs_review','reason':reason});continue
        layer=LAYERS[row['role']][0]
        if c['kind']=='ink_region':regions.setdefault(layer,np.zeros_like(ink))[:] |= p['effective']
        else:removed|=p['effective']
        shared_all|=p['shared'];accepted.append(p)
        ledger.append({'id':row['id'],'observation_id':c['id'],'status':'reviewed_pixel_export_ready','layer':layer,
            'owned_pixels':int(p['effective'].sum()),'source_support_max_error_px':p['error'],
            'shared_source_pixels_retained':int(p['shared'].sum()),'physical_geometry_approved':False})
    region_union=np.zeros_like(ink)
    for mask in regions.values():region_union|=mask
    if np.any(removed&(protected|text)) or np.any(removed&region_union):raise ValueError('Protection/partition invariant')
    residual=ink&~removed&~region_union
    if not np.array_equal(residual|removed|region_union,ink):raise ValueError('Source partition incomplete')
    visible=residual|region_union
    for p in accepted:
        if p['observation']['kind']!='ink_region':visible|=p['paint']
    rollback=not same_pixel_partitions(ink,visible)
    if rollback:
        vectors={p['review']['id'] for p in accepted if p['observation']['kind']!='ink_region'}
        for r in ledger:
            if r['id'] in vectors:r.update(status='needs_review',reason='global_pixel_connectivity_changed')
        accepted=[p for p in accepted if p['observation']['kind']=='ink_region']
        removed[:]=False;shared_all[:]=False;residual=ink&~region_union
    return {'accepted':accepted,'ledger':sorted(ledger,key=lambda r:r['id']),'removed':removed,'regions':regions,
        'residual':residual,'shared':shared_all,'protected':protected|text,'raster_work':roi_work,
        'owned_pixel_visits':owned_visits,'curve_samples':sample_count,'conflicting_pixels':int(conflicts.sum()),
        'global_pixel_topology_preserved':True,'global_topology_rollback':rollback}


def attrs(layer):return {'layer':layer,'color':256,'linetype':'BYLAYER','lineweight':-1}


def provenance(entity,sha,id_,role,review_ref='',error=0.,piece=0,*,path_id=None,observation_sha='',bound=0.,height=0):
    entity.set_xdata(APPID,[(1000,'source='+sha),(1000,'observation='+id_),(1000,'role='+role),
        (1000,'review='+review_ref),(1000,'path='+(path_id or id_)),
        (1000,'observation_sha256='+observation_sha),(1000,'source_height='+str(height)),
        (1000,'frame=processed_pixel_centres_xright_ydown'),(1040,float(error)),(1040,float(bound)),(1070,piece)])


def snapshot(entity):
    result=semantic_entity_snapshot(entity)
    if entity.dxftype()=='SOLID':result['vertices']=[list(entity.dxf.get('vtx'+str(i))) for i in range(4)]
    if entity.dxftype()=='POINT':result['location']=list(entity.dxf.location)
    result['source_xdata']=[list(t) for t in entity.get_xdata(APPID)]
    return result


def layer_mask(doc,layer,shape):
    h,w=shape;out=np.zeros(shape,bool)
    for e in doc.modelspace().query('SOLID'):
        if e.dxf.layer!=layer:continue
        p=np.asarray([list(e.dxf.get('vtx'+str(i)))[:2] for i in range(4)])
        a,b=int(round(p[:,0].min()+.5)),int(round(p[:,0].max()+.5))
        t,u=int(round(h-.5-p[:,1].max())),int(round(h-.5-p[:,1].min()))
        if not (0<=a<b<=w and 0<=t<u<=h):raise ValueError('Invalid raster rectangle')
        expected={(a-.5,h-.5-u),(b-.5,h-.5-u),(a-.5,h-.5-t),(b-.5,h-.5-t)}
        if set(map(tuple,p))!=expected:raise ValueError('Raster SOLID is not an exact rectangle')
        if out[t:u,a:b].any():raise ValueError('Duplicate raster rectangle ink')
        out[t:u,a:b]=True
    return out


def role_graph_roundtrips(packet,mapping,loaded,height):
    """Use qualified v3 QA on approved geometry and actual reread DXF, per role.

    Exact serialization does not certify an unresolved source boundary as a part.
    """
    from .boundary_certificates_v3 import build,CertificateCache
    from .robust_graph_qa_v1 import compare_graphs
    by_id={p['observation']['id']:p['observation'] for p in packet['accepted']}
    groups={}
    def flip(p):return [float(p[0]),height-1-float(p[1])]
    for row in mapping:
        c=by_id[row['observation_id']];kind=c['kind']
        if kind=='ink_region':continue
        for i,handle in enumerate(row['handles']):
            e=loaded.entitydb[handle];base={'id':'export-'+handle,'path_id':row['path_id']}
            if kind.startswith('arrow_'):
                if kind=='arrow_open':a={**base,'kind':'line','points_px':[c['points_px'][0],c['points_px'][i+1]]}
                else:a={**base,'kind':'polyline','closed':True,'points_px':c['points_px']}
            else:a={**base,**{k:c[k] for k in ('kind','points_px','center_px','radius_px','start_angle_deg','sweep_angle_deg','closed') if k in c}}
            if e.dxftype()=='LINE':b={**base,'kind':'line','points_px':[flip(e.dxf.start),flip(e.dxf.end)]}
            elif e.dxftype()=='LWPOLYLINE':b={**base,'kind':'polyline','closed':bool(e.closed),'points_px':[flip(p) for p in e.get_points('xy')]}
            elif e.dxftype()=='SOLID':b={**base,'kind':'polyline','closed':True,'points_px':[flip(e.dxf.get('vtx'+str(j))) for j in range(3)]}
            else:
                b={**base,'kind':'circle' if e.dxftype()=='CIRCLE' else 'arc','center_px':flip(e.dxf.center),'radius_px':e.dxf.radius}
                if e.dxftype()=='ARC':b.update(start_angle_deg=(-e.dxf.end_angle)%360,sweep_angle_deg=(e.dxf.end_angle-e.dxf.start_angle)%360)
            pair=groups.setdefault(row['role'],([],[]));pair[0].append(a);pair[1].append(b)
    reports={}
    for role,(before,after) in groups.items():
        cache=CertificateCache();a,b=build(before,cache=cache),build(after,cache=cache)
        normalize=lambda g:{k:v for k,v in g.items() if k not in ('certificate_stats','certificate_cache_peak_entries','topology_predicates')}
        equal=_same_snapshot(normalize(a),normalize(b))
        if not equal:raise ValueError('DXF role graph changed: '+role)
        reports[role]={'geometry_graph_equal':True,'qa':compare_graphs(a,b),
            'source_nesting_complete':a.get('nesting_complete',False),
            'logical_predicates_before_after':[a.get('topology_predicates'),b.get('topology_predicates')],
            'physical_geometry_approved':False,'scope':'serialization preservation per role, not full drawing/role-classifier admission'}
    return reports


def export_pixel_layers(binary_path,observations,review,output):
    binary_path,output=Path(binary_path),Path(output)
    if output.exists():raise FileExistsError(output)
    sha=hashlib.sha256(binary_path.read_bytes()).hexdigest()
    with Image.open(binary_path) as image:
        if image.width*image.height>4194304:raise ValueError('Source pixel budget')
        binary=np.array(image.convert('L'))
    if not np.isin(binary,[0,255]).all():raise ValueError('Recorded binary source required')
    ink=binary==0;packet=compile_layers(ink,observations,review,sha)
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8),connectivity=8)
    isolated=np.zeros_like(ink)
    for i,s in enumerate(stats[1:],1):
        if s[4]==1:isolated[int(s[1]),int(s[0])]=True
    isolated&=packet['residual'];residual=packet['residual']&~isolated
    masks={'SOURCE_INK':ink,'UNKNOWN_INK':residual,**packet['regions']}
    rectangles={layer:merged_runs(mask,limit=50000) for layer,mask in masks.items() if layer not in packet['regions']}
    region_runs=[(p,merged_runs(p['effective'],limit=50000)) for p in packet['accepted'] if p['observation']['kind']=='ink_region']
    native_count=sum(2 if p['observation']['kind']=='arrow_open' else 0 if p['observation']['kind']=='ink_region' else 1 for p in packet['accepted'])
    if sum(map(len,rectangles.values()))+sum(len(runs) for _,runs in region_runs)+native_count+int(isolated.sum())+1>50000:raise ValueError('DXF entity budget')
    doc=ezdxf.new('R2013');doc.units=0;doc.appids.new(APPID);model=doc.modelspace();h,w=ink.shape
    for layer,weight in list(LAYERS.values())+[('UNKNOWN_INK',0),('UNKNOWN_POINTS',0),('SOURCE_INK',0)]:
        if layer not in doc.layers:doc.layers.new(layer,dxfattribs={'color':8 if layer.startswith('UNKNOWN') else 7,'lineweight':weight})
    doc.layers.get('SOURCE_INK').off();doc.layers.get('SOURCE_INK').lock();doc.layers.get('SOURCE_INK').dxf.plot=0
    def point(p):return float(p[0]),h-1-float(p[1])
    expected=[];mapping=[]
    for layer,runs in rectangles.items():
        for i,(a,t,b,u) in enumerate(runs):
            e=model.add_solid([(a-.5,h-.5-u),(b-.5,h-.5-u),(a-.5,h-.5-t),(b-.5,h-.5-t)],dxfattribs=attrs(layer))
            provenance(e,sha,f'{layer}-run-{i}','source_ink' if layer=='SOURCE_INK' else 'text_ink' if layer=='TEXT_INK' else 'unknown',height=h)
            expected.append(snapshot(e))
    for p,runs in region_runs:
        c,r=p['observation'],p['review'];layer=LAYERS[r['role']][0];handles=[]
        for a,t,b,u in runs:
            e=model.add_solid([(a-.5,h-.5-u),(b-.5,h-.5-u),(a-.5,h-.5-t),(b-.5,h-.5-t)],dxfattribs=attrs(layer))
            provenance(e,sha,c['id'],r['role'],r['review_ref'],path_id=c.get('path_id',c['id']),
                observation_sha=object_hash(c),height=h)
            expected.append(snapshot(e));handles.append(e.dxf.handle)
        mapping.append({'observation_id':c['id'],'path_id':c.get('path_id',c['id']),'observation_sha256':object_hash(c),
            'review_id':r['id'],'review_ref':r['review_ref'],'role':r['role'],'layer':layer,
            'handles':handles,'types':['SOLID']*len(handles),'uncertainty_px':0.,'native_text':False})
    for y,x in zip(*np.nonzero(isolated)):
        e=model.add_point(point([x,y]),dxfattribs=attrs('UNKNOWN_POINTS'))
        provenance(e,sha,f'pixel-{x}-{y}','unknown',height=h);expected.append(snapshot(e))
    for p in packet['accepted']:
        c,r=p['observation'],p['review'];kind=c['kind'];layer=LAYERS[r['role']][0];entities=[]
        if kind=='ink_region':continue
        if kind=='line':entities=[model.add_line(*map(point,c['points_px']),dxfattribs=attrs(layer))]
        elif kind=='polyline':entities=[model.add_lwpolyline(list(map(point,c['points_px'])),close=c.get('closed',False),dxfattribs=attrs(layer))]
        elif kind=='circle':entities=[model.add_circle(point(c['center_px']),c['radius_px'],dxfattribs=attrs(layer))]
        elif kind=='arc':entities=[model.add_arc(point(c['center_px']),c['radius_px'],-(c['start_angle_deg']+c['sweep_angle_deg']),-c['start_angle_deg'],dxfattribs=attrs(layer))]
        elif kind=='arrow_filled':entities=[model.add_solid(list(map(point,c['points_px'])),dxfattribs=attrs(layer))]
        elif kind=='arrow_open':entities=[model.add_line(point(c['points_px'][0]),point(q),dxfattribs=attrs(layer)) for q in c['points_px'][1:]]
        for i,e in enumerate(entities):
            provenance(e,sha,c['id'],r['role'],r['review_ref'],p['error'],i,path_id=c.get('path_id',c['id']),
                observation_sha=object_hash(c),bound=r.get('uncertainty_px',1.5),height=h)
            expected.append(snapshot(e))
        mapping.append({'observation_id':c['id'],'path_id':c.get('path_id',c['id']),'observation_sha256':object_hash(c),
            'review_id':r['id'],'review_ref':r['review_ref'],'role':r['role'],'layer':layer,
            'handles':[e.dxf.handle for e in entities],'types':[e.dxftype() for e in entities],'uncertainty_px':r.get('uncertainty_px',1.5)})
    _configure_view(doc,(-.5,-.5,w-.5,h-.5),pixel_units=True)
    doc.header['$LWDISPLAY']=True;doc.header['$PDMODE']=2;doc.header['$PDSIZE']=.7
    if hashlib.sha256(binary_path.read_bytes()).hexdigest()!=sha:raise ValueError('Source changed during export')
    doc.saveas(output);loaded=ezdxf.readfile(output);audit=loaded.audit()
    if audit.errors or audit.fixes:raise ValueError('DXF requires repairs')
    if loaded.units!=0 or loaded.modelspace().query('DIMENSION HATCH IMAGE'):raise ValueError('Unexpected semantic/physical primitive')
    for state in expected:
        e=loaded.entitydb[state['handle']]
        if not _same_snapshot(state,snapshot(e)):raise ValueError('DXF geometry/provenance roundtrip mismatch')
    if not loaded.layers.get('SOURCE_INK').is_off() or loaded.layers.get('SOURCE_INK').dxf.plot!=0:raise ValueError('Original source must remain hidden/nonplotting')
    for layer,mask in masks.items():
        if not np.array_equal(layer_mask(loaded,layer,ink.shape),mask):raise ValueError('Raster layer roundtrip mismatch')
    points=np.zeros_like(ink)
    for e in loaded.modelspace().query('POINT[layer=="UNKNOWN_POINTS"]'):
        x,y=tuple(e.dxf.location)[:2];ix,iy=int(round(x)),int(round(h-1-y))
        if points[iy,ix]:raise ValueError('Duplicate unknown POINT')
        points[iy,ix]=True
    if not np.array_equal(points,isolated):raise ValueError('Unknown point roundtrip mismatch')
    graph_checks=role_graph_roundtrips(packet,mapping,loaded,h)
    report={'schema_version':POLICY['version'],'source_sha256':sha,'observations_sha256':object_hash(observations),
        'review_sha256':object_hash(review),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'status':'needs_review','units':0,'coordinate_space':'processed pixels, DXF y=height-1-source_y',
        'accepted_ids':[p['observation']['id'] for p in packet['accepted']], 'ledger':packet['ledger'],'entities':mapping,
        'input_ink_pixels':int(ink.sum()),'replaced_ink_pixels':int(packet['removed'].sum()),
        'classified_text_pixels':int(sum(m.sum() for m in packet['regions'].values())),
        'unknown_ink_pixels':int(residual.sum()),'unknown_points':int(isolated.sum()),
        'shared_source_pixels_retained':int((packet['shared']&packet['residual']).sum()),
        'conflicting_pixels':packet['conflicting_pixels'],'exact_source_partition':True,'protected_pixels_erased':0,
        'global_pixel_topology_preserved':packet['global_pixel_topology_preserved'],
        'global_topology_rollback':packet['global_topology_rollback'],
        'raster_roundtrip_exact':True,'native_roundtrip_exact':True,'audit_errors':0,'audit_fixes':0,
        'role_graph_roundtrips':graph_checks,
        'layer_counts':{l.dxf.name:dict(Counter(e.dxftype() for e in loaded.modelspace() if e.dxf.layer==l.dxf.name)) for l in loaded.layers},
        'raster_work':packet['raster_work'],'owned_pixel_visits':packet['owned_pixel_visits'],'curve_samples':packet['curve_samples'],
        'source_layer_off':True,'all_source_ink_archived':True,'automatic_role_profile_admitted':False,
        'physical_geometry_approved':False,'manual_CAD_checked':False,'production_changed':False,'native_OCR_text':0}
    return report,packet
