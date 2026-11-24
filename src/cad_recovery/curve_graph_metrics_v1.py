"""Native centerline QA, with abstentions and unitless DXF round trips explicit."""
from collections import Counter
import math

import cv2
import numpy as np

from .curve_graph_v1 import sample,curve_distance
from .centerline_comparison_v1 import native_points,topology
from .stroke_curve_control_v1 import maximum_matching,distance_to_polyline,sample_polyline


def native(candidate,matrix):
    m=np.asarray(matrix,float);scale=m[0,0]
    if not (np.allclose(m[:2,:2],np.eye(2)*scale) and scale>0 and np.allclose(m[2],[0,0,1])):
        raise ValueError('Only the preregistered positive uniform affine scale is supported')
    c={**candidate}
    if c['kind'] in ('line','polyline'):c['points_px']=native_points(c['points_px'],matrix).tolist()
    else:c['center_px']=native_points([c['center_px']],matrix)[0].tolist();c['radius_px']*=scale
    return c


def distance(points,candidate):
    return (distance_to_polyline(points,candidate['points_px']) if candidate['kind'] in ('line','polyline')
            else curve_distance(points,candidate))


def target_candidate(target):
    c={**target}
    if c['kind']=='line':c['points_px']=c.pop('points')
    else:c['center_px']=c.pop('center');c['radius_px']=c.pop('radius')
    return c


def pair_errors(target,predicted):
    a=sample(target);b=sample(predicted)
    forward=distance(a,predicted);reverse=distance(b,target)
    errors=np.r_[forward,reverse]
    result={'p95_px':float(np.percentile(errors,95)),'max_px':float(errors.max()),
        'coverage':float((forward<=2).mean()),'sampling_spacing_px':.5,'distance_sampling_bound_px':.25}
    if target['kind']!='circle':
        result['endpoint_error_px']=float(min(max(np.linalg.norm(a[0]-b[0]),np.linalg.norm(a[-1]-b[-1])),
            max(np.linalg.norm(a[0]-b[-1]),np.linalg.norm(a[-1]-b[0]))))
    if target['kind'] in ('arc','circle'):
        result.update(center_error_px=float(np.linalg.norm(np.asarray(target['center_px'])-predicted['center_px'])),
            radius_error_px=abs(target['radius_px']-predicted['radius_px']))
        if target['kind']=='arc':result['sweep_error_deg']=abs(target['sweep_angle_deg']-predicted['sweep_angle_deg'])
    else:
        u=np.diff(np.array(target['points_px']),axis=0)[0];v=np.diff(np.array(predicted['points_px']),axis=0)[0]
        result['angle_error_deg']=math.degrees(math.acos(float(np.clip(abs(u@v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))))
    return result


def qualified(errors):
    return (errors['p95_px']<=2 and errors['coverage']>=.8 and errors.get('endpoint_error_px',0)<=3
        and errors.get('angle_error_deg',0)<=5 and errors.get('center_error_px',0)<=2
        and errors.get('radius_error_px',0)<=2 and errors.get('sweep_error_deg',0)<=5)


def score(case,candidates,matrix):
    predictions=[native(c,matrix) for c in candidates if c['kind'] in ('line','arc','circle')]
    dense=case['scope']=='analytical_control'
    targets=[target_candidate(t) for t in case.get('labels',{}).get('geometry',[])] if dense else []
    edges=[];details=[]
    for target in targets:
        values={}
        for j,p in enumerate(predictions):
            if p['kind']!=target['kind']:continue
            if p['kind']!='line' and (np.linalg.norm(np.asarray(p['center_px'])-target['center_px'])>2
                                    or abs(p['radius_px']-target['radius_px'])>2):continue
            e=pair_errors(target,p)
            if qualified(e):values[j]=e
        edges.append(sorted(values,key=lambda j:values[j]['p95_px']));details.append(values)
    assignment=maximum_matching(edges)
    matches=[{'target':targets[i]['id'],'candidate':predictions[j]['id'],'kind':targets[i]['kind'],**details[i][j]}
        for i,j in assignment.items()]
    counts={}
    for kind in ('line','arc','circle'):
        total=sum(c['kind']==kind for c in targets);proposed=sum(c['kind']==kind for c in predictions)
        tp=sum(m['kind']==kind for m in matches)
        counts[kind]={'targets':total,'predictions':proposed,'TP':tp,'FP':proposed-tp if dense else None,
            'recall':tp/total if total else None,'precision':tp/proposed if dense and proposed else None}
    guides=[]
    for guide in ([] if dense else case.get('geometry_guides',[])):
        if 'points' not in guide:continue
        points=sample_polyline(guide['points'],.5);d=np.full(len(points),np.inf)
        for c in predictions:d=np.minimum(d,distance(points,c))
        guides.append({'id':guide['id'],'coverage':float((d<=guide.get('band_px',8)).mean()),
            'p95_px':float(np.percentile(d,95)) if np.isfinite(d).all() else None})
    gaps=[]
    for gap in case.get('labels',{}).get('gaps',[]):
        if 'points' not in gap:continue
        points=sample_polyline(gap['points'],.5)
        gaps.append({'id':gap['id'],'bridged':any(float((distance(points,c)<=2).mean())>=.8 for c in predictions)})
    # An open sector must never receive matching credit from a completed circle.
    completions=[]
    for t in targets:
        if t['kind']!='arc':continue
        completions.extend(p['id'] for p in predictions if p['kind']=='circle'
            and np.linalg.norm(np.asarray(p['center_px'])-t['center_px'])<=2 and abs(p['radius_px']-t['radius_px'])<=2)
    return {'counts':counts,'matches':matches,'guides':guides,'gaps':gaps,
        'duplicate_eligible':sum(max(0,len(e)-1) for e in edges),'false_circle_completions':sorted(set(completions)),
        'unresolved_paths':sum(c['kind']=='polyline' for c in candidates),'full_real_precision':None,
        'physical_geometry_approved':False,'units':'native_px','mm':'not_evaluable'}


def simplification(candidates):
    rows=[]
    for c in candidates:
        if 'support_points' not in c:continue
        observed=np.array(c['support_points']);predicted=sample(c)
        a=distance(observed,c);b=distance_to_polyline(predicted,observed);errors=np.r_[a,b]
        rows.append({'id':c['id'],'kind':c['kind'],'p95_px':float(np.percentile(errors,95)),
            'max_px':float(errors.max()),'sampling_bound_px':.25,'support_vertex_count':len(observed),
            'fitted_parameter_count':4 if c['kind']=='line' else 5 if c['kind']=='arc' else 3,
            'observed_closed_path':c['observed_closed_path'],'radial_or_normal_p95_px':c['fit_p95_px']})
    return rows


def rendered_topology(candidates,shape):
    raster=np.zeros(shape,np.uint8)
    for c in candidates:
        p=np.rint(sample(c)).astype(np.int32)
        if len(p)>1:cv2.polylines(raster,[p],c['kind']=='circle' or c.get('closed',False),255,1)
    return {'topology':topology(raster),'scope':'1px diagnostic raster; quantization may change adjacency'}


def dxf_roundtrip(candidates,height,path):
    import ezdxf
    doc=ezdxf.new('R2013');doc.units=0;model=doc.modelspace();expected=[]
    layers={'line':'CV_LINES','arc':'CV_ARCS','circle':'CV_CIRCLES','polyline':'UNRESOLVED_PATHS'}
    for name in layers.values():doc.layers.new(name)
    def flip(point):return [point[0],height-1-point[1]]
    for c in candidates:
        attrs={'layer':layers[c['kind']]}
        if c['kind']=='line':e=model.add_line(*[flip(p) for p in c['points_px']],dxfattribs=attrs)
        elif c['kind']=='circle':e=model.add_circle(flip(c['center_px']),c['radius_px'],dxfattribs=attrs)
        elif c['kind']=='arc':e=model.add_arc(flip(c['center_px']),c['radius_px'],
            -(c['start_angle_deg']+c['sweep_angle_deg']),-c['start_angle_deg'],dxfattribs=attrs)
        else:e=model.add_lwpolyline([flip(p) for p in c['points_px']],close=c.get('closed',False),dxfattribs=attrs)
        expected.append((c,e.dxf.handle))
    doc.saveas(path);loaded=ezdxf.readfile(path);audit=loaded.audit();errors=[];displacements=[]
    for c,handle in expected:
        e=loaded.entitydb[handle];r={**c}
        if c['kind']=='line':r['points_px']=[flip(list(e.dxf.start)[:2]),flip(list(e.dxf.end)[:2])]
        elif c['kind']=='polyline':r['points_px']=[flip(p[:2]) for p in e.get_points()];r['closed']=bool(e.closed)
        else:
            r['center_px']=flip(list(e.dxf.center)[:2]);r['radius_px']=e.dxf.radius
            if c['kind']=='arc':r['start_angle_deg']=(-e.dxf.end_angle)%360;r['sweep_angle_deg']=(e.dxf.end_angle-e.dxf.start_angle)%360
        if c['kind'] in ('line','polyline'):delta=float(np.abs(np.array(c['points_px'])-r['points_px']).max())
        else:
            delta=max(float(np.abs(np.array(c['center_px'])-r['center_px']).max()),abs(c['radius_px']-r['radius_px']))
            if c['kind']=='arc':
                angle=abs(((c['start_angle_deg']-r['start_angle_deg']+180)%360)-180)
                delta=max(delta,c['radius_px']*math.radians(angle+abs(c['sweep_angle_deg']-r['sweep_angle_deg'])))
        displacements.append(delta)
        if e.dxf.layer!=layers[c['kind']]:errors.append(c['id'])
    return {'entities':len(expected),'counts':dict(Counter(e.dxftype() for e in loaded.modelspace())),
        'audit_errors':len(audit.errors),'audit_fixes':len(audit.fixes),'layer_mismatch_ids':errors,
        'max_coordinate_displacement_px':max(displacements,default=0.),'INSUNITS':loaded.units,
        'manual_CAD_checked':False,'semantic_CAD_admission':False}
