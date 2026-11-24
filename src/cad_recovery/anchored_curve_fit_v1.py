"""Endpoint-constrained observational fitting, with explicit curve sensitivity.

No vertex snapping, nominal dimensions, semantic roles or noise erasure.
Only points on one observed skeleton path can be simplified together.
"""
import copy
import math
from collections import Counter

import cv2
import numpy as np

from .curve_graph_v1 import native_support,sample,width_evidence
from .geometry_detection import _fit_circle
from .stroke_curve_control_v1 import distance_to_polyline
from .simplification_qa_v1 import source_paths,guard
from .curve_graph_metrics_v1 import native

POLICY={'version':'anchored-curve-fit/1.0','fit_p95_input_px':1.25,'fit_max_input_px':2.,
    'minimum_typed_length_input_px':8.,'minimum_arc_sweep_deg':20.,'minimum_sagitta_input_px':1.,
    'curve_line_RMSE_ratio':.65,'maximum_radius_in_sides':4.,'maximum_fit_points':1024,
    'maximum_fit_calls':8192,'maximum_depth':8,'maximum_merge_passes':3,
    'robust_iterations':12,'backtracking_steps':8,'Huber_delta_input_px':.5,
    'radial_resolution_input_px':.5,'curve_parameter_sensitivity_native_px_max':2.,
    'input_pixels_max':4194304,'endpoints':'exact observed path endpoints, not independent projections',
    'uncertainty':'local model sensitivity, not calibrated confidence or guaranteed physical error',
    'ownership':'all inferred geometry unclassified; semantic promotion requires separate evidence',
    'production_admitted':False,'physical_geometry_approved':False}


def huber(residual):
    r=np.abs(residual);return float(np.mean(np.where(r<=.5,.5*r*r,.5*(r-.25))))


def circle_solution(points,closed,max_radius,anchor=None):
    """Robust radial fit; an open arc center is constrained to the endpoint bisector."""
    selected=points[np.linspace(0,len(points)-1,min(len(points),1024)).astype(int)]
    initial=_fit_circle(selected)
    if initial is None:return None
    center,radius,_=initial
    if closed and anchor is not None:
        anchor=np.asarray(anchor,float)
        for _ in range(12):
            radius=float(np.linalg.norm(center-anchor));delta=center-selected;length=np.linalg.norm(delta,axis=1)
            if radius<2 or np.any(length<1e-8):return None
            residual=length-radius;jacobian=delta/length[:,None]-(center-anchor)/radius
            weights=np.minimum(1.,.5/np.maximum(np.abs(residual),1e-12))
            update=np.linalg.lstsq(jacobian*np.sqrt(weights[:,None]),-residual*np.sqrt(weights),rcond=None)[0]
            old=huber(residual)
            for step in range(8):
                proposed=center+update/2**step;r=float(np.linalg.norm(proposed-anchor))
                if 2<=r<=max_radius and huber(np.linalg.norm(selected-proposed,axis=1)-r)<=old:
                    center=proposed;break
            else:break
            if np.linalg.norm(update)<1e-7:break
        radius=float(np.linalg.norm(center-anchor));delta=center-selected;length=np.linalg.norm(delta,axis=1)
        if radius<2 or np.any(length<1e-8):return None
        jacobian=delta/length[:,None]-(center-anchor)/radius
        singular=np.linalg.svd(jacobian/math.sqrt(len(selected)),compute_uv=False)
        sensitivity=.5/max(float(singular[-1]),1e-12)
    elif closed:
        for _ in range(12):
            delta=selected-center;length=np.linalg.norm(delta,axis=1)
            if np.any(length<1e-8):return None
            residual=length-radius;jacobian=np.c_[-delta/length[:,None],-np.ones(len(selected))]
            weights=np.minimum(1.,.5/np.maximum(np.abs(residual),1e-12))
            update=np.linalg.lstsq(jacobian*np.sqrt(weights[:,None]),-residual*np.sqrt(weights),rcond=None)[0]
            old=huber(residual)
            for step in range(8):
                proposed_center=center+update[:2]/2**step;proposed_radius=radius+update[2]/2**step
                if 2<=proposed_radius<=max_radius and huber(np.linalg.norm(selected-proposed_center,axis=1)-proposed_radius)<=old:
                    center,radius=proposed_center,float(proposed_radius);break
            else:break
            if np.linalg.norm(update)<1e-7:break
        delta=selected-center;length=np.linalg.norm(delta,axis=1)
        if np.any(length<1e-8):return None
        jacobian=np.c_[-delta/length[:,None],-np.ones(len(selected))]
        singular=np.linalg.svd(jacobian/math.sqrt(len(selected)),compute_uv=False)
        sensitivity=.5/max(float(singular[-1]),1e-12)
    else:
        a,b=points[[0,-1]];chord=b-a;length=float(np.linalg.norm(chord))
        if length<2:return None
        midpoint=(a+b)/2;normal=np.array([-chord[1],chord[0]])/length
        offset=float((center-midpoint)@normal)
        if abs(offset)>max_radius:return None
        for _ in range(12):
            center=midpoint+offset*normal;radius=math.hypot(length/2,offset)
            d=np.linalg.norm(selected-center,axis=1)
            if np.any(d<1e-8):return None
            residual=d-radius;jacobian=((center-selected)@normal)/d-offset/radius
            weights=np.minimum(1.,.5/np.maximum(np.abs(residual),1e-12))
            denom=float(np.sum(weights*jacobian**2))
            if denom<1e-12:return None
            update=-float(np.sum(weights*jacobian*residual))/denom;old=huber(residual)
            for step in range(8):
                proposed=offset+update/2**step;r=math.hypot(length/2,proposed)
                if 2<=r<=max_radius and huber(np.linalg.norm(selected-(midpoint+proposed*normal),axis=1)-r)<=old:
                    offset=proposed;break
            else:break
            if abs(update)<1e-7:break
        center=midpoint+offset*normal;radius=math.hypot(length/2,offset)
        d=np.linalg.norm(selected-center,axis=1)
        if np.any(d<1e-8):return None
        jacobian=((center-selected)@normal)/d-offset/radius
        sensitivity=.5/max(float(np.sqrt(np.mean(jacobian**2))),1e-12)
    if not np.isfinite(center).all() or not 2<=radius<=max_radius or not math.isfinite(sensitivity):return None
    return center,radius,sensitivity


def fit(points,closed,ink,*,conditioned,native_scale,protected_anchor=False):
    points=np.asarray(points,float)
    if len(points)<3:return None
    a,b=points[[0,-1]];chord=b-a;length=float(np.linalg.norm(chord));line=None;line_error=None
    if not closed and length>=8:
        errors=distance_to_polyline(points,[a,b]);line_error=float(np.sqrt(np.mean(errors**2)))
        proposed={'kind':'line','points_px':[a.tolist(),b.tolist()]}
        if np.percentile(errors,95)<=1.25 and errors.max()<=2 and native_support(proposed,ink):
            line={**proposed,'fit_p95_px':float(np.percentile(errors,95)),'fit_max_px':float(errors.max()),
                'fit_rmse_px':line_error,'endpoints_fixed':True,'source_support':1.}
    solution=circle_solution(points,closed,max(ink.shape)*4,points[0] if closed and protected_anchor else None)
    if solution is not None:
        center,radius,sensitivity=solution;radial=np.abs(np.linalg.norm(points-center,axis=1)-radius)
        angles=np.unwrap(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))
        if closed:
            candidate={'kind':'circle','center_px':center.tolist(),'radius_px':radius};sweep=2*math.pi
        else:
            # Read the actual traversal; never choose a missing sector by its bounding circle.
            delta=float(angles[-1]-angles[0]);start=angles[0] if delta>0 else angles[-1];sweep=abs(delta)
            candidate={'kind':'arc','center_px':center.tolist(),'radius_px':radius,
                'start_angle_deg':math.degrees(start)%360,'sweep_angle_deg':math.degrees(sweep)}
        error=float(np.sqrt(np.mean(radial**2)));sagitta=radius*(1-math.cos(min(sweep,math.pi)/2))
        eligible=closed or math.radians(20)<=sweep<2*math.pi-1e-7 and sagitta>=1 and (line is None or error<=line_error*.65)
        if eligible and np.percentile(radial,95)<=1.25 and radial.max()<=2 and native_support(candidate,ink):
            if not conditioned or sensitivity*native_scale<=2:
                return {**candidate,'fit_p95_px':float(np.percentile(radial,95)),'fit_max_px':float(radial.max()),
                    'fit_rmse_px':error,'source_support':1.,'endpoints_fixed':not closed,
                    'parameter_sensitivity_native_px':sensitivity*native_scale,
                    'radial_resolution_input_px':.5,'uncertainty_calibrated':False,
                    'observed_sweep_angle_deg':math.degrees(abs(float(angles[-1]-angles[0]))),
                    'closed_anchor_fixed':bool(closed and protected_anchor),
                    'observed_closed_path':closed}
    if line is not None:return {**line,'observed_closed_path':False}
    return None


def unresolved(points,closed,reason):
    return {'kind':'polyline','points_px':points.tolist(),'closed':closed,'reason':reason,
        'review_status':'unresolved','semantic_role':'unknown','physical_geometry_approved':False}


def split(points,closed,ink,account,conditioned,native_scale,depth=0,protected_anchor=False):
    if account['calls']>=8192:return [unresolved(points,closed,'fit_budget')]
    account['calls']+=1;length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    c=fit(points,closed,ink,conditioned=conditioned,native_scale=native_scale,protected_anchor=protected_anchor) if length>=8 else None
    if c is not None:return [{**c,'support_points':points.tolist()}]
    if closed or depth>=8 or len(points)<12 or length<16:return [unresolved(points,closed,'unsupported_or_uncertain_path')]
    midpoint=len(points)//2
    return split(points[:midpoint+1],False,ink,account,conditioned,native_scale,depth+1)+split(points[midpoint:],False,ink,account,conditioned,native_scale,depth+1)


def merge(parts,ink,account,conditioned,native_scale):
    for _ in range(3):
        output=[];i=0;changed=False
        while i<len(parts):
            if i+1<len(parts) and account['calls']<8192:
                a,b=parts[i:i+2];p=np.asarray(a.get('support_points',a.get('points_px')));q=np.asarray(b.get('support_points',b.get('points_px')))
                if np.array_equal(p[-1],q[0]) and not a.get('observed_closed_path',a.get('closed',False)) and not b.get('observed_closed_path',b.get('closed',False)):
                    joined=np.r_[p,q[1:]];account['calls']+=1
                    c=fit(joined,False,ink,conditioned=conditioned,native_scale=native_scale)
                    if c is not None:output.append({**c,'support_points':joined.tolist()});i+=2;changed=True;continue
            output.append(parts[i]);i+=1
        parts=output
        if not changed:break
    return parts


def extract(ink,skeleton,*,conditioned,input_to_native):
    if ink.dtype!=np.uint8 or skeleton.dtype!=np.uint8 or ink.shape!=skeleton.shape or ink.ndim!=2 or not 0<ink.size<=4194304:
        raise ValueError('Bounded aligned uint8 ink/skeleton required')
    if np.any((skeleton>0)&(ink==0)):raise ValueError('Skeleton outside observed ink')
    source,stats=source_paths(skeleton,np.eye(3));account={'calls':0};candidates=[]
    incidence=Counter(tuple(p) for c in source for p in (c['points_px'][0],c['points_px'][-1]) if len(c['points_px'])>1)
    scale=float(input_to_native[0][0])
    # Validate the full affine contract; no anisotropic or reflected radius scaling.
    native({'kind':'line','points_px':[[0,0],[1,1]]},input_to_native)
    for c in source:
        p=np.asarray(c['points_px'],float)
        protected_anchor=c['closed'] and incidence[tuple(p[0])]>2
        parts=split(p,c['closed'],ink,account,conditioned,scale,protected_anchor=protected_anchor)
        parts=merge(parts,ink,account,conditioned,scale)
        for part in parts:
            part.update(path_id=c['path_id'],classification='unclassified',semantic_role='unknown',
                semantic_ownership_confirmed=False,physical_geometry_approved=False,
                review_status='unresolved' if part['kind']=='polyline' else 'proposal_requires_review')
        candidates.extend(parts)
    distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    for i,c in enumerate(candidates):
        c['id']=f'{"conditioned" if conditioned else "anchored"}-{i:05d}'
        c['local_width']=width_evidence(c,ink,distance)
    return {'candidates':candidates,'source':source,'analysis':{**stats,'fit_calls':account['calls'],
        'search_incomplete':account['calls']>=8192,'isolated_observations_retained':stats['isolated_pixels']}}


def select_guarded(source,candidates,input_to_native,*,graphs=None):
    result=guard([native(c,input_to_native) for c in source],[native(c,input_to_native) for c in candidates],graphs=graphs)
    decision=result['decision']
    return {'candidates':copy.deepcopy(candidates if decision['accepted'] else source),'decision':decision,
        'exact_source_fallback':not decision['accepted'],'source_pixels_erased':False,
        'semantic_ownership_confirmed':False,'production_admitted':False}
