"""Bounded head-first proposals, validated against untouched native source ink."""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from . import arrow_path_recovery_v1 as previous
from .arrow_detection import _profile,_runs
from .family_catalog_v2 import require

POLICY={'version':'native-head-completion/1.0','opening_sides':[3,5,7,9,11],
    'maximum_pixels':25000000,'maximum_morphology_pixels':40000000,
    'maximum_components_per_scale':256,'maximum_proposals':512,'maximum_profile_pixels':8000000,
    'minimum_component_area':8,'maximum_component_extent_px':192,
    'minimum_head_length_px':6,'maximum_head_length_px':192,
    'minimum_head_aspect':1.8,'maximum_head_aspect':8,
    'maximum_side_length_ratio':1.4,'minimum_shaft_run_px':8,
    'maximum_shaft_probe_px':40,'minimum_base_to_shaft_ratio':2,
    'maximum_base_transition_in_strokes':2,'maximum_base_transition_in_head_lengths':.2,
    'minimum_tip_ink_radius_px':2,'tip_ink_radius_in_strokes':1,
    'manual_references_used':False,'production_admitted':False,'physical_geometry_approved':False,
    'confidence_calibrated':False,'short_shaft_is_not_a_complete_arc':True}


def triangle_proposal(contour,identity):
    area,triangle=cv2.minEnclosingTriangle(contour)
    if triangle is None or not np.isfinite(triangle).all():return None
    points=triangle.reshape(3,2).astype(float)
    edges=[np.linalg.norm(points[(i+1)%3]-points[(i+2)%3]) for i in range(3)]
    index=int(np.argmin(edges));tip=points[index];base=np.array([points[(index+1)%3],points[(index+2)%3]])
    center=base.mean(axis=0);delta=tip-center;length=float(np.linalg.norm(delta));width=float(edges[index])
    if not POLICY['minimum_head_length_px']<=length<=POLICY['maximum_head_length_px'] or width<=0:return None
    aspect=length/width;sides=np.linalg.norm(base-tip,axis=1)
    if not POLICY['minimum_head_aspect']<=aspect<=POLICY['maximum_head_aspect'] or sides.max()/sides.min()>POLICY['maximum_side_length_ratio']:return None
    return {'id':identity,'tip_px':tip.tolist(),'direction':(delta/length).tolist(),
        'polygon_px':np.vstack([tip,base]).tolist(),'classification':'filled_arrow_candidate',
        'uncertainty_px':max(2.,width*.25),'confidence':.55,
        'evidence':{'method':POLICY['version'],'path_kind':'head_seeded_native_shaft',
            'head_length_px':length,'base_width_px':width,'confidence_calibrated':False}}


def shaft_evidence(candidate,mask,stroke):
    polygon=np.asarray(candidate['polygon_px']);base=polygon[1:].mean(axis=0);unit=-np.asarray(candidate['direction'])
    length=candidate['evidence']['head_length_px'];width=candidate['evidence']['base_width_px']
    span=max(12,min(POLICY['maximum_shaft_probe_px'],length*.6))
    along,widths,midpoints=_profile(mask,base,unit,span)
    narrow=(along>=2)&(along<=span)&(widths>0)&(widths<=min(width*.65,max(3,stroke*2)))
    transition=min(length*POLICY['maximum_base_transition_in_head_lengths'],max(5,stroke*POLICY['maximum_base_transition_in_strokes']))
    runs=[(begin,end) for begin,end in _runs(narrow) if along[begin]<=transition
          and along[end-1]-along[begin]>=POLICY['minimum_shaft_run_px']
          and (widths[(along>=0)&(along<=along[begin])]>0).all()]
    if not runs:return False,{'reason':'no_connected_thin_shaft_behind_base'}
    begin,end=max(runs,key=lambda run:run[1]-run[0]);local_width=float(np.median(widths[begin:end]))
    if width/local_width<POLICY['minimum_base_to_shaft_ratio']:return False,{'reason':'head_has_no_base_width_contrast'}
    return True,{'reason':'observed_native_thin_shaft','observed_length_px':float(along[end-1]-along[begin]),
        'width_px':local_width,'base_to_shaft_ratio':width/local_width,
        'maximum_center_offset_px':float(np.max(np.abs(midpoints[begin:end]))),
        'complete_line_or_arc_inferred':False}


def tip_has_native_ink(candidate,mask,stroke):
    # Enclosing triangles may extrapolate a sharp apex into white paper.
    # Interior fill alone cannot validate such a vertex. Do not snap it to ink.
    tip=np.asarray(candidate['tip_px'])
    radius=max(POLICY['minimum_tip_ink_radius_px'],stroke*POLICY['tip_ink_radius_in_strokes'])
    left,top=np.maximum(0,np.floor(tip-radius)).astype(int)
    right,bottom=np.minimum([mask.shape[1],mask.shape[0]],np.ceil(tip+radius)+1).astype(int)
    if right<=left or bottom<=top:return False
    yy,xx=np.indices((bottom-top,right-left))
    disk=(xx+left-tip[0])**2+(yy+top-tip[1])**2<=radius**2
    return bool(np.any((mask[top:bottom,left:right]>0)&disk))


def proposals(gray,*,stroke_px,text_polygons=()):
    require(isinstance(gray,np.ndarray) and gray.ndim==2 and gray.dtype==np.uint8
        and 0<gray.size<=POLICY['maximum_pixels'],'Expected bounded native uint8 raster')
    require(np.isfinite(stroke_px) and 1<=stroke_px<=24,'Invalid native stroke estimate')
    for polygon in text_polygons:
        points=np.asarray(polygon,float)
        require(points.ndim==2 and points.shape[0]>=3 and points.shape[1]==2
                and np.isfinite(points).all() and np.abs(points).max()<=1e7, 'Invalid saved text polygon')
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    accepted=[];reasons=Counter();limits=[];morphology_pixels=profiles=examined=0
    for side in POLICY['opening_sides']:
        if morphology_pixels+gray.size>POLICY['maximum_morphology_pixels']:
            limits.append('morphology_pixel_budget');break
        opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((side,side),np.uint8));morphology_pixels+=gray.size
        contours,_=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours,key=lambda c:-cv2.contourArea(c))
        eligible=[]
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour)
            if cv2.contourArea(contour)>=POLICY['minimum_component_area'] and max(w,h)<=POLICY['maximum_component_extent_px']:eligible.append(contour)
        if len(eligible)>POLICY['maximum_components_per_scale']:limits.append('component_budget')
        for contour in eligible[:POLICY['maximum_components_per_scale']]:
            if examined>=POLICY['maximum_proposals']:
                limits.append('proposal_budget');break
            examined+=1;c=triangle_proposal(contour,'head-seed-'+str(examined))
            if c is None:reasons['not_elongated_triangle']+=1;continue
            if not tip_has_native_ink(c,mask,stroke_px):
                reasons['apex_without_native_ink']+=1;continue
            valid,check=previous.native_head_check(c,mask,stroke_px,text_polygons)
            if not valid:reasons[check['reason']]+=1;continue
            cost=81*(math.ceil(max(12,min(40,c['evidence']['head_length_px']*.6)))+97)
            if profiles+cost>POLICY['maximum_profile_pixels']:
                limits.append('shaft_profile_budget');break
            profiles+=cost;valid,shaft=shaft_evidence(c,mask,stroke_px)
            if not valid:reasons[shaft['reason']]+=1;continue
            c['evidence'].update(native_head_check=check,native_shaft=shaft,opening_side_px=side,
                                 native_tip_ink=True)
            accepted.append(c)
        if 'proposal_budget' in limits or 'shaft_profile_budget' in limits:break
    limits=sorted(set(limits))
    return {'candidates':previous.combine(accepted),'analysis':{'morphology_pixels':morphology_pixels,
        'components_examined':examined,'shaft_profile_pixels':profiles,
        'rejections':dict(reasons),'limits':limits},
        'search_incomplete':bool(limits),'production_admitted':False,'physical_geometry_approved':False}


def append_preserving_baseline(baseline,new):
    values=deepcopy(baseline)
    for c in deepcopy(new):
        same=False
        for old in values:
            if np.dot(c['direction'],old['direction'])<.94:continue
            a=np.asarray(c['polygon_px'],np.float32);b=np.asarray(old['polygon_px'],np.float32)
            overlap,_=cv2.intersectConvexConvex(a,b);denom=abs(cv2.contourArea(a))+abs(cv2.contourArea(b))-overlap
            if np.linalg.norm(np.asarray(c['tip_px'])-old['tip_px'])<=5 or denom>0 and overlap/denom>=.25:
                same=True;break
        if not same:values.append(c)
    for i,c in enumerate(values,1):c['id']='completed-arrow-'+str(i).zfill(5)
    return values


def search(gray,*,lines=(),arcs=(),stroke_px=3,text_polygons=()):
    baseline=previous.search(gray,lines=lines,arcs=arcs,stroke_px=stroke_px,text_polygons=text_polygons)
    extra=proposals(gray,stroke_px=stroke_px,text_polygons=text_polygons)
    return {'schema_version':'arrow-head-completion/1.0','policy':POLICY,
        'candidates':append_preserving_baseline(baseline['candidates'],extra['candidates']),
        'analysis':{'baseline':baseline['analysis'],'head_first':extra['analysis'],
            'baseline_candidates':len(baseline['candidates']),'head_first_candidates':len(extra['candidates'])},
        'search_incomplete':baseline['search_incomplete'] or extra['search_incomplete'],
        'production_admitted':False,'physical_geometry_approved':False,'CAD_promotions':0}
