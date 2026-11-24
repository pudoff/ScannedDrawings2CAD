"""Head-first source proposals with short/curved-path confirmation.

New hypotheses are appended to immutable prior observations. Shape, path and
text-context gates are separate views; none grants semantic or CAD approval.
"""
from collections import Counter
from copy import deepcopy
import math
import cv2
import numpy as np
from .stroke_roles_v1 import checked_gray,foreground
from .local_width_heads_v1 import triangle,profiles,append_preserving
from .arrow_path_recovery_v1 import native_head_check
from .native_head_bounds_v1 import safe_validator
from .head_source_context_v1 import analyze

POLICY={'version':'curved-source-heads/1.0','opening_radii_px':[7,5,4,3,2,1],
    'maximum_components_per_scale':256,'maximum_proposals':512,'maximum_profile_samples':8000000,
    'maximum_morphology_pixels':40000000,'maximum_contour_extent_px':192,
    'maximum_taper_relative_error':.30,'minimum_triangle_taper_slope':.1,
    'maximum_apex_spread_in_pens':1.,
    'minimum_base_to_path_width_ratio':2.5,'maximum_candidates':512,
    'source_erasure_allowed':False,'production_admitted':False,'physical_geometry_approved':False}


def shape(p,mask,audit):
    base=p[1:].mean(0);length=float(np.linalg.norm(p[0]-base));breadth=float(np.linalg.norm(p[1]-p[2]))
    cost=math.ceil((min(40.,max(14.,length))+length+1)/.5)*math.ceil((2*min(96.,breadth+3)+1)/.5)
    if audit['profile_samples']+cost>POLICY['maximum_profile_samples']:
        audit['limits'].append('profile_samples');return None,'profile_budget'
    along,widths,centers,cost=profiles(mask,p);audit['profile_samples']+=cost
    selected=(along>=.2*length)&(along<=.75*length)
    if selected.sum()<4 or np.mean(widths[selected]>0)<.95:return None,'head_profile_missing'
    slope,intercept=np.polyfit(along[selected],widths[selected],1)
    expected=slope*along[selected]+intercept
    error=float(np.median(np.abs(widths[selected]-expected)/np.maximum(1.,expected)))
    if slope>=-POLICY['minimum_triangle_taper_slope'] or error>POLICY['maximum_taper_relative_error']:return None,'not_triangle_taper'
    root=float(-intercept/slope)
    if not .8*length<=root<=1.2*length:return None,'apex_extrapolation'
    roots=[]
    for lo,hi in ((.2,.6),(.4,.8)):
        region=(along>=lo*length)&(along<=hi*length)
        if region.sum()<4 or np.mean(widths[region]>0)<.95:return None,'unstable_taper_support'
        m,b=np.polyfit(along[region],widths[region],1)
        if m>=-POLICY['minimum_triangle_taper_slope']:return None,'unstable_taper_slope'
        roots.append(float(-b/m))
    pen=max(1.,min(6.,breadth/4))
    if max(roots+[root])-min(roots+[root])>max(2.,POLICY['maximum_apex_spread_in_pens']*pen):return None,'unstable_taper_apex'
    # Locate the broad end in actual source widths, rather than requiring a
    # straight shaft behind the enclosing triangle's extrapolated base.
    window=np.flatnonzero((along>=-.3*length)&(along<=.35*length)&np.isfinite(centers))
    if not len(window):return None,'base_unobserved'
    station=int(window[np.argmax(widths[window])]);offset=float(along[station]);unit=(p[0]-base)/length
    normal=np.array([-unit[1],unit[0]]);newbase=base+offset*unit
    width=float(intercept+slope*offset)
    if width<=0 or root-offset<5:return None,'head_extent'
    polygon=np.array([base+root*unit,newbase+normal*width/2,newbase-normal*width/2])
    head={'tip_px':polygon[0].tolist(),'direction':unit.tolist(),'polygon_px':polygon.tolist(),
        'classification':'filled_arrow_candidate','uncertainty_px':max(1.,min(6.,width/4)),
        'confidence':.5,'evidence':{'method':POLICY['version'],'path_kind':'source_unclassified',
            'local_shaft_width_px':max(1.,min(6.,width/4)),'taper_relative_error':error,
            'head_length_px':float(root-offset),'base_width_px':width,'confidence_calibrated':False}}
    valid,check=safe_validator(native_head_check,[])(head,mask,head['evidence']['local_shaft_width_px'],())
    if not valid:return None,check['reason']
    head['evidence']['native_head_check']=check
    tip=np.asarray(head['tip_px']);radius=head['uncertainty_px']
    left,top=np.maximum(0,np.floor(tip-radius)).astype(int)
    right,bottom=np.minimum(mask.shape[::-1],np.ceil(tip+radius)+1).astype(int)
    if right<=left or bottom<=top:return None,'apex_outside_source'
    yy,xx=np.indices((bottom-top,right-left));near=(xx+left-tip[0])**2+(yy+top-tip[1])**2<=radius**2
    if not np.any(near&(mask[top:bottom,left:right]>0)):return None,'apex_without_source_ink'
    return head,None


def search(gray,*,baseline=(),prior_observations=()):
    checked_gray(gray)
    if len(baseline)>512 or len({h['id'] for h in baseline})!=len(baseline):raise ValueError('Bounded unique observations required')
    if len(prior_observations)>512 or len({h['id'] for h in prior_observations})!=len(prior_observations):raise ValueError('Bounded unique prior observations required')
    mask=foreground(gray);audit={'morphology_pixels':0,'components_examined':0,'profile_samples':0,'limits':[]}
    proposals=[];reasons=Counter()
    for radius in POLICY['opening_radii_px']:
        if audit['morphology_pixels']+gray.size>POLICY['maximum_morphology_pixels']:
            audit['limits'].append('morphology_pixels');break
        opened=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
        audit['morphology_pixels']+=int(gray.size)
        contours,_=cv2.findContours(opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours=[c for c in contours if cv2.contourArea(c)>=4 and max(cv2.boundingRect(c)[2:])<=192]
        contours.sort(key=lambda c:(cv2.boundingRect(c)[1],cv2.boundingRect(c)[0],-cv2.contourArea(c)))
        if len(contours)>POLICY['maximum_components_per_scale']:
            audit['limits'].append('components_per_scale');break
        for contour in contours:
            audit['components_examined']+=1;p=triangle(contour)
            if p is None:reasons['not_triangle_proposal']+=1;continue
            h,reason=shape(p,mask,audit)
            if reason:reasons[reason]+=1
            if audit['limits']:break
            if h is None:continue
            _,novel=append_preserving(list(prior_observations or baseline)+proposals,[h])
            if not novel:continue
            if len(baseline)+len(proposals)>=POLICY['maximum_candidates']:
                audit['limits'].append('candidates');break
            h['id']='curved-source-head-'+str(len(proposals)+1).zfill(5)
            while h['id'] in {x['id'] for x in baseline}:h['id']+='-new'
            h['evidence']['opening_radius_px']=radius;proposals.append(h)
        if audit['limits']:break
    context=analyze(gray,proposals);path=[];guarded=[]
    dt=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    for h,d in zip(proposals,context['decisions']):
        route=d['source_path'];supported=d['independent_stem_supported']
        if not supported:continue
        points=np.asarray(route.get('path_px',[]),int)
        measured=float(np.median(2*dt[points[:,1],points[:,0]]-1)) if len(points) else h['evidence']['local_shaft_width_px']
        if h['evidence']['base_width_px']/max(1.,measured)<POLICY['minimum_base_to_path_width_ratio']:continue
        head=deepcopy(h);head['evidence'].update(source_path=route,observed_path_width_px=measured)
        path.append(head)
        # A novel head in an aligned glyph group needs independent semantic
        # review even if it has just one long stem (e.g. a serif digit).
        d['novel_head_text_context_abstention']=bool(d['aligned_peer_groups'])
        if not d['positive_text_component_counterevidence'] and not d['aligned_peer_groups']:guarded.append(deepcopy(head))
    complete=not audit['limits'] and context['complete']
    audit['rejections']=dict(reasons);audit['source_context']=context['audit']
    if not complete:path=[];guarded=[]
    return {'raw_observations':deepcopy(list(baseline)),'proposals':proposals,'decisions':context['decisions'],
        'shape_only':deepcopy(list(baseline))+deepcopy(proposals) if complete else deepcopy(list(baseline)),
        'path_only':deepcopy(list(baseline))+path,'guarded':deepcopy(list(baseline))+guarded,
        'audit':audit,'complete':complete,'source_pixels_erased':0,'production_admitted':False,'physical_geometry_approved':False}
