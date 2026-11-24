"""Experimental source-backed arrow heads on normalized straight/arc strips.

The historical detector runs unchanged on finite normalized strips. Every mapped
head is checked against the native raster, and arcs need fresh head-boundary
evidence before they can provide a search path. Nothing is accepted into CAD.
"""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from .arrow_detection import _line_arrows, _profile, _runs, MARGIN, PROFILE_RADIUS
from .family_catalog_v2 import require
from .open_arc_search_v1 import radial_profile, endpoint_event

POLICY = {
    'version':'normalized-native-arrow-paths/1.0','target_stroke_px':3,
    'maximum_pixels':25000000,'maximum_paths':512,'maximum_path_length_px':8192,
    'maximum_sample_pixels':24000000,'maximum_candidates':512,
    'maximum_head_halo_px':192,'head_halo_in_strokes':24,
    'native_triangle_ink_min':.85,'native_flank_ink_max':.35,
    'minimum_arc_ink_support':.95,'maximum_arc_paths':32,
    'maximum_native_forward_shift_in_uncertainties':1,
    'maximum_native_backward_shift_in_head_lengths':.5,
    'maximum_transverse_axis_dot':.5,
    'minimum_native_transverse_flank_support':.75,
    'manual_references_used':False,'physical_geometry_approved':False,
    'production_admitted':False,'confidence_calibrated':False,
}


def path_for(candidate):
    if candidate['kind']=='line':
        points=np.asarray(candidate['points_px'],float);delta=points[1]-points[0]
        length=float(np.linalg.norm(delta))
        return {'kind':'line','id':candidate['id'],'first':points[0],
                'unit':delta/length,'length':length}
    radius=float(candidate['radius_px']);sweep=math.radians(candidate['sweep_angle_deg'])
    return {'kind':'arc','id':candidate['id'],'center':np.asarray(candidate['center_px'],float),
            'radius':radius,'start':math.radians(candidate['start_angle_deg']),
            'length':radius*sweep}


def coordinates(path,u,normal_offset):
    u,normal_offset=np.broadcast_arrays(np.asarray(u,float),np.asarray(normal_offset,float))
    if path['kind']=='line':
        unit=path['unit'];normal=np.array([-unit[1],unit[0]])
        return path['first']+u[...,None]*unit+normal_offset[...,None]*normal
    angle=path['start']+u/path['radius']
    radial=np.stack((np.cos(angle),np.sin(angle)),axis=-1)
    # Normal to the increasing-angle tangent points towards the circle centre.
    return path['center']+(path['radius']-normal_offset[...,None])*radial


def arc_head_evidence(mask,path,stroke,distance):
    if path['kind']!='arc':return True,[]
    u=np.linspace(0,path['length'],max(20,math.ceil(path['length']/2)))
    points=coordinates(path,u,0);pixels=np.rint(points).astype(int)
    inside=(pixels>=0).all(axis=1)&(pixels[:,0]<mask.shape[1])&(pixels[:,1]<mask.shape[0])
    if not inside.all():return False,['arc_outside_source']
    if np.mean(distance[pixels[:,1],pixels[:,0]]<=max(1.5,stroke*.4))<POLICY['minimum_arc_ink_support']:
        return False,['arc_without_native_ink']
    profile=radial_profile(mask,path['center'],path['radius'],stroke)
    evidence=[]
    for endpoint,(position,outward) in enumerate([(0,-1),(path['length'],1)]):
        angle=path['start']+position/path['radius']
        index=int(round(angle/(2*np.pi)*len(profile['supported'])))%len(profile['supported'])
        event=endpoint_event(mask,coordinates(path,position,0),stroke,profile,index,outward)
        if event=='abrupt_thick_head_boundary':evidence.append({'endpoint':endpoint,'event':event})
    return bool(evidence),evidence or ['arc_has_no_source_head_boundary']


def normalized_strip(gray,path,stroke):
    scale=min(1.,POLICY['target_stroke_px']/stroke)
    halo=min(POLICY['maximum_head_halo_px'],POLICY['head_halo_in_strokes']*stroke)
    length=(path['length']+2*halo)*scale
    width=math.ceil(length)+2*MARGIN+1
    along=(np.arange(width)-MARGIN)/scale-halo
    across=(np.arange(2*PROFILE_RADIUS+1)-PROFILE_RADIUS)/scale
    points=coordinates(path,along[None,:],across[:,None])
    strip=cv2.remap(gray,points[:,:,0].astype(np.float32),points[:,:,1].astype(np.float32),
                    cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
    _,mask=cv2.threshold(strip,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    offsets=np.zeros(width,float)
    if path['kind']=='arc':
        line={'id':'unrolled','points_px':[[MARGIN,PROFILE_RADIUS],[MARGIN+length,PROFILE_RADIUS]]}
        _,widths,midpoints=_profile(mask,np.array(line['points_px'][0],float),np.array([1.,0.]),length)
        # The profile spans exactly the strip columns apart from integer rounding.
        size=min(width,len(midpoints));radius=path['radius']*scale
        maximum=max(3.,(halo*scale)**2/(2*radius)+POLICY['target_stroke_px'])
        eligible=(widths[:size]>=4)&(widths[:size]<PROFILE_RADIUS*1.5)&(np.abs(midpoints[:size])<=maximum)
        offsets[:size][eligible]=midpoints[:size][eligible]
        offsets=cv2.medianBlur(offsets.astype(np.float32)[None,:],5)[0].astype(float)
        yy,xx=np.indices(mask.shape,dtype=np.float32)
        strip=cv2.remap(strip,xx,yy+offsets.astype(np.float32)[None,:],cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=255)
        _,mask=cv2.threshold(strip,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    return mask,{'scale':scale,'halo':halo,'length':length,'offsets':offsets,'sample_pixels':int(points.shape[0]*points.shape[1])}


def localize_tip(candidate,mask):
    """Use a unique complete junction near the measured-taper tip hypothesis."""
    result=deepcopy(candidate);e=candidate['evidence']
    shaft=e['shaft_width_px'];base=e['base_width_px'];length=e['arrow_length_px']
    fraction=(shaft-1)/max(base-1,1e-9)
    if not 0<=fraction<.5:return result,'original_tip_low_taper_contrast'
    tip=np.asarray(candidate['tip_px']);unit=np.asarray(candidate['direction'])
    # Include the raw/modelled apex range, but never accept a junction clipped
    # by that range (the adjacent extension line was the old ambiguous case).
    first=tip-unit*(fraction*length+2*shaft);span=fraction*length+4*shaft
    along,widths,_=_profile(mask,first,unit,span)
    high=widths>=max(base*1.5,shaft*4)
    junctions=[]
    for begin,end in _runs(high):
        low,upper=float(along[begin]),float(along[end-1])
        if 0<low and upper<span and end-begin<=2*shaft+3:
            junctions.append(float(np.mean(along[begin:end])))
    if len(junctions)!=1:return result,'original_tip_without_unique_local_junction'
    fixed=first+unit*junctions[0]
    result['tip_px']=fixed.tolist();result['polygon_px'][0]=fixed.tolist()
    return result,'unique_taper_local_raster_junction'


def native_candidate(candidate,path,transform):
    polygon=np.asarray(candidate['polygon_px'],float)
    x=polygon[:,0]
    offset=np.interp(x,np.arange(len(transform['offsets'])),transform['offsets'])
    u=(x-MARGIN)/transform['scale']-transform['halo']
    v=(polygon[:,1]-PROFILE_RADIUS+offset)/transform['scale']
    native=coordinates(path,u,v)
    delta=native[0]-native[1:].mean(axis=0);length=float(np.linalg.norm(delta))
    if length<=1:return None
    return {'tip_px':native[0].tolist(),'direction':(delta/length).tolist(),'polygon_px':native.tolist(),
        'classification':'filled_arrow_candidate','uncertainty_px':max(1.5,candidate['uncertainty_px']/transform['scale']),
        'confidence':candidate['confidence'],'evidence':{'method':POLICY['version'],
            'path_kind':path['kind'],'path_id':path['id'],'normalized_scale':transform['scale'],
            'legacy_profile_evidence':candidate['evidence'],'confidence_calibrated':False,
            'observed_shaft_support':candidate['evidence']['shaft_support_ratio']}}


def native_head_check(candidate,mask,stroke,text_polygons):
    points=np.asarray(candidate['polygon_px'],float)
    if not np.isfinite(points).all() or (points<0).any() or (points[:,0]>=mask.shape[1]).any() or (points[:,1]>=mask.shape[0]).any():
        return False,{'reason':'native_head_outside_source'}
    left,top=np.floor(points.min(axis=0)).astype(int);right,bottom=np.ceil(points.max(axis=0)).astype(int)+1
    triangle=np.zeros((bottom-top,right-left),np.uint8)
    cv2.fillConvexPoly(triangle,np.rint(points-[left,top]).astype(np.int32),255)
    interior=cv2.erode(triangle,np.ones((3,3),np.uint8))>0
    if interior.sum()<6:return False,{'reason':'native_triangle_too_small'}
    coverage=float(np.mean(mask[top:bottom,left:right][interior]>0))
    if coverage<POLICY['native_triangle_ink_min']:return False,{'reason':'native_triangle_not_filled','triangle_ink':coverage}
    tip=points[0];base=points[1:].mean(axis=0);unit=np.asarray(candidate['direction']);normal=np.array([-unit[1],unit[0]])
    half_width=float(np.linalg.norm(points[1]-points[2])/2)
    t=np.linspace(.2,.8,16);center=tip[None,:]+t[:,None]*(base-tip)
    outside=half_width*t+max(2,stroke)
    probes=np.rint(np.r_[center+outside[:,None]*normal,center-outside[:,None]*normal]).astype(int)
    inside=(probes>=0).all(axis=1)&(probes[:,0]<mask.shape[1])&(probes[:,1]<mask.shape[0])
    if not inside.all():return False,{'reason':'native_flanks_clipped'}
    flank=float(np.mean(mask[probes[:,1],probes[:,0]]>0))
    if flank>POLICY['native_flank_ink_max']:return False,{'reason':'native_flanks_not_background','triangle_ink':coverage,'flank_ink':flank}
    for polygon in text_polygons:
        contour=np.asarray(polygon,np.float32)
        if cv2.pointPolygonTest(contour,tuple(tip),False)>=0 or cv2.pointPolygonTest(contour,tuple(points.mean(axis=0)),False)>=0:
            return False,{'reason':'native_head_inside_saved_text'}
    return True,{'reason':'native_filled_triangle_and_clear_flanks','triangle_ink':coverage,'flank_ink':flank}


def native_junction(candidate,mask,stroke,lines,text_polygons):
    """A 2-D junction can survive a clipped/saturated unrolled width profile."""
    tip=np.asarray(candidate['tip_px']);unit=np.asarray(candidate['direction'])
    polygon=np.asarray(candidate['polygon_px']);base=polygon[1:].mean(axis=0)
    length=float(np.linalg.norm(tip-base));hypotheses=[]
    for line in lines:
        if line.get('kind')!='line':continue
        points=np.asarray(line['points_px']);delta=points[1]-points[0]
        extent=float(np.linalg.norm(delta))
        if extent<max(20,2*stroke):continue
        direction=delta/extent
        if abs(direction@unit)>POLICY['maximum_transverse_axis_dot']:continue
        t,s=np.linalg.solve(np.column_stack((unit,-direction)),points[0]-tip)
        if not (-max(3,candidate['uncertainty_px']*POLICY['maximum_native_forward_shift_in_uncertainties'])
                <=-t<=length*POLICY['maximum_native_backward_shift_in_head_lengths'] and -stroke<=s<=extent+stroke):continue
        point=tip+t*unit
        # A fitted segment is a hypothesis. Demand raster ink on both sides
        # beyond the head itself before using its intersection as an apex.
        half_width=float(np.linalg.norm(polygon[1]-polygon[2])/2)
        offsets=np.linspace(half_width+stroke,half_width+3*stroke,8)
        probes=np.rint(point+np.r_[-offsets,offsets][:,None]*direction).astype(int)
        if (probes<1).any() or (probes[:,0]>=mask.shape[1]-1).any() or (probes[:,1]>=mask.shape[0]-1).any():continue
        observed=np.array([np.any(mask[y-1:y+2,x-1:x+2]) for x,y in probes]).reshape(2,-1).mean(axis=1)
        if (observed<POLICY['minimum_native_transverse_flank_support']).any():continue
        result=deepcopy(candidate);result['tip_px']=point.tolist();result['polygon_px'][0]=point.tolist()
        actual=point-base;norm=float(np.linalg.norm(actual))
        if norm<=1:continue
        result['direction']=(actual/norm).tolist()
        valid,check=native_head_check(result,mask,stroke,text_polygons)
        if not valid:continue
        for previous in hypotheses:
            if np.linalg.norm(np.asarray(previous['tip_px'])-point)<=max(2,stroke/2):
                previous['evidence']['native_junction_lines'].append(line['id']);break
        else:
            result['evidence'].update(tip_localization='native_unique_transverse_line_junction',
                native_junction_lines=[line['id']],native_transverse_flank_support=observed.tolist(),native_head_check=check)
            hypotheses.append(result)
    if len(hypotheses)==1:return hypotheses[0]
    candidate['evidence']['native_junction_hypotheses']=len(hypotheses)
    return candidate


def combine(candidates):
    """Merge observations of one physical head, never opposite directions."""
    output=[]
    for value in deepcopy(candidates):
        polygon=np.asarray(value['polygon_px'],np.float32)
        for index,old in enumerate(output):
            if np.dot(value['direction'],old['direction'])<.96:continue
            other=np.asarray(old['polygon_px'],np.float32)
            area,_=cv2.intersectConvexConvex(polygon,other)
            denominator=abs(cv2.contourArea(polygon))+abs(cv2.contourArea(other))-area
            close=np.linalg.norm(np.asarray(value['tip_px'])-old['tip_px'])<=3
            if close or (denominator>0 and area/denominator>=.35):
                quality=lambda c:(c['evidence'].get('tip_localization')=='unique_taper_local_raster_junction',
                                  c['evidence'].get('native_head_check',{}).get('triangle_ink',0))
                if quality(value)>quality(old):output[index]=value
                break
        else:output.append(value)
    for index,value in enumerate(output,1):value['id']=f'path-arrow-{index:05d}'
    return output


def search(gray,*,lines=(),arcs=(),stroke_px=3,include_arcs=True,text_polygons=()):
    require(isinstance(gray,np.ndarray) and gray.ndim==2 and gray.dtype==np.uint8
            and 0<gray.size<=POLICY['maximum_pixels'],'Expected bounded uint8 gray raster')
    require(np.isfinite(stroke_px) and 1<=stroke_px<=24,'Invalid source stroke width')
    for polygon in text_polygons:
        p=np.asarray(polygon)
        require(p.ndim==2 and p.shape[1]==2 and len(p)>=3 and np.isfinite(p).all(),'Invalid saved text polygon')
    for c in lines:
        if c.get('kind')=='line':
            p=np.asarray(c['points_px']);require(p.shape==(2,2) and np.isfinite(p).all() and np.linalg.norm(p[1]-p[0])>0 and c.get('id'),'Invalid supporting line')
    for c in arcs:
        require(c.get('kind')=='arc' and np.asarray(c['center_px']).shape==(2,)
            and np.isfinite([*c['center_px'],c['radius_px'],c['start_angle_deg'],c['sweep_angle_deg']]).all()
            and c['radius_px']>0 and 0<c['sweep_angle_deg']<360 and c.get('id'),'Invalid open-arc proposal')
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    distance=cv2.distanceTransform(255-mask,cv2.DIST_L2,3)
    paths=[path_for(c) for c in lines if c.get('kind')=='line']
    paths.sort(key=lambda p:-p['length'])
    reasons=Counter();decisions=[];limits=[];candidates=[];sampled=0
    if include_arcs:
        if len(arcs)>POLICY['maximum_arc_paths']:limits.append('arc_path_budget')
        paths=[path_for(c) for c in arcs[:POLICY['maximum_arc_paths']]]+paths
    if len(paths)>POLICY['maximum_paths']:limits.append('path_count_budget')
    for path in paths[:POLICY['maximum_paths']]:
        if path['length']>POLICY['maximum_path_length_px']:
            limits.append('path_length_budget');continue
        justified,events=arc_head_evidence(mask,path,stroke_px,distance)
        if not justified:
            reasons['arc_without_revalidated_head_boundary']+=1
            decisions.append({'path_id':path['id'],'kind':path['kind'],'status':'excluded','events':events});continue
        scale=min(1.,POLICY['target_stroke_px']/stroke_px)
        halo=min(POLICY['maximum_head_halo_px'],POLICY['head_halo_in_strokes']*stroke_px)
        cost=(2*PROFILE_RADIUS+1)*(math.ceil((path['length']+2*halo)*scale)+2*MARGIN+1)
        if sampled+cost>POLICY['maximum_sample_pixels']:
            limits.append('sampling_pixel_budget');break
        strip,transform=normalized_strip(gray,path,stroke_px);sampled+=transform['sample_pixels']
        support_line={'id':path['id'],'points_px':[[MARGIN,PROFILE_RADIUS],[MARGIN+transform['length'],PROFILE_RADIUS]]}
        raw=_line_arrows(strip,support_line)
        for proposal in raw:
            localized,method=localize_tip(proposal,strip)
            candidate=native_candidate(localized,path,transform)
            if candidate is None:reasons['degenerate_native_head']+=1;continue
            candidate['evidence']['tip_localization']=method
            if method!='unique_taper_local_raster_junction':
                candidate=native_junction(candidate,mask,stroke_px,lines,text_polygons)
            allowed,check=native_head_check(candidate,mask,stroke_px,text_polygons)
            if not allowed:reasons[check['reason']]+=1;continue
            candidate['evidence'].update(native_head_check=check,
                                          arc_head_boundary_events=events if path['kind']=='arc' else [])
            candidates.append(candidate)
        decisions.append({'path_id':path['id'],'kind':path['kind'],'status':'examined',
                          'normalized_scale':transform['scale'],'strip_pixels':transform['sample_pixels'],
                          'legacy_strip_candidates':len(raw)})
        if len(candidates)>=POLICY['maximum_candidates']:
            limits.append('candidate_budget');break
    candidates=combine(candidates[:POLICY['maximum_candidates']])
    return {'schema_version':'arrow-path-recovery/1.0','policy':POLICY,'candidates':candidates,
        'analysis':{'paths_assigned':len(paths),'paths_examined':sum(r['status']=='examined' for r in decisions),
                    'sample_pixels':sampled,'rejections':dict(reasons),'limits':sorted(set(limits)),'path_decisions':decisions},
        'search_incomplete':bool(limits),'production_admitted':False,'physical_geometry_approved':False,'CAD_promotions':0}
