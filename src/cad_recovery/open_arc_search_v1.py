"""Bounded contour-window proposals and native-pixel validation of open arcs."""
from collections import Counter
import math

import cv2
import numpy as np

from .family_catalog_v2 import require
from .geometry_detection import _fit_circle, _observed_angles

POLICY = {
    'version':'contour-window-open-arcs/1.0', 'maximum_pixels':25000000,
    'maximum_contours':256, 'maximum_contour_points':250000,
    'windows_px':[256,128,64,32], 'maximum_window_fits':4096,
    'maximum_seed_count':512, 'maximum_refinements':96, 'maximum_candidates':96,
    'minimum_radius_px':10, 'minimum_seed_sweep_deg':12, 'maximum_seed_sweep_deg':160,
    'maximum_seed_rmse_px':1.5, 'minimum_sagitta_px':.75,
    'maximum_sagitta_to_chord':.25, 'angular_samples':720,
    'maximum_radial_band_px':32, 'thin_width_in_strokes':1.8,
    'maximum_junction_gap_in_strokes':3,
    'minimum_arc_sweep_deg':25, 'maximum_arc_sweep_deg':300,
    'minimum_arc_length_px':36, 'minimum_source_support':.95,
    'minimum_tangent_length_in_radii':.75, 'text_containment_fraction':.8,
    'minimum_endpoint_width_jump_ratio':1.8,
    'endpoint_boundary_probes':180,
    'continuous_circle_inferred':False,'physical_geometry_approved':False,
    'production_admitted':False,'manual_guides_used_for_search':False,
}


def _runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def circular_runs(mask, maximum_gap):
    mask=np.asarray(mask,bool).copy();size=len(mask)
    if not mask.any():return []
    if mask.all():return [(0,size)]
    # Three copies expose gaps crossing 0 degrees without treating the cut as an endpoint.
    for begin,end in _runs(~np.tile(mask,3)):
        if begin>0 and end<3*size and end-begin<=maximum_gap:
            mask[np.arange(begin,end)%size]=True
    if mask.all():return [(0,size)]
    cut=int(np.flatnonzero(~mask)[0]);rolled=np.roll(mask,-cut)
    return [((begin+cut)%size,end-begin) for begin,end in _runs(rolled)]


def contour_seeds(mask):
    contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    contours=sorted(contours,key=len,reverse=True)
    reasons=Counter();windows=[];points_examined=0;limited=[]
    for identity,contour in enumerate(contours[:POLICY['maximum_contours']]):
        points=contour.reshape(-1,2).astype(float)
        if points_examined+len(points)>POLICY['maximum_contour_points']:
            limited.append('contour_point_budget');break
        points_examined+=len(points)
        for width in POLICY['windows_px']:
            if len(points)<width:continue
            for begin in range(0,len(points)-width+1,max(8,width//4)):
                piece=points[begin:begin+width]
                chord=piece[-1]-piece[0];length=float(np.linalg.norm(chord))
                if length<12:continue
                midpoint=piece[len(piece)//2]
                sagitta=abs(float(chord[0]*(midpoint[1]-piece[0,1])-chord[1]*(midpoint[0]-piece[0,0])))/length
                if not POLICY['minimum_sagitta_px']<=sagitta<=POLICY['maximum_sagitta_to_chord']*length:
                    reasons['straight_or_sharp_window']+=1;continue
                windows.append((identity,begin,width,piece))
    if len(contours)>POLICY['maximum_contours']:limited.append('contour_count_budget')
    windows.sort(key=lambda row:(-row[2],row[0],row[1]))
    seeds=[];examined=0
    for identity,begin,width,points in windows[:POLICY['maximum_window_fits']]:
        examined+=1;fitted=_fit_circle(points)
        if fitted is None:reasons['degenerate_fit']+=1;continue
        center,radius,residual=fitted
        if not POLICY['minimum_radius_px']<=radius<=max(mask.shape)*2:
            reasons['radius']+=1;continue
        _,sweep=_observed_angles(points,center)
        rmse=float(np.sqrt(np.mean(residual**2)))
        if not POLICY['minimum_seed_sweep_deg']<=math.degrees(sweep)<=POLICY['maximum_seed_sweep_deg'] or rmse>POLICY['maximum_seed_rmse_px']:
            reasons['window_fit_or_sweep']+=1;continue
        if seeds:
            old_centers=np.asarray([s['center'] for s in seeds]);old_radii=np.asarray([s['radius'] for s in seeds])
            close=(np.linalg.norm(old_centers-center,axis=1)<=max(3,radius*.05)) & (np.abs(old_radii-radius)<=max(2,radius*.05))
            if close.any():reasons['duplicate_seed']+=1;continue
        if len(seeds)>=POLICY['maximum_seed_count']:
            limited.append('seed_budget');break
        seeds.append({'center':center.tolist(),'radius':radius,'window_points':width,
                      'contour_id':identity,'window_start':begin,'seed_rmse_px':rmse})
    if len(windows)>POLICY['maximum_window_fits']:limited.append('window_fit_budget')
    return seeds,{'contours_total':len(contours),'contour_points_examined':points_examined,
        'window_proposals':len(windows),'window_fits':examined,'seed_count':len(seeds),
        'seed_rejections':dict(reasons),'limits':limited}


def radial_profile(mask,center,radius,stroke):
    count=POLICY['angular_samples']
    angles=np.arange(count)*2*np.pi/count
    unit=np.c_[np.cos(angles),np.sin(angles)]
    band=min(POLICY['maximum_radial_band_px'],max(8,math.ceil(stroke*2),math.ceil(radius*.03)))
    offsets=np.arange(-band,band+1)
    points=np.asarray(center)+(radius+offsets[:,None])[:,:,None]*unit[None]
    strip=cv2.remap(mask,points[:,:,0].astype(np.float32),points[:,:,1].astype(np.float32),
                    cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0
    order=np.argsort(np.abs(offsets),kind='stable')
    selected=order[np.argmax(strip[order],axis=0)];columns=np.arange(count)
    rows=np.arange(len(offsets))[:,None]
    left=np.maximum.accumulate(np.where(~strip,rows,-1),axis=0)[selected,columns]+1
    right=np.minimum.accumulate(np.where(~strip,rows,len(offsets))[::-1],axis=0)[::-1][selected,columns]-1
    widths=right-left+1
    radii=radius+(offsets[np.clip(left,0,len(offsets)-1)]+offsets[np.clip(right,0,len(offsets)-1)])/2
    supported=strip.any(axis=0)&(left>0)&(right<len(offsets)-1)&(widths<=max(6,stroke*POLICY['thin_width_in_strokes']))
    observed=np.asarray(center)+radii[:,None]*unit
    maximum_gap=max(1,int(stroke*POLICY['maximum_junction_gap_in_strokes']/radius/(2*np.pi/count)))
    runs=circular_runs(supported,maximum_gap)
    runs=[(start,length) for start,length in runs if POLICY['minimum_arc_sweep_deg']<=length*360/count<=POLICY['maximum_arc_sweep_deg']
          and radius*length*2*np.pi/count>=POLICY['minimum_arc_length_px']]
    return {'angles':angles,'observed':observed,'supported':supported,'widths':widths,
            'truncated':(left==0)|(right==len(offsets)-1),'has_ink':strip.any(axis=0),
            'sample_arc_step_px':radius*2*np.pi/count,
            'runs':runs,'band':band,'closed_or_nearly_closed':bool(supported.mean()>.9)}


def endpoint_event(mask,point,stroke,profile,index,outward):
    count=POLICY['angular_samples']
    steps=max(1,min(8,math.ceil(stroke/profile['sample_arc_step_px'])))
    inside=(index-outward*np.arange(1,steps+1))%count
    baseline=float(np.median(profile['widths'][inside]))
    for probe in (index+outward*np.arange(1,steps+1))%count:
        if (profile['has_ink'][probe] and not profile['truncated'][probe]
                and profile['widths'][probe]>=POLICY['minimum_endpoint_width_jump_ratio']*max(1,baseline)
                and profile['widths'][probe]-baseline>=stroke):
            return 'abrupt_thick_head_boundary'
    # A real free end meets a surrounding disk boundary once. A continuing
    # S-curve meets it twice; falling outside a radial band is not an endpoint.
    radius=max(6,2*stroke);angles=np.arange(POLICY['endpoint_boundary_probes'])*2*np.pi/POLICY['endpoint_boundary_probes']
    pixels=np.rint(np.asarray(point)+radius*np.c_[np.cos(angles),np.sin(angles)]).astype(int)
    valid=(pixels>=0).all(axis=1)&(pixels[:,0]<mask.shape[1])&(pixels[:,1]<mask.shape[0])
    if not valid.all():return None
    boundary=mask[pixels[:,1],pixels[:,0]]>0
    if len(circular_runs(boundary,1))==1:return 'observed_free_stroke_end'
    return None


def tangent_endpoints(center,radius,start,sweep,lines,stroke,band):
    original=[start,start+sweep];result=list(original);evidence=[None,None]
    maximum_angle=1.5*math.sqrt(2*radius*band)/radius
    for line in lines:
        if line.get('kind')!='line':continue
        points=np.asarray(line['points_px'],float);delta=points[1]-points[0];length=float(np.linalg.norm(delta))
        if length<max(40,radius*POLICY['minimum_tangent_length_in_radii']):continue
        unit=delta/length;projection=float((np.asarray(center)-points[0])@unit)
        foot=points[0]+projection*unit;normal=foot-center;distance=float(np.linalg.norm(normal))
        if abs(distance-radius)>max(2,stroke) or not -stroke<=projection<=length+stroke:continue
        theta=float(np.arctan2(normal[1],normal[0]))
        for endpoint in (0,1):
            angle=theta+round((original[endpoint]-theta)/(2*np.pi))*2*np.pi
            error=abs(angle-original[endpoint])
            if error<=maximum_angle and (evidence[endpoint] is None or error<evidence[endpoint]['angle_error_rad']):
                result[endpoint]=angle;evidence[endpoint]={'geometry_id':line['id'],'angle_error_rad':error,
                    'distance_to_center_px':distance,'method':'observed_straight_tangent'}
    return result[0],result[1]-result[0],evidence


def refine_seed(mask,seed,stroke,lines,distance,text_polygons):
    center=np.asarray(seed['center'],float);radius=float(seed['radius'])
    for _ in range(3):
        profile=radial_profile(mask,center,radius,stroke)
        if profile['closed_or_nearly_closed']:return None,'closed_ring_not_open_arc'
        if not profile['runs']:return None,'no_continuous_thin_arc'
        start,count=max(profile['runs'],key=lambda r:r[1])
        indices=(start+np.arange(count))%POLICY['angular_samples']
        inner=indices[int(count*.2):max(int(count*.8),int(count*.2)+3)]
        points=profile['observed'][inner[profile['supported'][inner]]]
        if len(points)<12:return None,'insufficient_central_arc_points'
        fitted=_fit_circle(points)
        if fitted is None:return None,'central_fit_degenerate'
        new_center,new_radius,residual=fitted
        if (not POLICY['minimum_radius_px']<=new_radius<=max(mask.shape)*2
                or np.linalg.norm(new_center-center)>max(20,radius*.3) or abs(new_radius-radius)>max(20,radius*.3)):
            return None,'unstable_central_fit'
        center,radius=new_center,new_radius
        if np.sqrt(np.mean(residual**2))>max(1.5,stroke*.35):return None,'central_radial_residual'
    profile=radial_profile(mask,center,radius,stroke)
    if profile['closed_or_nearly_closed'] or not profile['runs']:return None,'missing_open_run_after_fit'
    first,length=max(profile['runs'],key=lambda r:r[1]);step=2*np.pi/POLICY['angular_samples']
    start,sweep,tangents=tangent_endpoints(center,radius,first*step,(length-1)*step,lines,stroke,profile['band'])
    if not POLICY['minimum_arc_sweep_deg']<=math.degrees(sweep)<=POLICY['maximum_arc_sweep_deg']:return None,'trimmed_arc_sweep'
    angles=np.linspace(start,start+sweep,max(24,math.ceil(radius*sweep/2)))
    points=center+radius*np.c_[np.cos(angles),np.sin(angles)]
    pixels=np.rint(points).astype(int)
    inside=(pixels>=0).all(axis=1)&(pixels[:,0]<mask.shape[1])&(pixels[:,1]<mask.shape[0])
    if not inside.all():return None,'arc_clipped_by_image'
    methods=[]
    for endpoint,index,outward in [(0,first,-1),(1,(first+length-1)%POLICY['angular_samples'],1)]:
        method='straight_tangent' if tangents[endpoint] else endpoint_event(
            mask,points[0 if endpoint==0 else -1],stroke,profile,index,outward)
        if method is None:return None,'unsupported_curve_endpoint'
        methods.append(method)
    support=float(np.mean(distance[pixels[:,1],pixels[:,0]]<=max(1.5,stroke*.4)))
    if support<POLICY['minimum_source_support']:return None,'source_ink_support'
    for polygon in text_polygons:
        contour=np.asarray(polygon,np.float32)
        if np.mean([cv2.pointPolygonTest(contour,tuple(p),False)>=0 for p in points])>=POLICY['text_containment_fraction']:
            return None,'mostly_inside_saved_OCR_polygon'
    return {'kind':'arc','center_px':center.tolist(),'radius_px':float(radius),
        'start_angle_deg':float(math.degrees(start)%360),'sweep_angle_deg':float(math.degrees(sweep)),
        'classification':'unclassified_open_arc_candidate','uncertainty_px':max(1.5,stroke/2),
        'evidence':{'method':POLICY['version'],'source_ink_support':support,'seed':seed,
            'endpoint_methods':methods,
            'tangent_supports':tangents,'continuous_circle_inferred':False}},'proposed_open_arc'


def search(gray,*,stroke_px,lines=(),text_polygons=()):
    require(isinstance(gray,np.ndarray) and gray.ndim==2 and gray.dtype==np.uint8
            and 0<gray.size<=POLICY['maximum_pixels'],'Expected bounded uint8 gray raster')
    require(np.isfinite(stroke_px) and 1<=stroke_px<=24,'Invalid source stroke width')
    for line in lines:
        if line.get('kind')=='line':
            require(np.asarray(line['points_px']).shape==(2,2) and np.isfinite(line['points_px']).all()
                    and np.linalg.norm(np.diff(line['points_px'],axis=0))>0 and line.get('id'),'Invalid tangent line')
    for polygon in text_polygons:
        require(np.asarray(polygon).ndim==2 and np.asarray(polygon).shape[1]==2
                and len(polygon)>=3 and np.isfinite(polygon).all(),'Invalid OCR proposal polygon')
    threshold,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    seeds,stats=contour_seeds(mask)
    ranked=[]
    for seed in seeds:
        profile=radial_profile(mask,np.asarray(seed['center']),seed['radius'],stroke_px)
        if not profile['closed_or_nearly_closed'] and profile['runs']:
            longest=max(length for _,length in profile['runs'])
            ranked.append((longest*seed['radius'],seed))
    ranked.sort(key=lambda r:-r[0]);candidates=[];reasons=Counter()
    distance=cv2.distanceTransform(255-mask,cv2.DIST_L2,3)
    for _,seed in ranked[:POLICY['maximum_refinements']]:
        curve,reason=refine_seed(mask,seed,stroke_px,lines,distance,text_polygons)
        if curve is None:reasons[reason]+=1;continue
        duplicate=False
        for old in candidates:
            if (np.linalg.norm(np.asarray(curve['center_px'])-old['center_px'])<=max(3,stroke_px)
                    and abs(curve['radius_px']-old['radius_px'])<=max(3,stroke_px)
                    and abs(curve['sweep_angle_deg']-old['sweep_angle_deg'])<10
                    and abs((curve['start_angle_deg']-old['start_angle_deg']+180)%360-180)<10):
                duplicate=True;break
        if duplicate:reasons['duplicate_arc']+=1;continue
        curve['id']=f'open-arc-{len(candidates)+1:05d}';candidates.append(curve)
        if len(candidates)>=POLICY['maximum_candidates']:stats['limits'].append('candidate_budget');break
    if len(ranked)>POLICY['maximum_refinements']:stats['limits'].append('refinement_budget')
    stats.update(screened_seeds=len(seeds),eligible_seeds=len(ranked),
                 refinements=min(len(ranked),POLICY['maximum_refinements']),refinement_rejections=dict(reasons))
    return {'schema_version':'open-arc-search/1.0','policy':POLICY,'source_size_px':[gray.shape[1],gray.shape[0]],
        'otsu_threshold':float(threshold),'source_stroke_width_px':float(stroke_px),
        'candidates':candidates,'analysis':stats,'search_incomplete':bool(stats['limits']),
        'production_admitted':False,'physical_geometry_approved':False,'CAD_promotions':0}
