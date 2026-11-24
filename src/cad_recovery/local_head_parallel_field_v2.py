"""Independent local repeat strokes cannot include the candidate own shaft."""
import math

import cv2
import numpy as np


def source_widths(roi, a, b, stroke, audit, maximum_pixels):
    delta = b-a; length = np.linalg.norm(delta); u = delta/length
    normal = np.array([-u[1],u[0]])
    radius = min(32, math.ceil(max(2.,1.5*stroke))+2)
    across = np.arange(-radius,radius+1)
    points = a+np.linspace(.15,.85,12)[:,None,None]*delta+across[None,:,None]*normal
    q = np.rint(points).astype(int); cost = int(np.prod(q.shape[:2]))
    if audit['context_pixels']+cost > maximum_pixels: return None
    audit['context_pixels'] += cost
    inside = (q>=0).all(2)&(q[:,:,0]<roi.shape[1])&(q[:,:,1]<roi.shape[0])
    ink = np.zeros(inside.shape,bool); ink[inside] = roi[q[:,:,1][inside],q[:,:,0][inside]]>0
    widths = []
    for row in ink:
        if not row[radius]: continue
        left=right=radius
        while left>0 and row[left-1]: left-=1
        while right<len(row)-1 and row[right+1]: right+=1
        if left==0 or right==len(row)-1: continue
        widths.append(right-left+1)
    return float(np.median(widths)) if len(widths)>=6 else float('inf')


def check(mask, head, audit, *, maximum_pixels=4000000):
    polygon=np.asarray(head['polygon_px'],float); base=polygon[1:].mean(0)
    length=float(np.linalg.norm(polygon[0]-base));stroke=float(head['evidence']['local_shaft_width_px'])
    margin=min(96.,2*length)
    lo=np.maximum([0,0],np.floor(polygon.min(0)-margin)).astype(int)
    hi=np.minimum(mask.shape[::-1],np.ceil(polygon.max(0)+margin+1)).astype(int)
    cost=int(np.prod(hi-lo))
    if audit['context_pixels']+cost>maximum_pixels:
        return {'available':False,'incomplete':True,'reason':'context_pixel_budget'}
    audit['context_pixels']+=cost
    original=mask[lo[1]:hi[1],lo[0]:hi[0]]; roi=original.copy()
    cv2.fillConvexPoly(roi,np.rint(polygon-lo).astype(np.int32),0)
    minimum=max(8,round(.6*length))
    raw=cv2.HoughLinesP(roi,1,np.pi/180,threshold=max(6,minimum//2),minLineLength=minimum,maxLineGap=1)
    if raw is None:return {'available':True,'incomplete':False,'reason':'no_parallel_source_field'}
    if len(raw)>256:return {'available':False,'incomplete':True,'reason':'context_segment_budget'}
    lines=[]
    for a,b in raw.reshape(-1,2,2).astype(float):
        delta=b-a;extent=float(np.linalg.norm(delta))
        if extent<minimum:continue
        observed_width=source_widths(original,a,b,stroke,audit,maximum_pixels)
        if observed_width is None:
            return {'available':False,'incomplete':True,'reason':'context_pixel_budget'}
        if observed_width>max(2.,1.5*stroke):continue
        lines.append({'a':a+lo,'b':b+lo,'u':delta/extent,'length':extent,'width':observed_width})
    audit['context_segments']+=len(lines)
    independent=[];head_unit=np.asarray(head['direction'],float)
    head_normal=np.array([-head_unit[1],head_unit[0]])
    for line in lines:
        midpoint=(line['a']+line['b'])/2
        longitudinal=sorted([float((line['a']-base)@head_unit),float((line['b']-base)@head_unit)])
        own_shaft=(abs(line['u']@head_unit)>=math.cos(math.radians(8))
            and abs(float((midpoint-base)@head_normal))<=max(2.,stroke)
            and longitudinal[0]<0 and longitudinal[1]>=-max(40.,2*length))
        if own_shaft:
            audit['context_own_shaft_segments_excluded']=audit.get('context_own_shaft_segments_excluded',0)+1
        else:independent.append(line)
    lines=independent
    center=polygon.mean(0)
    for seed in sorted(lines,key=lambda x:-x['length']):
        u=seed['u'];n=np.array([-u[1],u[0]]);station=float(center@u)
        # Removing the head polygon can occlude a stripe. Only its projected
        # footprint, plus stroke uncertainty, may extend the source intervals.
        # Distant unrelated fragments cannot enlarge this allowance.
        tolerance=stroke+.5*float(np.ptp(polygon@u))
        members=[]
        for line in lines:
            if abs(u@line['u'])<math.cos(math.radians(8)):continue
            ends=sorted([float(line['a']@u),float(line['b']@u)])
            if not ends[0]-tolerance<=station<=ends[1]+tolerance:continue
            members.append((float(((line['a']+line['b'])/2)@n),line))
        unique=[]
        for offset,line in sorted(members,key=lambda x:x[0]):
            if unique and offset-unique[-1][0]<max(2.,1.5*stroke):continue
            unique.append((offset,line))
        # Check local triples; a distant outer frame must not extend a hatch
        # field, and a genuine regular subfield must survive extra frame edges.
        for start in range(len(unique)-2):
            group=unique[start:start+3];offsets=np.array([x[0] for x in group]);gaps=np.diff(offsets)
            # One stripe may be hidden by the removed head polygon. Two source
            # spacings may therefore differ by a factor of two, with at most
            # one observed stroke of raster/localization uncertainty.
            if gaps.max()>2*gaps.min()+stroke:continue
            transverse=float(center@n)
            if not offsets[0]-stroke<=transverse<=offsets[-1]+stroke:continue
            points=np.array([p for _,line in group for p in (line['a'],line['b'])],np.float32)
            hull=cv2.convexHull(points)
            low=max(min(float(line['a']@u),float(line['b']@u))-tolerance for _,line in group)
            high=min(max(float(line['a']@u),float(line['b']@u))+tolerance for _,line in group)
            if low>high:continue
            field=np.array([low*u+offsets[0]*n,high*u+offsets[0]*n,
                            high*u+offsets[-1]*n,low*u+offsets[-1]*n],np.float32)
            direction=-np.asarray(head['direction'],float)
            along=np.arange(0,min(96.,max(40.,2*length))+.5,.5)
            tail=base+along[:,None]*direction;q=np.rint(tail).astype(int)
            inside=(q>=lo+math.ceil(stroke)).all(1)&(q<hi-math.ceil(stroke)).all(1)
            supported=np.zeros(len(q),bool);supported[inside]=mask[q[:,1][inside],q[:,0][inside]]>0
            # Continuing along a hatch stripe past a short Hough segment is not
            # independent shaft evidence. A protected tail must actually leave
            # the repeated transverse band, not just its inferred end cap.
            transverse_tail=tail@n
            outside=(transverse_tail<offsets[0]-stroke)|(transverse_tail>offsets[-1]+stroke)
            run=outside&supported&np.logical_and.accumulate(supported)&inside
            changes=np.diff(np.r_[False,run,False].astype(int))
            protected=any((b-a)*.5>=max(6.,2*stroke) for a,b in
                          zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
            evidence={'parallel_offsets':3,'spacing_px':gaps.tolist(),'head_station_px':station,
                'co_supported_at_head_station':True,'longitudinal_tolerance_px':tolerance,
                'station_support_includes_head_occlusion_allowance':True,
                'source_segments':[[x['a'].tolist(),x['b'].tolist()] for _,x in group],
                'source_widths_px':[x['width'] for _,x in group],'hull_px':hull.reshape(-1,2).tolist(),
                'local_field_px':field.tolist(),'common_longitudinal_interval_px':[low,high],
                'field_construction':'source_interval_intersection_with_explicit_head_occlusion_allowance',
                'tail_protection_requires_transverse_exit':True,
                'one_head_occluded_stripe_allowed':True,
                'own_tail_outside_parallel_field':protected}
            if not protected:
                return {'available':False,'incomplete':False,'reason':'ambiguous_local_regular_parallel_field','support':evidence}
            # Other overlapping local triples must also permit the observation.
    return {'available':True,'incomplete':False,'reason':'no_unresolved_local_regular_field'}
