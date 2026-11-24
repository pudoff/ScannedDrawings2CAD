"""Bounded per-line width observations on unchanged binary ink; no role inference."""
import math
import cv2
import numpy as np


def line_width(points, foreground, distance=None):
    points=np.asarray(points,float)
    if (points.shape!=(2,2) or not np.isfinite(points).all() or foreground.dtype!=np.uint8
            or foreground.ndim!=2 or foreground.size>4194304):
        raise ValueError('Bounded finite pixel line and uint8 mask required')
    delta=points[1]-points[0]; length=float(np.linalg.norm(delta))
    if length<=0: raise ValueError('Zero length line')
    distance=cv2.distanceTransform(foreground,cv2.DIST_L2,5) if distance is None else distance
    if distance.shape!=foreground.shape or not np.isfinite(distance).all():
        raise ValueError('Invalid distance field')
    normal=np.array([-delta[1],delta[0]])/length
    samples=points[0]+np.linspace(.12,.88,min(128,max(3,int(length/8))))[:,None]*delta
    diameters=[]; widths=[]; offset_ambiguities=0; supported=0
    offsets=np.arange(-32,32.01,.5)
    for sample in samples:
        xy=np.rint(sample).astype(int); x,y=xy
        if not (0<=x<foreground.shape[1] and 0<=y<foreground.shape[0]) or not foreground[y,x]:
            continue
        supported+=1
        diameter=2*float(distance[y,x])
        if diameter<=64:diameters.append(diameter)
        probes=np.rint(sample+offsets[:,None]*normal).astype(int)
        valid=(probes[:,0]>=0)&(probes[:,0]<foreground.shape[1])&(probes[:,1]>=0)&(probes[:,1]<foreground.shape[0])
        hit=np.zeros(len(probes),bool);hit[valid]=foreground[probes[valid,1],probes[valid,0]]>0
        center=len(offsets)//2
        if not hit[center]:continue
        left=right=center
        while left>0 and hit[left-1]:left-=1
        while right<len(hit)-1 and hit[right+1]:right+=1
        if left==0 or right==len(hit)-1:
            offset_ambiguities+=1;continue
        widths.append(float(offsets[right]-offsets[left]+.5))
    coverage=supported/len(samples)
    q=np.percentile(widths,[10,50,90]).tolist() if widths else [None]*3
    return {'method':'normal_connected_ink_span_and_DT_diameter/1.0', 'coordinate_space':'input_pixel_centers',
        'sample_count':len(samples),'supported_samples':supported,'coverage':coverage,
        'normal_span_p10_p50_p90_px':q,'dt_diameter_median_px':float(np.median(diameters)) if diameters else None,
        'width_px':q[1], 'uncertainty_px':None if not widths else max(.5,(q[2]-q[0])/2),
        'heterogeneous':bool(widths and q[2]>max(1.,q[0])*1.75),
        'clipped_normal_samples':offset_ambiguities,'line_role':'unknown','physical_width_mm':None,
        'width_missing_is_not_fallback_constant':True}
