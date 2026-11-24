"""Perpendicular source-ink widths outside observed junction footprints."""
from collections import defaultdict
import math
import cv2
import numpy as np


def junction_distance(geometry,shape):
    incidence=defaultdict(set)
    for c in geometry:
        points=np.asarray(c['support_points'],int)
        if len(points)<2:continue
        for p in (points[0],points[-1]):incidence[tuple(p)].add(c['id'])
    mask=np.zeros(shape,np.uint8)
    for (x,y),ids in incidence.items():
        if len(ids)>=3:mask[y,x]=255
    return cv2.distanceTransform(255-mask,cv2.DIST_L2,5) if mask.any() else np.full(shape,np.inf,np.float32)


def measure(ends,points,gray,work,junctions,*,maximum_samples=2000000):
    delta=ends[1]-ends[0];length=float(np.linalg.norm(delta))
    if length==0:return None
    normal=np.array([-delta[1],delta[0]])/length;q=np.asarray(points,int)
    radius=max(1.,float(np.quantile(work['distance'][q[:,1],q[:,0]],.95)))
    extent=2*radius+math.sqrt(.5);positions=np.arange(-extent,extent+.25,.25)
    widths=[];sites=[];excluded=0
    for fraction in np.linspace(.25,.75,9):
        center=ends[0]+fraction*delta;origin_pixel=np.rint(center).astype(int)
        if junctions[origin_pixel[1],origin_pixel[0]]<=2*radius+math.sqrt(.5):excluded+=1;continue
        sampled=np.rint(center+positions[:,None]*normal).astype(int);work['samples']+=len(sampled)
        if work['samples']>maximum_samples:return None
        if (sampled<0).any() or (sampled[:,0]>=gray.shape[1]).any() or (sampled[:,1]>=gray.shape[0]).any():continue
        values=work['ink'][sampled[:,1],sampled[:,0]]>0;origin=int(np.argmin(abs(positions)))
        if not values[origin]:continue
        lo=origin;hi=origin
        while lo>0 and values[lo-1]:lo-=1
        while hi+1<len(values) and values[hi+1]:hi+=1
        if lo==0 or hi==len(values)-1:continue
        widths.append((hi-lo+1)*.25);sites.append(float(fraction))
    if len(widths)<5:return None
    return {'pen_px':float(np.quantile(widths,.25)),'sections_px':widths,'endpoints_px':ends.tolist(),
        'normal':normal.tolist(),'site_fractions':sites,'excluded_observed_junction_sites':excluded,
        'minimum_valid_sites':5,'width_measurement':'source_ink_sections_outside_observed_junctions'}
