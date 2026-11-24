"""Measure annotation shafts outside bound heads and proposed material ink."""
import math
import numpy as np
from .local_contour_profiles_v1 import measure as previous


def measure(ends, support, gray, work, junctions, *, excluded, maximum_samples=2000000):
    old=previous(ends,support,gray,work,junctions,maximum_samples=maximum_samples)
    if work['samples']>maximum_samples:return None
    delta=ends[1]-ends[0];length=float(np.linalg.norm(delta))
    if not length:return None
    if old is not None:
        low,high=np.quantile(old['sections_px'],[.25,.95])
        sites=np.rint(ends[0]+np.asarray(old['site_fractions'])[:,None]*delta).astype(int)
        if low>=1 and high<=2*low and not excluded[sites[:,1],sites[:,0]].any():
            return old
    q=np.asarray(support,int);normal=np.array([-delta[1],delta[0]])/length
    sizes=[];sites=[];fractions=[];used=set()
    for fraction in np.linspace(.1,.9,33):
        p=q[np.argmin(np.linalg.norm(q-(ends[0]+fraction*delta),axis=1))]
        if tuple(p) in used or excluded[p[1],p[0]]:continue
        used.add(tuple(p));radius=max(1.,float(work['distance'][p[1],p[0]]))
        if junctions[p[1],p[0]]<=2*radius+math.sqrt(.5):continue
        extent=2*radius+math.sqrt(.5);positions=np.arange(-extent,extent+.25,.25)
        pixels=np.rint(p+positions[:,None]*normal).astype(int);work['samples']+=len(pixels)
        if work['samples']>maximum_samples:return None
        if (pixels<0).any() or (pixels[:,0]>=gray.shape[1]).any() or (pixels[:,1]>=gray.shape[0]).any():continue
        values=work['ink'][pixels[:,1],pixels[:,0]]>0;zero=int(np.argmin(abs(positions)))
        if not values[zero]:continue
        low=high=zero
        while low>0 and values[low-1]:low-=1
        while high+1<len(values) and values[high+1]:high+=1
        if low==0 or high==len(values)-1:continue
        run=pixels[low:high+1]
        if excluded[run[:,1],run[:,0]].any():continue
        sizes.append(float((high-low+1)*.25));sites.append(p.tolist());fractions.append(float(fraction))
    if len(sizes)<5:return None
    low,high=map(float,np.quantile(sizes,[.25,.95]))
    span=float(np.ptp((np.asarray(sites)-ends[0])@(delta/length)))
    if low<1 or high>2*low or span<6*low:return None
    return {'pen_px':low,'sections_px':sizes,'endpoints_px':ends.tolist(),'normal':normal.tolist(),
        'source_centres_px':sites,'site_fractions':fractions,'minimum_valid_sites':5,
        'observed_shaft_span_px':span,'excluded_heads_and_material_proposals':True,
        'width_measurement':'source_centred_clean_annotation_shaft'}
