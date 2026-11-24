"""Keep stable legacy profiles; resolve rejected staircase phase at source centres."""
import math
import numpy as np
from .local_contour_profiles_v1 import measure as previous,junction_distance


def measure(ends,points,gray,work,junctions,*,maximum_samples=2000000):
    result=previous(ends,points,gray,work,junctions,maximum_samples=maximum_samples)
    if result is not None:
        low,high=np.quantile(result['sections_px'],[.25,.95])
        if low>=1. and high<=2*low:return result
    delta=ends[1]-ends[0];length=float(np.linalg.norm(delta))
    if length==0:return None
    normal=np.array([-delta[1],delta[0]])/length;q=np.asarray(points,int)
    radius=max(1.,float(np.quantile(work['distance'][q[:,1],q[:,0]],.95)))
    extent=2*radius+math.sqrt(.5);positions=np.arange(-extent,extent+.25,.25)
    widths=[];sites=[];centres=[];excluded=0
    for fraction in np.linspace(.25,.75,9):
        target=ends[0]+fraction*delta;center=q[np.argmin(np.linalg.norm(q-target,axis=1))]
        if junctions[center[1],center[0]]<=2*radius+math.sqrt(.5):excluded+=1;continue
        sampled=np.rint(center+positions[:,None]*normal).astype(int);work['samples']+=len(sampled)
        if work['samples']>maximum_samples:return None
        if (sampled<0).any() or (sampled[:,0]>=gray.shape[1]).any() or (sampled[:,1]>=gray.shape[0]).any():continue
        values=work['ink'][sampled[:,1],sampled[:,0]]>0;origin=int(np.argmin(abs(positions)))
        if not values[origin]:continue
        lo=origin;hi=origin
        while lo>0 and values[lo-1]:lo-=1
        while hi+1<len(values) and values[hi+1]:hi+=1
        if lo==0 or hi==len(values)-1:continue
        widths.append((hi-lo+1)*.25);sites.append(float(fraction));centres.append(center.tolist())
    if len(widths)<5:return None
    return {'pen_px':float(np.quantile(widths,.25)),'sections_px':widths,'endpoints_px':ends.tolist(),
        'normal':normal.tolist(),'site_fractions':sites,'source_centres_px':centres,
        'excluded_observed_junction_sites':excluded,'minimum_valid_sites':5,
        'width_measurement':'source_centred_sections_after_unstable_legacy_profile'}
