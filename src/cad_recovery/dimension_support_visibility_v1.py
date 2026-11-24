"""Separate visible-region coverage from coverage of an entire source segment."""
import numpy as np
from .dimension_support_metrics_v1 import segment_coverage

def clip_segment(a,b,box):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float);d=b-a
    if not np.isfinite(np.r_[a,b,box]).all() or np.linalg.norm(d)==0:
        raise ValueError('Finite nonzero reference segment required')
    lo,hi=0.,1.
    for p,q in [(-d[0],a[0]-box[0]),(d[0],box[2]-a[0]),(-d[1],a[1]-box[1]),(d[1],box[3]-a[1])]:
        if p==0:
            if q<0:return None
        elif p<0:lo=max(lo,float(q/p))
        else:hi=min(hi,float(q/p))
    if lo>=hi:return None
    return a+lo*d,a+hi*d,hi-lo

def visible_coverage(a,b,source_box,geometry,tolerance):
    origin=np.array(source_box[:2]);clipped=clip_segment(a,b,source_box)
    full=segment_coverage(np.array(a)-origin,np.array(b)-origin,geometry,tolerance)
    if clipped is None:return {'source_segment_fraction_visible':0.,'visible_segment_coverage':None,'full_reference_coverage':full}
    first,second,fraction=clipped
    return {'source_segment_fraction_visible':fraction,
            'visible_segment_coverage':segment_coverage(first-origin,second-origin,geometry,tolerance),
            'full_reference_coverage':full}
