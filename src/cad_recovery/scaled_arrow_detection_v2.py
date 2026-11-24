"""Clip a disposable sampling segment to pixel centers; preserve source CV.

This corrects a frame-contract failure in v1. All normalization, taper,
confidence and resource coefficients remain unchanged.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .scaled_arrow_detection_v1 import detect as original_detect,POLICY as NORMALIZATION
from . import dimension_association_v3 as v3

POLICY={'version':'scaled-arrow-detection/2.0','normalization_policy':NORMALIZATION,
        'boundary_contract':'Liang-Barsky clipping for sampling only; original primitive and ID retained',
        'threshold_changes':False,'production_admitted':False}


def clip_segment(points,size):
    a,b=np.array(points,float);delta=b-a
    if not np.isfinite(delta).all():raise ValueError('Unrepresentable geometry extent')
    lo,hi=0.,1.
    for i in range(2):
        if abs(delta[i])<1e-12:
            if a[i]<0 or a[i]>size[i]-1:return None
            continue
        start,end=sorted(((0-a[i])/delta[i],(size[i]-1-a[i])/delta[i]))
        lo=max(lo,start);hi=min(hi,end)
        if hi<=lo:return None
    # Bound rounding noise at the clipping plane without changing orientation
    # or creating a contour edge along the image border.
    return np.clip(np.array([a+lo*delta,a+hi*delta]),[0,0],np.array(size)-1).tolist()


def detect(source_image,*,geometry,text_polygons=()):
    source_image=Path(source_image);common={k:geometry.get(k) for k in ('source_sha256','processed_size_px')}
    v3._validate(source_image,geometry,{**common,'lines':[],'notations':[]},{**common,'candidates':[]})
    view=deepcopy(geometry);candidates=[];clips=[]
    for g in view['candidates']:
        if g['kind']!='line':candidates.append(g);continue
        clipped=clip_segment(g['points_px'],view['processed_size_px'])
        if clipped is None:
            clips.append({'geometry_id':g['id'],'status':'no_source_intersection','original_points_px':g['points_px']})
            continue
        if not np.array_equal(np.array(clipped),np.array(g['points_px'])):
            clips.append({'geometry_id':g['id'],'status':'sampling_segment_clipped',
                          'original_points_px':deepcopy(g['points_px']),'sampling_points_px':clipped})
        g['points_px']=clipped;candidates.append(g)
    view['candidates']=candidates
    report=original_detect(source_image,geometry=view,text_polygons=text_polygons)
    report.update(algorithm_version=POLICY['version'],boundary_policy=deepcopy(POLICY),sampling_geometry_clips=clips)
    return report
