"""Posthoc whole-entity agreement, separate from reviewed local corridor scores.

Reference CAD outside reviewed corridors is diagnostic evidence, never automatic GT.
"""
from collections import Counter
import math

import numpy as np
from ezdxf.path import make_path

from .stroke_curve_control_v1 import distance_to_polyline, maximum_matching

KINDS={'LINE','ARC','CIRCLE','LWPOLYLINE','POLYLINE','ELLIPSE','SPLINE'}
IGNORED_LAYERS={'SOURCE_INK','CV_CANDIDATES','OCR_OBSERVATIONS'}


def project(points,matrix):
    return np.asarray(points,float)@matrix[:2,:2].T+matrix[:2,2]


def parameters(entity,matrix):
    if entity.dxftype() not in ('ARC','CIRCLE'):return {}
    scale=float(np.mean(np.linalg.svd(matrix[:2,:2],compute_uv=False)))
    return dict(center=project([list(entity.dxf.center)[:2]],matrix)[0].tolist(),
        radius=float(entity.dxf.radius*scale),
        sweep_deg=float((entity.dxf.end_angle-entity.dxf.start_angle)%360) if entity.dxftype()=='ARC' else 360.)


def points(entity,matrix):
    scale=float(np.linalg.norm(matrix[:2,:2],ord=2))
    p=project([(v.x,v.y) for v in make_path(entity).flattening(.125/max(scale,1e-9))],matrix)
    samples=[]
    for a,b in zip(p,p[1:]):
        count=max(2,int(math.ceil(float(np.linalg.norm(b-a))*2))+1)
        if count>24000:raise ValueError('QA path sampling limit exceeded')
        samples.append(np.linspace(a,b,count)[:-1])
    if not samples:return p
    result=np.concatenate([*samples,p[-1:]])
    if len(result)>24000:raise ValueError('QA entity sampling limit exceeded')
    return result


def catalog(doc,matrix):
    result=[];skipped=Counter()
    for entity in doc.modelspace():
        if entity.dxftype() not in KINDS or entity.dxf.layer in IGNORED_LAYERS:continue
        try:p=points(entity,matrix)
        except (ValueError,TypeError,NotImplementedError) as exc:
            skipped[type(exc).__name__]+=1;continue
        if len(p)<2:continue
        result.append(dict(handle=entity.dxf.handle,kind=entity.dxftype(),layer=entity.dxf.layer,
            points=p,low=p.min(0),high=p.max(0),parameters=parameters(entity,matrix),entity=entity))
    return result,dict(skipped)


def compare(targets,actual,*,distance=2.,coverage=.8,endpoints=3.):
    """A whole match needs both directions and its own endpoints, one handle only."""
    rows=[];edges=[]
    for target in targets:
        expected=target['points'];options={};fragments=[];union=np.full(len(expected),np.inf)
        low=expected.min(0)-distance;high=expected.max(0)+distance
        for candidate in actual:
            if (candidate['high']<low).any() or (candidate['low']>high).any():continue
            p=candidate['points'];forward=distance_to_polyline(expected,p);reverse=distance_to_polyline(p,expected)
            union=np.minimum(union,forward);a=float(np.mean(forward<=distance));b=float(np.mean(reverse<=distance))
            endpoint_error=None
            if target['kind'] not in ('CIRCLE','ELLIPSE'):
                endpoint_error=min(float(np.linalg.norm(p[[0,-1]]-expected[[0,-1]],axis=1).max()),
                                   float(np.linalg.norm(p[[-1,0]]-expected[[0,-1]],axis=1).max()))
            circular=True;centre_error=radius_error=sweep_error=None
            expected_parameters=target.get('parameters',{});actual_parameters=candidate.get('parameters',{})
            if expected_parameters and actual_parameters:
                centre_error=float(np.linalg.norm(np.array(expected_parameters['center'])-actual_parameters['center']))
                radius_error=abs(expected_parameters['radius']-actual_parameters['radius'])
                sweep_error=abs(expected_parameters['sweep_deg']-actual_parameters['sweep_deg'])
                circular=centre_error<=distance and radius_error<=distance and sweep_error<=5.
            whole=min(a,b)>=coverage and (endpoint_error is None or endpoint_error<=endpoints) and circular
            detail=dict(handle=candidate['handle'],type=candidate['kind'],layer=candidate['layer'],
                reference_coverage=a,candidate_coverage=b,endpoint_error_px=endpoint_error,
                same_type=candidate['kind']==target['kind'],
                centre_error_px=centre_error,radius_error_px=radius_error,sweep_error_deg=sweep_error,
                forward_p95_px=float(np.percentile(forward,95)),reverse_p95_px=float(np.percentile(reverse,95)))
            if whole:options[candidate['handle']]=detail
            elif b>=coverage and a>0:fragments.append(detail)
        edges.append(sorted(options,key=lambda key:(not options[key]['same_type'],options[key]['forward_p95_px'],key)))
        rows.append(dict(id=target['id'],reference_kind=target['kind'],reference_handle=target.get('reference_handle'),
            role=target.get('role'),whole_options=options,fragments=fragments,
            union_reference_coverage=float(np.mean(union<=distance)),match=None,
            reference_full_object_source_truth_approved=False))
    for index,handle in maximum_matching(edges).items():rows[index]['match']=rows[index]['whole_options'][handle]
    return dict(targets=len(rows),whole_geometry_matches=sum(r['match'] is not None for r in rows),
        whole_same_type_matches=sum(bool(r['match'] and r['match']['same_type']) for r in rows),
        fragmented_without_whole_match=sum(bool(not r['match'] and r['fragments']) for r in rows),
        full_union_without_whole_object=sum(not r['match'] and r['union_reference_coverage']>=coverage for r in rows),
        duplicate_whole_options=sum(max(0,len(r['whole_options'])-1) for r in rows),rows=rows,
        metrics=dict(distance_px=distance,bidirectional_coverage=coverage,endpoints_px=endpoints,circular_parameters_px=distance,sweep_deg=5.),
        source_CAD_agreement_diagnostic_only=True,full_page_precision=None)
