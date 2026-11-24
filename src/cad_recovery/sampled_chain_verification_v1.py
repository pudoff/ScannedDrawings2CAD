"""c.10 carrier/face checks with a bounded triangle coverage estimator.

Geometry search is unchanged. This version avoids allocating a head-sized
bitmap and retains a strict .85 coverage threshold for the estimator.
"""
from copy import deepcopy
import numpy as np
from .dimension_support_v2 import propose as geometry_propose
from .dimension_support_v1 import PixelSupport,_demote,v3
from .triangle_sampling_v1 import fraction,POLICY as TRIANGLE_POLICY

POLICY={'version':'sampled-chain-verification/1.0','minimum_face_ink':.85,
        'minimum_carrier_ink':.85,'minimum_head_ink':.65,'triangle_policy':TRIANGLE_POLICY,
        'production_admitted':False,'physical_geometry_approved':False}


def propose(source_image,*,geometry,ocr,arrows,baseline,arm='all'):
    report=geometry_propose(source_image,geometry=geometry,ocr=ocr,arrows=arrows,baseline=baseline,arm=arm)
    report['algorithm_version']=POLICY['version'];report['sampled_verification_policy']=deepcopy(POLICY)
    pixels=PixelSupport(source_image);geo={g['id']:g for g in geometry['candidates']}
    heads={a['id']:a for a in arrows['candidates']}
    faces={f['id']:f for f in (report.get('contour_topology') or {}).get('faces',[])}
    cache={};head_cache={};events=[];complete=True
    def segment(points):
        key=tuple(tuple(p) for p in points)
        if key not in cache:cache[key]=pixels.segment(*np.array(points,dtype=float))
        return cache[key]
    for row in report['associations']:
        if row['status']!='proposed':continue
        fractions={'face':[],'lines':{},'arrows':{}};valid=True
        try:
            for fid in row['evidence'].get('contour_face_ids',[]):
                for edge in faces[fid]['boundaries']:
                    value=segment(edge['points_px']);fractions['face'].append(value)
                    valid &= value>=POLICY['minimum_face_ink']
            for gid in sorted(set(row['dimension_line_candidate_ids']+row['extension_line_candidate_ids'])):
                if geo[gid]['kind']!='line':raise ValueError('Chain carrier must be a line')
                value=segment(geo[gid]['points_px']);fractions['lines'][gid]=value
                valid &= value>=POLICY['minimum_carrier_ink']
            for aid in row['arrow_candidate_ids']:
                if aid not in head_cache:head_cache[aid]=fraction(pixels,heads[aid])
                value=head_cache[aid];fractions['arrows'][aid]=value
                valid &= value>=POLICY['minimum_head_ink']
        except v3._SearchLimit:
            complete=False;valid=False;fractions['budget_exhausted']=True
        row['evidence']['observed_chain_pixels']=fractions
        if not valid:_demote(row,'sampled_chain_pixel_support_missing_or_incomplete')
        events.append({'association_id':row['id'],'supported':bool(valid),'fractions':fractions})
    if not complete:
        report['complete']=False
        for row in report['associations']:_demote(row,'sampled_pixel_verification_incomplete')
    report['pixel_verification']={'complete':complete,'samples':pixels.samples,'events':events,
        'triangle_sampling':deepcopy(TRIANGLE_POLICY)}
    return report
