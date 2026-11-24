"""Conservatively abstain on cropped strips without transverse part evidence.

The frozen first-stage context filter remains independently replayable. This stage
only vetoes its additional hypotheses, retaining source pixels and inherited work.
"""
import math
import numpy as np
from .open_reference_context_v1 import Context, filter_context as first_stage

POLICY={'version':'open-reference-context/2.0','parallel_angle_deg':8.,
        'minimum_transverse_angle_deg':20.,'minimum_transverse_extent_body_pens':2.,
        'border_tolerance_body_pens':1.5,'maximum_track_checks':250000,
        'changes_pixel_roles':False,'physical_geometry_approved':False,'production_admitted':False}


def part_structure(context, chain, component, threshold, audit):
    shaft=next(t for t in context.previous['tracks'] if t['id']==chain['shaft_track'])
    p=np.asarray(shaft['points'],float);axis=(p[1]-p[0])/np.linalg.norm(p[1]-p[0]);normal=np.array([-axis[1],axis[0]])
    pen=min(r['body_width'] for r in chain['local_reference_evidence']['references'])
    tolerance=POLICY['border_tolerance_body_pens']*pen+2**.5/2
    clipped=[];transverse=[]
    for track,_,direction,_ in context.wide_tracks(component,threshold):
        audit['track_checks']+=1
        if audit['track_checks']>POLICY['maximum_track_checks']:
            audit['limits'].append('track_checks');break
        # All ends come from actual observed source paths; fitted endpoints may
        # extend past a break and cannot be used as clipping evidence.
        points=np.asarray([context.lookup[key]['points_px'][end] for key in track['interval_ids']
                           if key in component and key in context.valid for end in (0,-1)])
        if not len(points):continue
        extent=float(np.ptp(points@normal));cosine=abs(float(direction@axis))
        if cosine<=math.cos(math.radians(POLICY['minimum_transverse_angle_deg'])) and extent>=POLICY['minimum_transverse_extent_body_pens']*pen:
            transverse.append({'track_id':track['id'],'source_interval_ids':list(track['interval_ids']),
                               'normal_extent_px':extent})
        if cosine<math.cos(math.radians(POLICY['parallel_angle_deg'])):continue
        along=points@axis;ends=points[[int(np.argmin(along)),int(np.argmax(along))]]
        distances=np.minimum.reduce([ends[:,0],ends[:,1],context.gray.shape[1]-1-ends[:,0],context.gray.shape[0]-1-ends[:,1]])
        if np.all(distances<=tolerance):
            clipped.append({'track_id':track['id'],'source_interval_ids':list(track['interval_ids']),
                            'observed_ends_px':ends.tolist(),'border_distances_px':distances.tolist()})
    return {'accepted':not clipped or bool(transverse),'clipped_reference_tracks':clipped,
            'transverse_part_witnesses':transverse,'observed_body_pen_px':pen,'border_tolerance_px':tolerance,
            'minimum_transverse_extent_px':POLICY['minimum_transverse_extent_body_pens']*pen}


def filter_context(gray,previous,inherited,basis_packet):
    stage1=first_stage(gray,previous,inherited,basis_packet)
    candidates=stage1['open_reference_chains'];audit={'track_checks':0,'limits':[]};decisions=[];accepted=[]
    context=Context(gray,previous) if candidates else None
    for index,chain in enumerate(candidates):
        component,threshold=context.component(chain)
        evidence=part_structure(context,chain,component,threshold,audit)
        admitted=evidence['accepted'] and not audit['limits'] and not context.audit['limits']
        if admitted:accepted.append(chain)
        decisions.append({'stage1_candidate_index':index,'head_ids':list(chain['head_ids']),
            'accepted':admitted,'reason':None if admitted else 'clipped_reference_strip_without_transverse_part_witness',
            'part_structure':evidence})
        if audit['limits'] or context.audit['limits']:break
    incomplete=stage1.get('search_incomplete',False) or bool(audit['limits']) or bool(context and context.audit['limits'])
    if incomplete:accepted=[]
    return {**stage1,'open_reference_chains':accepted,'dimension_chains':list(inherited['dimension_chains'])+accepted,
            'stage1_open_reference_chains':candidates,'part_structure_decisions':decisions,
            'part_structure_audit':{**audit,'component':context.audit if context else None},
            'search_incomplete':incomplete,'production_admitted':False,'physical_geometry_approved':False}
