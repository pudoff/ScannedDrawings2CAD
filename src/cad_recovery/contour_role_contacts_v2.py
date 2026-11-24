"""Retain independently supported PART outside a transverse annotation core.

Direct negative owners and every frame/grid/uncertified negative keep the
original exclusion. Only a proximity-only loss can be reconsidered.
"""
from copy import deepcopy
import math
import cv2
import numpy as np
from .contour_role_competition_v2 import guard as previous
from .contour_role_competition_v1 import guard as hard_guard
from .local_contour_profiles_v1 import measure,junction_distance
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import foreground

ANNOTATION={'dimension_line','extension_line','leader_line','axis_construction'}
POLICY={'version':'contour-role-contacts/2.0','maximum_profile_samples':2000000,'minimum_crossing_deg':20.,
    'core_clearance':'observed_q95_perpendicular_half_width_plus_sqrt_half_pixel',
    'direct_negative_preserved':True,'frame_grid_and_uncertified_veto_preserved':True,
    'positive_created':False,'geometry_modified':False}


def guard(gray,graph,owner,ambiguous,proposed,context,*,positive,contacts):
    if (contacts.get('policy',{}).get('version')!='contour-contact-witness/2.0'
            or positive.get('source_sha256')!=sha(gray) or positive.get('source_owner_sha256')!=sha(owner)
            or contacts.get('source_sha256')!=sha(gray) or contacts.get('source_owner_sha256')!=sha(owner)
            or contacts.get('source_graph_sha256')!=positive.get('source_graph_sha256')
            or contacts.get('input_mask_sha256')!=sha(positive['mask']) or contacts.get('output_mask_sha256')!=sha(proposed)
            or contacts.get('source_intervals_sha256')!=object_sha([{k:v for k,v in p.items() if k not in ('linear_context','width_px')} for p in graph['proposals']])):
        raise ValueError('Unbound shared-contact evidence')
    result=previous(gray,graph,owner,ambiguous,proposed,context)
    audit={'limits':[],'measured_tracks':0,'profile_samples':0,'restored_pixels':0}
    output=result['mask'].copy();before=object_sha([graph,context]);snapshot=sha(proposed)
    if result['complete'] and contacts['complete'] and positive['complete']:
        items=graph['proposals'];numbers={p['id']:n for n,p in enumerate(items,1)};tracks={t['id']:t for t in graph['tracks']}
        negatives=set(result['negative_source_owner_numbers']);soft={};hard=set(negatives)
        ink=foreground(gray);work={'ink':ink,'distance':cv2.distanceTransform(ink,cv2.DIST_L2,5),'samples':0}
        geometry=[{'id':p['id'],'support_points':p['points_px']} for p in items];junctions=junction_distance(geometry,gray.shape)
        for line in context['context']['lines']:
            hints=set(line['alternatives']);negative=hints&set(result['policy']['negative_roles'])
            if not negative or negative-ANNOTATION or 'part_contour' in hints:continue
            track=tracks[line['id']];ns={numbers[i] for i in track['interval_ids']}
            if ns&set(context['frame_conflict_owner_numbers']):continue
            points=np.asarray(track['points'],float);support=np.concatenate([np.asarray(items[n-1]['points_px'],int) for n in ns])
            profile=measure(points,support,gray,work,junctions,maximum_samples=POLICY['maximum_profile_samples'])
            if work['samples']>POLICY['maximum_profile_samples']:audit['limits'].append('contact_profile_budget');break
            if profile is None:continue
            delta=points[1]-points[0];u=delta/np.linalg.norm(delta);radius=float(np.quantile(profile['sections_px'],.95)/2)+math.sqrt(.5)
            audit['measured_tracks']+=1
            for n in ns:soft[n]={'u':u,'radius':radius}
        hard-=set(soft)
        # All other sources retain the full old clearance. In particular, a
        # nearby annotation cannot conceal a slightly more distant frame/grid.
        hard_result=hard_guard(gray,graph,owner,ambiguous,proposed,{'complete':context['complete'],
            'frame_conflict_owner_numbers':sorted(hard),'other_conflict_owner_numbers':[],'context':{'lines':[]}})
        seeds=np.zeros_like(gray);source=np.zeros(gray.shape,np.uint16)
        for n in soft:
            q=np.asarray(items[n-1]['points_px'],int);seeds[q[:,1],q[:,0]]=255;source[q[:,1],q[:,0]]=n
        if seeds.any() and not audit['limits']:
            distances,labels=cv2.distanceTransformWithLabels(255-seeds,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
            lookup=np.zeros(int(labels.max())+1,np.uint16);where=seeds>0;lookup[labels[where]]=source[where];nearest=lookup[labels]
            eligible=(proposed>0)&(output==0)&(hard_result['mask']>0)&~np.isin(owner,list(negatives))
            cosine=math.cos(math.radians(POLICY['minimum_crossing_deg']))
            for obj in positive['objects']:
                if obj['id'] in contacts['rejected_object_ids']:continue
                ends=np.asarray(obj['points_px'],float);u=(ends[1]-ends[0])/np.linalg.norm(ends[1]-ends[0])
                selected=eligible&np.isin(owner,obj['source_owner_numbers'])
                for n in np.unique(nearest[selected]):
                    info=soft.get(int(n))
                    if info is not None and abs(float(u@info['u']))<=cosine:
                        output[selected&(nearest==n)&(distances>info['radius'])]=255
        audit['profile_samples']=work['samples'];audit['restored_pixels']=int(np.count_nonzero(output)-np.count_nonzero(result['mask']))
    complete=bool(result['complete'] and contacts['complete'] and not audit['limits'])
    if not complete:output[:]=0
    if before!=object_sha([graph,context]) or snapshot!=sha(proposed):raise ValueError('Contact resolution changed observations')
    return {**result,'mask':output,'complete':complete,'previous_policy':result['policy'],'policy':deepcopy(POLICY),
        'previous_withdrawn_pixels':result['withdrawn_pixels'],
        'withdrawn_pixels':int(np.count_nonzero(proposed)-np.count_nonzero(output)),'contact_resolution':audit}
