"""Reject material-corner witnesses explained by two thick through paths."""
from copy import deepcopy
import math
import cv2
import numpy as np
from .cad_object_assembly_v5 import assemble
from .local_contour_probes_v1 import probes
from .local_contour_profiles_v1 import junction_distance,measure
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import foreground
from .local_contour_witness_v1 import _contact

POLICY={'version':'contour-contact-witness/1.0','maximum_objects':4096,'maximum_checks':250000,
    'maximum_profile_samples':2000000,'direction_deg':8.,'maximum_pen_ratio':1.67,
    'junction_search_radius_in_pens':4.,'maximum_local_raster_pixels':64000000,
    'minimum_return_length_in_pens':6.,'positive_created':False,'geometry_modified':False}


def disjoint(a,b):
    return not any(x['candidate_id']==y['candidate_id'] and
        max(x['source_index_interval'][0],y['source_index_interval'][0])<min(x['source_index_interval'][1],y['source_index_interval'][1])
        for x in a for y in b)


def analyze(packet,positive,*,protected=None):
    gray=packet['gray'];raw=positive['mask'];shape=gray.shape
    if (positive.get('policy',{}).get('version')!='local-contour-witness/1.0'
            or positive.get('source_sha256')!=sha(gray) or positive.get('source_graph_sha256')!=object_sha(packet['graph'])
            or positive.get('source_owner_sha256')!=sha(packet['source_owner'])
            or raw.shape!=shape or raw.dtype!=np.uint8 or not np.isin(raw,[0,255]).all()
            or positive.get('raw_positive_pixels')!=int(np.count_nonzero(raw))):
        raise ValueError('Contact witness is not bound to the original local positive')
    before=object_sha([packet['geometry'],packet['graph'],packet['heads'],{k:v for k,v in positive.items() if k!='mask'}])
    snapshots=[sha(a) for a in (gray,raw,packet['roles'],packet['source_owner'],packet['ambiguous_owner_mask'])]
    blocked=np.zeros(shape,bool) if protected is None else protected.copy()
    if blocked.shape!=shape or blocked.dtype!=bool:raise ValueError('Aligned protection required')
    blocked|=packet['roles']==6
    for head in packet['heads']:
        mask=np.zeros(shape,np.uint8);cv2.fillPoly(mask,[np.rint(head['polygon_px']).astype(np.int32)],255);blocked|=mask>0
    audit={'checks':0,'profile_samples':0,'local_raster_pixels':0,'limits':[]};decisions=[];selected={o['id']:o for o in positive['objects']}
    if not positive['complete']:audit['limits'].append('positive_search_incomplete')
    result=assemble(gray,packet['geometry'],roles=packet['roles'],protected=blocked)
    objects,probe_audit=probes(result,packet['geometry'],maximum_segments=POLICY['maximum_objects'])
    if not result['search_complete'] or not probe_audit['complete'] or len(objects)>POLICY['maximum_objects']:
        audit['limits'].append('contact_source_observations_incomplete')
    ink=foreground(gray);work={'ink':ink,'distance':cv2.distanceTransform(ink,cv2.DIST_L2,5),'samples':0}
    junctions=junction_distance(packet['geometry'],shape);profiles={};accepted=set();rejected=set()
    cosine=math.cos(math.radians(POLICY['direction_deg']))
    for witness in positive['witnesses'] if not audit['limits'] else []:
        center=np.mean(witness['contact']['points_px'],axis=0);opposites=[]
        for identity in witness['branch_object_ids']:
            if identity not in selected:raise ValueError('Foreign positive branch')
            branch=selected[identity];ends=np.asarray(branch['points_px'],float);near=int(np.argmin(np.linalg.norm(ends-center,axis=1)))
            vector=ends[1-near]-ends[near];u=vector/np.linalg.norm(vector);pen=branch['observed_pen_px'];choices=[]
            reach=POLICY['junction_search_radius_in_pens']*pen+2*math.sqrt(.5)
            for obj in objects:
                audit['checks']+=1
                if audit['checks']>POLICY['maximum_checks']:audit['limits'].append('contact_check_budget');break
                if obj['kind']!='line' or not disjoint(branch['source_members'],obj['source_members']):continue
                points=np.asarray(obj['points_px'],float);close=int(np.argmin(np.linalg.norm(points-center,axis=1)))
                if np.linalg.norm(points[close]-center)>reach:continue
                delta=points[1-close]-points[close];length=float(np.linalg.norm(delta))
                if not length or float(delta@u/length)>-cosine:continue
                if obj['id'] not in profiles:
                    profiles[obj['id']]=measure(points,np.asarray(obj['support_points'],int),gray,work,junctions,
                        maximum_samples=POLICY['maximum_profile_samples'])
                profile=profiles[obj['id']]
                if work['samples']>POLICY['maximum_profile_samples']:audit['limits'].append('contact_profile_budget');break
                if profile is None:continue
                width=profile['pen_px']
                if length<POLICY['minimum_return_length_in_pens']*width or not 1/POLICY['maximum_pen_ratio']<=width/pen<=POLICY['maximum_pen_ratio']:continue
                connected=_contact(ink,blocked,np.rint(ends[near]).astype(int),np.rint(points[close]).astype(int),pen,audit)
                if audit['local_raster_pixels']>POLICY['maximum_local_raster_pixels']:audit['limits'].append('contact_raster_budget');break
                if connected is None:continue
                choices.append({'object_id':obj['id'],'source_members':obj['source_members'],'profile':profile})
            opposites.append(choices)
            if audit['limits']:break
        if audit['limits']:break
        crossing=any(disjoint(a['source_members'],b['source_members']) for a in opposites[0] for b in opposites[1])
        ids=witness['branch_object_ids'];(rejected if crossing else accepted).update(ids)
        decisions.append({'branch_object_ids':ids,'contact_px':center.tolist(),'two_thick_through_paths':crossing,
            'opposite_branches':opposites})
    audit['profile_samples']=work['samples'];remove=rejected-accepted;candidate=np.zeros_like(raw)
    source_lookup={c['id']:c for c in packet['geometry']}
    # Reconstruct the original per-object footprint for accepted witnesses only.
    # A coarse interval shared by accepted/rejected chords cannot release all
    # of its pixels merely because one chord remains supported.
    for identity in sorted(accepted) if not audit['limits'] else []:
        obj=selected[identity];parts=[]
        for member in obj['source_members']:
            c=source_lookup[member['candidate_id']];a,b=member['source_index_interval'];origin=c['source_index_interval'][0]
            if not origin<=a<=b<=c['source_index_interval'][1]:raise ValueError('Foreign positive source range')
            parts.extend(c['support_points'][a-origin:b-origin+1])
        q=np.asarray(parts,int);radius=obj['observed_pen_px']/2+math.sqrt(.5);pad=math.ceil(radius)+1
        lo=np.maximum(0,q.min(0)-pad);hi=np.minimum(shape[::-1],q.max(0)+pad+1)
        audit['local_raster_pixels']+=int(np.prod(hi-lo))
        if audit['local_raster_pixels']>POLICY['maximum_local_raster_pixels']:audit['limits'].append('contact_raster_budget');break
        seed=np.zeros(tuple((hi-lo)[::-1]),np.uint8);seed[q[:,1]-lo[1],q[:,0]-lo[0]]=255
        distance=cv2.distanceTransform(255-seed,cv2.DIST_L2,5);region=np.s_[lo[1]:hi[1],lo[0]:hi[0]]
        retained=(raw[region]>0)&np.isin(packet['source_owner'][region],obj['source_owner_numbers'])&(distance<=radius)
        candidate[region][retained]=255
    if audit['limits']:candidate[:]=0
    if before!=object_sha([packet['geometry'],packet['graph'],packet['heads'],{k:v for k,v in positive.items() if k!='mask'}]) or snapshots!=[sha(a) for a in (gray,raw,packet['roles'],packet['source_owner'],packet['ambiguous_owner_mask'])]:
        raise ValueError('Contact witness changed source observations')
    return {'mask':candidate,'complete':not audit['limits'],'decisions':decisions,'audit':audit,
        'rejected_object_ids':sorted(remove),'withdrawn_pixels':int(np.count_nonzero(raw)-np.count_nonzero(candidate)),
        'source_sha256':sha(gray),'source_graph_sha256':object_sha(packet['graph']),
        'source_owner_sha256':sha(packet['source_owner']),'input_mask_sha256':sha(raw),
        'source_intervals_sha256':object_sha([{k:v for k,v in p.items() if k not in ('linear_context','width_px')} for p in packet['graph']['proposals']]),
        'output_mask_sha256':sha(candidate),'policy':deepcopy(POLICY),'source_immutable':True}
