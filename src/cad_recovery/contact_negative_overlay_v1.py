"""Withdraw proposed CV roles using bound through-path or clipped-line evidence.

Original ink, CAD geometry, known input roles and manual protection are intact.
This stage can correct a previous CV proposal; it never trusts its persistence
as proof that it is material. No expected labels or filenames are inputs.
"""
from copy import deepcopy
import math
import cv2
import numpy as np
from .cad_object_assembly_v5 import assemble
from .local_contour_probes_v1 import probes
from .local_contour_witness_v2 import inventory
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import foreground
from .local_contour_profiles_v2 import measure,junction_distance
from .contour_contact_witness_v3 import disjoint

POLICY={'version':'contact-negative-overlay/1.0','maximum_pixels':4000000,
        'maximum_objects':4096,'maximum_source_points':2000000,
        'maximum_profile_samples':2000000,
        'maximum_local_raster_pixels':64000000,'minimum_clipped_length_in_pens':12.,
        'geometry_modified':False,'known_roles_modified':False,'source_owner_remapped':False}


def analyze(packet,proposed,*,positive,contacts,protected):
    gray=packet['gray'];inventory(packet)
    if gray.size>POLICY['maximum_pixels']:raise ValueError('Bounded negative overlay required')
    if (positive.get('source_sha256')!=sha(gray) or positive.get('source_graph_sha256')!=object_sha(packet['graph'])
            or positive.get('source_owner_sha256')!=sha(packet['source_owner'])
            or positive.get('source_heads_sha256')!=object_sha(packet['heads'])):
        raise ValueError('Unbound negative overlay proof')
    if contacts is not None and (contacts.get('source_sha256')!=sha(gray) or contacts.get('source_heads_sha256')!=object_sha(packet['heads'])
            or contacts.get('source_graph_sha256')!=object_sha(packet['graph'])
            or contacts.get('source_owner_sha256')!=sha(packet['source_owner'])
            or contacts.get('input_mask_sha256')!=sha(positive['mask'])):
        raise ValueError('Unbound negative contact proof')
    if not positive['complete'] or (contacts is not None and not contacts['complete']):
        raise ValueError('Incomplete negative topology proof')
    if proposed.shape!=gray.shape or proposed.dtype!=np.uint8 or not np.isin(proposed,[0,255]).all():
        raise ValueError('Aligned binary CV proposal required')
    arrays=(gray,proposed,packet['roles'],packet['source_owner'],packet['ambiguous_owner_mask'],protected)
    snapshots=[sha(a) for a in arrays];before=object_sha([packet['geometry'],packet['graph'],packet['heads'],contacts])
    audit={'limits':[],'source_points':0,'local_raster_pixels':0};reasons=[]
    blocked=protected.copy()|(packet['roles']==6)
    for h in packet['heads']:
        p=np.asarray(h['polygon_px'],float);m=np.zeros_like(gray);cv2.fillPoly(m,[np.rint(p).astype('int32')],255);blocked|=m>0
    assembled=assemble(gray,packet['geometry'],roles=packet['roles'],protected=blocked)
    objects,search=probes(assembled,packet['geometry'],maximum_segments=POLICY['maximum_objects'])
    if not assembled['search_complete'] or not search['complete'] or len(objects)>POLICY['maximum_objects']:
        audit['limits'].append('negative_geometry_search')
    ink=foreground(gray);dt=cv2.distanceTransform(ink,cv2.DIST_L2,5);source_lookup={c['id']:c for c in packet['geometry']}
    profile_work={'ink':ink,'distance':dt,'samples':0};junctions=junction_distance(packet['geometry'],gray.shape)
    rejected=set(contacts['rejected_object_ids']) if contacts else set()
    targets=[{**o,'negative_reason':'two_source_thick_through_paths'} for o in positive['objects'] if o['id'] in rejected]
    height,width=gray.shape
    for obj in objects if not audit['limits'] else []:
        if obj['kind']!='line':continue
        q=np.asarray(obj['support_points'],int);ends=np.asarray(obj['points_px'],float)
        reach=float(np.quantile(dt[q[:,1],q[:,0]],.95))+math.sqrt(.5)
        clip_points=[]
        for point in q[[0,-1]]:
            if min(point[0],point[1],width-1-point[0],height-1-point[1])>reach:continue
            pad=math.ceil(reach)+1;lo=np.maximum(0,point-pad);hi=np.minimum(gray.shape[::-1],point+pad+1)
            audit['local_raster_pixels']+=int(np.prod(hi-lo))
            if audit['local_raster_pixels']>POLICY['maximum_local_raster_pixels']:
                audit['limits'].append('negative_clip_raster_budget');break
            local=ink[lo[1]:hi[1],lo[0]:hi[0]]
            _,labels=cv2.connectedComponents(local,connectivity=8);site=point-lo;label=int(labels[site[1],site[0]])
            yy,xx=np.nonzero((labels==label)&(labels>0));source=np.c_[xx+lo[0],yy+lo[1]]
            at_edge=(source[:,0]==0)|(source[:,1]==0)|(source[:,0]==width-1)|(source[:,1]==height-1)
            at_edge &= np.linalg.norm(source-point,axis=1)<=reach
            clip_points.extend(source[at_edge].tolist())
        if not clip_points:continue
        pen=max(1.,float(np.quantile(dt[q[:,1],q[:,0]],.25)*2-1))
        if np.linalg.norm(ends[1]-ends[0])<POLICY['minimum_clipped_length_in_pens']*pen:continue
        targets.append({'id':obj['id'],'source_members':obj['source_members'],
                        'observed_pen_px':float(2*np.quantile(dt[q[:,1],q[:,0]],.95)),
                        'points_px':obj['points_px'],
                        'observed_clip_pixels':clip_points,
                        'negative_reason':'source_LINE_clipped_at_input_edge'})
    withdrawn=np.zeros_like(gray)
    material=[]
    for obj in positive['objects']:
        if obj['id'] in rejected or obj.get('witness_only'):continue
        points=[]
        for m in obj['source_members']:
            c=source_lookup[m['candidate_id']];a,b=m['source_index_interval'];origin=c['source_index_interval'][0]
            points.extend(c['support_points'][a-origin:b-origin+1])
        material.append((obj,np.asarray(points,int)))
    for target in targets if not audit['limits'] else []:
        points=[];numbers=set()
        for member in target['source_members']:
            c=source_lookup[member['candidate_id']];a,b=member['source_index_interval'];origin=c['source_index_interval'][0]
            if not origin<=a<=b<=c['source_index_interval'][1]:raise ValueError('Foreign negative source interval')
            points.extend(c['support_points'][a-origin:b-origin+1])
            numbers|={n for n,p in enumerate(packet['graph']['proposals'],1) if p['parent_primitive_id']==c['id']
                      and max(a,p['source_index_interval'][0])<min(b,p['source_index_interval'][1])}
        q=np.asarray(points,int);audit['source_points']+=len(q)
        if audit['source_points']>POLICY['maximum_source_points']:
            audit['limits'].append('negative_source_budget');break
        radius=target['observed_pen_px']/2+math.sqrt(.5);pad=math.ceil(radius)+1
        lo=np.maximum(0,q.min(0)-pad);hi=np.minimum(gray.shape[::-1],q.max(0)+pad+1)
        audit['local_raster_pixels']+=int(np.prod(hi-lo))
        if audit['local_raster_pixels']>POLICY['maximum_local_raster_pixels']:
            audit['limits'].append('negative_raster_budget');break
        seed=np.zeros(tuple((hi-lo)[::-1]),np.uint8);seed[q[:,1]-lo[1],q[:,0]-lo[0]]=255
        distances=cv2.distanceTransform(255-seed,cv2.DIST_L2,5);region=np.s_[lo[1]:hi[1],lo[0]:hi[0]]
        hit=(proposed[region]>0)&np.isin(packet['source_owner'][region],list(numbers))&(distances<=radius)
        retained=0;profile=None
        if target['negative_reason']=='source_LINE_clipped_at_input_edge':
            profile=measure(np.asarray(target['points_px'],float),q,gray,profile_work,junctions,maximum_samples=POLICY['maximum_profile_samples'])
            if profile is not None:
                negative_pen=profile['pen_px']
                for body,support in material:
                    if not disjoint(body['source_members'],target['source_members']):continue
                    inside=((support>=lo)&(support<hi)).all(1)
                    if not inside.any():continue
                    audit['local_raster_pixels']+=int(np.prod(hi-lo))
                    if audit['local_raster_pixels']>POLICY['maximum_local_raster_pixels']:
                        audit['limits'].append('negative_competition_budget');break
                    body_seed=np.zeros_like(seed);s=support[inside]-lo;body_seed[s[:,1],s[:,0]]=255
                    body_distance=cv2.distanceTransform(255-body_seed,cv2.DIST_L2,5)
                    body_pen=body['observed_pen_px']
                    # Both candidates retain their own measured pen. Preserve
                    # an existing proposal only when independent material
                    # support is closer after that normalisation; no owner is
                    # remapped and no new positive pixel is created.
                    keep=hit&(body_distance<=body_pen/2+math.sqrt(.5))&(body_distance/body_pen<distances/negative_pen)
                    retained+=int(keep.sum());hit[keep]=False
            if profile_work['samples']>POLICY['maximum_profile_samples']:audit['limits'].append('negative_profile_budget')
        withdrawn[region][hit]=255
        reasons.append({'object_id':target['id'],'reason':target['negative_reason'],
            'source_members':target['source_members'],'source_owner_numbers':sorted(numbers),'withdrawn_pixels':int(hit.sum())})
        if 'observed_clip_pixels' in target:reasons[-1]['observed_clip_pixels']=target['observed_clip_pixels']
        reasons[-1]['retained_independent_material_pixels']=retained
        reasons[-1]['coverage_radius_px']=radius
        reasons[-1]['shaft_profile']=profile
    if audit['limits']:withdrawn[:]=0
    mask=proposed.copy();mask[withdrawn>0]=0
    if before!=object_sha([packet['geometry'],packet['graph'],packet['heads'],contacts]) or snapshots!=[sha(a) for a in arrays]:
        raise ValueError('Negative overlay changed source observations')
    return {'mask':mask,'withdrawn_mask':withdrawn,'complete':not audit['limits'],'reasons':reasons,
            'source_sha256':sha(gray),'input_mask_sha256':sha(proposed),'audit':audit,'policy':deepcopy(POLICY)}
