"""Attach explicit source-reviewed primitive/relationship supervision to fixed nodes.

Source truth and its candidate coverage remain separate; rejected/missing objects
and contacts stay in the inventory. Never mine negatives from unreviewed pairs.
"""
from collections import Counter
import math

import cv2
import numpy as np

from .role_relation_preparation_v1 import validate_review


def samples(obj):
    g=obj['geometry_px']; kind=obj['primitive']
    if kind in ('LINE','CENTER_LINE'):
        runs=g.get('visible_runs',[[g['start'],g['end']]])
        arrays=[]
        for a,b in runs:
            a,b=np.asarray(a,float),np.asarray(b,float)
            arrays.append(np.linspace(a,b,max(2,math.ceil(np.linalg.norm(b-a)*2)+1)))
        return np.concatenate(arrays)
    if kind in ('CIRCLE','ARC'):
        a,b=(0,360) if kind=='CIRCLE' else (g['start_deg'],g['end_deg'])
        if not 0<b-a<=360 or not g['radius']>0:raise ValueError('Invalid reviewed circular span')
        angles=np.deg2rad(np.linspace(a,b,max(8,math.ceil(g['radius']*math.radians(b-a)*2))))
        return np.asarray(g['center'])+g['radius']*np.c_[np.cos(angles),np.sin(angles)]
    raise ValueError('Unsupported reviewed primitive')


def point_distance(points,obj):
    p=np.asarray(points,float);g=obj['geometry_px'];kind=obj['primitive']
    if kind in ('LINE','CENTER_LINE'):
        a,b=np.asarray(g['start'],float),np.asarray(g['end'],float);d=b-a
        if d@d<=0:raise ValueError('Degenerate reviewed straight primitive')
        t=np.clip((p-a)@d/(d@d),0,1)
        return np.linalg.norm(p-(a+t[:,None]*d),axis=1)
    if kind in ('CIRCLE','ARC'):
        q=p-np.asarray(g['center']); distance=np.abs(np.linalg.norm(q,axis=1)-g['radius'])
        if kind=='ARC':
            span=g['end_deg']-g['start_deg'];angle=(np.rad2deg(np.arctan2(q[:,1],q[:,0]))-g['start_deg'])%360
            endpoints=np.asarray(g['center'])+g['radius']*np.c_[np.cos(np.deg2rad([g['start_deg'],g['end_deg']])),np.sin(np.deg2rad([g['start_deg'],g['end_deg']]))]
            outside=angle>span
            distance[outside]=np.minimum(np.linalg.norm(p[outside]-endpoints[0],axis=1),np.linalg.norm(p[outside]-endpoints[1],axis=1))
        return distance
    raise ValueError('Unsupported reviewed primitive')


def support(gray,obj,policy):
    pts=samples(obj);h,w=gray.shape
    inside=bool(np.all((pts>3)&(pts<np.array([w-4,h-4]))))
    q=np.round(pts).astype(int);q=np.clip(q,[0,0],[w-1,h-1])
    dt=cv2.distanceTransform((gray>=240).astype(np.uint8),cv2.DIST_L2,5)
    distances=dt[q[:,1],q[:,0]]
    fraction=float(np.mean(distances<=policy['distance_px']))
    return dict(source_support_fraction=fraction,source_support_p95_px=float(np.quantile(distances,.95)),
        inside_crop_with_margin=inside,source_QA_pass=inside and fraction>=policy['minimum_fraction'],
        sample_count=len(q),primitive_truth_automatically_derived=False)


def assemble_patch(reading,gray,graph,rows,targets,policy):
    ids=graph['node_ids']; points=[np.asarray(r['points_px'],float) for r in rows]
    if len(rows)!=len(ids) or len(targets)!=len(ids):raise ValueError('Node identity mismatch')
    raw=reading['objects'];by_id={o['id']:o for o in raw}
    if len(by_id)!=len(raw):raise ValueError('Duplicate source primitive ID')
    candidates=[[] for _ in ids];inventories=[]
    for obj in raw:
        if obj.get('physical_geometry_approved') is not False or obj.get('coordinate_regression_admitted') is not False:
            raise ValueError('Physical coordinate regression not admitted')
        qa=support(gray,obj,policy); possible=[]; conflicts=[];clipped=[]
        if qa['source_QA_pass']:
            for i,p in enumerate(points):
                if np.max(point_distance(p,obj))>policy['distance_px']:continue
                if rows[i]['target_reason']=='clipped_source_path':clipped.append(ids[i]);continue
                if obj['role_id']!=255 and int(targets[i]) not in (255,obj['role_id']):conflicts.append(ids[i]);continue
                possible.append(i);candidates[i].append(obj['id'])
        inventories.append(dict(**obj,source_QA=qa,candidate_indices=possible,
            role_conflict_nodes=conflicts,boundary_clipped_nodes=clipped))
    ambiguous={ids[i]:value for i,value in enumerate(candidates) if len(value)>1}
    objects=[]
    for record in inventories:
        matched=[i for i in record.pop('candidate_indices') if len(candidates[i])==1]
        node_ids=[ids[i] for i in matched];record['nodes']=node_ids
        mask=np.ones(gray.shape,np.uint8)
        for i in matched:
            q=points[i].astype(int);mask[q[:,1],q[:,0]]=0
        q=np.round(samples(record)).astype(int);q=np.clip(q,[0,0],[gray.shape[1]-1,gray.shape[0]-1])
        dt=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        coverage=float(np.mean(dt[q[:,1],q[:,0]]<=policy['distance_px'])) if matched else 0.
        record.update(candidate_source_coverage=coverage,
            candidate_whole_coverage_at_least_0_80=bool(matched and coverage>=.8),
            semantic_group_admitted=record['source_QA']['source_QA_pass'] and bool(node_ids),
            coordinate_regression_admitted=False)
        if record['semantic_group_admitted']:
            objects.append({k:record[k] for k in ['id','primitive','nodes','complete_visible_primitive','physical_geometry_approved']})
    edges={tuple(sorted((a,b))) for a,b,kind in graph['directed_edges']}
    membership={n:obj['id'] for obj in objects for n in obj['nodes']}
    relations=[];used=set()
    for a,b in sorted(edges):
        if ids[a] in membership and membership.get(ids[a])==membership.get(ids[b]):
            relations.append(dict(nodes=[ids[a],ids[b]],relation='same_primitive',source_evidence_reviewed=True,
                object_id=membership[ids[a]],derivation='explicit reviewed object identity; source adjacency alone is insufficient'))
            used.add((a,b))
    by_inventory={o['id']:o for o in inventories};contacts=[]
    id_index={n:i for i,n in enumerate(ids)}
    for event in reading['contacts']:
        a,b=event['objects']
        if a not in by_id or b not in by_id or a==b:raise ValueError('Invalid reviewed contact objects')
        if event['relation']=='dimension_extension' and [by_id[a]['role_id'],by_id[b]['role_id']]!=[2,3]:
            raise ValueError('Source DIM/EXT roles must be explicit')
        left=by_inventory[a]['nodes'];right=by_inventory[b]['nodes'];pairs=[]
        location=np.asarray(event['point_px'],float)
        for ln in left:
            for rn in right:
                i,j=id_index[ln],id_index[rn];key=tuple(sorted((i,j)))
                if key not in edges or key in used:continue
                cost=float(np.min(np.linalg.norm(points[i]-location,axis=1))+np.min(np.linalg.norm(points[j]-location,axis=1)))
                pairs.append((cost,i,j))
        selected=min(pairs) if pairs else None
        result=dict(**event,source_evidence_reviewed=True,candidate_pair_available=selected is not None,
                    missing_counts_as_source_candidate_miss=selected is None)
        if selected:
            cost,i,j=selected;used.add(tuple(sorted((i,j))))
            row=dict(nodes=[ids[i],ids[j]],relation=event['relation'],source_evidence_reviewed=True,
                     physical_contact_id=event['id'])
            if event['relation']=='dimension_extension':row.update(dimension_endpoint=ids[i],extension_endpoint=ids[j])
            relations.append(row);result.update(selected_nodes=[ids[i],ids[j]],nearest_contact_sum_distance_px=cost)
        contacts.append(result)
    review=dict(patch_id=reading['patch_id'],family=reading['family'],source=reading['source'],
        image=reading['image'],status='complete',authority='assistant_source_review',human_verified=False,
        objects=objects,relations=relations,unknown_regions=reading['unknown_regions'],
        partial_supervision=True,physical_geometry_approved=False)
    counts=validate_review(review,ids)
    return dict(review=review,counts=counts,source_objects=inventories,source_contacts=contacts,
        ambiguous_node_membership=ambiguous,
        source_objects_missing_candidates=sum(not o['nodes'] for o in inventories if o['source_QA']['source_QA_pass']),
        source_contacts_missing_candidates=sum(not e['candidate_pair_available'] for e in contacts),
        source_QA_rejected_objects=sum(not o['source_QA']['source_QA_pass'] for o in inventories),
        source_object_types=dict(Counter(o['primitive'] for o in inventories)),
        absent_pairs_are_negative=False)
