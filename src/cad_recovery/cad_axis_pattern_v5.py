"""Bounded exhaustive axis search with an exact necessary-condition filter."""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from .cad_axis_pattern_v4 import POLICY as PREVIOUS_POLICY, _edges, _measure, _partition
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha


POLICY = {**PREVIOUS_POLICY, 'version': 'cad-axis-pattern/5.0',
          'broadphase': 'centroid_normal_distance_and_seed_direction',
          'broadphase_pair_ceiling': PREVIOUS_POLICY['maximum_candidates'] * PREVIOUS_POLICY['maximum_support_candidates']}


class CollinearSupports:
    """Each rejection is necessary under the unchanged all-points test."""

    def __init__(self, supports):
        self.supports = supports
        self.centers = np.asarray([c['center'] for c in supports], float).reshape(-1, 2)
        self.directions = np.asarray([c['u'] for c in supports], float).reshape(-1, 2)
        self.has_direction = np.array([c['length'] >= 2 * POLICY['max_error_px'] for c in supports], bool)
        self.indices = np.array([c['index'] for c in supports], int)

    def query(self, origin, unit, used, audit):
        audit['broadphase_pairs'] += len(self.supports)
        if audit['broadphase_pairs'] > POLICY['broadphase_pair_ceiling']:
            audit.update(complete=False, reason='broadphase_budget')
            return []
        normal = np.array([-unit[1], unit[0]])
        # If every point is inside the strip, their mean is inside it too.
        # Padding only makes the prefilter conservative at a floating-point boundary.
        rounding_slack = 64 * np.finfo(float).eps * max(1., np.max(abs(self.centers), initial=0.), np.max(abs(origin)))
        close = np.abs((self.centers - origin) @ normal) <= POLICY['max_error_px'] + rounding_slack
        aligned = np.abs(self.directions @ unit) >= math.cos(math.radians(POLICY['angle_deg'])) - rounding_slack
        possible = close & (~self.has_direction | aligned) & ~np.isin(self.indices, list(used))
        return [self.supports[i] for i in np.flatnonzero(possible)]


def apply(result,gray,roles,geometry,heads=(),*,protected=None,text_protection=None):
    if result.get('source_sha256')!=sha(gray) or roles.shape!=gray.shape or roles.dtype!=np.uint8:
        raise ValueError('Axis assembly belongs to another source or role frame')
    for mask in (protected,text_protection):
        if mask is not None and (mask.shape!=gray.shape or mask.dtype!=bool):raise ValueError('Aligned boolean axis protection required')
    result=deepcopy(result);before=_edges(result['objects']);objects=result['objects'];lookup={c['id']:c for c in geometry}
    upstream_complete = result['search_complete']
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    dt=cv2.distanceTransform(ink,cv2.DIST_L2,5);support=cv2.dilate(ink,np.ones((3,3),np.uint8));candidates=[]
    for i,obj in enumerate(objects):
        p=np.asarray(obj['support_points'],int)
        if obj['kind'] not in ('line','polyline') or len(p)<3 or obj.get('closed'):continue
        known=set(map(int,roles[p[:,1],p[:,0]]))&set(range(1,10))
        if known-{5}:continue
        center=p.mean(0);_,_,vectors=np.linalg.svd(p-center,full_matrices=False);u=vectors[0];t=(p-center)@u
        pen=max(1.,float(np.median(dt[p[:,1],p[:,0]])*2-1))
        errors=np.linalg.norm(p-center-t[:,None]*u,axis=1)
        if np.ptp(t)<max(1.,pen/2) or errors.max()>POLICY['max_error_px'] or np.percentile(errors,95)>POLICY['p95_error_px']:continue
        candidates.append({'index':i,'p':p,'center':center,'u':u,'length':float(np.ptp(t)),'pen':pen})
    supports=candidates
    # A tiny dot does not have an independently reliable direction. It may
    # support a measured long-stroke seed, but must not consume the seed budget.
    candidates=[c for c in supports if c['length']>=2*POLICY['max_error_px']]
    audit={'policy':deepcopy(POLICY),'candidate_count':len(candidates),'support_candidate_count':len(supports),
        'short_undirected_support_count':len(supports)-len(candidates), 'input_assembly_complete':upstream_complete,
        'accepted':[],'pair_checks':0,'broadphase_pairs':0,'evaluated_seeds':0,'complete':True}
    if not upstream_complete:
        audit.update(complete=False, reason='input_assembly_incomplete')
        result['axis_pattern'] = audit
        return result
    if len(candidates)>POLICY['maximum_candidates'] or len(supports)>POLICY['maximum_support_candidates']:
        audit.update(complete=False,reason='candidate_budget');result['axis_pattern']=audit;result['search_complete']=False;return result
    index = CollinearSupports(supports)
    used=set();new=[]
    for seed in sorted(candidates,key=lambda c:(-c['length'],c['index'])):
        if seed['index'] in used:continue
        audit['evaluated_seeds'] += 1
        origin=seed['center'];u=seed['u'].copy()
        if u[np.argmax(abs(u))]<0:u=-u
        group=[]
        for candidate in index.query(origin, u, used, audit):
            if audit['pair_checks']>=POLICY['maximum_pair_checks']:audit.update(complete=False,reason='pair_budget');break
            audit['pair_checks']+=1
            if candidate['length']>=2*POLICY['max_error_px'] and abs(float(candidate['u']@u))<math.cos(math.radians(POLICY['angle_deg'])):continue
            q=candidate['p'];t=(q-origin)@u
            if np.linalg.norm(q-origin-t[:,None]*u,axis=1).max()<=POLICY['max_error_px']:group.append(candidate)
        if not audit['complete']:break
        if len(group)<POLICY['minimum_strokes']:continue
        # Collinear geometry elsewhere on the sheet is not an axis extension.
        # Split only at source-gap outliers, using the measured gap distribution.
        spans=sorted(((float(((c['p']-origin)@u).min()),float(((c['p']-origin)@u).max()),c) for c in group),
            key=lambda span:(span[0],span[1],span[2]['index']))
        gaps=[];end=spans[0][1]
        for left,right,_ in spans[1:]:
            if left>end:gaps.append(left-end)
            end=max(end,right)
        if len(gaps)>=4:
            q1,q3=np.quantile(gaps,[.25,.75]);limit=float(q3+1.5*(q3-q1)+2*POLICY['max_error_px'])
            clusters=[[]];end=spans[0][1]
            for left,right,c in spans:
                if clusters[-1] and left-end>limit:clusters.append([])
                clusters[-1].append(c);end=max(end,right) if len(clusters[-1])>1 else right
            group=next(g for g in clusters if any(c['index']==seed['index'] for c in g))
        if len(group)<POLICY['minimum_strokes']:continue
        p=np.concatenate([c['p'] for c in group]);origin=p.mean(0);_,_,vectors=np.linalg.svd(p-origin,full_matrices=False);u=vectors[0]
        if u[np.argmax(abs(u))]<0:u=-u
        t=(p-origin)@u
        if np.linalg.norm(p-origin-t[:,None]*u,axis=1).max()>2:continue
        lo,hi=float(t.min()),float(t.max());pen=float(np.median([c['pen'] for c in group]));blocked=False
        for head in heads:
            polygon=np.asarray(head.get('polygon_px',[]),float);direction=np.asarray(head.get('direction',[0,0]),float)
            if polygon.shape!=(3,2) or direction.shape!=(2,) or not np.isfinite(polygon).all() or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:continue
            direction=direction/np.linalg.norm(direction)
            if abs(float(direction@u))<math.cos(math.radians(20)):continue
            along=(polygon-origin)@u;normal=polygon-origin-along[:,None]*u
            if along.max()>=lo-2 and along.min()<=hi+2 and np.linalg.norm(normal,axis=1).min()<=pen/2+2:blocked=True;break
        if blocked:continue
        evidence=_measure(ink,origin,u,lo,hi,pen)
        if evidence is None:continue
        blocked_mask=np.zeros_like(gray,bool) if protected is None else protected.copy()
        transverse=[]
        for head in heads:
            polygon=np.asarray(head.get('polygon_px',[]),float);direction=np.asarray(head.get('direction',[0,0]),float)
            if polygon.shape!=(3,2) or direction.shape!=(2,) or not np.isfinite(polygon).all() or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:continue
            direction=direction/np.linalg.norm(direction)
            if abs(float(direction@u))>=math.cos(math.radians(20)):continue
            mask=np.zeros_like(gray);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255);area=mask>0
            if area.any() and np.mean(ink[area]>0)>=.7:
                blocked_mask[area]=False;transverse.append(head.get('id'))
        blocked_mask|=roles==6
        if text_protection is not None:blocked_mask|=text_protection
        samples=np.arange(evidence['lo'],evidence['hi']+.01,.5);q=np.rint(origin+samples[:,None]*u).astype(int)
        if blocked_mask[q[:,1],q[:,0]].any():continue
        evidence['transverse_heads_retained_as_source']=transverse
        indices=[c['index'] for c in group];split=_partition(objects,indices,origin,u,evidence,lookup,support)
        if split is None:continue
        members,points,ends,errors,remainders=split
        identity='axis-pattern/'+object_sha({'source':sha(gray),'members':members,'pattern':evidence})[:24]
        new.append({**objects[indices[0]],'kind':'line','id':identity,'role':5,
            'points_px':ends.tolist(),'support_points':points.tolist(),'source_members':members,
            'path_ids':list(dict.fromkeys(m['path_id'] for m in members)),'linetype_pattern_px':evidence['pattern_px'],
            'axis_pattern_evidence':evidence,'fit_max_px':float(errors.max()),'fit_p95_px':float(np.percentile(errors,95)),
            'status':'automatic_axis_pattern_draft','semantic_ownership_confirmed':False})
        new+=remainders;used.update(indices);audit['accepted'].append(identity)
        if len(audit['accepted'])>POLICY['maximum_axes']:audit.update(complete=False,reason='axis_budget');break
    if not audit['complete']:
        audit['discarded_provisional_axes'] = audit['accepted']
        audit['accepted'] = []
        result['axis_pattern']=audit;result['search_complete']=False;return result
    result['objects']=[o for i,o in enumerate(objects) if i not in used]+new
    if _edges(result['objects'])!=before:raise ValueError('Axis pattern changed source edge ownership')
    result.update(axis_pattern=audit,object_count=len(result['objects']),kind_counts=dict(Counter(o['kind'] for o in result['objects'])))
    return result
