"""Assemble source-bound draft CAD objects without guessing part semantics."""
from collections import Counter, defaultdict
from copy import deepcopy
import math

import cv2
import numpy as np

from .anchored_curve_fit_v1 import fit
from .curve_graph_v1 import sample
from .curve_graph_metrics_v1 import distance as geometry_distance
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha

POLICY = {'version':'cad-object-assembly/1.0','maximum_pixels':4194304,
    'maximum_candidates':8192,'maximum_source_points':2000000,
    'maximum_path_points':120000,'fit_max_px':2.,'fit_p95_px':1.25,
    'maximum_curve_samples':120000,'maximum_fit_path_points':8192,
    'maximum_junction_degree':32,'fit_sampling_corridor_px':1,
    'join':'exact_endpoint_or_unique_straight_junction_continuation',
    'continuation_tolerance_deg':8.,'maximum_continuation_pen_ratio':1.67,
    'units':'working_pixels_unitless','physical_geometry_approved':False}


def _reverse_error(candidate,points):
    """Measured distance to this source walk, not to unrelated image ink."""
    samples = sample(candidate)
    if len(samples)>POLICY['maximum_curve_samples']: return None
    low = np.minimum(points.min(0),np.floor(samples.min(0)).astype(int))-4
    high = np.maximum(points.max(0),np.ceil(samples.max(0)).astype(int))+5
    if np.prod(high-low)>POLICY['maximum_pixels']+200000: return None
    mask = np.full(tuple((high-low)[::-1]),255,np.uint8)
    q = points-low; mask[q[:,1],q[:,0]] = 0
    distance = cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    q = np.rint(samples).astype(int)-low
    return float(distance[q[:,1],q[:,0]].max(initial=0))


def _points(value):
    points = np.asarray(value,float)
    if points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or not np.isfinite(points).all():
        raise ValueError('Finite nonempty source points required')
    if not np.array_equal(points,np.rint(points)):
        raise ValueError('Source points must be observed integer pixel centres')
    if len(points)>1 and (np.any(np.abs(np.diff(points,axis=0)).max(1)>1) or
                         np.any(np.abs(np.diff(points,axis=0)).max(1)==0)):
        raise ValueError('Source points must follow adjacent pixels without gaps')
    return points.astype(np.int32)


def compact(points):
    """Remove only exactly collinear same-direction interior vertices."""
    p = np.asarray(points,float)
    if len(p)<3: return p.tolist()
    delta = np.diff(p,axis=0)
    cross = delta[:-1,0]*delta[1:,1]-delta[:-1,1]*delta[1:,0]
    forward = np.sum(delta[:-1]*delta[1:],axis=1)>0
    keep = np.r_[True,(cross!=0)|~forward,True]
    return p[keep].tolist()


def _paths(gray,geometry,roles,protected):
    groups = defaultdict(list); ids = set(); count = 0
    for c in geometry:
        if c['id'] in ids: raise ValueError('Duplicate candidate ID')
        ids.add(c['id']); p = _points(c.get('support_points',c.get('points_px')))
        count += len(p)
        if count>POLICY['maximum_source_points']: raise ValueError('Source point budget exceeded')
        h,w = gray.shape
        if (p<0).any() or (p[:,0]>=w).any() or (p[:,1]>=h).any(): raise ValueError('Source outside raster')
        lo,hi = c['source_index_interval']
        if type(lo) is not int or type(hi) is not int or lo<0 or hi-lo+1!=len(p):
            raise ValueError('Invalid source index interval')
        groups[c['path_id']].append((lo,hi,c,p))
    _,ink = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    output = []
    for path,parts in sorted(groups.items()):
        parts.sort(key=lambda x:(x[0],x[1],x[2]['id']))
        if parts[0][0]!=0: raise ValueError('Source partition must start at zero')
        p = parts[0][3].copy()
        observation_ids = {c.get('source_observation_id',path) for _,_,c,_ in parts}
        if len(observation_ids)!=1: raise ValueError('Path has conflicting source observation IDs')
        for a,b in zip(parts,parts[1:]):
            if a[1]!=b[0] or not np.array_equal(a[3][-1],b[3][0]):
                raise ValueError('Source partition gap, overlap or conflicting endpoint')
            p = np.r_[p,b[3][1:]]
        if len(p)>POLICY['maximum_path_points']: raise ValueError('Path point budget exceeded')
        edges = [tuple(sorted((tuple(a),tuple(b)))) for a,b in zip(p,p[1:])]
        if len(edges)!=len(set(edges)): raise ValueError('Retraced source edges are ambiguous')
        if not np.all(ink[p[:,1],p[:,0]]): raise ValueError('Source support outside observed ink')
        blocked = bool(np.any(protected[p[:,1],p[:,0]]))
        known = set(map(int,roles[p[:,1],p[:,0]])) & set(range(1,10))
        closed = len(p)>2 and np.array_equal(p[0],p[-1])
        output.append({'path_ids':[path],'points':p,'closed':closed,'blocked':blocked,
            'roles':known,'members':[{'candidate_id':c['id'],'source_observation_id':c.get('source_observation_id',path),
                'path_id':path,'source_index_interval':[lo,hi],
                'interval_ids':c.get('interval_ids',[c['interval_id']] if c.get('interval_id') else [])}
                for lo,hi,c,_ in parts]})
    return output,ink


def _join(paths,ink):
    endpoints = defaultdict(list); interiors = set()
    for i,path in enumerate(paths):
        p = path['points']
        interiors.update(map(tuple,p if path['closed'] else p[1:-1]))
        if path['closed'] or len(p)<2: continue
        for end in (0,-1): endpoints[tuple(p[end])].append(i)
    junction_nodes = {point for point,indices in endpoints.items() if len(set(indices))>2 or point in interiors}
    links = defaultdict(list); dt = cv2.distanceTransform(ink,cv2.DIST_L2,5)
    def compatible(a,b):
        return not (a['blocked'] or b['blocked']) and len(a['roles']|b['roles'])<=1
    def direction_and_pen(path,node):
        p = path['points']
        if tuple(p[-1])==node: p = p[::-1]
        distance = np.linalg.norm(p-p[0],axis=1)
        candidates = np.flatnonzero(distance>=8)
        index = int(candidates[0]) if len(candidates) else int(np.argmax(distance))
        direction = p[index].astype(float)-p[0]
        if np.linalg.norm(direction)==0: return None
        q = p[1:index+1] if index else p[:1]
        return direction/np.linalg.norm(direction),max(1.,float(np.median(dt[q[:,1],q[:,0]])*2-1))
    def connect(point,a,b):
        links[a].append((point,b)); links[b].append((point,a))
    for point,indices in endpoints.items():
        indices = sorted(set(indices))
        if len(indices)>POLICY['maximum_junction_degree']: continue
        if len(indices)==2 and point not in interiors and compatible(*(paths[i] for i in indices)):
            connect(point,*indices); continue
        # At T/X nodes keep every branch. Only uniquely reciprocal straight
        # continuations become one entity; intersection is retained geometrically.
        options = defaultdict(list)
        tangents = {i:direction_and_pen(paths[i],point) for i in indices}
        for position,a in enumerate(indices):
            for b in indices[position+1:]:
                u,v = tangents[a],tangents[b]
                if not compatible(paths[a],paths[b]) or u is None or v is None: continue
                if (float(u[0]@v[0])<=-math.cos(math.radians(POLICY['continuation_tolerance_deg'])) and
                    max(u[1],v[1])/min(u[1],v[1])<=POLICY['maximum_continuation_pen_ratio']):
                    options[a].append(b); options[b].append(a)
        for a in indices:
            if len(options[a])==1:
                b = options[a][0]
                if a<b and options[b]==[a]: connect(point,a,b)
    visited = set(); output = []
    def traverse(start):
        path = deepcopy(paths[start]); current = start; visited.add(start)
        available = [tuple(path['points'][0]),tuple(path['points'][-1])]
        # Start from a terminal side where possible; cycles keep deterministic order.
        if any(node==available[0] for node,_ in links[start]) and not any(node==available[1] for node,_ in links[start]):
            path['points'] = path['points'][::-1].copy()
        while True:
            end = tuple(path['points'][-1])
            options = [(node,i) for node,i in links[current] if node==end and i not in visited]
            if len(options)!=1: break
            _,next_id = options[0]; following = paths[next_id]; q = following['points']
            if tuple(q[0])!=end: q = q[::-1]
            if len(path['points'])+len(q)-1>POLICY['maximum_path_points']: break
            path['points'] = np.r_[path['points'],q[1:]]
            path['path_ids'] += following['path_ids']; path['members'] += deepcopy(following['members'])
            path['roles'] |= following['roles']; visited.add(next_id); current = next_id
        path['closed'] = len(path['points'])>2 and np.array_equal(path['points'][0],path['points'][-1])
        return path
    # Components with terminal endpoints must be started before degree-two interiors.
    order = sorted(range(len(paths)),key=lambda i:(len(links[i])==2,i))
    for i in order:
        if i not in visited: output.append(traverse(i))
    for path in output:
        path['junction_anchors'] = np.asarray([p for p in path['points'] if tuple(p) in junction_nodes],float).reshape(-1,2)
    return output


def assemble(gray,geometry,*,roles=None,protected=None):
    if gray.dtype!=np.uint8 or gray.ndim!=2 or not 0<gray.size<=POLICY['maximum_pixels']:
        raise ValueError('Bounded uint8 gray required')
    if len(geometry)>POLICY['maximum_candidates']: raise ValueError('Candidate budget exceeded')
    roles = np.full_like(gray,255) if roles is None else roles
    protected = np.zeros_like(gray,bool) if protected is None else protected
    if roles.shape!=gray.shape or roles.dtype!=np.uint8 or protected.shape!=gray.shape or protected.dtype!=bool:
        raise ValueError('Aligned uint8 roles and boolean protection required')
    if not np.isin(roles,list(range(10))+[255]).all(): raise ValueError('Unsupported source role value')
    before = object_sha(geometry); pixels = sha(gray); protection = sha(protected.astype(np.uint8)); role_sha = sha(roles)
    excluded = protected | (roles==6)
    paths,ink = _paths(gray,geometry,roles,excluded)
    joined = _join(paths,ink); objects = []; withheld = []
    # A digital source walk need not share the Bresenham phase of its chord.
    # The sampling corridor never changes source; reverse distance below is
    # measured against this owned walk, not against other nearby ink.
    fit_support = cv2.dilate(ink,np.ones((3,3),np.uint8))
    for path in joined:
        p = path['points']; members = path['members']
        if path['blocked']:
            withheld.append({'source_members':members,'reason':'protected_source_path'}); continue
        # Mixed-role paths remain exact observations, not newly simplified role objects.
        typed = None
        length = float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        if len(p)<=POLICY['maximum_fit_path_points'] and length>=8 and len(path['roles'])<=1:
            anchors = path['junction_anchors']
            typed = fit(p,path['closed'],fit_support,conditioned=True,native_scale=1.,
                protected_anchor=bool(len(anchors) and np.any(np.all(anchors==p[0],axis=1))))
            if typed and typed['kind'] in ('circle','arc'):
                sweep = math.radians(typed.get('sweep_angle_deg',360))
                if typed['radius_px']*sweep/.5>POLICY['maximum_curve_samples']: typed = None
            if typed and len(anchors) and np.max(geometry_distance(anchors,typed))>1e-7:
                typed = None  # retain the exact walk rather than detach a branch
            if typed:
                reverse = _reverse_error(typed,p)
                # Query rounding contributes at most sqrt(0.5) pixels.
                if reverse is None or reverse>POLICY['fit_max_px']+math.sqrt(.5): typed = None
                else: typed['reverse_source_pixel_distance_px'] = reverse
        if typed is None:
            if len(p)==1: typed = {'kind':'point','points_px':p.tolist(),'fit_max_px':0.,'fit_p95_px':0.}
            else: typed = {'kind':'polyline','points_px':compact(p),'closed':path['closed'],
                          'fit_max_px':0.,'fit_p95_px':0.}
        identity = 'cad-object/'+object_sha({'source':pixels,'members':members,'geometry':typed})[:24]
        objects.append({**typed,'id':identity,'source_members':members,'path_ids':path['path_ids'],
            'support_points':p.tolist(),'role':next(iter(path['roles'])) if len(path['roles'])==1 else 255,
            'status':'draft_requires_review','semantic_ownership_confirmed':False,
            'physical_geometry_approved':False})
    if before!=object_sha(geometry) or pixels!=sha(gray) or protection!=sha(protected.astype(np.uint8)) or role_sha!=sha(roles):
        raise ValueError('Object assembly changed input observations')
    return {'objects':objects,'withheld':withheld,'source_sha256':pixels,'policy':deepcopy(POLICY),
        'input_candidate_count':len(geometry),'input_source_path_count':len(paths),
        'joined_path_count':len(joined),'object_count':len(objects),
        'kind_counts':dict(Counter(o['kind'] for o in objects)),
        'multi_candidate_objects':sum(len(o['source_members'])>1 for o in objects),
        'input_immutable':True,'physical_geometry_approved':False,'units':'working_pixels_unitless'}
