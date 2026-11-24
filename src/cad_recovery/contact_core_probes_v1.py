"""Join only observed thick-core source edges into analysis-only corner witnesses.

No raster flood fill, interpolated edge, semantic owner or CAD object is created.
The original paths remain intact. New chains can witness an old clean branch,
but cannot by themselves release any source pixels as PART.
"""
from collections import defaultdict
import math

import cv2
import numpy as np

from .local_contour_probes_v1 import probes as previous
from .local_contour_profiles_v2 import measure, junction_distance
from .stroke_curve_control_v1 import distance_to_polyline
from .stroke_roles_v1 import foreground
from .part_admission_v1 import object_sha

POLICY = {'version':'contact-core-probes/1.0', 'maximum_edge_visits':2000000,
          'maximum_checks':250000, 'maximum_families':64, 'maximum_models':512,
          'maximum_profile_samples':2000000, 'maximum_raster_pixels':4000000,
          'minimum_pen_ratio':1.5, 'maximum_pen_ratio':1.67,
          'minimum_length_in_pens':6., 'pruned_terminal_length_in_pens':.5,
          'fit_max_px':2., 'fit_p95_px':1.25, 'witness_only':True}


def profile_junctions(obj, shape, original):
    if not obj.get('witness_only'):
        return original
    seed=np.zeros(shape,np.uint8)
    for x,y in obj['core_junction_points_px']:
        seed[y,x]=255
    return cv2.distanceTransform(255-seed,cv2.DIST_L2,5) if seed.any() else np.full(shape,np.inf,np.float32)


def probes(result, geometry, *, gray, protected, maximum_segments=4096):
    objects, audit = previous(result, geometry, maximum_segments=maximum_segments)
    work = {'limits':[], 'edge_visits':0, 'checks':0, 'profile_samples':0, 'families':[], 'added_models':0}
    audit = {**audit, 'core_search':work}
    if gray.size > POLICY['maximum_raster_pixels']:
        work['limits'].append('core_raster_budget'); audit['complete']=False; return objects,audit
    ink = foreground(gray); distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    profile_work={'ink':ink,'distance':distance,'samples':0};junctions=junction_distance(geometry,gray.shape)
    measured=[]
    for obj in objects:
        if obj['kind']!='line':continue
        p=np.asarray(obj['support_points'],int);ends=np.asarray(obj['points_px'],float)
        profile=measure(ends,p,gray,profile_work,junctions,maximum_samples=POLICY['maximum_profile_samples'])
        if profile_work['samples']>POLICY['maximum_profile_samples']:
            work['limits'].append('core_profile_budget');break
        if profile is None:continue
        low,high=map(float,np.quantile(profile['sections_px'],[.25,.95]))
        if low>=1. and high<=2*low and np.linalg.norm(ends[1]-ends[0])>=6*low:
            measured.append((low,high,obj))
    work['profile_samples']=profile_work['samples']
    minimum=min((hi for _,hi,_ in measured),default=math.inf)
    families=sorted({pen for pen,_,_ in measured if pen>=POLICY['minimum_pen_ratio']*minimum})
    work['families']=families
    if len(families)>POLICY['maximum_families']:work['limits'].append('core_family_budget')
    edges={};lookup={c['id']:c for c in geometry}
    for c in geometry:
        points=np.asarray(c['support_points'],int);origin=c['source_index_interval'][0]
        for index,(a,b) in enumerate(zip(points,points[1:]),origin):
            if np.max(abs(a-b))!=1:continue
            key=tuple(sorted((tuple(a),tuple(b))))
            edges.setdefault(key,[]).append((c['id'],index))
    extra=[]
    for pen in families if not work['limits'] else []:
        adjacency=defaultdict(set)
        # DT measures distance to background pixel centres. Use its diameter
        # consistently; subtracting an extra pixel disconnects staircase cores.
        threshold=pen/(2*POLICY['maximum_pen_ratio'])
        for a,b in edges:
            work['edge_visits']+=1
            if work['edge_visits']>POLICY['maximum_edge_visits']:
                work['limits'].append('core_edge_budget');break
            if (distance[a[1],a[0]]>=threshold and distance[b[1],b[0]]>=threshold
                    and ink[a[1],a[0]] and ink[b[1],b[0]] and not protected[a[1],a[0]] and not protected[b[1],b[0]]):
                adjacency[a].add(b);adjacency[b].add(a)
        if work['limits']:break
        # Remove only short terminal twigs in this analysis graph. All such
        # edges remain present in the immutable source and the normal exporter.
        for _ in range(2):
            remove=set()
            for start in sorted(adjacency):
                if len(adjacency[start])!=1:continue
                route=[start];length=0.;node=start;prior=None
                while True:
                    choices=adjacency[node]-({prior} if prior is not None else set())
                    if len(choices)!=1:break
                    nxt=next(iter(choices));length+=math.dist(node,nxt);route.append(nxt)
                    if length>POLICY['pruned_terminal_length_in_pens']*pen or len(adjacency[nxt])!=2:break
                    prior,node=node,nxt
                if len(adjacency[route[-1]])>=3 and length<=POLICY['pruned_terminal_length_in_pens']*pen:
                    remove.update(tuple(sorted((a,b))) for a,b in zip(route,route[1:]))
            for a,b in remove:adjacency[a].discard(b);adjacency[b].discard(a)
        visited=set();paths=[]
        for start in sorted(adjacency):
            if len(adjacency[start])==2:continue
            for nxt in sorted(adjacency[start]):
                edge=tuple(sorted((start,nxt)))
                if edge in visited:continue
                route=[start,nxt];visited.add(edge);prior,node=start,nxt
                while len(adjacency[node])==2:
                    following=next(iter(adjacency[node]-{prior}));edge=tuple(sorted((node,following)))
                    if edge in visited:break
                    visited.add(edge);route.append(following);prior,node=node,following
                paths.append(route)
        # Closed cores cannot witness the open material corner being studied.
        for route in paths:
            q=np.asarray(route,int)
            if len(q)<2:continue
            vertices=cv2.approxPolyDP(q.astype(np.float32),1.25,False).reshape(-1,2)
            indices=[0];position=0
            for v in vertices[1:-1]:
                matches=np.flatnonzero(np.all(q[position+1:]==v,axis=1))
                if len(matches):position+=int(matches[0])+1;indices.append(position)
            indices.append(len(q)-1)
            for low,high in zip(indices,indices[1:]):
                p=q[low:high+1];ends=p[[0,-1]].astype(float);length=float(np.linalg.norm(ends[1]-ends[0]))
                error=distance_to_polyline(p,ends)
                if length<POLICY['minimum_length_in_pens']*pen or error.max()>POLICY['fit_max_px'] or np.percentile(error,95)>POLICY['fit_p95_px']:continue
                duplicate=False
                for old in objects+extra:
                    if old['kind']!='line':continue
                    other=np.asarray(old['points_px'],float)
                    # Containment in the original segment's expanded AABB is
                    # necessary for the strict endpoint-distance test below.
                    # Reject distant/short segments before any costly fit.
                    if (ends.min(0)<other.min(0)-2).any() or (ends.max(0)>other.max(0)+2).any():continue
                    work['checks']+=1
                    if work['checks']>POLICY['maximum_checks']:
                        work['limits'].append('core_model_budget');break
                    if distance_to_polyline(ends,other).max()<=2 and distance_to_polyline(p,other).max()<=2:
                        duplicate=True;break
                if work['limits']:break
                if duplicate:continue
                spans=defaultdict(set)
                for a,b in zip(p,p[1:]):
                    for identity,index in edges[tuple(sorted((tuple(a),tuple(b))))]:spans[identity].add(index)
                members=[]
                for identity,values in sorted(spans.items()):
                    c=lookup[identity];groups=[]
                    for index in sorted(values):
                        if groups and index==groups[-1][-1]+1:groups[-1].append(index)
                        else:groups.append([index])
                    for group in groups:
                        a,b=group[0],group[-1]+1
                        ranges=[{'id':r['id'],'source_index_interval':[max(a,r['source_index_interval'][0]),min(b,r['source_index_interval'][1])]}
                            for r in c.get('graph_interval_ranges',[]) if max(a,r['source_index_interval'][0])<min(b,r['source_index_interval'][1])]
                        members.append({'candidate_id':identity,'source_observation_id':c['source_observation_id'],'path_id':c['path_id'],
                            'source_index_interval':[a,b],'graph_interval_ranges':ranges,
                            'interval_ids':[r['id'] for r in ranges] if ranges else c.get('interval_ids',[])})
                extra.append({'id':'contact-core/'+object_sha([members,ends.tolist()])[:24], 'kind':'line',
                    'points_px':ends.tolist(),'support_points':p.tolist(),'source_members':members,
                    'analysis_probe_only':True,'witness_only':True,'core_pen_family_px':pen,
                    'core_junction_points_px':[list(map(int,node)) for node in sorted(adjacency) if len(adjacency[node])>=3]})
                work['provisional_models']=len(extra)
                if len(extra)>POLICY['maximum_models']:
                    work['limits'].append('core_object_budget');break
            if work['limits']:break
        if work['limits']:break
    if work['limits']:audit['complete']=False;return objects,audit
    work['added_models']=len(extra)
    return objects+extra,audit
