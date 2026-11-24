"""Source-only geometric atoms and bounded geodesic relation proposals.

Atoms partition observed skeleton edges. Relation proposals never fill a raster
gap or declare that adjoining LINE/ARC pieces are one CAD entity.
"""
from collections import Counter, defaultdict
import heapq
import math
import re

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .curve_graph_v1 import graph_paths
from .geometry_detection import _fit_circle, _observed_angles
from .role_interval_dataset_v1 import _features

POLICY=dict(maximum_pixels=1024**2,maximum_skeleton_pixels=120000,maximum_paths=4096,
    maximum_nodes=16384,maximum_fit_calls=16384,maximum_depth=8,maximum_fit_points=1024,
    fit_p95_px=1.25,fit_max_px=2.,minimum_arc_sweep_deg=20.,minimum_arc_sagitta_px=1.,
    maximum_radius_in_sides=4,minimum_atom_edges=1,maximum_directed_edges=1000000,
    geodesic_radius_in_source_widths=32,maximum_geodesic_radius_px=128,
    maximum_geodesic_expansions=2000000,incidence_type=6,geodesic_type=7,
    old_nodes_and_features_preserved=True,labels_in_extractor=False,
    relation_proposals_are_CAD_geometry=False,physical_geometry_approved=False)


def classify(points,closed,side):
    p=np.asarray(points,float);chosen=p[np.linspace(0,len(p)-1,min(len(p),POLICY['maximum_fit_points'])).astype(int)]
    mean=chosen.mean(0);_,_,vectors=np.linalg.svd(chosen-mean,full_matrices=False)
    axis=vectors[0];normal=np.array([-axis[1],axis[0]])
    error=np.abs((p-mean)@normal)
    line_ok=not closed and np.quantile(error,.95)<=POLICY['fit_p95_px'] and error.max()<=POLICY['fit_max_px']
    # Keep unambiguously straight pieces straight; a tiny curve is not an ARC.
    circle=_fit_circle(chosen) if len(chosen)>=3 else None
    if circle is not None:
        center,radius,_=circle;residual=np.abs(np.linalg.norm(p-center,axis=1)-radius)
        if (2<=radius<=side*POLICY['maximum_radius_in_sides']
                and np.quantile(residual,.95)<=POLICY['fit_p95_px'] and residual.max()<=POLICY['fit_max_px']):
            start,sweep=_observed_angles(p,center)
            sagitta=radius*(1-math.cos(min(sweep,math.pi)/2))
            if closed or (math.degrees(sweep)>=POLICY['minimum_arc_sweep_deg']
                    and sagitta>=POLICY['minimum_arc_sagitta_px'] and
                    (not line_ok or np.mean(residual**2)<=.65**2*np.mean(error**2))):
                return 'CIRCLE' if closed else 'ARC'
    return 'LINE' if line_ok else None


def split_path(points,closed,side,account,start=0,depth=0):
    """Half-open edge ranges share endpoints; no source edge is lost or added."""
    p=np.asarray(points,float)
    if len(p)<2:return []
    if account['calls']>=POLICY['maximum_fit_calls']:
        raise ValueError('Atomic fit-call budget exceeded')
    account['calls']+=1;kind=classify(p,closed,side)
    if kind is not None or depth>=POLICY['maximum_depth'] or len(p)<=3:
        return [dict(start=start,stop=start+len(p)-1,kind=kind or 'RESIDUAL')]
    middle=(len(p)-1)//2
    return (split_path(p[:middle+1],False,side,account,start,depth+1)
            +split_path(p[middle:],False,side,account,start+middle,depth+1))


def topology_relations(paths,node_ranges,initial_edges,width):
    """Walk the observed, compressed skeleton, not a nearest-K image graph."""
    owners=defaultdict(set);end_owners=defaultdict(set);positions=defaultdict(set)
    for node,(path_id,start,stop) in enumerate(node_ranges):
        p=np.asarray(paths[path_id]['points'],int)
        if not 0<=start<stop<len(p):raise ValueError('Invalid source node range')
        for q in p[start:stop+1]:owners[tuple(q)].add(node)
        for offset in (start,stop):
            q=tuple(p[offset]);end_owners[q].add(node);positions[path_id].add(offset)
    edges=set(initial_edges)
    def connect(a,b,kind):
        if a!=b:edges.update(((a,b,kind),(b,a,kind)))
    def guard():
        if len(edges)>POLICY['maximum_directed_edges']:raise ValueError('Atomic graph edge budget exceeded')
    for nodes in owners.values():
        ordered=sorted(nodes)
        for i,a in enumerate(ordered):
            for b in ordered[i+1:]:connect(a,b,POLICY['incidence_type'])
        guard()
    adjacency=defaultdict(dict)
    for path_id,path in enumerate(paths):
        p=np.asarray(path['points'],int)
        cumulative=np.r_[0.,np.linalg.norm(np.diff(p,axis=0),axis=1).cumsum()]
        stops=sorted(positions[path_id]|{0,len(p)-1})
        for a,b in zip(stops,stops[1:]):
            left,right=tuple(p[a]),tuple(p[b]);distance=float(cumulative[b]-cumulative[a])
            if left==right:continue
            adjacency[left][right]=min(adjacency[left].get(right,float('inf')),distance)
            adjacency[right][left]=min(adjacency[right].get(left,float('inf')),distance)
    radius=min(POLICY['maximum_geodesic_radius_px'],POLICY['geodesic_radius_in_source_widths']*width)
    expansions=0
    for origin in sorted(end_owners):
        best={origin:0.};queue=[(0.,origin)]
        while queue:
            distance,point=heapq.heappop(queue)
            if distance!=best.get(point):continue
            expansions+=1
            if expansions>POLICY['maximum_geodesic_expansions']:raise ValueError('Atomic geodesic budget exceeded')
            if point>origin:
                for a in end_owners[origin]:
                    for b in end_owners.get(point,()):connect(a,b,POLICY['geodesic_type'])
            for other,cost in adjacency[point].items():
                proposed=distance+cost
                if proposed<=radius and proposed<best.get(other,float('inf')):
                    best[other]=proposed;heapq.heappush(queue,(proposed,other))
        guard()
    return [list(e) for e in sorted(edges)],dict(geodesic_expansions=expansions,geodesic_radius_px=float(radius),
        source_endpoint_positions=len(end_owners),compressed_vertices=len(adjacency),
        no_nearest_neighbour_cutoff=True,white_gap_edges_added=0)


def build(gray,legacy_nodes,legacy_features,base_graph,*,patch_to_source,source_size):
    if gray.dtype!=np.uint8 or gray.ndim!=2 or not 0<gray.size<=POLICY['maximum_pixels']:
        raise ValueError('Bounded original grayscale source required')
    matrix=np.asarray(patch_to_source,float);size=np.asarray(source_size)
    if (matrix.shape!=(3,3) or not np.array_equal(matrix[:2,:2],np.eye(2))
            or not np.array_equal(matrix[2],[0,0,1]) or not np.array_equal(matrix[:2,2],np.floor(matrix[:2,2]))
            or len(size)!=2 or (size<=0).any() or (matrix[:2,2]<0).any()
            or (matrix[:2,2]+gray.shape[::-1]>size).any()):raise ValueError('Exact source crop translation required')
    x=np.asarray(legacy_features)
    if x.shape!=(len(legacy_nodes),21) or len(base_graph['node_ids'])!=len(legacy_nodes):
        raise ValueError('Legacy observations differ')
    ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    skeleton=skeletonize(ink>0,method='zhang');paths,trace=graph_paths(skeleton)
    if trace['search_incomplete']:raise ValueError('Incomplete original source paths')
    distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    widths=np.maximum(1.,2*distance[skeleton]-1);width=float(np.median(widths)) if len(widths) else 1.
    integral=cv2.integral((ink>0).astype(np.uint8),sdepth=cv2.CV_64F)
    nodes=[];features=[];ranges=[]
    def append(identity,path_id,start,stop,kind,is_old):
        path=paths[path_id];p=np.asarray(path['points'],int)[start:stop+1]
        clipped=bool(((np.asarray(path['points'])==[0,0])|(np.asarray(path['points'])==np.asarray(gray.shape[::-1])-1)).any())
        f=np.asarray(_features(p,path,gray,distance,width,matrix,source_size,integral),dtype=x.dtype)
        nodes.append(dict(id=identity,points_px=p.tolist(),path_index=path_id,start=start,stop=stop,
            source_path_clipped=clipped,kind=kind,legacy=is_old))
        features.append(f);ranges.append((path_id,start,stop))
    for i,record in enumerate(legacy_nodes):
        match=re.fullmatch(r'path-(\d+)/(\d+)-(\d+)',record['id'])
        if not match:raise ValueError('Invalid original interval identity')
        path_id,start,stop=map(int,match.groups())
        if path_id>=len(paths):raise ValueError('Original source path missing')
        append(record['id'],path_id,start,stop,'LEGACY_INTERVAL',True)
        if nodes[-1]['points_px']!=record['points_px'] or not np.array_equal(features[-1],x[i]):
            raise ValueError('Original coordinates or features changed')
    account={'calls':0};partition_edges=0
    for path_id,path in enumerate(paths):
        p=np.asarray(path['points'],int)
        parts=split_path(p,path['closed'],max(gray.shape),account)
        expected=0
        for part in parts:
            start,stop=part['start'],part['stop']
            if start!=expected:raise ValueError('Source edge partition has a gap or overlap')
            append(f'atom-{path_id:05d}/{start}-{stop}',path_id,start,stop,part['kind'],False)
            expected=stop;partition_edges+=stop-start
            if len(nodes)>POLICY['maximum_nodes']:raise ValueError('Atomic node budget exceeded')
        if expected!=len(p)-1:raise ValueError('Source path tail omitted')
    if partition_edges!=trace['traced_edges']:raise ValueError('Source edge partition differs')
    if base_graph['node_ids']!=[r['id'] for r in legacy_nodes]:raise ValueError('Old graph identity differs')
    edges,topology=topology_relations(paths,ranges,{tuple(e) for e in base_graph['directed_edges']},width)
    return dict(node_ids=[r['id'] for r in nodes],directed_edges=edges,nodes=nodes,
        legacy_node_count=len(legacy_nodes),atom_count=len(nodes)-len(legacy_nodes),
        source_trace=trace,source_partition_edges=partition_edges,source_edge_partition_exact=True,
        proposed_atom_kinds=dict(Counter(r['kind'] for r in nodes if not r['legacy'])),
        fit_calls=account['calls'],source_topology=topology,graph_from_source_only=True,
        physical_geometry_approved=False,CAD_export_changed=False),np.asarray(features,dtype=x.dtype)
