"""Source-line localization atop the frozen topology-preserving graph.

The digital graph core and fitted centerline intersection are distinct views.
Failed/curved/short support keeps the original graph observation unchanged.
"""
from copy import deepcopy
import math
import numpy as np
import cv2
from .junction_topology_v1 import prepare as topology_prepare,adjacency_bits,OFFSETS
from .raster_junctions import _fit_branch,_connectivity,_topology

POLICY={'version':'junction-topology/2.0','maximum_trace_steps':200000,'maximum_trace_length_px':96.,
    'maximum_condition_number':6.,'maximum_intersection_residual_px':1.5,
    'source_erasure_allowed':False,'production_admitted':False,'physical_geometry_approved':False}


def trace_branches(node,bits,audit):
    cluster={tuple(int(v) for v in p[::-1]) for p in node['source_cluster_px']};paths=[]
    for cy,cx in sorted(cluster):
        for bit,(dy,dx) in enumerate(OFFSETS):
            if not int(bits[cy,cx])&(1<<bit) or (cy+dy,cx+dx) in cluster:continue
            previous=(cy,cx);point=(cy+dy,cx+dx);path=[[cx,cy]];distance=0.
            while distance<=96:
                audit['trace_steps']+=1
                if audit['trace_steps']>POLICY['maximum_trace_steps']:
                    audit['limits'].append('trace_steps');return []
                path.append([point[1],point[0]])
                options=[(point[0]+dy,point[1]+dx) for bit,(dy,dx) in enumerate(OFFSETS)
                    if int(bits[point])&(1<<bit) and (point[0]+dy,point[1]+dx)!=previous]
                if len(options)!=1 or options[0] in cluster:break
                other=options[0];distance+=math.dist(point,other);previous,point=point,other
            paths.append(np.asarray(path,float))
    return paths


def localize_nodes(packet):
    bits=adjacency_bits(packet['skeleton']);dt=cv2.distanceTransform(packet['ink'],cv2.DIST_L2,5)
    rgb=np.repeat((255-packet['ink'])[:,:,None],3,axis=2);output=[];decisions=[]
    audit={'trace_steps':0,'limits':[]}
    for node in packet['nodes']:
        result=deepcopy(node)
        if node['kind']!='junction':output.append(result);continue
        point=np.asarray(node['point_px']);branches=[];paths=trace_branches(node,bits,audit)
        for path in paths:
            if len(path)<6:break
            along=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
            pixels=np.rint(path).astype(int);pen=float(np.median(2*dt[pixels[:,1],pixels[:,0]]-1));pen=max(1.,pen)
            low=max(3.,min(2*pen,.25*along[-1]));high=min(52.,.9*along[-1])
            selected=path[(along>=low)&(along<=high)]
            if len(selected)<4 or high-low<6:break
            center=selected.mean(0);_,_,vectors=np.linalg.svd(selected-center,full_matrices=False)
            direction=vectors[0]
            if direction@(selected[-1]-point)<0:direction=-direction
            search=min(64.,max(4.,1.5*pen))
            branch=_fit_branch(rgb,point,direction,search,low,high)
            branches.append(branch)
        evidence={'node_id':node['id'],'topological_point_px':node['point_px'],'status':'unchanged_insufficient_source_branches',
            'branches':branches}
        if len(branches)==node['degree'] and all(b['status']=='supported' for b in branches) and not audit['limits']:
            normals=np.asarray([b['fit']['normal_xy'] for b in branches]);centers=np.asarray([b['fit']['point_xy'] for b in branches])
            singular=np.linalg.svd(normals,compute_uv=False);condition=float(singular[0]/singular[-1]) if singular[-1]>1e-12 else math.inf
            fitted=np.linalg.lstsq(normals,np.sum(normals*centers,axis=1),rcond=None)[0]
            residual=np.abs(np.sum(normals*(fitted-centers),axis=1));pen=max(b['median_stroke_width_px'] for b in branches)
            displacement=float(np.linalg.norm(fitted-point))
            evidence.update(intersection_candidate_px=fitted.tolist(),displacement_px=displacement,
                condition_number=condition if math.isfinite(condition) else None,residual_px=residual.tolist())
            if condition<=6 and residual.max()<=1.5 and displacement<=max(8.,2*pen) and (fitted>=0).all() and (fitted<packet['ink'].shape[::-1]).all():
                inner=max(4.,min(12.,1.5*pen));connected=_connectivity(rgb,fitted,branches,inner);rings=_topology(rgb,fitted,branches)
                evidence.update(connectivity=connected,topology_rings=rings)
                if all(c['connected'] for c in connected) and all(r['matches_requested_topology'] for r in rings):
                    result['topological_point_px']=node['point_px'];result['point_px']=fitted.tolist()
                    result['position_evidence']='independently_fitted_connected_source_branches'
                    evidence['status']='localized_from_source_branches'
        output.append(result);decisions.append(evidence)
    if audit['limits']:output=deepcopy(packet['nodes'])
    return output,decisions,audit


def prepare(gray):
    packet=topology_prepare(gray)
    if not packet['complete']:
        return {**packet,'localization_decisions':[],'localization_audit':{'trace_steps':0,'limits':[]}}
    nodes,decisions,audit=localize_nodes(packet)
    return {**packet,'topological_nodes':packet['nodes'],'nodes':nodes,'localization_decisions':decisions,
        'localization_audit':audit,'complete':packet['complete'] and not audit['limits']}
