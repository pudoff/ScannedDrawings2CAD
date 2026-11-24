"""Require two observed opposing transverse half-branches at a split LINE node."""
import math
import numpy as np
from .cad_primitive_fit_v2 import POLICY


def certify(items,index,i,j,p,q,internal,direction):
    vectors=[];branches=[];midpoint=(p+q)/2
    for k in index.near(p,q):
        if k in (i,j) or k in internal:continue
        item=items[k];points=item['points']
        near=(np.max(abs(points-p),axis=1)<=1)|(np.max(abs(points-q),axis=1)<=1)
        contact=np.flatnonzero(near)
        if not len(contact):continue
        model=item.get('model')
        if not model or model['kind']!='line':return None
        ends=np.asarray(model['points_px'],float);axis=ends[1]-ends[0];length=np.linalg.norm(axis)
        if not length:return None
        axis/=length
        for samples in (points[:contact[0]+1][::-1],points[contact[-1]:]):
            delta=samples-midpoint;distance=np.linalg.norm(delta,axis=1)
            if not len(distance) or distance.max()<POLICY['minimum_length_px']:continue
            vector=axis*np.sign(float(delta[int(np.argmax(distance))]@axis))
            if not np.linalg.norm(vector):return None
            vectors.append(vector);branches.append(dict(path_ids=item['path_ids'],direction=vector.tolist()))
    cosine=math.cos(math.radians(POLICY['tangent_tolerance_deg']))
    if len(vectors)!=2 or float(vectors[0]@vectors[1])>-cosine:return None
    if any(abs(float(vector@direction))>=cosine for vector in vectors):return None
    return dict(rule='two_opposing_observed_transverse_LINE_branches',branches=branches,
        source_nodes=[p.astype(int).tolist(),q.astype(int).tolist()],
        geometry_role_assigned=False,physical_geometry_approved=False)
