"""Require original-pixel curvature before reassigning a short LINE at an ARC corner."""
import math
import cv2
import numpy as np

from .cad_primitive_fit_v2 import POLICY, checked, sector
from .native_short_arc_evidence_v1 import assess
from .source_arrow_observation_v1 import transform


def refine(items, fitter, gray, roles, protected, source, audit):
    if source is None:
        audit['reason']='original_frame_unavailable'; return items
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    inverse=np.linalg.inv(source.matrix); scale=np.linalg.svd(source.matrix[:2,:2],compute_uv=False).min()
    plans=[]; boxes=[]
    for index in range(len(items)-2):
        triple=items[index:index+3]; models=[v['model'] for v in triple]
        if not all(models): continue
        kinds=[v['kind'] for v in models]; reverse=kinds==['arc','line','line']
        if reverse: triple=triple[::-1]
        elif kinds!=['line','line','arc']: continue
        line,tail,arc=triple; curve=arc['model']
        p=[v['points'][::-1] if reverse else v['points'] for v in triple]
        lengths=[np.linalg.norm(np.diff(v,axis=0),axis=1).sum() for v in p]
        if (not line['path_ids']==tail['path_ids']==arc['path_ids']
            or not np.array_equal(p[0][-1],p[1][0]) or not np.array_equal(p[1][-1],p[2][0])
            or not lengths[1]<min(lengths[0],lengths[2]) or len(set().union(*(v['roles'] for v in triple)))>1
            or not curve.get('locally_stable',True) or not curve.get('circle_family_support')):
            continue
        delta=p[0][-1]-p[0][0]; short=p[1][-1]-p[1][0]
        if min(np.linalg.norm(delta),np.linalg.norm(short))==0: continue
        dot=abs(float(delta@short))/(np.linalg.norm(delta)*np.linalg.norm(short))
        if dot>=math.cos(math.radians(POLICY['tangent_tolerance_deg'])): continue
        model=sector(p[1],np.asarray(curve['center_px']),curve['radius_px'])
        if model is None or checked(model,p[1],fitter.support) is None: continue
        q=p[1].astype(int)
        if protected[q[:,1],q[:,0]].any() or np.any(roles[q[:,1],q[:,0]]==6): continue
        width=float(np.median(np.maximum(1.,2*distance[q[:,1],q[:,0]]-1)))
        native=transform(p[1],inverse); margin=math.ceil((width/2+POLICY['max_px'])/scale)+2
        low=np.floor(native.min(0)-margin).astype(int); high=np.ceil(native.max(0)+margin+1).astype(int)
        if ((low<0).any() or (high>source.size).any() or max(high-low)>1024): continue
        if len(plans)>=64:
            audit['search_complete']=False; return items
        boxes.append([*low.tolist(),*high.tolist()]); plans.append((index+1,tail,curve,width,model))
    audit['candidates']=len(plans)
    windows=source.windows(boxes) if boxes else []
    for (index,tail,curve,width,model),box,window in zip(plans,boxes,windows,strict=True):
        proof=assess(window,np.asarray(box[:2]),source.matrix,tail['points'],curve,width)
        decision=dict(index=index,paths=tail['path_ids'],native=proof,box=box,
            input=source.record,source=source.source,source_points=tail['points'].tolist())
        audit['decisions'].append(decision)
        if proof['accepted']:
            tail['model']={**curve,**model,'parameter_evidence':'native_short_curvature_with_independent_circle_context'}
            audit['accepted']+=1
    return items
