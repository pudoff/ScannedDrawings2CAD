"""Source-reviewed primitive references and one-to-one geometry/type/layer diagnostics."""
import math
import cv2
import numpy as np
from .experimental_export_v1 import ROLE_LAYERS
from .stroke_curve_control_v1 import maximum_matching
from .whole_primitive_qa_v1 import compare

LAYER_ROLES={v:k for k,v in ROLE_LAYERS.items()}


def target(obj,size):
    if (obj['kind'] not in ('LINE','CIRCLE') or obj['role'] not in (1,2,3,4,5,7,8,9)
            or obj['complete_visible_primitive'] is not True or obj['physical_geometry_approved']
            or obj['human_verified'] or len(obj['observation'])<20):
        raise ValueError('Only explicitly reviewed complete pixel primitives, without physical approval, are admitted')
    if obj['kind']=='LINE':
        vertices=np.asarray(obj['points'],float)
        if vertices.shape!=(2,2):raise ValueError('A LINE cannot be a compound path')
        length=float(np.linalg.norm(vertices[1]-vertices[0]))
        if length<8:raise ValueError('Tiny or degenerate line is not a complete-object target')
        p=np.linspace(vertices[0],vertices[1],max(2,math.ceil(length*2)+1));params={}
    else:
        center=np.asarray(obj['center'],float);radius=obj['radius']
        if center.shape!=(2,) or not np.isfinite(radius) or radius<=3:raise ValueError('Tiny or invalid circle')
        angles=np.linspace(0,2*np.pi,max(24,math.ceil(4*np.pi*radius)+1))
        p=center+radius*np.column_stack([np.cos(angles),np.sin(angles)])
        params=dict(center=center.tolist(),radius=radius,sweep_deg=360.)
    if not np.isfinite(p).all() or (p<0).any() or (p>np.array(size)-1).any():
        raise ValueError('Complete primitive leaves source reference frame')
    return dict(id=obj['id'],kind=obj['kind'],role=obj['role'],points=p,parameters=params)


def source_support(obj,gray):
    t=target(obj,list(gray.shape[::-1]));p=np.rint(t['points']).astype(int)
    distance=cv2.distanceTransform((gray>=240).astype(np.uint8),cv2.DIST_L2,5)
    d=distance[p[:,1],p[:,0]];coverage=float(np.mean(d<=2))
    endpoint=float(d[[0,-1]].max()) if obj['kind']=='LINE' else None
    required=.35 if obj['role']==5 else .90
    return dict(id=obj['id'],source_ink_support=coverage,source_distance_p95_px=float(np.percentile(d,95)),
        endpoint_ink_distance_px=endpoint,minimum_source_support=required,
        source_support_pass=coverage>=required and (endpoint is None or endpoint<=2),
        source_support_is_not_semantic_or_physical_verification=True)


def assignments(rows,objects):
    """Each criterion gets its own maximum matching; one handle cannot serve two targets."""
    maps={}
    for criterion in ('geometry','type','type_and_layer'):
        edges=[]
        for row,obj in zip(rows,objects,strict=True):
            options=row['whole_options']
            edges.append(sorted(h for h,d in options.items()
                if (criterion=='geometry' or d['same_type']) and
                   (criterion!='type_and_layer' or LAYER_ROLES.get(d['layer'])==obj['role'])))
        maps[criterion]=maximum_matching(edges)
    return maps


def evaluate(objects,actual,size):
    targets=[target(obj,size) for obj in objects]
    result=compare(targets,actual);maps=assignments(result['rows'],objects)
    for i,(row,obj) in enumerate(zip(result['rows'],objects,strict=True)):
        row['source_reviewed_primitive']=obj
        row['criterion_matches']={k:(row['whole_options'][m[i]] if i in m else None) for k,m in maps.items()}
    result.update(whole_geometry_matches=len(maps['geometry']),whole_same_type_matches=len(maps['type']),
        whole_type_and_layer_matches=len(maps['type_and_layer']),
        matching='separate maximum-cardinality geometry / geometry+type / geometry+type+layer, one handle per criterion',
        source_CAD_agreement_diagnostic_only=False,
        source_reviewed_partial_evaluation=True,full_page_precision=None,
        human_verified=False,physical_geometry_approved=False,full_sheet_ground_truth=False)
    return result
