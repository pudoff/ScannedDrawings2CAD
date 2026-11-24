"""Assign semantic intervals on frozen CAD models, without assembling or fitting.

The baseline already includes source alignment, completion, axis handling and
its old role guard. A new role map may subdivide a model, never refit it.
"""
from collections import Counter
from copy import deepcopy

import numpy as np

from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .primitive_role_guard_v1 import apply as role_guard, edge_inventory, point_inventory
from .cad_object_assembly_v1 import compact


POLICY={'version':'geometry-preserving-role-export/1.0','maximum_objects':16384,
    'numerical_parameter_epsilon':1e-9,'refit_performed':False,
    'source_geometry_changed':False,'physical_geometry_approved':False}


def _model(obj):
    keys={'line':('points_px',),'arc':('center_px','radius_px','start_angle_deg','sweep_angle_deg'),
        'circle':('center_px','radius_px'),'polyline':('points_px','closed'),
        'point':('points_px',)}.get(obj['kind'])
    if keys is None:raise ValueError('Unsupported frozen CAD model')
    return {'kind':obj['kind'],**{k:deepcopy(obj.get(k)) for k in keys}}


def _partition(parent,children):
    """Return exact model intervals; epsilon covers only floating arithmetic."""
    eps=POLICY['numerical_parameter_epsilon'];kind=parent['kind'];spans=[]
    if len(children)==1 and _model(children[0])==_model(parent):
        return [[0.,1.]]
    if kind=='line':
        ends=np.asarray(parent['points_px'],float);delta=ends[1]-ends[0]
        square=float(delta@delta)
        if square<=0:raise ValueError('Degenerate frozen LINE')
        for child in children:
            if child['kind']!='line':raise ValueError('Role update changed LINE kind')
            p=np.asarray(child['points_px'],float)
            positions=(p-ends[0])@delta/square
            projected=ends[0]+positions[:,None]*delta
            if not np.allclose(p,projected,rtol=0,atol=eps):
                raise ValueError('Role update moved frozen LINE')
            spans.append(sorted(map(float,positions)))
    elif kind in ('arc','circle'):
        start=float(parent.get('start_angle_deg',0.))
        sweep=float(parent.get('sweep_angle_deg',360.))
        for child in children:
            if child['kind']!='arc' or child['center_px']!=parent['center_px'] or child['radius_px']!=parent['radius_px']:
                raise ValueError('Role update changed frozen circular model')
            low=(float(child['start_angle_deg'])-start)%360.
            if abs(low-360.)<=eps:low=0.
            high=low+float(child['sweep_angle_deg'])
            if not high>low:raise ValueError('Degenerate frozen ARC interval')
            if kind=='circle' and high>360.+eps:
                spans.extend([[low/360.,1.],[0.,(high-360.)/360.]])
            else:spans.append([low/sweep,high/sweep])
    elif kind=='polyline':
        # A polyline has no fitted parameters: its compact source walk must
        # itself be the frozen geometry before source-index subdivision.
        if compact(np.asarray(parent['support_points'],int))!=parent['points_px']:
            raise ValueError('Cannot subdivide a polyline displaced from its source walk')
        edges=Counter(tuple(sorted((tuple(a),tuple(b)))) for a,b in
            zip(parent['support_points'],parent['support_points'][1:]))
        actual=Counter()
        for child in children:
            if child['kind']!='polyline' or compact(np.asarray(child['support_points'],int))!=child['points_px']:
                raise ValueError('Role update moved frozen polyline')
            actual.update(tuple(sorted((tuple(a),tuple(b)))) for a,b in
                zip(child['support_points'],child['support_points'][1:]))
        if actual!=edges:raise ValueError('Role update changed polyline coverage')
        return None
    else:raise ValueError('Role update changed an indivisible frozen model')
    spans.sort()
    if (not spans or abs(spans[0][0])>eps or abs(spans[-1][1]-1)>eps
            or any(b-a<=0 for a,b in spans)
            or any(abs(a[1]-b[0])>eps for a,b in zip(spans,spans[1:]))):
        raise ValueError('Role update overlaps or omits frozen model intervals')
    return spans


def verify_partition(baseline,result):
    if (len(baseline['objects'])>POLICY['maximum_objects'] or len(result['objects'])>POLICY['maximum_objects']):
        raise ValueError('Frozen model object budget exceeded')
    parents={o['id']:o for o in baseline['objects']}
    if len(parents)!=len(baseline['objects']):raise ValueError('Unique frozen CAD identities required')
    children={identity:[] for identity in parents}
    for child in result['objects']:
        identity=child['id'] if child['id'] in parents else child.get('semantic_role_guard',{}).get('parent_object_id')
        if identity not in parents:raise ValueError('Role output has no frozen model parent')
        children[identity].append(child)
    records=[]
    for identity,parent in parents.items():
        pieces=children[identity]
        if not pieces:raise ValueError('Role update lost a frozen model')
        if (edge_inventory(pieces)!=edge_inventory([parent]) or point_inventory(pieces)!=point_inventory([parent])):
            raise ValueError('Role update changed frozen source membership')
        spans=_partition(parent,pieces)
        if len(pieces)>1 or any(p['role']!=parent['role'] for p in pieces):
            records.append({'parent_object_id':identity,'kind':parent['kind'],
                'parent_model_sha256':object_sha(_model(parent)),
                'output_object_ids':[p['id'] for p in pieces],'normalized_model_intervals':spans})
    return {'policy':deepcopy(POLICY),'input_models':len(parents),'output_objects':len(result['objects']),
        'frozen_model_partition_exact':True,'source_membership_exact':True,
        'baseline_assembly_sha256':object_sha(baseline),'changed_role_or_partition':records,
        'geometry_refits':0,'moved_models':0}


def apply(baseline,gray,geometry,roles):
    """Apply the existing semantic guard after all CAD geometry is frozen."""
    before=object_sha([baseline,geometry]);pixels=[sha(gray),sha(roles)]
    result=role_guard(baseline,gray,geometry,roles)
    proof=verify_partition(baseline,result)
    proof.update(source_gray_sha256=sha(gray),final_roles_sha256=sha(roles),
        source_geometry_sha256=object_sha(geometry))
    result['geometry_preserving_role_export']=proof
    if before!=object_sha([baseline,geometry]) or pixels!=[sha(gray),sha(roles)]:
        raise ValueError('Frozen role export mutated its inputs')
    return result
