"""Position, branch-count and domain-specific connectivity diagnostics.

Only explicitly reviewed scopes are denominators. Neither geometric incidence
nor raster connectivity supplies a missing semantic label.
"""
import math
import numpy as np
from .real_chain_metrics_v1 import assignment,inside,point_native

POLICY={'version':'junction-metrics/1.0','maximum_targets':64,'maximum_predictions':512,
        'maximum_matching_edge_visits':2000000,'position_tolerance_native_px':2.,
        'physical_geometry_approved':False,'semantic_inference_from_incidence':False}


def score_nodes(targets,predictions,matrix,*,scopes,unknown_boxes=(),tolerance=2.):
    if len(targets)>64 or len(predictions)>512:raise ValueError('Bounded node matching required')
    if not math.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive finite tolerance required')
    m=np.asarray(matrix,float)
    if m.shape!=(3,3) or not np.isfinite(m).all() or not np.allclose(m[2],[0,0,1]):raise ValueError('Finite affine transform required')
    for group in (targets,predictions):
        if len({n['id'] for n in group})!=len(group):raise ValueError('Unique node IDs required')
        for n in group:
            p=np.asarray(n['point_px'],float)
            if p.shape!=(2,) or not np.isfinite(p).all():raise ValueError('Finite node point required')
    transformed=[{**p,'point_px':point_native(p['point_px'],m).tolist()} for p in predictions]
    def admitted(n):return inside(n['point_px'],scopes) and not inside(n['point_px'],unknown_boxes)
    refs=[n for n in targets if admitted(n)];candidates=[n for n in transformed if admitted(n)]
    def kind(n):return n.get('kind','endpoint' if n['degree']==1 else 'junction')
    costs={};errors={}
    for i,t in enumerate(refs):
        for j,p in enumerate(candidates):
            error=float(np.linalg.norm(np.asarray(t['point_px'])-p['point_px']))
            if kind(t)==kind(p) and error<=tolerance:costs[i,j]=(error/tolerance)**2;errors[i,j]=error
    matching=assignment(len(refs),len(candidates),costs,POLICY['maximum_matching_edge_visits'])
    if matching['incomplete']:
        return {'complete':False,'reason':'matching_budget','tp':None,'fp':None,'fn':None,'physical_geometry_approved':False}
    pairs=[{'target_id':refs[i]['id'],'prediction_id':candidates[j]['id'],'position_error_native_px':errors[i,j],
        'target_degree':refs[i]['degree'],'observed_degree':candidates[j].get('degree'),
        'degree_correct':candidates[j].get('degree')==refs[i]['degree'],'kind':kind(refs[i])} for i,j in matching['pairs']]
    matched_t={p['target_id'] for p in pairs};matched_p={p['prediction_id'] for p in pairs}
    missing=[t['id'] for t in refs if t['id'] not in matched_t];extra=[p['id'] for p in candidates if p['id'] not in matched_p]
    errors=[p['position_error_native_px'] for p in pairs]
    by_kind={k:{'tp':sum(p['kind']==k for p in pairs),'fn':sum(kind(t)==k and t['id'] in missing for t in refs),
        'fp':sum(kind(p)==k and p['id'] in extra for p in candidates)} for k in ('endpoint','junction','isolated')}
    return {'complete':True,'tp':len(pairs),'fp':len(extra),'fn':len(missing),'pairs':pairs,'missing_ids':missing,'extra_ids':extra,
        'degree_errors':sum(not p['degree_correct'] for p in pairs),'by_kind':by_kind,
        'position_error_median_native_px':float(np.median(errors)) if errors else None,
        'position_error_max_native_px':max(errors) if errors else None,'matching_edge_visits':matching['edge_visits'],
        'outside_or_unknown_targets':len(targets)-len(refs),'outside_or_unknown_predictions':len(predictions)-len(candidates),
        'full_page_precision':None,'physical_geometry_approved':False}


def score_connectivity(expected,predicted):
    if set(expected)!={'pixel','primitive','semantic'} or set(predicted)!=set(expected):raise ValueError('Three separate domains required')
    result={}
    for domain,target in expected.items():
        value=predicted[domain]
        if target is not None and type(target) is not bool or value is not None and type(value) is not bool:
            raise ValueError('Connectivity is bool or unknown, never a coerced scalar')
        result[domain]={'false_join':int(target is False and value is True),'false_break':int(target is True and value is False),
            'unresolved_prediction':int(target is not None and value is None),'unknown_target':int(target is None),
            'correct':int(target is not None and value is target),'unsupported_positive':int(target is None and value is True)}
    return result
