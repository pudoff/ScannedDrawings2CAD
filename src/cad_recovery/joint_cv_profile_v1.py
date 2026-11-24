"""Shadow role composition on frozen source intervals, with immutable baseline."""
from copy import deepcopy
import math
import numpy as np
from .source_chain_profile_v2 import analyze as routes
from .head_junction_ownership_v1 import apply as junction_guard
from .head_width_contract_v1 import normalize


def run_roles(gray,graph,base,heads):
    inherited=base['pixel_roles'];text=np.where(inherited==6,255,0).astype(np.uint8)
    raw=routes(gray,graph,inherited,text,heads)
    # Source-role hypotheses remain visible; the composed view cannot alter a
    # previously assigned role, even when the old heuristic is incorrect.
    candidate={**raw,'pixel_roles':raw['pixel_roles'].copy(),'pixel_interval_owner':raw['pixel_interval_owner'].copy()}
    known=inherited!=255;conflicts=int(np.count_nonzero(known&(candidate['pixel_roles']!=inherited)))
    candidate['pixel_roles'][known]=inherited[known];candidate['pixel_interval_owner'][known]=0
    approved,decisions,allowed=chain_authority(graph,base['dimension_chains'],raw['dimension_chains'])
    new=candidate['pixel_roles']!=inherited
    good=np.isin(candidate['pixel_interval_owner'],allowed)
    withdrawn=new&~good
    candidate['pixel_roles'][withdrawn]=inherited[withdrawn];candidate['pixel_interval_owner'][withdrawn]=0
    candidate['dimension_chains']=approved
    result=junction_guard(gray,graph,inherited,candidate,heads)
    result['raw_known_role_conflicts_withdrawn']=conflicts
    result['source_chain_hypotheses']=deepcopy(raw['dimension_chains'])
    result['chain_authority_decisions']=decisions
    result['chain_authority_withdrawn_pixels']=int(withdrawn.sum())
    return result


def analyze(gray,graph,base,previous_heads,joint_heads,*,prior_binding_ids=None):
    prefix=[h['id'] for h in previous_heads]
    if joint_heads[:len(previous_heads)]!=list(previous_heads):raise ValueError('Exact accepted-head prefix required')
    width=normalize(gray,joint_heads,immutable_ids=prefix)
    allowed=set(prefix) if prior_binding_ids is None else set(prior_binding_ids)
    if not allowed<=set(prefix):raise ValueError('Prior binding IDs must refer to the old prefix')
    old_role_heads=[h for h in previous_heads if h['id'] in allowed]
    joint_role_heads=[h for h in width['role_heads'] if h['id'] in allowed or h['id'] not in set(prefix)]
    old=run_roles(gray,graph,base,old_role_heads)
    shared=joint_role_heads==old_role_heads
    joint=old if shared else run_roles(gray,graph,base,joint_role_heads)
    packets={}
    for name,packet in [('replay_old',old),('joint',joint)]:
        # Rerunning a role producer is not a new canonical-chain verifier.
        # Competing source chains remain scored diagnostic hypotheses.
        p={**packet,'dimension_chains':deepcopy(base['dimension_chains'])}
        if not width['complete'] and name=='joint':
            p={**p,'pixel_roles':base['pixel_roles'].copy(),'pixel_interval_owner':np.zeros_like(packet['pixel_interval_owner']),
                'dimension_chains':deepcopy(base['dimension_chains']),'search_incomplete':True,'added_pixels':0}
        packets[name]=p
    return {'width_contract':width,**packets,'shared_role_evaluation':shared,'role_evaluations':1 if shared else 2,
        'prior_binding_ids':sorted(allowed),'joint_role_head_ids':[h['id'] for h in joint_role_heads],
        'source_pixels_erased':0,'production_admitted':False,'physical_geometry_approved':False,
        'new_junction_positions_used_for_role_authority':False}


def chain_key(chain):
    return (chain.get('shaft_track'),tuple(sorted(chain.get('head_ids',[]))))


def chain_authority(graph,inherited,candidates):
    identities={p['id']:i+1 for i,p in enumerate(graph['proposals'])};boundaries={b['id']:b for b in graph['boundaries']['boundaries']}
    existing={}
    for c in inherited:existing.setdefault(chain_key(c),[]).append(c)
    admitted=[];records=[];allowed=set();forbidden=set();tolerance=max(1.,graph['stroke_px'])+math.sqrt(.5)
    for c in candidates:
        source=set(c['shaft_interval_ids'])|{i for g in c['extension_interval_ids'] for i in g}
        body=boundaries.get(c.get('boundary_id'));source|=set(body['owners']) if body else set()
        witnesses=existing.get(chain_key(c),[]);p=np.asarray(c['feet_px'],float);errors=[]
        if p.shape!=(2,2) or not np.isfinite(p).all():raise ValueError('Finite source-chain feet required')
        for w in witnesses:
            q=np.asarray(w['feet_px'],float)
            if q.shape==(2,2) and np.isfinite(q).all():errors.append(min(float(np.linalg.norm(p-q,axis=1).max()),float(np.linalg.norm(p-q[::-1],axis=1).max())))
        compatible=bool(errors and min(errors)<=tolerance and source<=set(identities))
        if compatible:admitted.append(deepcopy(c));allowed|={identities[i] for i in source}
        else:forbidden|={identities[i] for i in source if i in identities}
        records.append({'shaft_track':c['shaft_track'],'head_ids':list(c['head_ids']),'canonical_witnesses':len(witnesses),
            'maximum_foot_displacement_px':min(errors) if errors else None,'source_width_tolerance_px':tolerance,
            'eligible_for_role_claims':compatible,'reason':'compatible_canonical_chain' if compatible else 'missing_or_conflicting_canonical_chain',
            'source_interval_ids':sorted(source),'canonical_chain_added':False})
    return admitted,records,sorted(allowed-forbidden)
