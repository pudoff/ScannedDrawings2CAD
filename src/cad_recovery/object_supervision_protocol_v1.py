"""Source endpoint candidates and a separate, reviewed-object FIT target channel.

Legacy pixel truth, source-event denominators and fit supervision are distinct.
No graph builder accepts truth, and no held object label may enter fitting.
"""
from collections import Counter, defaultdict
import math

import numpy as np

POLICY = dict(endpoint_neighbours=16, radius_in_source_widths=64, maximum_radius_px=256,
              maximum_nodes_per_patch=4096, maximum_directed_edges_per_node_budget=96,
              additional_relation_type=5, candidate_edges_are_geometry=False,
              minimum_fit_units=16, minimum_fit_groups=2,
              minimum_held_units=8, minimum_held_groups=2)
KINDS=('same_primitive','dimension_extension','reviewed_unrelated')


def endpoint_graph(features, points, base_graph):
    """Additional relation candidates from observed endpoints only, never labels."""
    x=np.asarray(features,float); n=len(x)
    if (x.shape!=(n,21) or not 0<n<=POLICY['maximum_nodes_per_patch']
            or not np.isfinite(x).all() or len(points)!=n or len(base_graph['node_ids'])!=n):
        raise ValueError('Invalid fixed source observations')
    clouds=[np.asarray(p,float) for p in points]
    if any(p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all() for p in clouds):
        raise ValueError('Invalid source path points')
    ends=np.asarray([[p[0],p[-1]] for p in clouds])
    widths=[]
    for p, feature in zip(clouds,x,strict=True):
        ratio=np.expm1(feature[0])
        if not np.isfinite(ratio) or ratio<=0:raise ValueError('Invalid source width descriptor')
        widths.append(max(1.,float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()/ratio)))
    edges={tuple(map(int,e)) for e in base_graph['directed_edges']}
    if any(not 0<=a<n or not 0<=b<n or a==b or not 0<=k<5 for a,b,k in edges):
        raise ValueError('Invalid original candidate edges')
    original=len(edges)
    for i in range(n):
        # Four endpoint distances, O(N) temporary storage per source node.
        distance=np.min(np.linalg.norm(ends[:,None,:,:]-ends[i][None,:,None,:],axis=-1),axis=(1,2))
        radius=min(POLICY['maximum_radius_px'],POLICY['radius_in_source_widths']*widths[i])
        choices=np.flatnonzero((distance<=radius)&(np.arange(n)!=i))
        order=choices[np.lexsort((choices,distance[choices]))[:POLICY['endpoint_neighbours']]]
        for j in order:
            edges.update(((i,int(j),5),(int(j),i,5)))
    if len(edges)>POLICY['maximum_directed_edges_per_node_budget']*n:
        raise ValueError('Endpoint graph budget exceeded; no silent truncation')
    return dict(node_ids=list(base_graph['node_ids']),directed_edges=[list(e) for e in sorted(edges)],
                old_directed_edges=original,added_directed_edges=len(edges)-original,
                graph_from_source_only=True,candidate_edges_are_geometry=False,
                maximum_radius_px=POLICY['maximum_radius_px'])


def role_channels(legacy, source_objects, node_ids):
    y=np.asarray(legacy)
    if y.dtype!=np.uint8 or len(y)!=len(node_ids) or not np.isin(y,[*range(1,10),255]).all():
        raise ValueError('Legacy source-role targets required')
    index={n:i for i,n in enumerate(node_ids)}
    if len(index)!=len(node_ids):raise ValueError('Repeated source node')
    owners=defaultdict(list)
    for obj in source_objects:
        if (not obj['source_QA']['source_QA_pass'] or not obj['complete_visible_primitive']
                or obj['physical_geometry_approved'] or obj['role_id'] not in range(1,10)):
            continue
        for node in obj['nodes']:
            if node not in index:raise ValueError('Object references absent source node')
            owners[index[node]].append(obj)
    fit=y.copy(); records=[];conflicts=[]
    for i,objects in sorted(owners.items()):
        if len(objects)!=1:
            conflicts.append(dict(node=node_ids[i],reason='multiple_reviewed_object_owners'));continue
        obj=objects[0];role=int(obj['role_id'])
        if int(y[i]) not in (255,role):
            conflicts.append(dict(node=node_ids[i],reason='reviewed_object_conflicts_with_known_pixel_role'));continue
        if y[i]==255:
            fit[i]=role;records.append(dict(node=node_ids[i],object_id=obj['id'],role=role,
                origin='reviewed_object_semantics_fit_only',legacy_pixel_target_remains=255))
    return fit,dict(additional_fit_targets=records,conflicts=conflicts,legacy_evaluation_targets_changed=False)


def pair_channels(supervision, graph, points):
    ids=graph['node_ids'];index={v:i for i,v in enumerate(ids)}
    edges={tuple(sorted((a,b))) for a,b,k in graph['directed_edges']}
    clouds=[np.asarray(p,float) for p in points]
    objects={o['id']:o for o in supervision['source_objects']}
    membership={n:o['id'] for o in objects.values() if o['semantic_group_admitted'] for n in o['nodes']}
    targets={}; event_rows=[]
    def put(a,b,kind,origin,unit,**extra):
        key=tuple(sorted((a,b)))
        if key not in edges:raise ValueError('Training pair absent from source-only candidates')
        old=targets.get(key)
        if old and old['relation']!=kind:raise ValueError('Conflicting derived relation targets')
        if old is None:targets[key]=dict(nodes=[ids[a],ids[b]],relation=kind,origin=origin,unit=unit,**extra)
    # Exact previous pairs remain available as the historical evaluation channel.
    old_pairs=supervision['review']['relations']
    for row in old_pairs:
        a,b=map(index.__getitem__,row['nodes'])
        kind=row['relation'];unit=('primitive:'+row['object_id'] if kind=='same_primitive'
                                  else 'contact:'+row['physical_contact_id'])
        direction={k:row[k] for k in ('dimension_endpoint','extension_endpoint') if k in row}
        put(a,b,kind,'previous_explicit_supervision',unit,**direction)
    # Source physical events may gain a representative candidate, not a new truth label.
    existing_events={r.get('physical_contact_id'):r for r in old_pairs if r.get('physical_contact_id')}
    explicit_object_pairs=set()
    for event in supervision['source_contacts']:
        left,right=event['objects'];explicit_object_pairs.add(tuple(sorted((left,right))))
        old=existing_events.get(event['id']);choices=[];location=np.asarray(event['point_px'],float)
        if old:chosen=list(map(index.__getitem__,old['nodes']))
        else:
            for ln in objects[left]['nodes']:
                for rn in objects[right]['nodes']:
                    a,b=index[ln],index[rn];key=tuple(sorted((a,b)))
                    if key not in edges:continue
                    cost=float(np.min(np.linalg.norm(clouds[a]-location,axis=1))+np.min(np.linalg.norm(clouds[b]-location,axis=1)))
                    choices.append((cost,a,b))
            chosen=list(min(choices)[1:]) if choices else None
            if chosen:
                direction=(dict(dimension_endpoint=ids[chosen[0]],extension_endpoint=ids[chosen[1]])
                           if event['relation']=='dimension_extension' else {})
                put(*chosen,event['relation'],'new_candidate_for_same_reviewed_event','contact:'+event['id'],**direction)
        event_rows.append(dict(id=event['id'],relation=event['relation'],objects=event['objects'],
            previous_available=event['candidate_pair_available'],candidate_available=chosen is not None,
            selected_nodes=[ids[i] for i in chosen] if chosen else None,
            missing_counts_as_FN=chosen is None))
    # Extra FIT negatives are entailed by TWO reviewed object identities and roles.
    # Missing annotations alone never establish either identity or role.
    alternatives=defaultdict(list)
    for a,b in sorted(edges):
        left,right=membership.get(ids[a]),membership.get(ids[b])
        if left is None or right is None or left==right:continue
        pair=tuple(sorted((left,right)))
        if pair in explicit_object_pairs:continue
        roles={objects[left]['role_id'],objects[right]['role_id']}
        if not all(r in range(1,10) for r in roles) or roles=={2,3}:continue
        distance=float(np.min(np.linalg.norm(clouds[a][[0,-1],None,:]-clouds[b][None,[0,-1],:],axis=-1)))
        alternatives[pair].append((distance,a,b))
    for pair,options in sorted(alternatives.items()):
        _,a,b=min(options)
        put(a,b,'reviewed_unrelated','entailed_from_two_reviewed_distinct_non_DIM_EXT_objects',
            'object-pair:'+pair[0]+'|'+pair[1],source_object_ids=list(pair))
    return dict(fit_pairs=list(targets.values()),source_events=event_rows,
        legacy_evaluation_pairs=old_pairs,additional_negatives_are_new_manual_events=False,
        unknown_object_or_DIM_EXT_pair_automatically_negative=False)


def mixed_role_folds(metadata,legacy,fit_targets):
    old=np.asarray(legacy);fit=np.asarray(fit_targets)
    if (len(metadata)!=len(old) or old.shape!=fit.shape or old.dtype!=np.uint8 or fit.dtype!=np.uint8
            or any(r['declared_split']!='train' for r in metadata)
            or not np.isin(old,[*range(1,10),255]).all() or not np.isin(fit,[*range(1,10),255]).all()
            or not np.array_equal(old[old!=255],fit[old!=255])):
        raise ValueError('Original train identities and strict pixel evaluation must be preserved')
    groups=np.array([r['canonical_family'] for r in metadata]);families=sorted(set(groups))
    if len(families)<3:raise ValueError('At least three train groups required')
    definitions=[];held={r:[] for r in range(1,10)}
    for family in families:
        train=np.flatnonzero((groups!=family)&(fit!=255));val=np.flatnonzero((groups==family)&(old!=255))
        roles={}
        for role in range(1,10):
            r=train[fit[train]==role];fg=sorted(set(groups[r]));count=int(np.count_nonzero(old[val]==role))
            ready=len(r)>=16 and len(fg)>=2
            roles[str(role)]=dict(fit_units=len(r),fit_groups=fg,legacy_held_units=count,fit_support_sufficient=ready)
            if ready and count:held[role].append(dict(family=family,units=count))
        definitions.append(dict(held_family=family,fit_indices=train.tolist(),held_legacy_indices=val.tolist(),
            held_object_semantic_diagnostic_indices=np.flatnonzero((groups==family)&(old==255)&(fit!=255)).tolist(),roles=roles))
    coverage={str(r):dict(evaluable_held_groups=len(v),evaluable_legacy_held_units=sum(a['units'] for a in v),
        ready=len(v)>=2 and sum(a['units'] for a in v)>=8) for r,v in held.items()}
    return definitions,coverage


def pair_folds(patches):
    groups=sorted({p['family'] for p in patches});definitions=[];coverage={k:[] for k in KINDS}
    for held in groups:
        fit=[(p,r) for p in patches if p['family']!=held for r in p['pairs']['fit_pairs']]
        legacy=[(p,r) for p in patches if p['family']==held for r in p['pairs']['legacy_evaluation_pairs']]
        support={}
        for kind in KINDS:
            # Count supervision units, not all correlated edges of one primitive.
            units={(p['patch_id'],r['unit']) for p,r in fit if r['relation']==kind}
            fg={p['family'] for p,r in fit if r['relation']==kind}
            count=sum(r['relation']==kind for p,r in legacy)
            ready=len(units)>=16 and len(fg)>=2
            support[kind]=dict(fit_distinct_source_units=len(units),fit_groups=sorted(fg),
                legacy_held_pairs=count,fit_support_sufficient=ready)
            if ready and count:coverage[kind].append(dict(family=held,units=count))
        # Each primitive or physical/object-pair unit has total fit loss weight 1.
        count=Counter((p['patch_id'],r['unit']) for p,r in fit)
        definitions.append(dict(held_family=held,support=support,fit_pairs=[dict(patch_id=p['patch_id'],**r,
            unit_weight=1/count[p['patch_id'],r['unit']]) for p,r in fit],
            held_legacy_pairs=[dict(patch_id=p['patch_id'],**r) for p,r in legacy]))
    return definitions,{k:dict(evaluable_held_groups=len(v),evaluable_legacy_held_pairs=sum(a['units'] for a in v),
        ready=len(v)>=2 and sum(a['units'] for a in v)>=8) for k,v in coverage.items()}
