"""Keep an existing whole ARC when only one partial automatic role is known.

This never joins geometry. A real role conflict or confirmed semantic owner
retains the previous split. Partial evidence stays attached, with UNKNOWN for
the whole primitive rather than a small semantic slice mistaken for its end.
"""
from collections import Counter, defaultdict
from copy import deepcopy

from .primitive_role_guard_v1 import apply as previous, edge_inventory, point_inventory
from .geometry_preserving_role_export_v1 import verify_partition


def apply(assembly, gray, geometry, roles):
    result = previous(assembly, gray, geometry, roles)
    children = defaultdict(list)
    parents = {obj['id']: obj for obj in assembly['objects']}
    for obj in result['objects']:
        identity = obj.get('semantic_role_guard', {}).get('parent_object_id', obj['id'])
        children[identity].append(obj)
    restored = {}; evidence = []
    for identity, pieces in children.items():
        parent = parents[identity]
        if (parent['kind'] != 'arc' or len(pieces) <= 1 or parent.get('semantic_ownership_confirmed')
                or any(obj.get('semantic_ownership_confirmed') for obj in pieces)):
            continue
        proof = pieces[0].get('semantic_role_guard', {}).get('parent_evidence')
        if not proof:
            continue
        known = {int(role) for role, count in proof['observed_role_counts'].items() if int(role) in range(1, 10) and count > 0}
        if len(known) != 1 or not proof['observed_role_counts'].get('255', 0):
            continue
        # This independently proves that the rejected role-only slices exactly
        # partition the already fitted ARC, not merely a nearby circular model.
        verify_partition(dict(objects=[parent]), dict(objects=pieces))
        obj = deepcopy(parent); obj['role'] = 255
        obj['semantic_role_guard'] = dict(**proof, decision='whole_ARC_partial_automatic_role_unknown',
            rejected_role_only_splits=[dict(assigned_role=p['role'],
                source_support_interval=p['semantic_role_guard']['source_support_interval']) for p in pieces])
        obj['semantic_role_guard']['assigned_role'] = 255
        restored[identity] = obj
        evidence.append(dict(object_id=identity, old_parts=len(pieces), role_evidence_preserved=True,
            whole_model_unchanged=True, physical_geometry_approved=False))
    emitted = set(); output = []
    for obj in result['objects']:
        identity = obj.get('semantic_role_guard', {}).get('parent_object_id', obj['id'])
        if identity in restored:
            if identity not in emitted:
                output.append(restored[identity]); emitted.add(identity)
        else:
            output.append(obj)
    if edge_inventory(output) != edge_inventory(assembly['objects']) or point_inventory(output) != point_inventory(assembly['objects']):
        raise ValueError('Whole ARC role decision changed source membership')
    result.update(objects=output, object_count=len(output), kind_counts=dict(Counter(obj['kind'] for obj in output)),
        multi_candidate_objects=sum(len({member['candidate_id'] for member in obj['source_members']}) > 1 for obj in output),
        arc_identity_guard=dict(version='arc-identity-role-guard/1.0', restored=evidence,
            merged_geometry=False, source_membership_exact=True))
    result['primitive_role_guard'].update(output_objects=len(output),
        decisions=dict(Counter(obj['semantic_role_guard']['decision'] for obj in output)),
        changed_role_objects=sum(obj['role'] != obj['semantic_role_guard']['previous_role'] for obj in output))
    return result
