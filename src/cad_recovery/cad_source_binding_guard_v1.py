"""Withhold invalid source intervals before FRAME and native ownership.

No interval guessing or CV changes: unbound objects remain unconverted raster ink.
"""
from collections import Counter
from copy import deepcopy


def apply(assembly, geometry):
    lookup = {c['id']: c for c in geometry}
    result = deepcopy(assembly)
    kept, rejected = [], []
    for obj in result['objects']:
        problems = []
        for member in obj['source_members']:
            candidate = lookup.get(member['candidate_id'])
            if candidate is None:
                problems.append({'member': member, 'reason': 'missing_candidate'})
                continue
            points = candidate.get('support_points', candidate.get('points_px', []))
            start = candidate['source_index_interval'][0]
            low, high = member['source_index_interval']
            if not (0 <= low-start <= high-start < len(points)):
                problems.append({'member': member, 'candidate_interval': candidate['source_index_interval'],
                                 'candidate_points': len(points), 'reason': 'out_of_range_source_interval'})
        if not obj['source_members']:
            problems.append({'reason': 'no_source_members'})
        if problems:
            rejected.append({'object_id': obj['id'], 'kind': obj['kind'], 'role': obj['role'],
                             'support_count': len(obj['support_points']), 'problems': problems})
        else:
            kept.append(obj)
    if not rejected:
        return result
    result.update(objects=kept, object_count=len(kept), kind_counts=dict(Counter(o['kind'] for o in kept)),
        multi_candidate_objects=sum(len({m['candidate_id'] for m in o['source_members']}) > 1 for o in kept),
        source_binding_export_guard={'version': '1.0', 'withheld': rejected,
            'source_intervals_repaired': False, 'withheld_ink': 'unconverted residual'})
    proof = result.get('primitive_role_guard')
    if proof is not None:
        proof.update(output_objects=len(kept),
            decisions=dict(Counter(o['semantic_role_guard']['decision'] for o in kept)),
            changed_role_objects=sum(o['role'] != o['semantic_role_guard']['previous_role'] for o in kept),
            export_source_binding_withheld=len(rejected))
    return result
