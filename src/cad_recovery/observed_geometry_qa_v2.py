"""The original strict QA with an exact, smaller boundary representation."""
from collections import defaultdict
from .curve_graph_metrics_v1 import native
from .part_admission_v1 import object_sha
from .robust_topology_v2 import build, POLICY
from .robust_graph_qa_v1 import guard, verified_provenance


def geometry_qa(source, geometry, matrix):
    before = object_sha([source, geometry])
    a, b = [[native(v, matrix) for v in values] for values in (source, geometry)]
    ga, gb = build(a), build(b)
    if ga['complete'] and gb['complete']:
        ends = defaultdict(list)
        for path in a:
            points = path['points_px']; ends[tuple(points[0])].append((path['path_id'], 0))
            if len(points) > 1: ends[tuple(points[-1])].append((path['path_id'], -1))
        evidence = [{'observed_native_px': list(point), 'estimated_native_px': list(point), 'incidences': incident}
            for point, incident in sorted(ends.items())]
        proof = verified_provenance(ga, gb, a, b, evidence)
        result = guard(a, b, graphs=(ga, gb), provenance=proof)
    else:
        result = {'decision': {'complete': False, 'accepted': False,
            'reasons': ['incomplete_observed_geometry_graph'], 'distance': None}}
    keys = ('complete', 'components', 'cycle_rank', 'bounded_faces', 'nesting_complete',
        'unresolved_face_reasons', 'nesting_failure_reasons', 'QA_representation')
    return {'decision': result['decision'], 'source_graph': {k: ga.get(k) for k in keys},
        'candidate_graph': {k: gb.get(k) for k in keys}, 'source_immutable': before == object_sha([source, geometry]),
        'fallback_not_applied_to_role_graph': True, 'policy': POLICY,
        'reason': 'QA only; selected geometry requires a fresh bound dependency rebuild'}
