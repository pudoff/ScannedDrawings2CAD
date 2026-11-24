"""Exact lattice compaction for QA only; incidence and budgets stay unchanged."""
from copy import deepcopy
import numpy as np
from . import primitive_graph_qa_v1 as kernel
from .robust_topology_v1 import build as original_build

POLICY = {'version': 'robust-topology/2.0', 'compaction': 'exact_monotone_collinear_half_pixel_lattice_only',
    'maximum_flat_vertices': 512, 'maximum_topology_predicates': 250000,
    'source_or_candidate_coordinates_changed': False, 'approximation_performed': False,
    'physical_geometry_approved': False}


def compact_points(points):
    """Delete only redundant interior vertices, with an exact integer proof.

    The half-pixel lattice covers native and AREA-half frames. Other coordinates
    use the original topology implementation. Integer determinants cannot overflow
    within the unchanged kernel coordinate budget (abs coordinate <=1e7).
    Endpoints and reversals are retained; intersections are reinserted analytically.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Invalid source points')
    scaled = p*2
    if np.abs(p).max() > 1e7 or not np.array_equal(scaled, np.rint(scaled)):
        return p.tolist(), 0
    lattice = scaled.astype(np.int64); stack = []
    for i in range(len(p)):
        while len(stack) >= 2:
            a, b = lattice[stack[-2]], lattice[stack[-1]]
            u, v = b-a, lattice[i]-b
            # Equality is exact, not an epsilon or an approximation tolerance.
            if int(u[0])*int(v[1])-int(u[1])*int(v[0]) != 0 or int(u[0])*int(v[0])+int(u[1])*int(v[1]) <= 0:
                break
            stack.pop()
        stack.append(i)
    return p[stack].tolist(), len(p)-len(stack)


def build(candidates):
    # Validate the uncompressed input first, so compaction cannot evade object,
    # original-segment, finite-coordinate or coordinate-extent budgets.
    try:
        _, _, definitions = kernel.segments(candidates)
    except kernel.Incomplete as exc:
        return {'complete': False, 'reason': str(exc), 'physical_geometry_approved': False}
    compacted = deepcopy(candidates); removed = 0
    for candidate in compacted:
        if candidate['kind'] in ('line', 'polyline'):
            candidate['points_px'], count = compact_points(candidate['points_px'])
            removed += count
    graph = original_build(compacted)
    graph['QA_representation'] = {'policy': POLICY, 'redundant_vertices_removed': removed,
        'original_definition_vertices': definitions, 'source_geometry_mutated': False}
    return graph
