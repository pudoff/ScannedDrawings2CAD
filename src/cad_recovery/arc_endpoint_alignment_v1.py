"""Search the existing source uncertainty for a shared LINE/ARC endpoint."""
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np

from .cad_object_assembly_v5 import _circle_line
from .cad_primitive_fit_v2 import checked, sector, POLICY as LIMITS
from .stroke_curve_control_v1 import distance_to_polyline
from .arc_joint_corner_v1 import refine_failed

POLICY = dict(version='arc-endpoint-alignment/1.0', maximum_tests=8192,
    angular_positions=2*LIMITS['maximum_refine_positions']-1,
    source_uncertainty_px=LIMITS['max_px'], threshold_changes=False)


def refine(items, support, audit):
    changes = refine_failed(items, support, audit)
    endings = defaultdict(list)
    for i, item in enumerate(items):
        p = item['points']
        if len(p) > 1 and not np.array_equal(p[0], p[-1]):
            endings[tuple(p[0])].append((i, 0)); endings[tuple(p[-1])].append((i, -1))
    for node, incident in endings.items():
        if len(incident) != 2:
            continue
        (i, ei), (j, ej) = incident; a, b = items[i], items[j]; ma, mb = a['model'], b['model']
        if not ma or not mb or {ma['kind'], mb['kind']} != {'line', 'arc'}:
            continue
        if len(a['roles'] | b['roles']) > 1 or not set(a['path_ids']) & set(b['path_ids']):
            continue
        li, le, ci, ce = (i, ei, j, ej) if ma['kind'] == 'line' else (j, ej, i, ei)
        line, arc = items[li]['model'], items[ci]['model']; point = np.asarray(node, float)
        center = np.asarray(arc['center_px']); radial = point-center; radius = arc['radius_px']
        choices = [(q, False) for q in _circle_line(arc, line, point)]
        if np.linalg.norm(radial) > 0:
            choices.append((center+radial/np.linalg.norm(radial)*radius, False))
        choices.sort(key=lambda row: float(np.linalg.norm(row[0]-point)))
        theta = math.atan2(radial[1], radial[0])
        delta = 2*math.asin(min(1., LIMITS['max_px']/(2*radius)))
        angles = theta+np.linspace(-delta, delta, POLICY['angular_positions'])
        # Baseline exact intersections are still preferred. The fallback moves
        # only this common endpoint; the opposite LINE endpoint stays frozen.
        choices.extend((center+radius*np.array([math.cos(t), math.sin(t)]), True) for t in angles)
        accepted = False
        for q, fallback in choices:
            if np.linalg.norm(q-point) > LIMITS['max_px']:
                continue
            if audit['tests'] >= POLICY['maximum_tests']:
                audit['search_complete'] = False; break
            audit['tests'] += 1
            new_line = deepcopy(line); ends = np.asarray(line['points_px']).copy(); ends[0 if le == 0 else 1] = q
            error = distance_to_polyline(items[li]['points'], ends)
            if error.max() > LIMITS['max_px'] or np.percentile(error, 95) > LIMITS['p95_px']:
                continue
            new_line.update(points_px=ends.tolist(), fit_max_px=float(error.max()), fit_p95_px=float(np.percentile(error, 95)))
            adjusted = items[ci]['points'].astype(float).copy(); adjusted[ce] = q
            model = sector(adjusted, center, radius)
            if model is None:
                continue
            new_arc = {**arc, **model}
            if not checked(new_line, items[li]['points'], support) or not checked(new_arc, items[ci]['points'], support):
                continue
            items[li]['model'] = new_line; items[ci]['model'] = new_arc; accepted = True
            changes.append(dict(source_node=list(map(int, node)), model_node=q.tolist(),
                distance_px=float(np.linalg.norm(q-point)), items=[li, ci], bounded_search_fallback=fallback))
            audit['fallbacks_accepted'] += int(fallback)
            break
        if not accepted:
            items[li]['model'] = None; items[ci]['model'] = None
    return changes
