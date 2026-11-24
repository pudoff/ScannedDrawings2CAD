"""Native-unit fitting and bounded joint nodes; every inferred role stays unknown."""
import copy
import math
from collections import defaultdict

import cv2
import numpy as np

from .anchored_curve_fit_v1 import circle_solution
from .curve_graph_v1 import native_support, sample, width_evidence
from .curve_graph_metrics_v1 import native, distance
from .simplification_qa_v1 import source_paths, guard
from .stroke_curve_control_v1 import distance_to_polyline

POLICY = {'version': 'joint-shape/1.0', 'fit_p95_native_px': 1.25, 'fit_max_native_px': 2.,
          'minimum_length_native_px': 12., 'minimum_sweep_deg': 20., 'minimum_sagitta_native_px': 1.,
          'sensitivity_native_px_max': 2., 'radial_resolution_native_px': .5,
          'maximum_fit_calls': 8192, 'maximum_points_per_fit': 1024,
          'maximum_parts_per_path': 2, 'maximum_points_per_path': 4096,
          'maximum_split_candidates': 9, 'sharp_corner_degrees': 35.,
          'maximum_node_move_native_px': 1., 'tangent_window_native_px': [6., 18.],
          'node_prior_weight': .25, 'maximum_pixels': 4194304,
          'uncertainty': 'bounded diagnostic search/sensitivity, not calibrated confidence or physical tolerance',
          'production_admitted': False, 'semantic_ownership_confirmed': False}


def transform_points(points, matrix):
    m = np.asarray(matrix, float)
    return np.asarray(points, float) * m[0, 0] + m[:2, 2]


def to_input(candidate, matrix):
    m = np.asarray(matrix, float)
    scale = m[0, 0]
    result = copy.deepcopy(candidate)
    if result['kind'] in ('line', 'polyline'):
        result['points_px'] = ((np.asarray(result['points_px']) - m[:2, 2]) / scale).tolist()
    else:
        result['center_px'] = ((np.asarray(result['center_px']) - m[:2, 2]) / scale).tolist()
        result['radius_px'] /= scale
    return result


def evidence_error(points, candidate):
    """Both directions use finite geometry, in native pixels."""
    modeled = sample(candidate, .5)
    errors = np.r_[distance(points, candidate), distance_to_polyline(modeled, points)]
    return {'p95': float(np.percentile(errors, 95)), 'maximum': float(errors.max()),
            'rmse': float(np.sqrt(np.mean(errors**2)))}


def fit_native(points, closed, ink, matrix, account, *, protected_anchor=False):
    if account['calls'] >= POLICY['maximum_fit_calls']:
        account['incomplete'] = True
        return None, 'fit_budget'
    account['calls'] += 1
    if len(points) < 3:
        return None, 'insufficient_points'
    if len(points) > 4096:
        account['incomplete'] = True
        return None, 'path_point_budget'
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if length < 12:
        return None, 'short_observation'
    a, b = points[[0, -1]]
    line, line_rmse = None, None
    if not closed and np.linalg.norm(b-a) >= 12:
        line = {'kind': 'line', 'points_px': [a.tolist(), b.tolist()]}
        error = evidence_error(points, line)
        line_rmse = error['rmse']
        if error['p95'] > 1.25 or error['maximum'] > 2 or not native_support(to_input(line, matrix), ink):
            line = None
        else:
            line['fit_error_native_px'] = error
    # This fit receives native coordinates, so all robust/sensitivity settings are native.
    max_radius = max(ink.shape) * float(matrix[0][0]) * 4
    solution = circle_solution(points, closed, max_radius, a if closed and protected_anchor else None)
    uncertain = False
    if solution is not None:
        center, radius, sensitivity = solution
        angles = np.unwrap(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))
        delta = float(angles[-1]-angles[0])
        sweep = 2*math.pi if closed else abs(delta)
        start = float(angles[0] if delta > 0 else angles[-1])
        curve = {'kind': 'circle' if closed else 'arc', 'center_px': center.tolist(), 'radius_px': radius}
        if not closed:
            curve.update(start_angle_deg=math.degrees(start) % 360, sweep_angle_deg=math.degrees(sweep))
        radial = np.abs(np.linalg.norm(points-center, axis=1)-radius)
        radial_rmse = float(np.sqrt(np.mean(radial**2)))
        eligible = closed or (math.radians(20) <= sweep < 2*math.pi-1e-7
                              and radius*(1-math.cos(min(sweep, math.pi)/2)) >= 1
                              and (line is None or radial_rmse <= line_rmse*.65))
        if eligible and np.percentile(radial, 95) <= 1.25 and radial.max() <= 2:
            error = evidence_error(points, curve)
            supported = error['p95'] <= 1.25 and error['maximum'] <= 2 and native_support(to_input(curve, matrix), ink)
            if supported and sensitivity <= 2:
                curve.update(fit_error_native_px=error, parameter_sensitivity_native_px=sensitivity,
                             uncertainty_calibrated=False, radial_resolution_native_px=.5,
                             closed_anchor_fixed=bool(closed and protected_anchor))
                return curve, None
            uncertain = supported and sensitivity > 2
    if line is not None:
        return line, None
    return None, 'uncertain_curve_no_fragmentation' if uncertain else 'no_supported_model'


def cut_candidates(points):
    if len(points) < 7:
        return []
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    chosen = []
    for fraction in np.linspace(.15, .85, 7):
        chosen.append(int(np.searchsorted(length, length[-1]*fraction)))
    angles = []
    for i in range(2, len(points)-2):
        left = max(0, int(np.searchsorted(length, length[i]-6)))
        right = min(len(points)-1, int(np.searchsorted(length, length[i]+6)))
        u, v = points[i]-points[left], points[right]-points[i]
        denominator = np.linalg.norm(u)*np.linalg.norm(v)
        angle = math.degrees(math.acos(float(np.clip(u@v/denominator, -1, 1)))) if denominator > 0 else 0.
        angles.append((angle, i))
    chosen.extend(i for _, i in sorted(angles, reverse=True)[:2])
    return sorted({i for i in chosen if length[i] >= 12 and length[-1]-length[i] >= 12})[:9]


def partition(points, closed, ink, matrix, account, *, protected_anchor=False):
    whole, reason = fit_native(points, closed, ink, matrix, account, protected_anchor=protected_anchor)
    if whole is not None:
        return [(0, len(points)-1, whole)], None
    if closed or reason in ('uncertain_curve_no_fragmentation', 'fit_budget'):
        return [], reason
    best = None
    for i in cut_candidates(points):
        left, _ = fit_native(points[:i+1], False, ink, matrix, account)
        right, _ = fit_native(points[i:], False, ink, matrix, account)
        if left is None or right is None:
            continue
        if left['kind'] == right['kind'] == 'line':
            u, v = np.diff(left['points_px'], axis=0)[0], np.diff(right['points_px'], axis=0)[0]
            angle = math.degrees(math.acos(float(np.clip(u@v/(np.linalg.norm(u)*np.linalg.norm(v)), -1, 1))))
            if angle < 35:
                continue  # A shallow/uncertain arc cannot become a collection of short LINEs.
        cost = left['fit_error_native_px']['rmse'] + right['fit_error_native_px']['rmse']
        if best is None or cost < best[0]:
            best = (cost, [(0, i, left), (i, len(points)-1, right)])
    # No recursive bisection fallback: unresolved source is retained as one addressable interval.
    return (best[1], None) if best else ([], reason)


def joint_nodes(source, matrix, ink, *, enabled):
    endpoints = defaultdict(list)
    native_paths = {c['path_id']: transform_points(c['points_px'], matrix) for c in source}
    for c in source:
        p = native_paths[c['path_id']]
        if len(p) == 1:
            endpoints[tuple(p[0])].append((c['path_id'], 0))
        else:
            endpoints[tuple(p[0])].append((c['path_id'], 0))
            endpoints[tuple(p[-1])].append((c['path_id'], -1))
    records = []
    for index, (point, incident) in enumerate(sorted(endpoints.items())):
        origin = np.asarray(point)
        proposal = origin.copy()
        equations, rhs = [], []
        # Only structural shared nodes are movable; open/isolated endpoints stay exact.
        if enabled and len(incident) >= 3:
            for owner, end in incident:
                p = native_paths[owner] if end == 0 else native_paths[owner][::-1]
                arclength = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
                local = p[(arclength >= 6) & (arclength <= 18)]
                if len(local) < 4:
                    continue
                center = local.mean(axis=0)
                _, _, axes = np.linalg.svd(local-center, full_matrices=False)
                normal = axes[1]
                if np.max(np.abs((local-center)@normal)) > .75:
                    continue
                equations.append(normal)
                rhs.append(float(normal@center))
            if len(equations) >= 2 and np.linalg.svd(equations, compute_uv=False)[-1] >= .25:
                a = np.r_[np.asarray(equations), np.eye(2)*.5]
                b = np.r_[rhs, origin*.5]
                candidate = np.linalg.lstsq(a, b, rcond=None)[0]
                # Backtracking into the registered native uncertainty region and observed ink.
                for attempt in range(8):
                    proposed = origin + (candidate-origin)/2**attempt
                    if np.linalg.norm(proposed-origin) > 1:
                        continue
                    q = np.rint((proposed-np.asarray(matrix)[:2, 2])/matrix[0][0]).astype(int)
                    if 0 <= q[0] < ink.shape[1] and 0 <= q[1] < ink.shape[0] and ink[q[1], q[0]]:
                        proposal = proposed
                        break
        for owner, end in incident:
            native_paths[owner][end] = proposal
        records.append({'id': f'shared-node-{index:05d}', 'observed_native_px': origin.tolist(),
                        'estimated_native_px': proposal.tolist(), 'incidences': [[o, e] for o, e in incident],
                        'degree_in_source_paths': len(incident), 'validated_endpoint_fixed': len(incident) < 3,
                        'movement_native_px': float(np.linalg.norm(proposal-origin)),
                        'search_radius_native_px': 1. if len(incident) >= 3 else 0.,
                        'calibrated_uncertainty': False})
    return native_paths, records


def extract(ink, skeleton, *, input_to_native, move_nodes):
    if (ink.dtype != np.uint8 or skeleton.dtype != np.uint8 or ink.shape != skeleton.shape
            or ink.ndim != 2 or not 0 < ink.size <= 4194304):
        raise ValueError('Bounded aligned uint8 ink/skeleton required')
    if np.any((skeleton > 0) & (ink == 0)):
        raise ValueError('Skeleton outside observed ink')
    native({'kind': 'line', 'points_px': [[0, 0], [1, 1]]}, input_to_native)
    source, stats = source_paths(skeleton, np.eye(3))
    native_paths, nodes = joint_nodes(source, input_to_native, ink, enabled=move_nodes)
    account = {'calls': 0, 'incomplete': False}
    output, intervals = [], []
    incidence = defaultdict(int)
    for c in source:
        if len(c['points_px']) > 1:
            incidence[tuple(c['points_px'][0])] += 1
            incidence[tuple(c['points_px'][-1])] += 1
    for c in source:
        points = native_paths[c['path_id']]
        parts, reason = partition(points, c['closed'], ink, input_to_native, account,
                                  protected_anchor=c['closed'] and incidence[tuple(c['points_px'][0])] > 2)
        if not parts:
            parts = [(0, len(points)-1, {'kind': 'polyline', 'points_px': points.tolist(),
                                        'closed': c['closed'], 'reason': reason})]
        for low, high, shape in parts:
            section = f'{c["path_id"]}:{low}-{high}'
            shape = to_input(shape, input_to_native)
            shape.update(id=f'{"joint" if move_nodes else "native-fixed"}-{len(output):05d}',
                         path_id=c['path_id'], source_observation_id=c['id'], interval_id=section,
                         source_index_interval=[low, high], support_points=c['points_px'][low:high+1],
                         classification='unclassified', semantic_role='unknown', semantic_ownership_confirmed=False,
                         physical_geometry_approved=False, review_status='unresolved' if shape['kind'] == 'polyline' else 'proposal_requires_review')
            output.append(shape)
            intervals.append({'id': section, 'source_path_id': c['path_id'], 'primitive_id': shape['id'],
                              'source_index_interval': [low, high], 'ownership': 'unknown'})
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    for c in output:
        c['local_width'] = width_evidence(c, ink, dt)
    return {'candidates': output, 'source': source, 'nodes': nodes, 'intervals': intervals,
            'analysis': {**stats, 'fit_calls': account['calls'], 'search_incomplete': account['incomplete'],
                         'maximum_parts_per_source_path': max((sum(v['path_id'] == c['path_id'] for v in output) for c in source), default=0),
                         'source_pixels_erased': 0, 'semantic_ownership_confirmed': False}}


def select_guarded(source, proposal, matrix, *, graphs=None):
    before = [native(c, matrix) for c in source]
    after = [native(c, matrix) for c in proposal['candidates']]
    decision = guard(before, after, graphs=graphs)['decision']
    accepted = decision['accepted'] and not proposal['analysis']['search_incomplete']
    return {'candidates': copy.deepcopy(proposal['candidates'] if accepted else source),
            'decision': decision, 'accepted': accepted, 'exact_source_fallback': not accepted,
            'production_admitted': False, 'semantic_ownership_confirmed': False}
