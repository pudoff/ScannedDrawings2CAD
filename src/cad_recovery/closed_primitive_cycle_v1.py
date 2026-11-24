"""Close an already observed primitive cycle, never a missing angular sector.

Only exact source endpoints are traversed. Confirmed LINEs, protected pixels,
semantic conflicts and ambiguous branches cannot be consumed by this pass.
"""
from collections import Counter, defaultdict
from copy import deepcopy

import cv2
import numpy as np

from .cad_primitive_fit_v2 import checked, POLICY as FIT_POLICY
from .anchored_curve_fit_v1 import circle_solution
from .part_admission_v1 import object_sha

POLICY = dict(version='closed-primitive-cycle/1.0', maximum_objects=16384,
    maximum_arc_seeds=256, maximum_cycle_objects=256,
    connectors='unresolved open source walks shorter than existing minimum typed length',
    endpoints='exact observed source pixel; no tolerance or added bridge',
    parameter_error_px=FIT_POLICY['max_px'], physical_geometry_approved=False)


def ledger(objects):
    return Counter((member['candidate_id'], edge) for obj in objects
        for member in obj['source_members'] for edge in range(*member['source_index_interval']))


def _eligible(obj, center, radius, protected):
    if obj['kind'] not in ('arc', 'polyline') or obj.get('closed'):
        return False
    p = np.asarray(obj['support_points'], float)
    if len(p) < 2 or np.array_equal(p[0], p[-1]):
        return False
    if obj['kind'] == 'polyline' and np.linalg.norm(np.diff(p, axis=0), axis=1).sum() >= FIT_POLICY['minimum_length_px']:
        return False
    if np.any(np.max(np.abs(np.diff(p, axis=0)), axis=1) != 1):
        return False
    q = p.astype(int)
    if not np.array_equal(p, q) or protected[q[:, 1], q[:, 0]].any():
        return False
    radial = np.abs(np.linalg.norm(p-center, axis=1)-radius)
    if radial.max() > FIT_POLICY['max_px'] or np.percentile(radial, 95) > FIT_POLICY['p95_px']:
        return False
    if obj['kind'] == 'arc':
        if (not obj.get('locally_stable', True)
                or np.linalg.norm(np.asarray(obj['center_px'])-center) > FIT_POLICY['max_px']
                or abs(obj['radius_px']-radius) > FIT_POLICY['max_px']):
            return False
    return True


def _walk(seed, objects, eligible):
    ends = defaultdict(list)
    for i in eligible:
        p = objects[i]['support_points']
        ends[tuple(p[0])].append((i, False)); ends[tuple(p[-1])].append((i, True))
    points = list(objects[seed]['support_points']); order = [seed]; used = {seed}
    start, node = tuple(points[0]), tuple(points[-1])
    while node != start:
        options = [(i, reverse) for i, reverse in ends[node] if i not in used]
        if len(options) != 1 or len(order) >= POLICY['maximum_cycle_objects']:
            return None
        i, reverse = options[0]
        p = objects[i]['support_points'][::-1] if reverse else objects[i]['support_points']
        order.append(i); used.add(i); points.extend(p[1:]); node = tuple(p[-1])
    if sum(objects[i]['kind'] == 'arc' for i in order) < 2:
        return None
    p = np.asarray(points, np.int32)
    if len(set(map(tuple, p[:-1]))) != len(p)-1:  # No figure-eight or repeated walk.
        return None
    return order, p


def apply(assembly, gray, *, roles, protected):
    if (gray.dtype != np.uint8 or gray.ndim != 2 or gray.size > 4194304
            or roles.shape != gray.shape or roles.dtype != np.uint8
            or protected.shape != gray.shape or protected.dtype != bool):
        raise ValueError('Bounded aligned grayscale, role and protection arrays required')
    before = object_sha(assembly); result = deepcopy(assembly); objects = result['objects']
    if len(objects) > POLICY['maximum_objects']:
        raise ValueError('Primitive cycle object budget exceeded')
    original = ledger(objects); blocked = protected | (roles == 6)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Same support envelope and fitting thresholds as the previous assembler.
    support = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    seeds = [i for i, obj in enumerate(objects) if obj['kind'] == 'arc']
    complete = len(seeds) <= POLICY['maximum_arc_seeds']
    consumed = set(); replacements = []; decisions = []
    for seed in seeds[:POLICY['maximum_arc_seeds']]:
        if seed in consumed:
            continue
        obj = objects[seed]; center = np.asarray(obj['center_px']); radius = obj['radius_px']
        eligible = [i for i, item in enumerate(objects) if i not in consumed and _eligible(item, center, radius, blocked)]
        if seed not in eligible:
            continue
        walk = _walk(seed, objects, eligible)
        if walk is None:
            continue
        order, points = walk
        known = {objects[i]['role'] for i in order} & set(range(1, 10))
        q = points.astype(int)
        known |= set(map(int, roles[q[:, 1], q[:, 0]])) & set(range(1, 10))
        if len(known) > 1:
            continue
        # A connected observed cycle must also make one monotonic full turn.
        angles = np.unwrap(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))
        if not np.isclose(abs(angles[-1]-angles[0]), 2*np.pi, atol=1e-6):
            continue
        progress = np.diff(angles)*np.sign(angles[-1]-angles[0])
        if (progress < -1e-9).any():
            continue
        solution = circle_solution(points, True, max(gray.shape)*4)
        if solution is None or solution[2] > FIT_POLICY['sensitivity_px']:
            continue
        fitted_center, fitted_radius, sensitivity = solution
        error = np.abs(np.linalg.norm(points-fitted_center, axis=1)-fitted_radius)
        if error.max() > FIT_POLICY['max_px'] or np.percentile(error, 95) > FIT_POLICY['p95_px']:
            continue
        model = checked(dict(kind='circle', center_px=fitted_center.tolist(), radius_px=float(fitted_radius),
            fit_max_px=float(error.max()), fit_p95_px=float(np.percentile(error, 95)),
            parameter_sensitivity_native_px=float(sensitivity),
            parameter_evidence='exact_closed_source_cycle_with_independent_ARC_sectors'), points, support)
        if model is None:
            continue
        members = [member for i in order for member in objects[i]['source_members']]
        evidence = dict(object_ids=[objects[i]['id'] for i in order],
            observed_closed_cycle=True, added_source_edges=0, absorbed_LINEs=0,
            independent_ARC_sectors=sum(objects[i]['kind'] == 'arc' for i in order))
        replacement = dict(**model, id='cad-closed-cycle/'+object_sha(dict(members=members, geometry=model))[:24],
            source_members=members, path_ids=list(dict.fromkeys(path for i in order for path in objects[i]['path_ids'])),
            support_points=points.tolist(), source_junction_links=[link for i in order for link in objects[i].get('source_junction_links', [])],
            circle_cycle_evidence=evidence, role=next(iter(known)) if known else 255,
            status='draft_requires_review', semantic_ownership_confirmed=False, physical_geometry_approved=False)
        replacements.append(replacement); consumed.update(order); decisions.append(dict(id=replacement['id'], **evidence))
    result['objects'] = [obj for i, obj in enumerate(objects) if i not in consumed]+replacements
    if original != ledger(result['objects']) or object_sha(assembly) != before:
        raise ValueError('Closed primitive cycle changed source-edge partition or input')
    result.update(object_count=len(result['objects']), kind_counts=dict(Counter(obj['kind'] for obj in result['objects'])),
        closed_primitive_cycles=dict(policy=POLICY, decisions=decisions, search_complete=complete),
        search_complete=assembly['search_complete'] and complete)
    return result
