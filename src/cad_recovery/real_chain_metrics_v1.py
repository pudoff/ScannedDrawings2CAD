"""Read-only local real-drawing diagnostics; targets are evaluation inputs only."""
from collections import deque
import math
import cv2
import numpy as np

POLICY = {'tip_tolerance_native_px': 8., 'direction_tolerance_deg': 20.,
          'foot_tolerance_native_px': 8., 'maximum_matching_edge_visits': 2000000,
          'maximum_targets': 64, 'maximum_candidates': 512,
          'full_page_precision': None, 'physical_geometry_approved': False}


def inside(point, boxes):
    return any(x <= point[0] < r and y <= point[1] < b for x, y, r, b in boxes)


def point_native(point, matrix):
    return (np.asarray(matrix, float) @ [*point, 1.])[:2]


def assignment(left_count, right_count, costs, maximum_visits=2000000):
    """Maximum cardinality, minimum cost via bounded residual shortest paths."""
    if left_count > 64 or right_count > 512:
        raise ValueError('Matching object budget')
    count = left_count + right_count + 2; source = count - 2; sink = count - 1
    graph = [[] for _ in range(count)]
    def edge(a, b, cost):
        forward = [b, len(graph[b]), 1, float(cost)]
        reverse = [a, len(graph[a]), 0, -float(cost)]
        graph[a].append(forward); graph[b].append(reverse)
        return forward
    for a in range(left_count): edge(source, a, 0)
    for b in range(right_count): edge(left_count + b, sink, 0)
    references = {}
    for (a, b), cost in sorted(costs.items()):
        if not (0 <= a < left_count and 0 <= b < right_count and math.isfinite(cost) and cost >= 0):
            raise ValueError('Finite nonnegative matching costs required')
        references[a, b] = edge(a, left_count + b, cost)
    visits = 0
    while True:
        distance = [math.inf] * count; distance[source] = 0
        previous = [None] * count; queue = deque([source]); queued = {source}
        while queue:
            a = queue.popleft(); queued.remove(a)
            for index, e in enumerate(graph[a]):
                visits += 1
                if visits > maximum_visits:
                    return {'pairs': [], 'incomplete': True, 'edge_visits': visits}
                b, _, capacity, cost = e
                if capacity and distance[a] + cost < distance[b] - 1e-12:
                    distance[b] = distance[a] + cost; previous[b] = (a, index)
                    if b not in queued: queue.append(b); queued.add(b)
        if previous[sink] is None: break
        b = sink
        while b != source:
            a, index = previous[b]; e = graph[a][index]
            e[2] -= 1; graph[b][e[1]][2] += 1; b = a
    return {'pairs': [(a, b) for (a, b), e in references.items() if not e[2]],
            'incomplete': False, 'edge_visits': visits}


def transformed_heads(heads, matrix):
    if len(heads) > 512 or len({h['id'] for h in heads}) != len(heads):
        raise ValueError('Unique bounded head IDs required')
    matrix = np.asarray(matrix, float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError('Finite affine transform required')
    result = []
    for h in heads:
        point = point_native(h['tip_px'], matrix); direction = matrix[:2, :2] @ h['direction']
        norm = np.linalg.norm(direction)
        if point.shape != (2,) or direction.shape != (2,) or not np.isfinite([*point, *direction]).all() or norm <= 0:
            raise ValueError('Finite directed head required')
        result.append({'id': h['id'], 'tip_px': point.tolist(), 'direction': (direction / norm).tolist()})
    return result


def head_metrics(review, heads, matrix):
    refs = sorted(review['heads'], key=lambda h: h['id'])
    if len(refs) > 64 or len({h['id'] for h in refs}) != len(refs): raise ValueError('Unique bounded targets required')
    candidates = sorted([h for h in transformed_heads(heads, matrix) if inside(h['tip_px'], review['head_scopes'])], key=lambda h: h['id'])
    costs = {}; geometry = {}
    for i, ref in enumerate(refs):
        direction = np.asarray(ref['direction'], float); norm = np.linalg.norm(direction)
        if not np.isfinite([*ref['tip_px'], *direction]).all() or not math.isclose(float(norm), 1., abs_tol=1e-6):
            raise ValueError('Finite unit reference direction required')
        if not inside(ref['tip_px'], review['head_scopes']): raise ValueError('Target outside reviewed scope')
        for j, candidate in enumerate(candidates):
            error = float(np.linalg.norm(np.array(candidate['tip_px']) - ref['tip_px']))
            angle = math.degrees(math.acos(float(np.clip(np.array(candidate['direction']) @ direction, -1, 1))))
            if error <= 8. and angle <= 20.:
                costs[i, j] = (error / 8.) ** 2 + (angle / 20.) ** 2
                geometry[i, j] = {'tip_error_native_px': error, 'direction_error_deg': angle}
    matched = assignment(len(refs), len(candidates), costs)
    pairs = [{'target_id': refs[i]['id'], 'candidate_id': candidates[j]['id'], **geometry[i, j]} for i, j in matched['pairs']]
    tp = len(pairs); fp = len(candidates) - tp; fn = len(refs) - tp
    return {'tp': tp, 'fp': fp, 'fn': fn, 'target_count': len(refs), 'scoped_candidates': len(candidates),
            'precision': tp / (tp + fp) if tp + fp else None, 'recall': tp / (tp + fn) if tp + fn else None,
            'pairs': pairs, 'search_incomplete': matched['incomplete'], 'edge_visits': matched['edge_visits'],
            'full_page_precision': None, 'outside_scope_candidates_unscored': len(heads) - len(candidates)}


def distance_segment(points, segment):
    points = np.asarray(points, float); segment = np.asarray(segment, float)
    start, end = segment; delta = end - start; squared = float(delta @ delta)
    if squared <= 0: raise ValueError('Nonzero shaft reference required')
    t = np.clip((points - start) @ delta / squared, 0., 1.)
    return np.linalg.norm(points - (start + t[:, None] * delta), axis=1)


def chain_metrics(review, heads, matrix, packet, previous, trace, matched_heads):
    transformed = {h['id']: h for h in transformed_heads(heads, matrix)}
    tracks = {t['id']: t for t in previous['tracks']}
    lookup = {p['target_id']: p['candidate_id'] for p in matched_heads['pairs']}
    targets = review['chains']; scopes = [t['head_scope'] for t in targets]
    scoped = []
    for index, c in enumerate(packet['dimension_chains']):
        tips = [transformed[h]['tip_px'] for h in c['head_ids'] if h in transformed]
        if len(tips) >= 2 and any(all(inside(p, [scope]) for p in tips) for scope in scopes):
            scoped.append((index, c))
    costs = {}; evidence = {}; diagnostics = []
    for i, target in enumerate(targets):
        ids = [lookup[h] for h in target['head_ids'] if h in lookup]
        common = [t for t, contacts in trace['contacts'].items() if len(ids) == 2 and set(ids) <= set(contacts)]
        closest = []
        for t in previous['tracks']:
            p = [point_native(v, matrix) for v in t['points']]
            error = float(distance_segment(target['shaft'], p).max())
            closest.append({'track_id': t['id'], 'max_reference_to_segment_native_px': error,
                            'contact_head_ids': trace['contacts'].get(t['id'], [])})
        feet = []
        for boundary in trace['eligible_boundaries']:
            polygon = np.asarray([point_native(v, matrix) for v in boundary['polygon']], np.float32)
            distances = [abs(cv2.pointPolygonTest(polygon, tuple(map(float, p)), True)) for p in target['feet_px']]
            feet.append({'boundary_id': boundary['id'], 'foot_distances_native_px': distances})
        for j, (_, c) in enumerate(scoped):
            if len(ids) != 2 or set(c['head_ids']) != set(ids) or c['shaft_track'] not in tracks: continue
            track = [point_native(v, matrix) for v in tracks[c['shaft_track']]['points']]
            shaft_error = float(distance_segment(target['shaft'], track).max())
            predicted = np.asarray([point_native(v, matrix) for v in c['feet_px']])
            if predicted.shape != (2, 2): continue
            expected = np.asarray(target['feet_px'])
            error = min(float(np.linalg.norm(predicted - expected, axis=1).max()),
                        float(np.linalg.norm(predicted[::-1] - expected, axis=1).max()))
            if shaft_error <= 8. and error <= 8.:
                costs[i, j] = (shaft_error / 8.) ** 2 + (error / 8.) ** 2
                evidence[i, j] = {'shaft_error_native_px': shaft_error, 'max_foot_error_native_px': error}
        reason = ('missing_detected_head' if len(ids) < 2 else
                  'no_common_head_to_shaft_contact' if not common else 'extension_or_reference_boundary_not_accepted')
        diagnostics.append({'target_id': target['id'], 'matched_head_ids': ids, 'common_shaft_tracks': common,
                            'closest_tracks': sorted(closest, key=lambda x: (x['max_reference_to_segment_native_px'], x['track_id']))[:3],
                            'closest_reference_boundaries': sorted(feet, key=lambda x: max(x['foot_distances_native_px']))[:3],
                            'first_unmet_observation': reason, 'topology': target['topology'],
                            'reference_kinds': target['reference_kinds'], 'whole_part_contour_visible': target['whole_part_contour_visible'],
                            'causal_counterfactual_proven': False})
    matched = assignment(len(targets), len(scoped), costs)
    pairs = [{'target_id': targets[i]['id'], 'proposal_index': scoped[j][0], **evidence[i, j]} for i, j in matched['pairs']]
    accepted = {p['target_id'] for p in pairs}
    for d in diagnostics:
        if d['target_id'] in accepted: d['first_unmet_observation'] = None
    tp = len(pairs); fp = len(scoped) - tp; fn = len(targets) - tp
    return {'tp': tp, 'fp': fp, 'fn': fn, 'target_count': len(targets), 'scoped_proposals': len(scoped),
            'precision': tp / (tp + fp) if tp + fp else None, 'recall': tp / (tp + fn) if tp + fn else None,
            'pairs': pairs, 'diagnostics': diagnostics, 'search_incomplete': matched['incomplete'],
            'outside_scope_proposals_unscored': len(packet['dimension_chains']) - len(scoped),
            'full_page_precision': None}
