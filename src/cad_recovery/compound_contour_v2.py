"""Bounded reciprocal continuation of observed paths; cycles confer no role."""
from collections import defaultdict
from copy import deepcopy
import math
import cv2
import numpy as np
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'compound-contour/2.0', 'maximum_edges': 4096,
          'maximum_source_points': 2000000, 'maximum_transition_checks': 250000,
          'maximum_walk_steps': 20000, 'maximum_output_points': 250000,
          'maximum_cycles': 512, 'maximum_width_ratio': 1.67,
          'minimum_turn_margin_deg': 15., 'maximum_turn_deg': 150.,
          'tangent_span_in_pens': 3., 'endpoint_join': 'exact_observed_coordinate',
          'role_or_dimension_required': False, 'geometry_modified': False,
          'semantic_PART_approved': False, 'physical_geometry_approved': False}


def assemble(gray, graph, *, source_paths=None):
    checked_gray(gray); before = object_sha([graph, source_paths]); gray_sha = sha(gray)
    proposals = graph['proposals']; lookup = {p['id']: n for n, p in enumerate(proposals, 1)}
    if len(lookup) != len(proposals): raise ValueError('Unique current interval IDs required')
    if len(proposals) > POLICY['maximum_edges']: raise ValueError('Bounded current interval domain required')
    if source_paths is None:
        records = [{'id': p['id'], 'points_px': p['points_px'], 'owner_numbers': [n]} for n, p in enumerate(proposals, 1)]
    else:
        if len(source_paths) > POLICY['maximum_edges']: raise ValueError('Bounded observed path domain required')
        members = defaultdict(list)
        for n, p in enumerate(proposals, 1): members[p.get('source_observation_id')].append(n)
        if len({p['id'] for p in source_paths}) != len(source_paths): raise ValueError('Unique source paths required')
        records = [{'id': p['id'], 'points_px': p['points_px'], 'owner_numbers': members[p['id']]} for p in source_paths if members[p['id']]]
    ink = foreground(gray); distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    audit = {'limits': [], 'source_points': 0, 'transition_checks': 0, 'walk_steps': 0, 'output_points': 0}
    rejected, edges, closed, incidence = [], [], [], defaultdict(list)
    if graph.get('search_incomplete'): audit['limits'].append('incomplete_source_graph')
    for record in records if not audit['limits'] else []:
        points = np.asarray(record['points_px'], float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 1 or not np.isfinite(points).all():
            raise ValueError('Finite observed path required')
        q = np.rint(points).astype(int)
        if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any():
            raise ValueError('Observed path outside raster')
        audit['source_points'] += len(points)
        if audit['source_points'] > POLICY['maximum_source_points']:
            audit['limits'].append('source_point_budget'); break
        if len(points) == 1:
            rejected.append({'path': record['id'], 'reason': 'isolated_point_is_not_an_edge', 'point_px': points[0].tolist()})
            continue
        samples = distance[q[:, 1], q[:, 0]]; values = samples[samples > 0]
        if len(values) < 2 or np.mean(ink[q[:, 1], q[:, 0]] > 0) < .95:
            rejected.append({'path': record['id'], 'reason': 'insufficient_observed_ink'}); continue
        pen = max(1., float(2*np.quantile(values, .25)-1))
        edge = {**record, 'points': points, 'pen': pen}
        if np.array_equal(points[0], points[-1]):
            if len(points) >= 4: closed.append(edge)
            continue
        index = len(edges); edges.append(edge)
        incidence[tuple(points[0])].append((index, 1)); incidence[tuple(points[-1])].append((index, -1))
    tangents = {}
    for i, edge in enumerate(edges):
        for sign in (1, -1):
            p = edge['points'] if sign == 1 else edge['points'][::-1]
            cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            j = min(len(p)-1, max(1, int(np.searchsorted(cumulative, max(2., POLICY['tangent_span_in_pens']*edge['pen'])))))
            vector = p[j]-p[0]; norm = float(np.linalg.norm(vector))
            tangents[i, sign] = vector/norm if norm > 0 else None
    raw_next, transitions = {}, []
    for i, edge in enumerate(edges) if not audit['limits'] else []:
        for sign in (1, -1):
            end = tuple(edge['points'][-1] if sign == 1 else edge['points'][0]); incoming = tangents[i, -sign]
            options = []
            for other, direction in incidence[end]:
                if other == i: continue
                audit['transition_checks'] += 1
                if audit['transition_checks'] > POLICY['maximum_transition_checks']:
                    audit['limits'].append('transition_budget'); break
                ratio = edges[other]['pen']/edge['pen']; tangent = tangents[other, direction]
                if incoming is None or tangent is None or not 1/POLICY['maximum_width_ratio'] <= ratio <= POLICY['maximum_width_ratio']: continue
                turn = math.degrees(math.acos(float(np.clip((-incoming) @ tangent, -1., 1.))))
                if turn <= POLICY['maximum_turn_deg']: options.append(((other, direction), turn))
            if audit['limits']: break
            options.sort(key=lambda item: item[1]); selected = None
            if options and (len(options) == 1 or options[1][1]-options[0][1] >= POLICY['minimum_turn_margin_deg']): selected = options[0][0]
            raw_next[i, sign] = selected
            transitions.append({'from': [i, sign], 'node': list(end), 'alternatives': [{'half': list(h), 'turn_deg': a} for h, a in options], 'selected': list(selected) if selected else None})
        if audit['limits']: break
    following = {}
    for half, nxt in raw_next.items():
        following[half] = nxt if nxt is not None and raw_next.get((nxt[0], -nxt[1])) == (half[0], -half[1]) else None
    for transition in transitions:
        nxt = following.get(tuple(transition['from']))
        transition['reciprocal_selected'] = list(nxt) if nxt is not None else None
    done, cycles, signatures = set(), [], set()
    def add_cycle(route, direct=None):
        chosen = [direct] if direct is not None else [edges[i] for i, _ in route]
        identity = tuple(sorted(e['id'] for e in chosen))
        if identity in signatures: return
        pieces = [direct['points']] if direct is not None else [edges[i]['points'] if s == 1 else edges[i]['points'][::-1] for i, s in route]
        polygon = np.concatenate([pieces[0]]+[p[1:] for p in pieces[1:]])
        if not np.array_equal(polygon[0], polygon[-1]): raise ValueError('Observed cycle did not close exactly')
        if len(polygon) < 4 or abs(cv2.contourArea(polygon.astype(np.float32))) == 0: return
        audit['output_points'] += len(polygon)
        if audit['output_points'] > POLICY['maximum_output_points'] or len(cycles) >= POLICY['maximum_cycles']:
            audit['limits'].append('cycle_output_budget'); return
        signatures.add(identity)
        cycles.append({'id': 'observed-cycle-'+str(len(cycles)), 'source_path_ids': list(identity),
                       'source_owner_numbers': sorted({n for e in chosen for n in e['owner_numbers']}),
                       'points_px': polygon.tolist(), 'component_path_count': len(chosen),
                       'closure': 'exact_shared_observed_nodes', 'requires_semantic_and_topology_QA': True})
    for edge in closed if not audit['limits'] else []: add_cycle([], direct=edge)
    for start in list(following) if not audit['limits'] else []:
        if start in done: continue
        path, positions, current = [], {}, start
        while current is not None and current not in done and current not in positions:
            audit['walk_steps'] += 1
            if audit['walk_steps'] > POLICY['maximum_walk_steps']:
                audit['limits'].append('walk_budget'); break
            positions[current] = len(path); path.append(current); current = following.get(current)
        if audit['limits']: break
        if current in positions:
            route = path[positions[current]:]
            nodes = [tuple(edges[i]['points'][0] if s == 1 else edges[i]['points'][-1]) for i, s in route]
            if len({i for i, _ in route}) == len(route) and len(set(nodes)) == len(nodes): add_cycle(route)
            else: rejected.append({'reason': 'repeated_edge_or_node_cycle', 'route': [list(h) for h in route]})
        done.update(path)
        if audit['limits']: break
    complete = not audit['limits']
    if not complete: cycles = []
    immutable = before == object_sha([graph, source_paths]) and gray_sha == sha(gray)
    if not immutable: raise ValueError('Compound search mutated source observations')
    descriptors = [{'id': e['id'], 'source_owner_numbers': e['owner_numbers'], 'observed_pen_px': e['pen'],
                    'endpoints_px': [e['points'][0].tolist(), e['points'][-1].tolist()]} for e in edges]
    return {'cycles': cycles, 'edges': descriptors, 'transitions': transitions, 'rejected': rejected, 'audit': audit,
            'complete': complete, 'source_immutable': immutable, 'policy': deepcopy(POLICY),
            'inherited_boundary_rejections': deepcopy(graph.get('boundaries', {}).get('rejected', [])),
            'roles_assigned': 0, 'geometry_modified': False, 'physical_geometry_approved': False}
