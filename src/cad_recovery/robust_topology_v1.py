"""Float64 boundary certificates atop the unchanged analytic incidence kernel."""
from collections import Counter
import math

import numpy as np

from . import primitive_graph_qa_v1 as kernel
from .cad_validation_topology_v1 import loop_data

POLICY = {'version': 'robust-topology/1.0', 'maximum_flat_vertices': 512,
          'maximum_topology_predicates': 250000, 'maximum_nesting_checks': 100000,
          'numeric_native_px_min': 1e-7, 'float64_envelope_multiplier': 64.,
          'contact_policy': 'unresolved, never strict containment',
          'non_simple_policy': 'unresolved', 'physical_geometry_approved': False}


class Unresolved(ValueError):
    pass


class Budget:
    def __init__(self):
        self.used = 0

    def spend(self, count):
        if self.used + count > 250000:
            raise Unresolved('topology_predicate_budget')
        self.used += count


def numeric_envelope(points):
    return max(1e-7, 64*np.finfo(np.float64).eps*max(1., float(np.abs(points).max())))


def segment_distances(a, b, c, d):
    """Vectorized float64 finite segment distances; ambiguous crossings fail closed."""
    origin = np.asarray(a)
    a, b, c, d = np.zeros(2), np.asarray(b)-origin, np.asarray(c)-origin, np.asarray(d)-origin

    def point_edge(p, q, r):
        delta = r-q
        denominator = np.sum(delta*delta, axis=-1)
        t = np.clip(np.sum((p-q)*delta, axis=-1)/np.maximum(denominator, 1e-300), 0, 1)
        return np.linalg.norm(p-q-t[..., None]*delta, axis=-1)

    distances = np.minimum.reduce([point_edge(a, c, d), point_edge(b, c, d),
                                   point_edge(c, a, b), point_edge(d, a, b)])

    def cross(u, v):
        return u[..., 0]*v[..., 1]-u[..., 1]*v[..., 0]

    p, q = cross(b-a, c-a), cross(b-a, d-a)
    r, s = cross(d-c, a-c), cross(d-c, b-c)
    bound = 64*np.finfo(float).eps*max(1., float(np.max(np.abs(np.r_[b[None], c, d])))**2)
    overlap = np.all(np.maximum(np.minimum(a, b), np.minimum(c, d)) <=
                     np.minimum(np.maximum(a, b), np.maximum(c, d)) + math.sqrt(bound), axis=1)
    opposite = ((p > bound) & (q < -bound) | (q > bound) & (p < -bound)) & ((r > bound) & (s < -bound) | (s > bound) & (r < -bound))
    uncertain = (np.abs(p) <= bound) | (np.abs(q) <= bound) | (np.abs(r) <= bound) | (np.abs(s) <= bound)
    return np.where(overlap & (opposite | uncertain), 0., distances)


def certify_simple(flat, chord, numeric, budget):
    p = np.asarray(flat, float)
    q = np.roll(p, -1, axis=0)
    for i in range(len(p)):
        indices = np.arange(i+2, len(p))
        if i == 0:
            indices = indices[indices != len(p)-1]
        if not len(indices):
            continue
        budget.spend(len(indices))
        if np.any(segment_distances(p[i], q[i], p[indices], q[indices]) <= 2*chord+numeric):
            raise Unresolved('self_intersection_or_unresolved_curve_clearance')


def face_boundaries(nodes, edges, adjacency, budget):
    def tangent(half):
        index, sign = half
        edge = edges[index]
        segment = edge['segment']
        t = edge['low'] if sign == 1 else edge['high']
        if segment['kind'] == 'line':
            d = sign*(segment['b']-segment['a'])
        else:
            angle = segment['start']+t*segment['sweep']
            d = sign*np.array([-math.sin(angle), math.cos(angle)])
        return math.atan2(d[1], d[0])

    ordered = [sorted(a, key=tangent) for a in adjacency]
    positions = [{h: i for i, h in enumerate(a)} for a in ordered]
    seen, output, unresolved = set(), [], []
    for index in range(len(edges)):
        for sign in (1, -1):
            first = (index, sign)
            if first in seen:
                continue
            walk, current = [], first
            while current not in seen:
                seen.add(current)
                walk.append(current)
                j, direction = current
                edge = edges[j]
                end = edge['b'] if direction == 1 else edge['a']
                order = ordered[end]
                current = order[(positions[end][(j, -direction)]-1) % len(order)]
            if current != first:
                raise kernel.Incomplete('face_walk_inconsistent')
            reduced = []
            for half in walk:
                if reduced and reduced[-1] == (half[0], -half[1]):
                    reduced.pop()
                else:
                    reduced.append(half)
            while len(reduced) > 1 and reduced[0] == (reduced[-1][0], -reduced[-1][1]):
                reduced = reduced[1:-1]
            if not reduced:
                continue
            vertices, pieces, owners = [], [], set()
            for j, direction in reduced:
                edge = edges[j]
                segment = edge['segment']
                low, high = (edge['low'], edge['high']) if direction == 1 else (edge['high'], edge['low'])
                a, b = kernel.point(segment, low), kernel.point(segment, high)
                bulge = math.tan(segment['sweep']*(high-low)/4) if segment['kind'] == 'arc' else 0.
                pieces.append((a, b, bulge))
                vertices.append(edge['a'] if direction == 1 else edge['b'])
                owners.update(edge['owners'])
            if len(set(vertices)) != len(vertices):
                unresolved.append('non_simple_boundary_walk')
                continue
            origin = pieces[0][0]
            local = [(tuple(a-origin), tuple(b-origin), bulge) for a, b, bulge in pieces]
            try:
                loop = loop_data(local, 1e-7)
                flat = np.asarray(loop['flat'], float)+origin
                numeric = numeric_envelope(flat)
                # Area is computed locally, avoiding subtraction of products of global coordinates.
                if loop['signed_area'] <= 1e-7:
                    continue
                if len(flat) > 512:
                    raise Unresolved('flattened_vertex_budget')
                certify_simple(flat, loop['chord_error_bound'], numeric, budget)
            except ValueError as exc:
                unresolved.append(str(exc))
                continue
            if len(output) >= kernel.POLICY['maximum_faces']:
                raise kernel.Incomplete('face_budget')
            output.append({'id': f'face-{len(output):05d}', 'area_px2': loop['signed_area'],
                'owners': sorted(owners), 'vertices': vertices, 'flat': flat.tolist(),
                'chord_error_bound_px': loop['chord_error_bound'], 'numeric_envelope_native_px': numeric,
                'simple': True, 'boundary_certified': True, 'parent': None, 'depth': None})
    return output, unresolved


def inside(point, polygon, budget):
    flat = np.asarray(polygon, float)-point
    following = np.roll(flat, -1, axis=0)
    budget.spend(len(flat))
    cross = flat[:, 0]*following[:, 1]-flat[:, 1]*following[:, 0]
    upward = (flat[:, 1] <= 0) & (following[:, 1] > 0) & (cross > 0)
    downward = (following[:, 1] <= 0) & (flat[:, 1] > 0) & (cross < 0)
    return bool(np.count_nonzero(upward)-np.count_nonzero(downward))


def relation(left, right, budget):
    a, b = np.asarray(left['flat']), np.asarray(right['flat'])
    envelope = left['chord_error_bound_px']+right['chord_error_bound_px'] + max(left['numeric_envelope_native_px'], right['numeric_envelope_native_px'])
    if np.any(a.min(axis=0) > b.max(axis=0)+envelope) or np.any(b.min(axis=0) > a.max(axis=0)+envelope):
        return 'disjoint'
    if set(left['vertices']) & set(right['vertices']):
        raise Unresolved('boundary_contact')
    following_b = np.roll(b, -1, axis=0)
    for p, q in zip(a, np.roll(a, -1, axis=0)):
        budget.spend(len(b))
        if np.any(segment_distances(p, q, b, following_b) <= envelope):
            raise Unresolved('boundary_contact_or_unresolved_curve_clearance')
    # Full boundary separation + certified simple polygons makes this winding test sufficient.
    left_in_right = inside(a[0], b, budget)
    right_in_left = inside(b[0], a, budget)
    if left_in_right and right_in_left:
        raise Unresolved('inconsistent_containment')
    return 'left_inside_right' if left_in_right else 'right_inside_left' if right_in_left else 'disjoint'


def nesting(loops, budget):
    uncertain, failures, parents = set(), [], {f['id']: [] for f in loops}
    checks = 0
    for i, left in enumerate(loops):
        for right in loops[i+1:]:
            checks += 1
            if checks > 100000:
                raise kernel.Incomplete('nesting_budget')
            try:
                state = relation(left, right, budget)
            except Unresolved as exc:
                uncertain.update((left['id'], right['id']))
                failures.append(str(exc))
                continue
            if state == 'left_inside_right':
                parents[left['id']].append(right)
            elif state == 'right_inside_left':
                parents[right['id']].append(left)
    for loop in loops:
        if loop['id'] not in uncertain and parents[loop['id']]:
            loop['parent'] = min(parents[loop['id']], key=lambda f: f['area_px2'])['id']
    lookup = {f['id']: f for f in loops}
    for loop in loops:
        parent, visited = loop['parent'], set()
        if loop['id'] in uncertain:
            continue
        while parent:
            if parent in visited:
                raise kernel.Incomplete('cyclic_nesting')
            visited.add(parent)
            parent = lookup[parent]['parent']
        loop['depth'] = len(visited)
    return checks, sorted(uncertain), failures


def build(candidates):
    """Reuse v1 analytic intersections/incidence; only boundary QA is versioned."""
    try:
        parts, isolated, definitions = kernel.segments(candidates)
        checks, entries = kernel.split_intersections(parts)
        nodes, edges, isolated_owners, duplicates = kernel.assemble(parts, isolated)
        adj = kernel.adjacency(nodes, edges)
        component, count = [-1]*len(nodes), 0
        for first in range(len(nodes)):
            if component[first] >= 0:
                continue
            stack, component[first] = [first], count
            while stack:
                v = stack.pop()
                for j, _ in adj[v]:
                    edge = edges[j]
                    w = edge['b'] if edge['a'] == v else edge['a']
                    if component[w] < 0:
                        component[w] = count
                        stack.append(w)
            count += 1
        budget = Budget()
        loops, unresolved = face_boundaries(nodes, edges, adj, budget)
        nest_checks, uncertain, nesting_failures = nesting(loops, budget)
        structural, component_owners = [], [set() for _ in range(count)]
        for i, p in enumerate(nodes):
            owners = set(isolated_owners[i])
            for j, _ in adj[i]:
                owners.update(edges[j]['owners'])
            component_owners[component[i]].update(owners)
            if len(adj[i]) != 2:
                structural.append({'id': f'node-{i:06d}', 'point_px': p.tolist(), 'degree': len(adj[i]),
                                   'owners': sorted(owners), 'component': component[i]})
        return {'complete': True, 'NoP': len(candidates), 'NoV_definition': definitions,
            'graph_vertices': len(nodes), 'graph_edges': len(edges), 'components': count,
            'cycle_rank': len(edges)-len(nodes)+count, 'degree_histogram': dict(sorted(Counter(map(len, adj)).items())),
            'structural_nodes': structural, 'component_owners': sorted(sorted(o) for o in component_owners),
            'faces': [{k: v for k, v in f.items() if k not in ('flat', 'vertices')} for f in loops],
            'bounded_faces': len(loops), 'nesting_complete': not unresolved and not uncertain,
            'unresolved_face_reasons': dict(Counter(unresolved)), 'uncertain_nesting_ids': uncertain,
            'nesting_failure_reasons': dict(Counter(nesting_failures)), 'topology_predicates': budget.used,
            'duplicate_geometric_pieces': duplicates, 'pair_checks': checks, 'grid_entries': entries,
            'nesting_checks': nest_checks, 'geometric_crossings_are_semantic_junctions': False,
            'endpoint_snapping_performed': False, 'physical_geometry_approved': False}
    except kernel.Incomplete as exc:
        return {'complete': False, 'reason': str(exc), 'physical_geometry_approved': False}
