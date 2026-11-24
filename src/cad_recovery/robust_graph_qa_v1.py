"""Owner-compatible node QA, optionally proven by actual source interval endpoints."""
from collections import defaultdict
import copy
import math

import numpy as np

from .robust_topology_v1 import build
from .simplification_qa_v1 import face_comparison, owner_errors
from .stroke_curve_control_v1 import maximum_matching

POLICY = {'version': 'robust-graph-qa/1.0', 'matching_native_px': 3.,
          'exact_node_epsilon_native_px': 1e-7, 'maximum_node_pair_checks': 250000,
          'maximum_bidirectional_native_px': 2., 'sampling_envelope_native_px': .25,
          'maximum_relative_face_area_change': .10, 'ambiguous_matching': 'reject',
          'semantic_ownership_confirmed': False, 'production_admitted': False}


def spatial_index(nodes, cell):
    bins = defaultdict(list)
    for i, n in enumerate(nodes):
        bins[tuple(np.floor(np.asarray(n['point_px'])/cell).astype(np.int64))].append(i)
    return bins


def nearby(point, bins, cell):
    x, y = np.floor(np.asarray(point)/cell).astype(np.int64)
    return [i for dx in (-1, 0, 1) for dy in (-1, 0, 1) for i in bins.get((x+dx, y+dy), [])]


def endpoints(candidate, point):
    if candidate['kind'] in ('line', 'polyline'):
        return min(math.dist(point, p) for p in (candidate['points_px'][0], candidate['points_px'][-1]))
    if candidate['kind'] == 'circle':
        return abs(math.dist(point, candidate['center_px'])-candidate['radius_px'])
    start = math.radians(candidate['start_angle_deg'])
    end = start+math.radians(candidate['sweep_angle_deg'])
    return min(math.dist(point, np.asarray(candidate['center_px'])+candidate['radius_px']*np.array([math.cos(a), math.sin(a)])) for a in (start, end))


def verified_provenance(before, after, source, candidates, evidence):
    """Check source IDs against original vertices and modeled endpoints, then graph nodes."""
    try:
        paths = {c['path_id']: c for c in source}
        pieces = defaultdict(list)
        for c in candidates:
            owner = c['path_id']
            lo, hi = c['source_index_interval']
            if owner not in paths or not 0 <= lo <= hi < len(paths[owner]['points_px']):
                raise ValueError('invalid_interval')
            if c['source_observation_id'] != paths[owner]['id']:
                raise ValueError('source_ID_changed')
            pieces[owner].append(c)
        if set(pieces) != set(paths):
            raise ValueError('source_paths_missing')
        for owner, selected in pieces.items():
            selected.sort(key=lambda c: c['source_index_interval'][0])
            if len(paths[owner]['points_px']) == 1 and len(selected) != 1:
                raise ValueError('duplicate_isolated_observation')
            if len(paths[owner]['points_px']) > 1 and any(c['source_index_interval'][0] == c['source_index_interval'][1] for c in selected):
                raise ValueError('empty_source_interval')
            if selected[0]['source_index_interval'][0] != 0 or selected[-1]['source_index_interval'][1] != len(paths[owner]['points_px'])-1:
                raise ValueError('interval_coverage')
            if any(a['source_index_interval'][1] != b['source_index_interval'][0] for a, b in zip(selected, selected[1:])):
                raise ValueError('interval_overlap_or_gap')
        declared = [{'point_px': n['observed_native_px'], **n} for n in evidence]
        observed_bins = spatial_index(declared, 1e-7)
        actual_bins = spatial_index(after['structural_nodes'], 1e-7)
        output, used = {}, set()
        for expected in before['structural_nodes']:
            options = [declared[i] for i in nearby(expected['point_px'], observed_bins, 1e-7)
                       if math.dist(declared[i]['point_px'], expected['point_px']) <= 1e-7]
            if len(options) != 1:
                raise ValueError('source_node_evidence_not_unique')
            declaration = options[0]
            target = declaration['estimated_native_px']
            if np.asarray(target).shape != (2,) or not np.isfinite(target).all():
                raise ValueError('nonfinite_node_evidence')
            if math.dist(target, expected['point_px']) > 1+1e-12:
                raise ValueError('registered_node_motion_exceeded')
            incidences = declaration['incidences']
            if sorted({owner for owner, _ in incidences}) != expected['owners']:
                raise ValueError('source_node_owners_changed')
            degree = 0 if len(incidences) == 1 and len(paths[incidences[0][0]]['points_px']) == 1 else len(incidences)
            if degree != expected['degree']:
                raise ValueError('source_node_degree_changed')
            for owner, end in incidences:
                if end not in (0, -1) or math.dist(paths[owner]['points_px'][end], expected['point_px']) > 1e-7:
                    raise ValueError('source_endpoint_not_observed')
                index = 0 if end == 0 else len(paths[owner]['points_px'])-1
                attached = [c for c in pieces[owner] if c['source_index_interval'][0 if end == 0 else 1] == index]
                if not attached or any(endpoints(c, target) > 1e-7 for c in attached):
                    raise ValueError('modeled_endpoint_contradicts_evidence')
            actual = [i for i in nearby(target, actual_bins, 1e-7)
                      if math.dist(after['structural_nodes'][i]['point_px'], target) <= 1e-7]
            if len(actual) != 1 or actual[0] in used:
                raise ValueError('modeled_graph_node_not_unique')
            found = after['structural_nodes'][actual[0]]
            if found['degree'] != expected['degree'] or found['owners'] != expected['owners']:
                raise ValueError('modeled_graph_incidence_changed')
            used.add(actual[0])
            output[expected['id']] = found['id']
        if len(used) != len(after['structural_nodes']):
            raise ValueError('extra_modeled_graph_nodes')
        return {'complete': True, 'mapping': output, 'method': 'verified_source_intervals_and_actual_endpoints'}
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return {'complete': False, 'reason': str(exc), 'mapping': {}, 'method': 'source_provenance_rejected'}


def alternating_cycle(edges, assignment):
    inverse = {v: k for k, v in assignment.items()}
    adjacency = {i: [inverse[j] for j in choices if j in inverse and assignment.get(i) != j] for i, choices in enumerate(edges)}
    colors = {}
    # Iterative DFS keeps the check bounded even on large near-node components.
    for start in adjacency:
        if colors.get(start):
            continue
        stack = [(start, iter(adjacency[start]))]
        colors[start] = 1
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                colors[node] = 2
                stack.pop()
            elif colors.get(child) == 1:
                return True
            elif not colors.get(child):
                colors[child] = 1
                stack.append((child, iter(adjacency[child])))
    return False


def match_nodes(before, after, provenance=None):
    output = {}
    total_work = 0
    for category, test in [('junction', lambda d: d >= 3), ('endpoint', lambda d: d == 1), ('isolated', lambda d: d == 0)]:
        a = [n for n in before['structural_nodes'] if test(n['degree'])]
        b = [n for n in after['structural_nodes'] if test(n['degree'])]
        bins = spatial_index(b, 3.)
        edges, work = [], 0
        for expected in a:
            options = []
            for j in nearby(expected['point_px'], bins, 3.):
                work += 1
                total_work += 1
                if total_work > 250000:
                    return None
                found = b[j]
                if (expected['owners'] == found['owners'] and expected['degree'] == found['degree']
                        and math.dist(expected['point_px'], found['point_px']) <= 3):
                    options.append(j)
            edges.append(sorted(options, key=lambda j: (math.dist(expected['point_px'], b[j]['point_px']), b[j]['id'])))
        if provenance and provenance['complete']:
            lookup = {n['id']: j for j, n in enumerate(b)}
            assignment = {i: lookup[provenance['mapping'][n['id']]] for i, n in enumerate(a)}
            ambiguous = False
            method = provenance['method']
        else:
            # Identical actual coordinates are stronger evidence than a proximity permutation.
            exact = [[j for j in choices if math.dist(a[i]['point_px'], b[j]['point_px']) <= 1e-7] for i, choices in enumerate(edges)]
            exact_assignment = maximum_matching(exact)
            if len(exact_assignment) == len(a) == len(b):
                assignment, ambiguous = exact_assignment, alternating_cycle(exact, exact_assignment)
                method = 'exact_coordinates_degree_and_owners'
            else:
                assignment = maximum_matching(edges)
                ambiguous = len(assignment) == len(a) == len(b) and alternating_cycle(edges, assignment)
                method = 'unique_compatible_bijection' if not ambiguous else 'ambiguous_compatible_bijections'
        matches = [{'before_id': a[i]['id'], 'after_id': b[j]['id'],
                    'distance_px': math.dist(a[i]['point_px'], b[j]['point_px']),
                    'degree_before': a[i]['degree'], 'degree_after': b[j]['degree'], 'owners_preserved': True}
                   for i, j in assignment.items()]
        output[category] = {'expected': len(a), 'predicted': len(b), 'matched': len(matches),
            'missed': len(a)-len(matches), 'extra': len(b)-len(matches), 'wrong_degree': 0, 'wrong_owners': 0,
            'ambiguous': bool(ambiguous), 'method': method, 'pair_checks': work,
            'RMSE_px': math.sqrt(sum(m['distance_px']**2 for m in matches)/len(matches)) if matches else None,
            'matches': matches}
    return output


def compare_graphs(before, after, *, provenance=None):
    if not before['complete'] or not after['complete']:
        return {'complete': False, 'accepted': False, 'reasons': ['incomplete_graph'], 'physical_geometry_approved': False}
    try:
        nodes = match_nodes(before, after, provenance)
    except RecursionError:
        return {'complete': False, 'accepted': False, 'reasons': ['node_matching_recursion_budget'], 'physical_geometry_approved': False}
    if nodes is None:
        return {'complete': False, 'accepted': False, 'reasons': ['node_matching_budget'], 'physical_geometry_approved': False}
    faces = face_comparison(before, after)
    reasons = []
    for name in ('components', 'cycle_rank', 'bounded_faces'):
        if before[name] != after[name]:
            reasons.append(name+'_changed')
    if before['component_owners'] != after['component_owners']:
        reasons.append('component_ownership_changed')
    if provenance and not provenance['complete']:
        reasons.append('invalid_source_provenance')
    for category, data in nodes.items():
        if data['missed'] or data['extra']:
            reasons.append(category+'_incidence_or_ownership_changed')
        if data['ambiguous']:
            reasons.append(category+'_matching_ambiguous')
    if not faces['complete']:
        reasons.append('unresolved_face_or_nesting')
    if faces['missing'] or faces['extra']:
        reasons.append('face_ownership_changed')
    if any(not f['parent_owners_preserved'] or not f['depth_preserved'] for f in faces['matches']):
        reasons.append('nesting_changed')
    if any(f['relative_area_change'] > .1 for f in faces['matches']):
        reasons.append('protected_face_area_changed')
    if after['duplicate_geometric_pieces'] > before['duplicate_geometric_pieces']:
        reasons.append('new_duplicate_pieces')
    return {'complete': True, 'accepted': not reasons, 'reasons': reasons, 'nodes': nodes, 'faces': faces,
            'provenance': provenance, 'physical_geometry_approved': False, 'production_admitted': False}


def guard(source, proposal, *, graphs=None, provenance=None, distance_evidence=None):
    before, after = graphs if graphs is not None else (build(source), build(proposal))
    result = compare_graphs(before, after, provenance=provenance)
    errors = distance_evidence if distance_evidence is not None else owner_errors(source, proposal)
    if not errors['complete']:
        result['reasons'].append('incomplete_geometric_error')
    if errors['max_px'] is not None and errors['max_px']+.25 > 2:
        result['reasons'].append('geometric_error_with_sampling_bound')
    result['accepted'] = not result['reasons']
    result['distance'] = errors
    result['selection'] = 'proposal' if result['accepted'] else 'unchanged_source'
    result['production_admitted'] = False
    return {'decision': result, 'selected': copy.deepcopy(proposal if result['accepted'] else source)}
