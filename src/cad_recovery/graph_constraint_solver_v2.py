"""Adapt explicit reviewed graph constraints to a view-scoped physical problem."""
from copy import deepcopy
import math

import numpy as np

from .geometry_constraints_v2 import ConstraintProblem, solve_constraints
from .graph_v2 import read_graph_v2


def build_graph_constraint_problem(graph, view_id):
    graph = read_graph_v2(graph)
    view = next((v for v in graph.views if v.id == view_id), None)
    if view is None:
        raise ValueError('Unknown constrained view')
    matrix = np.asarray(view.model_matrix())
    constraints = [c for c in graph.constraints if c.view_id == view_id]
    if not constraints:
        raise ValueError('No explicit constraints in selected view')
    if any(c.status != 'accepted' or c.evidence.status != 'verified' for c in constraints):
        raise ValueError('Graph constraints need both acceptance and verified field evidence')
    used = {key for c in constraints for key in c.entity_ids}
    geometries = {g.id: g for g in graph.primitives if g.id in used}
    if any(g.status == 'rejected' for g in geometries.values()):
        raise ValueError('Rejected source geometry cannot enter a solver problem')
    points, circles, bindings, lookup = {}, {}, [], {}

    def add_point(entity, field, coordinates):
        identifier = 'p'+str(len(points))
        points[identifier] = (matrix @ [*coordinates, 1])[:2].tolist()
        lookup[(entity.id, field)] = identifier
        bindings.append({'point_id': identifier, 'entity_id': entity.id, 'field': field,
                         'view_id': view_id, 'source_ref': entity.evidence.source_ref,
                         'original_view_xy': list(coordinates), 'initial_model_mm': points[identifier]})

    for entity in geometries.values():
        if entity.kind == 'line':
            add_point(entity, 'start', entity.start)
            add_point(entity, 'end', entity.end)
        elif entity.kind in ('circle', 'arc'):
            if not math.isclose(*view.model_scale_xy, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError('A nonuniform view transform cannot preserve circular constraints')
            add_point(entity, 'center', entity.center)
            circles[entity.id] = {'center_point': lookup[(entity.id, 'center')],
                                  'radius': entity.radius*view.model_scale_xy[0]}
        else:
            raise ValueError('Constraint adapter currently supports explicit LINE/CIRCLE/ARC parameters only')
    converted = []
    for c in constraints:
        params = c.parameters
        row = {'id': c.id, 'kind': c.kind, 'evidence_ref': c.evidence.review_ref,
               'verified': True, 'tolerance': params.get('tolerance', 1e-5)}
        common = {'tolerance'}
        if c.kind in ('fixed', 'distance', 'projected_distance', 'coincident', 'symmetric'):
            allowed = common | {'point_refs'}
            refs = params.get('point_refs', [])
            if not refs or any(set(r) != {'entity_id', 'field'} for r in refs):
                raise ValueError('Point constraints require explicit entity/field references')
            if {r['entity_id'] for r in refs} != set(c.entity_ids):
                raise ValueError('Point references must cover exactly the declared geometry entities')
            if any((r['entity_id'], r['field']) not in lookup for r in refs):
                raise ValueError('Unsupported geometry field reference')
            row['point_ids'] = [lookup[(r['entity_id'], r['field'])] for r in refs]
            if c.kind == 'fixed':
                row.update(kind='fixed_point', target=params.get('target_mm'))
                allowed.add('target_mm')
            elif c.kind == 'symmetric':
                row['kind'] = 'symmetry'
            elif c.kind in ('distance', 'projected_distance'):
                row['value'] = c.value
                if c.kind == 'projected_distance':
                    row['measurement_axis'] = params.get('measurement_axis')
                    allowed.add('measurement_axis')
        elif c.kind in ('radius', 'diameter', 'concentric', 'equal_radius'):
            allowed = common
            if any(key not in circles for key in c.entity_ids):
                raise ValueError('Circular constraint requires circular geometry')
            row['circle_ids'] = c.entity_ids
            if c.kind in ('radius', 'diameter'):
                row['value'] = c.value
        elif c.kind in ('parallel', 'perpendicular'):
            allowed = common
            if len(c.entity_ids) != 2 or any(geometries[key].kind != 'line' for key in c.entity_ids):
                raise ValueError('Direction relation requires two explicit lines')
            row['point_ids'] = [lookup[(key, field)] for key in c.entity_ids for field in ('start', 'end')]
        elif c.kind == 'tangent':
            allowed = common | {'side', 'finite_segment', 'tangent_mode'}
            lines = [key for key in c.entity_ids if geometries[key].kind == 'line']
            rings = [key for key in c.entity_ids if geometries[key].kind == 'circle']
            if len(lines) == 1 and len(rings) == 1 and len(c.entity_ids) == 2:
                row.update(kind='line_circle_tangent', circle_ids=rings,
                           point_ids=[lookup[(lines[0], f)] for f in ('start', 'end')],
                           side=params.get('side', 1), finite_segment=params.get('finite_segment', True))
            elif len(rings) == 2 and len(c.entity_ids) == 2:
                row.update(kind='circle_tangent', circle_ids=c.entity_ids,
                           tangent_mode=params.get('tangent_mode', 'external'))
            else:
                raise ValueError('Tangent adapter requires circles; finite ARC tangency is not inferred from its support circle')
        else:
            raise ValueError('Unsupported graph constraint kind')
        if set(params)-allowed:
            raise ValueError('Unknown constraint parameters must not be silently discarded')
        if c.kind not in ('distance', 'projected_distance', 'radius', 'diameter') and c.value is not None:
            raise ValueError('Unexpected scalar value on geometric relation')
        converted.append(row)
    problem = ConstraintProblem(view_id=view_id, points=points, circles=circles, constraints=converted)
    provenance = {'view_id': view_id, 'view_to_model_matrix': matrix.tolist(),
                  'view_scale_evidence': list(view.scale_evidence_refs), 'point_bindings': bindings,
                  'included_primitive_ids': list(geometries),
                  'unmodeled_primitive_ids': [g.id for g in graph.primitives if g.view_id == view_id and g.id not in used],
                  'unmodeled_arc_sweeps': [g.id for g in geometries.values() if g.kind == 'arc'],
                  'constraint_evidence': {c.id: c.evidence.model_dump(mode='json') for c in constraints},
                  'source': graph.source.model_dump(mode='json'), 'graph_modified': False}
    return problem, provenance


def solve_graph_view(graph, view_id):
    problem, provenance = build_graph_constraint_problem(graph, view_id)
    result = solve_constraints(problem)
    result['graph_binding'] = deepcopy(provenance)
    return result
