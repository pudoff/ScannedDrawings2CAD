"""Review-bound scalar corrections with explicit dimension dependency bindings.

The solver proposes a local branch. Only an approval of the exact proposal can
produce a derivative graph; neither step approves source accuracy or training.
"""
from copy import deepcopy
import hashlib
import json
import math
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exporter_v2 import dimension_measurement
from .geometry_constraints_v2 import _constraint_residual
from .graph_constraint_solver_v2 import build_graph_constraint_problem, solve_graph_view
from .graph_v2 import read_graph_v2
from .topology_v2 import TopologyBudget, validate_contour, validate_hatch_boundaries

Number = Annotated[float, Field(ge=-1e10, le=1e10, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, le=1e10, allow_inf_nan=False)]
Digest = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Identifier = Annotated[str, Field(min_length=1, max_length=1000, pattern=r'\S')]


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)


class CorrectionRequest(Model):
    schema_version: Literal['1.0'] = '1.0'
    parent_graph_sha256: Digest
    view_id: Identifier
    constraint_id: Identifier
    expected_old_value: Positive
    new_value: Positive
    parameter_evidence_ref: Identifier
    max_point_displacement_mm: Positive = 10
    max_radius_change_mm: Positive = 10


class PointBinding(Model):
    entity_id: Identifier
    field: Literal['start', 'end', 'center', 'line_fraction', 'circle_point']
    fraction: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] | None = None
    angle_deg: Number | None = None

    @model_validator(mode='after')
    def arguments(self):
        if (self.fraction is not None) != (self.field == 'line_fraction'):
            raise ValueError('Only a line_fraction binding has a fraction')
        if (self.angle_deg is not None) != (self.field == 'circle_point'):
            raise ValueError('Only a circle_point binding has a local angle')
        return self


class DimensionBinding(Model):
    dimension_id: Identifier
    points: list[PointBinding] = Field(min_length=2, max_length=3)
    # A nominal is changed only with an explicit choice for this dimension.
    nominal_policy: Literal['preserve', 'derive_from_reviewed_geometry'] = 'preserve'


class CorrectionBindings(Model):
    schema_version: Literal['1.0'] = '1.0'
    dimensions: list[DimensionBinding] = Field(default_factory=list, max_length=256)

    @model_validator(mode='after')
    def unique(self):
        if len({d.dimension_id for d in self.dimensions}) != len(self.dimensions):
            raise ValueError('Duplicate dimension binding')
        return self


class CorrectionApproval(Model):
    schema_version: Literal['1.0'] = '1.0'
    proposal_sha256: Digest
    review_ref: Identifier
    scope: Literal['analytical_control', 'reviewed_correction']
    branch_reviewed: bool = Field(strict=True)
    derived_dimensions_reviewed: bool = Field(strict=True)

    @model_validator(mode='after')
    def confirmed(self):
        if not self.branch_reviewed or not self.derived_dimensions_reviewed:
            raise ValueError('Explicit branch and derived-dimension review is required')
        return self


def canonical_sha256(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode='json')
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _point(binding, geometries):
    entity = geometries.get(binding.entity_id)
    if entity is None:
        raise ValueError('Binding refers to unknown geometry')
    if entity.kind == 'line' and binding.field in ('start', 'end', 'line_fraction'):
        if binding.field != 'line_fraction':
            return list(getattr(entity, binding.field))
        return [(1-binding.fraction)*a+binding.fraction*b for a, b in zip(entity.start, entity.end)]
    if entity.kind == 'circle' and binding.field in ('center', 'circle_point'):
        if binding.field == 'center':
            return list(entity.center)
        angle = math.radians(binding.angle_deg)
        return [entity.center[0]+entity.radius*math.cos(angle), entity.center[1]+entity.radius*math.sin(angle)]
    raise ValueError('Binding kind does not match a supported finite geometry field')


def _measure(dimension, matrix):
    points = [(matrix @ [*p, 1])[:2] for p in dimension.points]
    if dimension.kind == 'angular':
        center, a, b = points
        if min(np.linalg.norm(a-center), np.linalg.norm(b-center)) <= 1e-9:
            raise ValueError('Degenerate angular support')
        first, second = math.atan2(*(a-center)[::-1]), math.atan2(*(b-center)[::-1])
        return math.degrees((second-first) % (2*math.pi))
    if dimension.kind in ('radius', 'diameter', 'aligned'):
        return float(np.linalg.norm(points[1]-points[0]))*(2 if dimension.kind == 'diameter' else 1)
    angle = math.radians(dimension.angle_deg)
    axis = matrix[:2, :2] @ [math.cos(angle), math.sin(angle)]
    axis /= np.linalg.norm(axis)
    return abs(float((points[1]-points[0]) @ axis))


def _check_exact_geometry(graph, view_id):
    """Evaluate the stored coordinates themselves, without re-optimizing them."""
    problem, _ = build_graph_constraint_problem(graph, view_id)
    points = {key: np.asarray(point) for key, point in problem.points.items()}
    circles = {key: (points[c.center_point], c.radius) for key, c in problem.circles.items()}
    rows = []
    for constraint in problem.constraints:
        raw = np.asarray(_constraint_residual(constraint, points, circles))
        okay = bool(np.isfinite(raw).all() and np.all(np.abs(raw) <= constraint.tolerance))
        if constraint.kind == 'line_circle_tangent' and constraint.finite_segment:
            a, b = [points[key] for key in constraint.point_ids]
            center = circles[constraint.circle_ids[0]][0]
            t = float((center-a) @ (b-a)/((b-a) @ (b-a)))
            okay &= 0 <= t <= 1
        rows.append({'id': constraint.id, 'residual': raw.tolist(),
                     'tolerance': constraint.tolerance, 'within_tolerance': okay})
    if not all(row['within_tolerance'] for row in rows):
        raise ValueError('Stored derivative geometry does not satisfy the reviewed constraints')
    return rows


def _boundaries(graph, view_id):
    geometries = {p.id: p for p in graph.primitives}
    contours = {c.id: c for c in graph.contours}
    budget, checked, areas = TopologyBudget(), [], []
    for contour in graph.contours:
        if contour.view_id == view_id:
            result = validate_contour(contour, geometries, budget=budget)
            checked.append({'id': contour.id, 'closed': result['closed'], 'signed_area_view_units': result['signed_area']})
    for hatch in graph.hatches:
        if hatch.view_id == view_id:
            _, area = validate_hatch_boundaries(contours[hatch.outer_contour_id],
                [contours[key] for key in hatch.hole_contour_ids], geometries, budget=budget)
            areas.append({'id': hatch.id, 'net_area_view_units': area})
    return {'contours': checked, 'hatches': areas, 'topology_predicates': budget.used}


def prepare_graph_correction(graph, request, bindings):
    """Return a non-approved proposal, or a reasoned rejection with solver data."""
    graph = read_graph_v2(graph)
    request = CorrectionRequest.model_validate(request)
    bindings = CorrectionBindings.model_validate(bindings)
    parent = graph.model_dump(mode='json')
    if canonical_sha256(graph) != request.parent_graph_sha256:
        raise ValueError('Stale correction: parent graph hash differs')
    constraint = next((c for c in graph.constraints if c.id == request.constraint_id), None)
    if constraint is None or constraint.view_id != request.view_id:
        raise ValueError('Correction constraint must exist in the selected view')
    if constraint.kind not in ('distance', 'projected_distance', 'radius', 'diameter'):
        raise ValueError('Only explicit scalar length constraints can be corrected')
    if constraint.value != request.expected_old_value or request.new_value == request.expected_old_value:
        raise ValueError('Old value must match exactly and the new value must change it')
    problem, provenance = build_graph_constraint_problem(graph, request.view_id)
    included = set(provenance['included_primitive_ids'])
    view = next(v for v in graph.views if v.id == request.view_id)
    old_geometry = {p.id: p for p in graph.primitives}
    if any(old_geometry[key].kind not in ('line', 'circle') for key in included):
        raise ValueError('Correction supports constrained LINE/CIRCLE only; ARC sweeps are not modeled')
    affected_dimensions = {d.id for d in graph.dimensions if included.intersection(d.measured_entity_ids)}
    if {d.dimension_id for d in bindings.dimensions} != affected_dimensions:
        raise ValueError('Bindings must cover exactly every dependent dimension')
    dimension_lookup = {d.id: d for d in graph.dimensions}
    for binding in bindings.dimensions:
        dim = dimension_lookup[binding.dimension_id]
        if len(binding.points) != len(dim.points) or any(p.entity_id not in dim.measured_entity_ids for p in binding.points):
            raise ValueError('Dimension bindings must identify its own explicit measured geometry')
        linear = np.asarray(view.model_matrix())[:2, :2]
        if any(math.dist(point, _point(ref, old_geometry)) > 1e-8 or
               np.linalg.norm(linear @ (np.asarray(point)-_point(ref, old_geometry))) > 1e-7
               for point, ref in zip(dim.points, binding.points)):
            raise ValueError('Dimension binding does not reproduce the original defining point')
        dimension_measurement(dim, old_geometry, np.asarray(view.model_matrix()))
    affected_contours = {c.id for c in graph.contours if included.intersection(e.primitive_id for e in c.edges)}
    affected_hatches = {h.id for h in graph.hatches if affected_contours.intersection([h.outer_contour_id, *h.hole_contour_ids])}
    touched = included | affected_dimensions | affected_contours | affected_hatches | {constraint.id}
    entities = [*graph.primitives, *graph.dimensions, *graph.contours, *graph.hatches, *graph.constraints]
    if any(e.status != 'accepted' or e.evidence.status != 'verified' for e in entities if e.id in touched):
        raise ValueError('Correction requires reviewed input objects, constraints and dependent annotations')
    _check_exact_geometry(graph, request.view_id)
    old_boundaries = _boundaries(graph, request.view_id)
    data = deepcopy(parent)
    next(c for c in data['constraints'] if c['id'] == constraint.id)['value'] = request.new_value
    changed = read_graph_v2(data)
    solver = solve_graph_view(changed, request.view_id)
    base = {'schema_version': '1.0', 'parent_graph_sha256': request.parent_graph_sha256,
            'request': request.model_dump(mode='json'), 'bindings': bindings.model_dump(mode='json'),
            'source_modified': False, 'source_accuracy_approved': False, 'training_eligible': False,
            'global_uniqueness_proven': False, 'requires_review': True, 'solver': solver}
    if solver['status'] != 'solved':
        return {**base, 'status': 'rejected', 'reason': 'Constraint solution is not fully determined and consistent', 'candidate_graph': None}
    try:
        inverse = np.linalg.inv(np.asarray(view.model_matrix()))
        rows = {p['id']: p for p in data['primitives']}
        for binding in provenance['point_bindings']:
            result = solver['proposal']['points'][binding['point_id']]
            if math.dist(result, binding['initial_model_mm']) > request.max_point_displacement_mm:
                raise ValueError('Point displacement exceeds the explicit correction limit')
            rows[binding['entity_id']][binding['field']] = (inverse @ [*result, 1])[:2].tolist()
        for key, circle in solver['proposal']['circles'].items():
            if abs(circle['radius']-problem.circles[key].radius) > request.max_radius_change_mm:
                raise ValueError('Radius adjustment exceeds the explicit correction limit')
            rows[key]['radius'] = circle['radius']/view.model_scale_xy[0]
        candidate = read_graph_v2(data)
        geometry = {p.id: p for p in candidate.primitives}
        dims = {d.id: d for d in candidate.dimensions}
        updates = []
        for binding in bindings.dimensions:
            dim = dims[binding.dimension_id]
            dim.points = [_point(ref, geometry) for ref in binding.points]
            if binding.nominal_policy == 'derive_from_reviewed_geometry':
                dim.nominal = _measure(dim, np.asarray(view.model_matrix()))
            dimension_measurement(dim, geometry, np.asarray(view.model_matrix()))
            updates.append({'id': dim.id, 'old_nominal': dimension_lookup[dim.id].nominal,
                            'new_nominal': dim.nominal, 'raw_text_preserved': dim.raw_text,
                            'nominal_policy': binding.nominal_policy})
        exact = _check_exact_geometry(candidate, request.view_id)
        boundaries = _boundaries(candidate, request.view_id)
        # Candidate values are observations until review of this exact branch.
        for collection in ('primitives', 'dimensions', 'contours', 'hatches', 'constraints'):
            for entity in getattr(candidate, collection):
                if entity.id in touched:
                    entity.status = 'proposed'
                    entity.evidence = entity.evidence.model_copy(update={'method': 'derived', 'status': 'observed',
                        'review_ref': None, 'source_region': None, 'uncertainty': None})
        result = {**base, 'status': 'needs_branch_review', 'candidate_graph': candidate.model_dump(mode='json'),
                  'touched_entity_ids': sorted(touched), 'dimension_updates': updates,
                  'stored_geometry_checks': exact, 'before_boundaries': old_boundaries, 'after_boundaries': boundaries,
                  'reason': 'Review this exact local branch and every derived nominal before applying.'}
        result['proposal_sha256'] = canonical_sha256(result)
        return result
    except (ValueError, np.linalg.LinAlgError) as error:
        return {**base, 'status': 'rejected', 'reason': str(error), 'candidate_graph': None}


def accept_graph_correction(graph, request, bindings, approval):
    """Recompute the proposal and apply an explicit review of its exact digest."""
    graph = read_graph_v2(graph)
    approval = CorrectionApproval.model_validate(approval)
    proposal = prepare_graph_correction(graph, request, bindings)
    if proposal['status'] != 'needs_branch_review' or proposal['proposal_sha256'] != approval.proposal_sha256:
        raise ValueError('Approval is stale or does not refer to this recomputed correction proposal')
    result = read_graph_v2(proposal['candidate_graph'])
    for collection in ('primitives', 'dimensions', 'contours', 'hatches', 'constraints'):
        for entity in getattr(result, collection):
            if entity.id in proposal['touched_entity_ids']:
                entity.status = 'accepted'
                entity.evidence = entity.evidence.model_copy(update={'status': 'verified',
                    'review_ref': approval.review_ref, 'source_ref': 'correction:'+approval.proposal_sha256})
    result.provenance['graph_correction_v1'] = {
        'parent_graph_sha256': proposal['parent_graph_sha256'], 'parent_graph': graph.model_dump(mode='json'),
        'request': proposal['request'], 'bindings': proposal['bindings'], 'approval': approval.model_dump(mode='json'),
        'proposal_sha256': approval.proposal_sha256, 'dimension_updates': proposal['dimension_updates'],
        'source_accuracy_approved': False, 'training_eligible': False,
        'global_uniqueness_proven': False, 'cad_associativity_verified': False}
    result = read_graph_v2(result.model_dump(mode='json'))
    _check_exact_geometry(result, proposal['request']['view_id'])
    _boundaries(result, proposal['request']['view_id'])
    return result
