"""Bounded view calibration and local constraint solving on explicit evidence.

No OCR string becomes a metric constraint implicitly. Solved coordinates are
proposals; rank establishes local determination, never global uniqueness.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(ge=-1e10, le=1e10, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, le=1e10, allow_inf_nan=False)]
Tolerance = Annotated[float, Field(ge=1e-9, le=1e6, allow_inf_nan=False)]
Point = tuple[Finite, Finite]
Identifier = Annotated[str, Field(min_length=1, max_length=256)]


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', validate_assignment=True)


class ScaleObservation(Model):
    id: Identifier
    first: Point
    second: Point
    kind: Literal['linear', 'aligned'] = 'linear'
    measurement_axis: Point | None = None
    measurement_axis_space: Literal['view_local', 'scaled_view'] = 'view_local'
    nominal_mm: Positive
    tolerance_mm: Tolerance
    evidence_ref: Identifier
    verified: bool = Field(default=False, strict=True)
    broken_view: bool = Field(default=False, strict=True)

    @model_validator(mode='after')
    def geometry(self):
        if math.dist(self.first, self.second) <= 1e-9:
            raise ValueError('Scale requires distinct measured supports')
        if self.kind == 'linear':
            if self.measurement_axis is None or abs(math.hypot(*self.measurement_axis)-1) > 1e-9:
                raise ValueError('Linear scale evidence needs an explicit unit measurement axis')
        elif self.measurement_axis is not None:
            raise ValueError('Aligned evidence uses the measured support direction')
        return self


class ScaleSourceBinding(Model):
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    page_index: int = Field(ge=0, strict=True)
    view_coordinate_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


class ScaleProblem(Model):
    schema_version: Literal['1.0'] = '1.0'
    view_id: Identifier
    coordinate_unit: Literal['pixel', 'paper_mm', 'model_mm']
    observations: list[ScaleObservation] = Field(min_length=1, max_length=128)
    source_binding: ScaleSourceBinding | None = None
    deformation_confirmed: bool = Field(default=False, strict=True)
    deformation_evidence_ref: Identifier | None = None

    @model_validator(mode='after')
    def identities(self):
        if len({o.id for o in self.observations}) != len(self.observations):
            raise ValueError('Duplicate scale observation identity')
        if self.deformation_confirmed != (self.deformation_evidence_ref is not None):
            raise ValueError('Anisotropic calibration requires explicit deformation evidence')
        if self.coordinate_unit == 'model_mm' and self.deformation_confirmed:
            raise ValueError('Physical model coordinates cannot be reinterpreted as a deformed source raster')
        return self


def _rank(jacobian):
    singular = np.linalg.svd(jacobian, compute_uv=False)
    threshold = max(jacobian.shape)*np.finfo(float).eps*singular[0]*100 if singular.size else 0
    return int(np.count_nonzero(singular > max(threshold, 1e-9)))


def _jacobian(function, x):
    baseline = function(x)
    jac = np.empty((len(baseline), len(x)))
    for i, value in enumerate(x):
        step = 1e-6*max(1., abs(value))
        trial = x.copy()
        trial[i] += step
        other = x.copy()
        other[i] -= step
        try:
            plus = function(trial)
        except ValueError:
            plus = None
        try:
            minus = function(other)
        except ValueError:
            minus = None
        if plus is not None and minus is not None:
            jac[:, i] = (plus-minus)/(2*step)
        elif plus is not None:
            jac[:, i] = (plus-baseline)/step
        elif minus is not None:
            jac[:, i] = (baseline-minus)/step
        else:
            raise ValueError('Both finite-difference directions are outside the valid domain')
    if not np.isfinite(jac).all():
        raise ValueError('Nonfinite constraint Jacobian')
    return jac


def _least_squares(function, initial, max_iterations=120):
    """Damped finite-difference Gauss-Newton with bounded iteration count."""
    x = np.asarray(initial, dtype=float).copy()
    residual = function(x)
    damping, converged, iterations = 1e-3, False, 0
    for iterations in range(1, max_iterations+1):
        jac = _jacobian(function, x)
        gradient = jac.T @ residual
        if np.linalg.norm(gradient, ord=np.inf) < 1e-8 or np.max(np.abs(residual)) < 1e-8:
            converged = True
            break
        scale = np.maximum(np.linalg.norm(jac, axis=0), 1e-9)
        normalized = jac/scale
        # lstsq on an augmented system avoids squaring the condition number.
        system = np.vstack([normalized, math.sqrt(damping)*np.eye(len(x))])
        rhs = np.concatenate([-residual, np.zeros(len(x))])
        step = np.linalg.lstsq(system, rhs, rcond=1e-12)[0]/scale
        candidate = x+step
        try:
            next_residual = function(candidate)
            improvement = float(residual @ residual-next_residual @ next_residual)
        except ValueError:
            improvement = -1
        if improvement > 0:
            x, residual = candidate, next_residual
            damping = max(1e-12, damping/3)
            if np.linalg.norm(step) < 1e-10*(1+np.linalg.norm(x)):
                converged = True
                break
        else:
            damping *= 10
            if damping > 1e14:
                break
    jac = _jacobian(function, x)
    return x, residual, jac, iterations, converged


def fit_view_scale(problem: ScaleProblem | dict):
    """Fit only the declared view, with isotropic scale tried first."""
    problem = ScaleProblem.model_validate(problem)
    observations = problem.observations
    result = {'schema_version': '1.0', 'view_id': problem.view_id,
              'coordinate_unit': problem.coordinate_unit, 'problem': problem.model_dump(mode='json'),
              'model_scale_xy': None, 'physical_scale_confirmed': False,
              'deformation_confirmed': False, 'deformation_evidence_ref': None,
              'scale_evidence_refs': [o.evidence_ref for o in observations],
              'original_coordinates_modified': False, 'nominals_modified': False,
              'requires_review': True, 'uncertainty_bound': None}
    if any(not o.verified or o.broken_view for o in observations):
        return {**result, 'status': 'unverified_evidence',
                'reason': 'Every nominal and measured support must be verified; broken spans are ineligible.'}
    directions, deltas, lengths = [], [], []
    for obs in observations:
        delta = np.subtract(obs.second, obs.first)
        direction = np.asarray(obs.measurement_axis) if obs.kind == 'linear' else delta/np.linalg.norm(delta)
        directions.append(direction)
        deltas.append(delta)
        lengths.append(abs(float(delta @ direction)))
    if min(lengths) <= 1e-9:
        return {**result, 'status': 'invalid_geometry', 'reason': 'Zero measured projection cannot calibrate a view.'}
    if len({o.evidence_ref for o in observations}) < 2 or _rank(np.asarray(directions)) < 2:
        return {**result, 'status': 'underdetermined',
                'reason': 'At least two separately evidenced, nonparallel measurement directions are required.'}
    nominals = np.array([o.nominal_mm for o in observations])
    tolerances = np.array([o.tolerance_mm for o in observations])
    lengths = np.asarray(lengths)
    squared_rows, squared_targets, linear_rows, linear_targets = [], [], [], []
    for obs, delta, direction in zip(observations, deltas, directions):
        row = None
        if obs.kind == 'aligned' or (obs.measurement_axis_space == 'view_local'
                and _cross(delta, direction) == 0.):
            row = delta*delta
        elif obs.kind == 'linear' and direction[0] == 0.:
            row = np.array([0., delta[1]**2])
        elif obs.kind == 'linear' and direction[1] == 0.:
            row = np.array([delta[0]**2, 0.])
        if row is not None and np.linalg.norm(row) > 0:
            norm = np.linalg.norm(row)
            squared_rows.append(row/norm)
            squared_targets.append(obs.nominal_mm**2/norm)
        if obs.kind == 'linear' and (obs.measurement_axis_space == 'scaled_view'
                                    or min(abs(direction)) == 0.):
            row = delta*direction
            if (np.all(row >= 0) or np.all(row <= 0)) and np.linalg.norm(row) > 0:
                norm = np.linalg.norm(row)
                linear_rows.append(abs(row)/norm)
                linear_targets.append(obs.nominal_mm/norm)
    squared_rank = _rank(np.asarray(squared_rows)) if squared_rows else 0
    linear_rank = _rank(np.asarray(linear_rows)) if linear_rows else 0
    axis_intervals = []
    for axis in (0, 1):
        controls = [(obs, abs(delta[axis])) for obs, delta, direction in zip(observations, deltas, directions)
                    if direction[1-axis] == 0.]
        if controls:
            lo = max(0., max((o.nominal_mm-o.tolerance_mm)/length for o, length in controls))
            hi = min((o.nominal_mm+o.tolerance_mm)/length for o, length in controls)
            axis_intervals.append({'axis': 'X' if axis == 0 else 'Y', 'lower': float(lo), 'upper': float(hi),
                                   'consistent': bool(lo <= hi), 'observation_ids': [o.id for o, _ in controls]})
    explicit_axis_conflict = any(not row['consistent'] for row in axis_intervals)
    factor = float((lengths/tolerances) @ (nominals/tolerances) / ((lengths/tolerances) @ (lengths/tolerances)))
    lower = max(0., float(np.max((nominals-tolerances)/lengths)))
    upper = float(np.min((nominals+tolerances)/lengths))
    feasible_interval = [lower, upper] if lower <= upper else None
    if feasible_interval and not lower <= factor <= upper:
        factor = (lower+upper)/2
    if problem.coordinate_unit == 'model_mm':
        factor = 1.
    if not math.isfinite(factor) or factor <= 0:
        return {**result, 'status': 'invalid_geometry',
                'reason': 'Scale estimation exceeds the supported floating-point domain.'}

    def measures(scales):
        values = []
        for obs, delta, direction in zip(observations, deltas, directions):
            if obs.kind == 'linear':
                if obs.measurement_axis_space == 'view_local':
                    scaled_axis = direction*scales
                    magnitude = np.max(np.abs(scaled_axis))
                    if magnitude <= 0 or not math.isfinite(magnitude):
                        raise ValueError('Degenerate scaled measurement axis')
                    scaled_axis = scaled_axis/magnitude
                    axis = scaled_axis/np.linalg.norm(scaled_axis)
                else:
                    axis = direction
                values.append(abs(float((delta*scales) @ axis)))
            else:
                values.append(float(np.linalg.norm(delta*scales)))
        return np.asarray(values)

    scales = np.array([factor, factor])
    method, iterations, converged = 'isotropic_verified_dimensions', 0, True
    uniform_residual = measures(scales)-nominals
    if np.any(abs(uniform_residual) > tolerances) and problem.deformation_confirmed:
        def residual(log_scales):
            if np.max(np.abs(log_scales)) > 30:
                raise ValueError('Calibration scale outside supported range')
            return (measures(np.exp(log_scales))-nominals)/tolerances
        initial = scales
        # Start from independently identifying conditions, not an arbitrary
        # branch of an absolute projected-distance equation.
        if linear_rank == 2:
            candidate = np.linalg.lstsq(np.asarray(linear_rows), linear_targets, rcond=1e-12)[0]
            if np.all(candidate > 0):
                initial = candidate
        elif squared_rank == 2:
            candidate = np.linalg.lstsq(np.asarray(squared_rows), squared_targets, rcond=1e-12)[0]
            if np.all(candidate > 0):
                initial = np.sqrt(candidate)
        try:
            fitted, _, jac, iterations, converged = _least_squares(residual, np.log(initial))
        except (ValueError, np.linalg.LinAlgError) as exc:
            return {**result, 'status': 'invalid_geometry', 'reason': 'Numerical calibration domain: '+str(exc)}
        if _rank(jac) < 2:
            return {**result, 'status': 'underdetermined',
                    'reason': 'Independent X/Y scale information is insufficient even with a deformation record.'}
        scales = np.exp(fitted)
        method = 'anisotropic_with_explicit_deformation_evidence'
    actual = measures(scales)
    checks = [{'id': o.id, 'nominal_mm': o.nominal_mm, 'computed_mm': float(value),
               'residual_mm': float(value-o.nominal_mm), 'tolerance_mm': o.tolerance_mm,
               'within_tolerance': bool(abs(value-o.nominal_mm) <= o.tolerance_mm)}
              for o, value in zip(observations, actual)]
    accepted = all(check['within_tolerance'] for check in checks)
    result.update(status='solved' if accepted else 'conflict' if converged or explicit_axis_conflict else 'not_converged', method=method, checks=checks,
                  isotropic_candidate=factor, isotropic_residuals_mm=uniform_residual.tolist(),
                  isotropic_feasible_interval=feasible_interval,
                  axis_scale_intervals=axis_intervals,
                  candidate_scale_xy=scales.tolist(), iterations=iterations, optimizer_converged=converged)
    result['anisotropic_identifiability'] = {'squared_equation_count': len(squared_rows),
                                           'squared_equation_rank': squared_rank,
                                           'known_sign_linear_equation_rank': linear_rank,
                                           'method': 'independent_verified_squared_or_known_sign_linear_scale_equations'}
    if accepted and problem.deformation_confirmed and max(squared_rank, linear_rank) < 2:
        return {**result, 'status': 'ambiguous_calibration',
                'reason': 'Nonlinear projected dimensions may admit several positive X/Y scales; independent axis or aligned controls are required.'}
    if accepted:
        bound = problem.source_binding is not None
        result.update(model_scale_xy=scales.tolist(), physical_scale_confirmed=bound, requires_review=not bound,
                      deformation_confirmed=method.startswith('anisotropic'),
                      deformation_evidence_ref=problem.deformation_evidence_ref if method.startswith('anisotropic') else None)
        if not bound:
            result['reason'] = 'Numerical calibration succeeded, but source and coordinate-frame binding is still required.'
    else:
        result['reason'] = 'Verified nominal constraints disagree; no scale or corrected nominal is published.'
    return result


class CircleVariable(Model):
    center_point: Identifier
    radius: Positive


class Constraint(Model):
    id: Identifier
    kind: Literal['fixed_point', 'coincident', 'distance', 'projected_distance', 'radius', 'diameter', 'horizontal', 'vertical',
                  'parallel', 'perpendicular', 'concentric', 'equal_radius', 'line_circle_tangent',
                  'circle_tangent', 'symmetry']
    point_ids: list[Identifier] = Field(default_factory=list, max_length=4)
    circle_ids: list[Identifier] = Field(default_factory=list, max_length=2)
    target: Point | None = None
    value: Positive | None = None
    measurement_axis: Point | None = None
    side: Literal[-1, 1] = 1
    tangent_mode: Literal['external', 'internal'] = 'external'
    finite_segment: bool = True
    tolerance: Tolerance = 1e-5
    evidence_ref: Identifier
    verified: bool = Field(default=False, strict=True)

    @model_validator(mode='after')
    def arity(self):
        point_counts = {'fixed_point': 1, 'coincident': 2, 'distance': 2, 'projected_distance': 2,
                        'horizontal': 2, 'vertical': 2, 'parallel': 4, 'perpendicular': 4,
                        'line_circle_tangent': 2, 'symmetry': 4}
        circle_counts = {'radius': 1, 'diameter': 1, 'concentric': 2, 'equal_radius': 2, 'circle_tangent': 2, 'line_circle_tangent': 1}
        if len(self.point_ids) != point_counts.get(self.kind, 0) or len(self.circle_ids) != circle_counts.get(self.kind, 0):
            raise ValueError('Constraint reference arity differs from its kind')
        if len(set(self.circle_ids)) != len(self.circle_ids):
            raise ValueError('A two-circle relation needs distinct circles')
        if (self.target is not None) != (self.kind == 'fixed_point'):
            raise ValueError('Only fixed points require a target coordinate')
        if (self.value is not None) != (self.kind in ('distance', 'projected_distance', 'radius', 'diameter')):
            raise ValueError('Only measured length constraints require a scalar value')
        if self.kind == 'projected_distance':
            if self.measurement_axis is None or abs(math.hypot(*self.measurement_axis)-1) > 1e-9:
                raise ValueError('Projected length requires a unit measurement axis')
        elif self.measurement_axis is not None:
            raise ValueError('Unexpected measurement axis')
        return self


class ConstraintProblem(Model):
    schema_version: Literal['1.0'] = '1.0'
    view_id: Identifier
    coordinate_unit: Literal['mm'] = 'mm'
    points: dict[Identifier, Point] = Field(min_length=1, max_length=64)
    circles: dict[Identifier, CircleVariable] = Field(default_factory=dict, max_length=32)
    constraints: list[Constraint] = Field(min_length=1, max_length=128)
    max_iterations: int = Field(default=120, ge=1, le=200)

    @model_validator(mode='after')
    def references(self):
        if len({c.id for c in self.constraints}) != len(self.constraints):
            raise ValueError('Constraint identities must be distinct')
        if any(c.center_point not in self.points for c in self.circles.values()):
            raise ValueError('Unknown circle center')
        for c in self.constraints:
            if not set(c.point_ids) <= self.points.keys() or not set(c.circle_ids) <= self.circles.keys():
                raise ValueError('Unknown constraint geometry reference')
        return self


def _unit(vector):
    norm = np.linalg.norm(vector)
    if norm <= 1e-10:
        raise ValueError('Degenerate direction in geometric constraint')
    return vector/norm


def _cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def _constraint_residual(c, points, circles):
    p = [points[key] for key in c.point_ids]
    rings = [circles[key] for key in c.circle_ids]
    kind = c.kind
    if kind == 'fixed_point': return p[0]-c.target
    if kind == 'coincident': return p[0]-p[1]
    if kind == 'distance': return [np.linalg.norm(p[1]-p[0])-c.value]
    if kind == 'projected_distance': return [abs(float((p[1]-p[0]) @ c.measurement_axis))-c.value]
    if kind == 'horizontal': return [p[1][1]-p[0][1]]
    if kind == 'vertical': return [p[1][0]-p[0][0]]
    if kind == 'radius': return [rings[0][1]-c.value]
    if kind == 'diameter': return [2*rings[0][1]-c.value]
    if kind in ('parallel', 'perpendicular'):
        a, b = _unit(p[1]-p[0]), _unit(p[3]-p[2])
        return [_cross(a, b) if kind == 'parallel' else float(a @ b)]
    if kind == 'concentric': return rings[0][0]-rings[1][0]
    if kind == 'equal_radius': return [rings[0][1]-rings[1][1]]
    if kind == 'line_circle_tangent':
        direction = _unit(p[1]-p[0])
        center, radius = rings[0]
        return [_cross(direction, center-p[0])-c.side*radius]
    if kind == 'circle_tangent':
        (a, ra), (b, rb) = rings
        expected = ra+rb if c.tangent_mode == 'external' else ra-rb
        if expected <= 0:
            raise ValueError('Internal tangency requires the first circle to enclose the second')
        return [np.linalg.norm(a-b)-expected]
    direction = _unit(p[3]-p[2])
    # Midpoint on the symmetry axis; paired displacement normal to that axis.
    return [_cross(direction, (p[0]+p[1])/2-p[2]), float((p[1]-p[0]) @ direction)]


def solve_constraints(problem: ConstraintProblem | dict):
    """Return a traceable local proposal, including residuals and free degrees."""
    problem = ConstraintProblem.model_validate(problem)
    original = problem.model_dump(mode='json')
    base = {'schema_version': '1.0', 'view_id': problem.view_id, 'problem': original,
            'original_geometry_modified': False, 'nominals_modified': False,
            'geometry_approved': False, 'training_eligible': False, 'requires_review': True,
            'global_uniqueness_proven': False, 'local_solution_only': True}
    if any(not c.verified for c in problem.constraints):
        return {**base, 'status': 'unverified_evidence', 'proposal': None,
                'reason': 'Only explicitly verified constraints can participate in the solver.'}
    point_ids, circle_ids = list(problem.points), list(problem.circles)
    count = 2*len(point_ids)
    initial = [value for key in point_ids for value in problem.points[key]]+[problem.circles[key].radius for key in circle_ids]

    def unpack(x):
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 1e10:
            raise ValueError('Coordinates outside bounded solver range')
        points = {key: x[2*i:2*i+2] for i, key in enumerate(point_ids)}
        circles = {key: (points[problem.circles[key].center_point], x[count+i]) for i, key in enumerate(circle_ids)}
        if any(r <= 1e-9 for _, r in circles.values()):
            raise ValueError('Circle radius must stay positive')
        return points, circles

    def residual(x):
        points, circles = unpack(x)
        values = np.concatenate([np.asarray(_constraint_residual(c, points, circles))/c.tolerance for c in problem.constraints])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite geometric residual')
        return values

    try:
        fitted, values, jac, iterations, converged = _least_squares(residual, initial, problem.max_iterations)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {**base, 'status': 'invalid_geometry', 'proposal': None, 'reason': str(exc)}
    points, circles = unpack(fitted)
    checks = []
    for c in problem.constraints:
        raw = np.asarray(_constraint_residual(c, points, circles))
        check = {'id': c.id, 'kind': c.kind, 'residual': raw.tolist(), 'tolerance': c.tolerance,
                 'residual_unit': 'dimensionless' if c.kind in ('parallel', 'perpendicular') else 'mm',
                 'within_tolerance': bool(np.all(np.abs(raw) <= c.tolerance)), 'evidence_ref': c.evidence_ref}
        if c.kind == 'line_circle_tangent' and c.finite_segment:
            a, b = (points[key] for key in c.point_ids)
            center = circles[c.circle_ids[0]][0]
            parameter = float((center-a) @ (b-a)/((b-a) @ (b-a)))
            check.update(contact_segment_parameter=parameter, contact_on_finite_segment=0 <= parameter <= 1)
            check['within_tolerance'] &= check['contact_on_finite_segment']
        checks.append(check)
    rank = _rank(jac)
    conflicts = [row['id'] for row in checks if not row['within_tolerance']]
    status = ('conflict' if converged else 'not_converged') if conflicts else 'underdetermined' if rank < len(initial) else 'solved'
    proposal = {'points': {key: value.tolist() for key, value in points.items()},
                'circles': {key: {'center_point': problem.circles[key].center_point, 'radius': float(value[1])}
                            for key, value in circles.items()}}
    return {**base, 'status': status, 'proposal': proposal, 'checks': checks,
            'constraints_outside_tolerance': conflicts, 'parameter_count': len(initial),
            'jacobian_rank': rank, 'local_free_degrees': len(initial)-rank,
            'iterations': iterations, 'optimizer_converged': converged,
            'max_normalized_residual': float(np.max(np.abs(values))),
            'max_coordinate_adjustment_mm': float(np.max(np.abs(fitted-np.asarray(initial)))),
            'reason': 'Review all residuals and branch choices; local numerical rank is not a proof of a unique global design.'}


def make_scale_source_binding(graph, view_id):
    """Bind authored calibration supports to the exact source page and view frame."""
    from .graph_v2 import read_graph_v2
    graph = read_graph_v2(graph)
    view = next((v for v in graph.views if v.id == view_id), None)
    if view is None:
        raise ValueError('Unknown calibration view')
    frame = {'source_sha256': graph.source.sha256, 'page_index': graph.source.page_index,
             'view_id': view.id, 'coordinate_space': view.coordinate_space,
             'source_to_view': view.source_to_view, 'view_to_sheet': view.view_to_sheet,
             'model_origin': view.model_origin, 'model_rotation_deg': view.model_rotation_deg}
    encoded = json.dumps(frame, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {'source_sha256': graph.source.sha256, 'page_index': graph.source.page_index,
            'view_coordinate_sha256': hashlib.sha256(encoded).hexdigest()}


def apply_view_calibration(graph, result):
    """Revalidate a calibration before returning a new graph; never edit in place."""
    from .graph_v2 import DrawingGraphV2, read_graph_v2
    graph = read_graph_v2(graph)
    fresh = fit_view_scale(result['problem'])
    if fresh != result or fresh['status'] != 'solved' or not fresh['physical_scale_confirmed']:
        raise ValueError('Only a reproduced successful calibration can update a view')
    if fresh['problem']['source_binding'] != make_scale_source_binding(graph, fresh['view_id']):
        raise ValueError('Calibration source, page or coordinate-frame revision differs from the graph')
    data = deepcopy(graph.model_dump(mode='json'))
    view = next((v for v in data['views'] if v['id'] == fresh['view_id']), None)
    if view is None or view['coordinate_space'] != fresh['coordinate_unit']:
        raise ValueError('Calibration view or coordinate space does not match the graph')
    previous_view = deepcopy(view)
    view.update(model_scale_xy=fresh['model_scale_xy'], physical_scale_confirmed=True,
                scale_evidence_refs=fresh['scale_evidence_refs'], deformation_confirmed=fresh['deformation_confirmed'],
                deformation_evidence_ref=fresh['deformation_evidence_ref'])
    history = data['provenance'].setdefault('view_calibration_v2', {})
    if not isinstance(history, dict) or fresh['view_id'] in history:
        raise ValueError('Preserve earlier calibration evidence; use an explicit new graph revision')
    history[fresh['view_id']] = {'previous_view': previous_view, 'result': deepcopy(fresh)}
    return DrawingGraphV2.model_validate(data)
