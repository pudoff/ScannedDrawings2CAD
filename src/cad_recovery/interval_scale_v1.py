"""Exact axis-projected scale compatibility under authored pixel intervals.

This module checks mathematical compatibility of explicitly reviewed inputs.
It neither authenticates review files nor approves physical scale. The caller
must verify evidence/source hashes against actual files before use.

Decimal input values are interpreted as Fraction(str(value)); in particular,
chains of 0.1 and 0.2 are exactly consistent with 0.3. Returned float interval
endpoints are rounded outward; exact rational strings are authoritative.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=256)]
SHA = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Coordinate = Annotated[float, Field(ge=0, le=1e12, allow_inf_nan=False)]
Nominal = Annotated[float, Field(ge=1e-9, le=1e12, allow_inf_nan=False)]


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, validate_assignment=True,
                              revalidate_instances='always', allow_inf_nan=False)


class ReviewRefV1(Model):
    path: Identifier
    sha256: SHA
    locator: Identifier

    @model_validator(mode='after')
    def contained_path(self):
        p = PurePosixPath(self.path)
        if '\\' in self.path or ':' in self.path or p.is_absolute() or '..' in p.parts or p.as_posix() != self.path:
            raise ValueError('Review path must be normalized and workspace-relative')
        return self


class SourceFrameBindingV1(Model):
    source_sha256: SHA
    page_index: int = Field(ge=0, le=100000, strict=True)
    frame_id: Identifier
    source_size_px: list[Annotated[int, Field(ge=2, le=1000000000001, strict=True)]] = Field(min_length=2, max_length=2)
    unit: Literal['source_pixel'] = 'source_pixel'
    axes: Literal['x_right_y_down'] = 'x_right_y_down'
    coordinate_convention: Literal['original_pixel_centers_inclusive_intervals'] = 'original_pixel_centers_inclusive_intervals'
    frame_sha256: SHA

    @model_validator(mode='after')
    def frame_hash(self):
        value = self.model_dump(mode='json', exclude={'frame_sha256'})
        if self.frame_sha256 != _digest(value):
            raise ValueError('Frame hash does not bind the source, page, coordinate convention and extent')
        return self


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def make_source_frame_binding_v1(source_sha256, page_index, frame_id, source_size_px):
    """Create a binding to original source pixel centers, never a fitted frame."""
    value = {'source_sha256': source_sha256, 'page_index': page_index, 'frame_id': frame_id,
             'source_size_px': list(source_size_px), 'unit': 'source_pixel', 'axes': 'x_right_y_down',
             'coordinate_convention': 'original_pixel_centers_inclusive_intervals'}
    return SourceFrameBindingV1.model_validate({**value, 'frame_sha256': _digest(value)})


class IntervalAnchorV1(Model):
    id: Identifier
    frame_sha256: SHA
    box_xyxy: list[Coordinate] = Field(min_length=4, max_length=4)
    reviewed: bool = Field(strict=True)
    review_refs: list[ReviewRefV1] = Field(min_length=1, max_length=8)
    bound_kind: Literal['authored_interval_not_statistical_or_physical_accuracy'] = 'authored_interval_not_statistical_or_physical_accuracy'

    @model_validator(mode='after')
    def interval(self):
        if not self.reviewed:
            raise ValueError('Unreviewed anchor intervals cannot calibrate or validate scale')
        x0, y0, x1, y1 = self.box_xyxy
        if x0 > x1 or y0 > y1:
            raise ValueError('Anchor boxes must have ordered inclusive bounds')
        return self


class ProjectedSpanV1(Model):
    id: Identifier
    frame_sha256: SHA
    anchor_a: Identifier
    anchor_b: Identifier
    axis: Literal['X', 'Y']
    source_direction: Literal['positive', 'negative']
    kind: Literal['projected_length'] = 'projected_length'
    nominal_mm: Nominal
    unit: Literal['mm']
    nominal_reviewed: bool = Field(strict=True)
    direction_reviewed: bool = Field(strict=True)
    review_refs: list[ReviewRefV1] = Field(min_length=1, max_length=8)
    broken_view: bool = Field(default=False, strict=True)

    @model_validator(mode='after')
    def reviewed_projection(self):
        if not self.nominal_reviewed or not self.direction_reviewed:
            raise ValueError('Printed nominal, unit and projection direction need explicit review')
        if self.anchor_a == self.anchor_b or self.broken_view:
            raise ValueError('Scale needs distinct supports and an unbroken projected span')
        return self


class HoldoutSpanV1(ProjectedSpanV1):
    independent_review: bool = Field(strict=True)
    independence_scope: Literal['review_not_used_to_select_or_fit_calibration'] = 'review_not_used_to_select_or_fit_calibration'

    @model_validator(mode='after')
    def independently_reviewed(self):
        if not self.independent_review:
            raise ValueError('Holdout review must be independent of selecting or fitting the calibration')
        return self


class IntervalScaleProblemV1(Model):
    schema_version: Literal['interval-scale/1.0'] = 'interval-scale/1.0'
    purpose: Literal['authored_interval_compatibility_not_physical_scale_approval'] = 'authored_interval_compatibility_not_physical_scale_approval'
    binding: SourceFrameBindingV1
    anchors: list[IntervalAnchorV1] = Field(min_length=3, max_length=128)
    calibration: list[ProjectedSpanV1] = Field(min_length=2, max_length=128)
    holdouts: list[HoldoutSpanV1] = Field(default_factory=list, max_length=128)

    @model_validator(mode='after')
    def references_and_review(self):
        anchors = {a.id: a for a in self.anchors}
        observations = [*self.calibration, *self.holdouts]
        if len(anchors) != len(self.anchors) or len({o.id for o in observations}) != len(observations):
            raise ValueError('Anchor and observation IDs must be distinct within their inventories')
        if {o.axis for o in self.calibration} != {'X', 'Y'}:
            raise ValueError('At least two independent projected directions X and Y must calibrate scale')
        evidence, duplicate_spans, artifact_revisions = set(), set(), {}
        for item in [*self.anchors, *observations]:
            if item.frame_sha256 != self.binding.frame_sha256:
                raise ValueError('Mixed source/page/frame evidence is not allowed')
            for ref in item.review_refs:
                if ref.path in artifact_revisions and artifact_revisions[ref.path] != ref.sha256:
                    raise ValueError('One review artifact path cannot claim multiple byte revisions')
                artifact_revisions[ref.path] = ref.sha256
                token = (ref.path, ref.sha256, ref.locator)
                if token in evidence:
                    raise ValueError('Duplicate evidence reference: each reviewed record needs its own explicit locator')
                evidence.add(token)
        for anchor in self.anchors:
            if anchor.box_xyxy[2] > self.binding.source_size_px[0]-1 or anchor.box_xyxy[3] > self.binding.source_size_px[1]-1:
                raise ValueError('Anchor interval lies outside original source pixel-center extent')
        for o in observations:
            if o.anchor_a not in anchors or o.anchor_b not in anchors:
                raise ValueError('Observation references an unknown anchor')
            # Reversing both support order and direction is the same observation.
            a, b = (o.anchor_a, o.anchor_b) if o.source_direction == 'positive' else (o.anchor_b, o.anchor_a)
            key = (a, b, o.axis, _fraction(o.nominal_mm))
            if key in duplicate_spans:
                raise ValueError('Repeated projected-span evidence cannot count as another calibration or holdout')
            duplicate_spans.add(key)
            low, _ = _span_bounds(o, anchors)
            if low <= 0:
                raise ValueError('Projected span touches or crosses zero; a positive reviewed direction is required')
        return self


def _fraction(value):
    return Fraction(str(value))


def _span_bounds(observation, anchors):
    axis = 0 if observation.axis == 'X' else 1
    a, b = anchors[observation.anchor_a].box_xyxy, anchors[observation.anchor_b].box_xyxy
    low = _fraction(b[axis])-_fraction(a[axis+2])
    high = _fraction(b[axis+2])-_fraction(a[axis])
    return (low, high) if observation.source_direction == 'positive' else (-high, -low)


def _outward(value, lower):
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError('Derived interval exceeds finite floating representation') from error
    if not math.isfinite(result):
        raise ValueError('Derived interval exceeds finite floating representation')
    actual = Fraction.from_float(result)
    if lower and actual > value:
        result = math.nextafter(result, -math.inf)
    elif not lower and actual < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise ValueError('Outward interval endpoint exceeds finite floating representation')
    return result


def _interval(low, high, unit):
    if low > high:
        return None
    return {'lower': _outward(low, True), 'upper': _outward(high, False),
            'lower_exact': str(low), 'upper_exact': str(high), 'unit': unit,
            'bounds': 'inclusive_exact_rationals_with_outward_rounded_float_display'}


def _components(observations, anchor_ids, axis):
    adjacency = defaultdict(list)
    for o in observations:
        if o.axis != axis:
            continue
        signed = _fraction(o.nominal_mm)*(1 if o.source_direction == 'positive' else -1)
        adjacency[o.anchor_a].append((o.anchor_b, signed, o.id))
        adjacency[o.anchor_b].append((o.anchor_a, -signed, o.id))
    seen, components = {}, []
    for start in sorted(anchor_ids):
        if start in seen:
            continue
        seen[start] = Fraction(0)
        queue, potentials = [start], {}
        while queue:
            node = queue.pop()
            potentials[node] = seen[node]
            for neighbor, nominal, identity in adjacency[node]:
                candidate = seen[node]+nominal
                if neighbor in seen:
                    if seen[neighbor] != candidate:
                        return None, {'code': 'inconsistent_shared_anchor_cycle', 'axis': axis,
                                      'observation_id': identity, 'exact_nominal_cycle_residual_mm': str(candidate-seen[neighbor])}
                else:
                    seen[neighbor] = candidate
                    queue.append(neighbor)
        components.append(potentials)
    return components, None


def _joint_feasible(observations, anchors, t_low, t_high):
    """Exact shared-anchor feasibility for t=px/mm and axis-specific translations.

    Each component has coordinate_i = translation + potential_i * t. Its
    original boxes require L_i-potential_i*t <= U_j-potential_j*t for all i,j.
    These linear inequalities in t are necessary AND sufficient; no marginal
    independence assumption or optimizer tolerance enters the calculation.
    """
    used_ids = {identity for o in observations for identity in (o.anchor_a, o.anchor_b)}
    anchors = {identity: anchors[identity] for identity in sorted(used_ids)}
    all_components = {}
    for axis, offset in [('X', 0), ('Y', 1)]:
        components, failure = _components(observations, anchors, axis)
        if failure:
            return {'feasible': False, 'reason': failure}
        all_components[axis] = components
        for potentials in components:
            for left, p_left in potentials.items():
                lower = _fraction(anchors[left].box_xyxy[offset])
                for right, p_right in potentials.items():
                    upper = _fraction(anchors[right].box_xyxy[offset+2])
                    coefficient, rhs = p_right-p_left, upper-lower
                    if coefficient > 0:
                        t_high = min(t_high, rhs/coefficient)
                    elif coefficient < 0:
                        t_low = max(t_low, rhs/coefficient)
                    elif rhs < 0:
                        return {'feasible': False, 'reason': {'code': 'shared_anchor_boxes_incompatible', 'axis': axis}}
                    if t_low > t_high or t_high <= 0:
                        return {'feasible': False, 'reason': {'code': 'shared_anchor_constraints_have_empty_intersection', 'axis': axis}}
    if t_low <= 0:
        raise ValueError('Internal positive scale bound lost')
    # One mathematical witness, explicitly not a fitted or observed source point.
    t = t_low
    coordinates = {identity: [None, None] for identity in anchors}
    for axis, offset in [('X', 0), ('Y', 1)]:
        for potentials in all_components[axis]:
            translation = max(_fraction(anchors[identity].box_xyxy[offset])-potential*t
                              for identity, potential in potentials.items())
            for identity, potential in potentials.items():
                value = translation+potential*t
                if not _fraction(anchors[identity].box_xyxy[offset]) <= value <= _fraction(anchors[identity].box_xyxy[offset+2]):
                    raise ValueError('Internal exact witness lies outside an original box')
                coordinates[identity][offset] = value
    for o in observations:
        offset = 0 if o.axis == 'X' else 1
        actual = (coordinates[o.anchor_b][offset]-coordinates[o.anchor_a][offset])*(1 if o.source_direction == 'positive' else -1)
        if actual != _fraction(o.nominal_mm)*t:
            raise ValueError('Internal exact witness does not satisfy a projected span')
    witness = {'purpose': 'mathematical_feasibility_witness_not_measured_or_corrected_coordinates',
               'selection': 'lowest_feasible_pixels_per_mm_and_component_translation',
               'pixels_per_mm_exact': str(t), 'mm_per_pixel_exact': str(1/t),
               'anchor_coordinates': {key: {'x_exact': str(value[0]), 'y_exact': str(value[1]),
                                            'approximate_xy': [float(value[0]), float(value[1])]}
                                      for key, value in sorted(coordinates.items())},
               'equations_verified_exact': True, 'source_boxes_unchanged': True,
               'physical_scale_confirmed': False, 'training_eligible': False}
    return {'feasible': True, 't_low': t_low, 't_high': t_high, 'witness': witness}


def read_interval_scale_problem_v1(value):
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.stat().st_size > 4*1024**2:
            raise ValueError('Interval problem JSON exceeds the 4 MiB budget')
        def pairs(items):
            out = {}
            for key, child in items:
                if key in out:
                    raise ValueError('Duplicate JSON key: '+key)
                out[key] = child
            return out
        value = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))
    if isinstance(value, IntervalScaleProblemV1):
        value = value.model_dump(mode='json')
    return IntervalScaleProblemV1.model_validate(value)


def solve_interval_scale_v1(problem):
    """Return compatibility, exact bounds and a witness; never a physical approval.

    Holdouts are checked only against the fixed calibration solution set. Their
    hypothetical compatibility witnesses do not replace the calibration interval
    or witness. Individual compatibility does not imply joint independence.
    """
    problem = read_interval_scale_problem_v1(problem)
    anchors = {a.id: a for a in problem.anchors}
    rows, scale_bounds = [], []
    for o in problem.calibration:
        low, high = _span_bounds(o, anchors)
        lower, upper = _fraction(o.nominal_mm)/high, _fraction(o.nominal_mm)/low
        scale_bounds.append((lower, upper))
        rows.append({'id': o.id, 'axis': o.axis, 'source_direction': o.source_direction,
                     'anchor_a': o.anchor_a, 'anchor_b': o.anchor_b, 'nominal_mm': o.nominal_mm,
                     'kind': o.kind, 'unit': o.unit,
                     'review_refs': [ref.model_dump(mode='json') for ref in o.review_refs],
                     'span_px': _interval(low, high, 'source_pixel'),
                     'scale_mm_per_px': _interval(lower, upper, 'mm_per_source_pixel'),
                     'used_for_calibration': True})
    marginal_low = max(x[0] for x in scale_bounds)
    marginal_high = min(x[1] for x in scale_bounds)
    marginal = _interval(marginal_low, marginal_high, 'mm_per_source_pixel')
    if marginal is None:
        joint = {'feasible': False, 'reason': {'code': 'projected_scale_marginal_intersection_empty'}}
    else:
        joint = _joint_feasible(problem.calibration, anchors, 1/marginal_high, 1/marginal_low)
    calibrated = {'status': 'jointly_compatible_under_authored_bounds' if joint['feasible'] else 'conflict',
                  'participating_anchor_ids': sorted({a for o in problem.calibration for a in (o.anchor_a, o.anchor_b)}),
                  'marginal_intersection': marginal, 'marginal_intersection_scope': 'necessary_compatibility_only',
                  'scale_mm_per_px': _interval(1/joint['t_high'], 1/joint['t_low'], 'mm_per_source_pixel') if joint['feasible'] else None,
                  'pixels_per_mm': _interval(joint['t_low'], joint['t_high'], 'source_pixel_per_mm') if joint['feasible'] else None,
                  'shared_anchor_correlations_checked': True, 'feasibility_witness': joint.get('witness'),
                  'conflict_reason': joint.get('reason'), 'observations': rows,
                  'coordinate_estimate_selected': False, 'physical_scale_confirmed': False}
    calibration_anchor_ids = {a for o in problem.calibration for a in (o.anchor_a, o.anchor_b)}
    holdouts = []
    for o in problem.holdouts:
        low, high = _span_bounds(o, anchors)
        check = _joint_feasible([*problem.calibration, o], anchors, joint['t_low'], joint['t_high']) if joint['feasible'] else None
        holdouts.append({'id': o.id, 'axis': o.axis, 'nominal_mm': o.nominal_mm,
            'anchor_a': o.anchor_a, 'anchor_b': o.anchor_b, 'source_direction': o.source_direction,
            'kind': o.kind, 'unit': o.unit, 'review_refs': [ref.model_dump(mode='json') for ref in o.review_refs],
            'independent_review_attested': o.independent_review, 'independence_scope': o.independence_scope,
            'status': ('compatible' if check['feasible'] else 'conflict') if check is not None else 'unresolved',
            'reason': check.get('reason') if check is not None else {'code': 'calibration_is_inconsistent'},
            'span_px': _interval(low, high, 'source_pixel'), 'used_for_calibration': False,
            'shared_calibration_anchor_ids': sorted({o.anchor_a, o.anchor_b} & calibration_anchor_ids),
            'compatibility_scope': 'existence_under_original_authored_boxes_not_statistical_or_physical_accuracy',
            'compatibility_witness': check.get('witness') if check is not None else None,
            'physical_scale_confirmed': False})
    combined = None
    if problem.holdouts and joint['feasible']:
        check = _joint_feasible([*problem.calibration, *problem.holdouts], anchors, joint['t_low'], joint['t_high'])
        combined = {'status': 'compatible' if check['feasible'] else 'conflict', 'used_for_calibration': False,
                    'reason': check.get('reason'), 'meaning': 'Diagnostic coexistence of all holdouts; original calibration is unchanged.'}
    return {'schema_version': 'interval-scale-result/1.0', 'purpose': problem.purpose,
            'binding': problem.binding.model_dump(mode='json'), 'problem_content_sha256': _digest(problem.model_dump(mode='json')),
            'calibration': calibrated, 'holdouts': holdouts, 'combined_holdout_check': combined,
            'input_anchors': [a.model_dump(mode='json') for a in problem.anchors],
            'coordinate_estimate_selected': False, 'physical_scale_confirmed': False,
            'annotation_approved': False, 'training_eligible': False, 'split': None,
            'limitations': ['Author intervals are assumed bounds, not validated accuracy or statistical confidence.',
                'Projected X/Y spans are used; a chamfer projection is not its diagonal length.',
                'Witness coordinates prove feasibility only and never replace observed source coordinates.',
                'Holdouts never fit calibration. Compatibility is existential under stated bounds.',
                'Independent review is an input attestation, not proof of temporal or statistical independence.',
                'Caller must verify actual source and review artifact hashes; syntactic pins are not customer approval.']}
