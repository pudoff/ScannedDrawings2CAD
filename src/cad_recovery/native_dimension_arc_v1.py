"""Source-pixel curve witnesses for linear dimensions tangent to an arc.

The result is a geometric anchor proposal, not a PART role or a CAD dimension.
Every accepted extremum is fitted from radial source sections on both sides.
"""
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np

from .part_admission_v1 import object_sha


POLICY = {'version': 'native-dimension-arc/1.0', 'maximum_roi_pixels': 4_000_000,
          'maximum_curves': 32, 'maximum_samples': 500_000, 'maximum_radial_reach_px': 128,
          'maximum_input_angular_intervals': 64,
          'sections_per_side': 48, 'minimum_sections_per_side': 8,
          'minimum_side_fraction': .8, 'minimum_angle_deg': 12., 'maximum_angle_deg': 45.,
          'minimum_curve_coverage': .95,
          'maximum_fit_residual_px': float(math.sqrt(2)),
          'maximum_tangent_angle_deg': 8., 'pixel_quantization_px': float(math.sqrt(.5)),
          'semantic_part_confirmed': False, 'physical_geometry_approved': False}


def _point(value, label):
    value = np.asarray(value, float)
    if value.shape != (2,) or not np.isfinite(value).all():
        raise ValueError(label+' must be a finite pixel point')
    return value


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(label+' must be finite and positive')
    return float(value)


def _cross(first, second):
    return float(first[0]*second[1]-first[1]*second[0])


def _in_arc(theta, curve):
    intervals = curve.get('input_angular_intervals_deg')
    if intervals is None and curve['kind'] == 'circle':
        return True
    angle = math.degrees(theta) % 360
    if intervals is None:
        intervals = [[curve['start_angle_deg'], curve['sweep_angle_deg']]]
    return any((angle-float(start)) % 360 <= float(sweep)+1e-8 for start, sweep in intervals)


def _sections(ink, center, radius, angles, reach, audit):
    offsets = np.arange(-reach, reach+.25, .5)
    rows, occupied = [], set()
    for theta in angles:
        radial = np.array([math.cos(theta), math.sin(theta)])
        seed = tuple(np.rint(center+radius*radial).astype(int))
        if seed in occupied:
            continue
        occupied.add(seed)
        pixels = np.rint(center+(radius+offsets)[:, None]*radial).astype(int)
        audit['samples'] += len(pixels)
        if audit['samples'] > POLICY['maximum_samples']:
            return None, 'radial_section_sample_budget'
        if ((pixels < 0).any() or (pixels[:, 0] >= ink.shape[1]).any()
                or (pixels[:, 1] >= ink.shape[0]).any()):
            return None, 'radial_section_clipped_by_roi'
        observed = ink[pixels[:, 1], pixels[:, 0]]
        changes = np.diff(np.r_[False, observed, False].astype(np.int8))
        runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        # A neighbouring concentric stroke cannot be selected by proximity.
        if len(runs) != 1:
            continue
        start, end = runs[0]
        if start == 0 or end == len(offsets):
            continue
        low, high = offsets[start]-.25, offsets[end-1]+.25
        rows.append({'theta': float(theta), 'point': center+(radius+(low+high)/2)*radial,
                     'width': float(high-low)})
    if len(rows) < POLICY['minimum_sections_per_side'] or len(rows) < POLICY['minimum_side_fraction']*len(occupied):
        return None, 'insufficient_unambiguous_arc_sections'
    return rows, None


def _fit(points):
    origin = points.mean(0)
    local = points-origin
    matrix = np.column_stack((2*local, np.ones(len(local))))
    solution, _, rank, _ = np.linalg.lstsq(matrix, np.square(local).sum(1), rcond=None)
    if rank != 3:
        return None
    center = origin+solution[:2]
    square = solution[2]+float(solution[:2] @ solution[:2])
    if square <= 0:
        return None
    radius = float(math.sqrt(square))
    residual = np.abs(np.linalg.norm(points-center, axis=1)-radius)
    if np.quantile(residual, .95) > POLICY['maximum_fit_residual_px']:
        return None
    return center, radius, float(np.quantile(residual, .95)), float(residual.max())


def _coverage(distance, center, radius, theta, stroke, audit):
    half_angle = math.radians(POLICY['maximum_angle_deg'])
    arc_length = 2*radius*half_angle
    count = max(3, math.ceil(arc_length*2)+1)
    audit['samples'] += count
    if audit['samples'] > POLICY['maximum_samples']:
        return None, 'arc_coverage_sample_budget'
    angles = theta+np.linspace(-half_angle, half_angle, count)
    points = center+radius*np.column_stack((np.cos(angles), np.sin(angles)))
    q = np.rint(points).astype(int)
    if ((q < 0).any() or (q[:, 0] >= distance.shape[1]).any()
            or (q[:, 1] >= distance.shape[0]).any()):
        return None, 'fitted_arc_coverage_clipped_by_roi'
    # These are fitted centre points, not approximate input geometry. Expanding
    # their search by half the stroke again could borrow tangent-line ink and
    # falsely cover a missing arc beside it.
    allowance = POLICY['pixel_quantization_px']
    allowed_gap = stroke+2*POLICY['pixel_quantization_px']
    visible = distance[q[:, 1], q[:, 0]] <= allowance
    changes = np.diff(np.r_[False, ~visible, False].astype(np.int8))
    gaps = np.flatnonzero(changes == -1)-np.flatnonzero(changes == 1)
    maximum_gap = float(gaps.max(initial=0)*arc_length/(count-1))
    proof = {'fraction': float(visible.mean()), 'maximum_unexplained_gap_px': maximum_gap,
             'allowed_unexplained_gap_px': allowed_gap, 'pixel_radius_px': allowance,
             'source_curve_samples': count}
    if proof['fraction'] < POLICY['minimum_curve_coverage'] or maximum_gap > allowed_gap:
        return proof, 'source_arc_between_flanks_and_extremum_missing'
    return proof, None


def observe(gray, curves, extension, dimension_axis):
    """All inputs and returned coordinates are original pixels of this ROI.

    Curves use center_px/radius_px and (for arcs) positive image-coordinate
    start_angle_deg/sweep_angle_deg. Extension points_px are source-refitted
    endpoints with measured width_px and uncertainty_px. Source members pass
    through unchanged; supplied prior roles confer no semantic authority.
    Optional input_angular_intervals_deg are [start, sweep] pairs from one
    authenticated exact circle family. Flanks and extrema must be inside their
    union; missing interval coverage between flanks still requires source ink.
    """
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError('Nonempty uint8 source ROI required')
    axis = _point(dimension_axis, 'dimension_axis')
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        raise ValueError('Nonzero dimension axis required')
    axis /= norm
    endpoints = np.asarray(extension['points_px'], float)
    if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
        raise ValueError('Finite extension endpoints required')
    delta = endpoints[1]-endpoints[0]
    length = float(np.linalg.norm(delta))
    if length <= 0:
        raise ValueError('Nondegenerate extension required')
    unit = delta/length
    width = _positive(extension['width_px'], 'extension.width_px')
    extension_uncertainty = _positive(extension['uncertainty_px'], 'extension.uncertainty_px')
    ids = [curve.get('id') for curve in curves]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique curve IDs required')
    report = {'schema_version': POLICY['version'], 'anchors': [], 'events': [], 'complete': True,
              'source_pixels_sha256': None,
              'source_size_px': list(gray.shape[::-1]), 'curves_sha256': object_sha(curves),
              'extension_sha256': object_sha(extension), 'dimension_axis': axis.tolist(),
              'policy': deepcopy(POLICY), 'audit': {'samples': 0, 'limits': []},
              'semantic_part_confirmed': False, 'physical_geometry_approved': False,
              'source_pixels_changed': 0, 'coordinate_space': 'original_roi_pixels'}
    if gray.size > POLICY['maximum_roi_pixels'] or len(curves) > POLICY['maximum_curves']:
        report['audit']['limits'].append('roi_or_curve_budget')
        report['complete'] = False
        return report
    report['source_pixels_sha256'] = hashlib.sha256(gray.tobytes()).hexdigest()
    if abs(float(unit @ axis)) > math.sin(math.radians(POLICY['maximum_tangent_angle_deg'])):
        report['events'].append({'reason': 'extension_not_perpendicular_to_dimension_axis'})
        return report
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
    distance = cv2.distanceTransform((~ink).astype(np.uint8)*255, cv2.DIST_L2, 5)
    for curve in curves:
        if curve.get('kind') not in ('arc', 'circle'):
            raise ValueError('Only arc/circle witnesses are accepted')
        center = _point(curve['center_px'], 'curve.center_px')
        radius = _positive(curve['radius_px'], 'curve.radius_px')
        uncertainty = _positive(curve['uncertainty_px'], 'curve.uncertainty_px')
        if curve['kind'] == 'arc':
            if not np.isfinite(curve['start_angle_deg']) or not 0 < _positive(curve['sweep_angle_deg'], 'arc.sweep') <= 360:
                raise ValueError('Finite arc angle range required')
        intervals = curve.get('input_angular_intervals_deg')
        if intervals is not None:
            if not isinstance(intervals, list) or not intervals:
                raise ValueError('Nonempty original angular intervals required')
            if len(intervals) > POLICY['maximum_input_angular_intervals']:
                report['audit']['limits'].append('input_angular_interval_budget')
                break
            for interval in intervals:
                if (not isinstance(interval, list) or len(interval) != 2
                        or not np.isfinite(interval[0])
                        or not 0 < _positive(interval[1], 'input angular sweep') <= 360):
                    raise ValueError('Original angular intervals must be [start, positive sweep]')
        reach = math.ceil(uncertainty+3*width+2)
        for sign in (-1, 1):
            nominal = center+sign*radius*axis
            offset = float((nominal-endpoints[0]) @ unit)
            perpendicular = abs(_cross(unit, nominal-endpoints[0]))
            candidate_bound = uncertainty+extension_uncertainty+POLICY['pixel_quantization_px']
            if perpendicular > candidate_bound or offset < -candidate_bound or offset > length+candidate_bound:
                continue
            event = {'curve_id': curve['id'], 'extremum_sign': sign, 'status': 'rejected', 'reasons': []}
            report['events'].append(event)
            if reach > POLICY['maximum_radial_reach_px']:
                report['audit']['limits'].append('radial_reach_budget')
                break
            theta = math.atan2(sign*axis[1], sign*axis[0])
            # A radial ray near a tangent can meet the extension as well as
            # the curve. Observe flanks outside that entire search corridor.
            minimum_angle = max(POLICY['minimum_angle_deg'],
                math.degrees(math.acos(radius/(radius+reach+2*width+extension_uncertainty)))+1.)
            if minimum_angle >= POLICY['maximum_angle_deg']:
                event['reasons'].append('curve_too_small_for_independent_tangent_flanks')
                continue
            offsets = np.radians(np.linspace(minimum_angle, POLICY['maximum_angle_deg'], POLICY['sections_per_side']))
            angles = [theta-offsets, theta+offsets]
            if not all(_in_arc(t, curve) for group in angles for t in group) or not _in_arc(theta, curve):
                event['reasons'].append('extremum_or_two_sided_neighbourhood_outside_arc_domain')
                continue
            sides = []
            for group in angles:
                rows, reason = _sections(ink, center, radius, group, reach, report['audit'])
                if reason:
                    event['reasons'].append(reason)
                    if reason.endswith('_budget'):
                        report['audit']['limits'].append(reason)
                    break
                sides.append(rows)
            if len(sides) != 2:
                continue
            rows = sides[0]+sides[1]
            points = np.asarray([row['point'] for row in rows])
            fitted = _fit(points)
            if fitted is None:
                event['reasons'].append('source_sections_do_not_fit_one_circle')
                continue
            actual_center, actual_radius, residual, maximum_residual = fitted
            support_angles = np.r_[angles[0], theta, angles[1]]
            directions = np.column_stack((np.cos(support_angles), np.sin(support_angles)))
            displacement = actual_center-center+(actual_radius-radius)*directions
            shift = float(np.linalg.norm(displacement, axis=1).max())
            if shift > uncertainty+POLICY['pixel_quantization_px']:
                event['reasons'].append('curve_refit_exceeds_input_uncertainty')
                continue
            native_anchor = actual_center+sign*actual_radius*axis
            # Residual fit quality is not a calibrated localization error.
            # Preserve the caller's bound instead of shrinking it to residuals.
            fitted_error = uncertainty+POLICY['pixel_quantization_px']
            gap = abs(_cross(unit, native_anchor-endpoints[0]))
            along = float((native_anchor-endpoints[0]) @ unit)
            if gap > extension_uncertainty+fitted_error or not -fitted_error <= along <= length+fitted_error:
                event['reasons'].append('fitted_extremum_not_on_observed_extension')
                continue
            # The fitted extremum must itself be supported, not only the flanks.
            q = np.rint(native_anchor).astype(int)
            measured_width = float(np.median([row['width'] for row in rows]))
            coverage, reason = _coverage(distance, actual_center, actual_radius, theta,
                                         measured_width, report['audit'])
            if reason:
                event['reasons'].append(reason)
                if reason.endswith('_budget'):
                    report['audit']['limits'].append(reason)
                continue
            if (q < 0).any() or q[0] >= gray.shape[1] or q[1] >= gray.shape[0] or not ink[q[1], q[0]]:
                event['reasons'].append('fitted_extremum_source_ink_missing')
                continue
            proof = {'curve_center_px': actual_center.tolist(), 'curve_radius_px': actual_radius,
                     'input_curve_center_px': center.tolist(), 'input_curve_radius_px': radius,
                     'input_uncertainty_px': uncertainty, 'refit_shift_bound_px': shift,
                     'refit_center_shift_px': float(np.linalg.norm(actual_center-center)),
                     'refit_radius_shift_px': float(abs(actual_radius-radius)),
                     'measured_curve_width_px': measured_width, 'fit_residual95_px': residual,
                     'fit_max_residual_px': maximum_residual, 'localization_uncertainty_px': fitted_error,
                     'two_sided_source_sections': [len(side) for side in sides],
                     'source_curve_coverage': coverage,
                     'source_section_points_px': points.tolist(),
                     'arc_domain_checked': True, 'extension_perpendicular_gap_px': gap,
                     'extension_along_px': along, 'extension_endpoint_distance_px': min(along, length-along),
                     'extension_terminal_confirmed': False, 'through_crossings_resolved': False,
                     'extension_pixels_rechecked_here': False,
                     'source_curve_geometry_confirmed': True, 'semantic_part_confirmed': False,
                     'curve_geometry_confirmation_scope': 'two_sided_extremum_neighbourhood_only',
                     'input_angular_intervals_deg': deepcopy(intervals),
                     'input_curve_ids': deepcopy(curve.get('input_curve_ids', [curve['id']])),
                     'source_interval_merge_approved': False,
                     'physical_geometry_approved': False, 'coordinate_space': 'original_roi_pixels'}
            report['anchors'].append({'curve_id': curve['id'], 'curve_kind': curve['kind'],
                'anchor_px': native_anchor.tolist(), 'extremum_sign': sign, 'evidence': proof,
                'source_members': deepcopy(curve.get('source_members', []))})
            event.update(status='source_curve_anchor_observed', reasons=[])
        if report['audit']['limits']:
            break
    if report['audit']['limits']:
        report.update(complete=False, anchors=[])
    return report
