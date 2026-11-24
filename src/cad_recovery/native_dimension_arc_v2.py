"""Bounded, contact-aware source observations around a tangent arc extremum.

Several independent flank extents are checked, including the frozen V1 extent.
No foreground is removed and a contact never grants exclusive curve ownership.
An anchor is a geometric observation only, not a semantic PART decision.
"""
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np

from . import native_dimension_arc_v1 as frozen
from .part_admission_v1 import object_sha


POLICY = {**deepcopy(frozen.POLICY),
          'version': 'native-dimension-arc/2.0',
          'flank_extents_deg': [45., 55., 65.],
          'maximum_flank_alternatives': 3,
          # This is shared by all alternatives and all curves, not per trial.
          'maximum_samples': 500_000,
          'alternative_selection': 'widest_agreeing_source_supported_extent',
          'exclusive_source_pixel_ownership_confirmed': False}


def _sections(ink, center, radius, angles, reach, audit):
    """Keep V1 section acceptance, while retaining every ambiguous profile."""
    offsets = np.arange(-reach, reach+.25, .5)
    rows, occupied, contacts = [], set(), []
    for theta in angles:
        radial = np.array([math.cos(theta), math.sin(theta)])
        seed = tuple(np.rint(center+radius*radial).astype(int))
        if seed in occupied:
            continue
        occupied.add(seed)
        pixels = np.rint(center+(radius+offsets)[:, None]*radial).astype(int)
        audit['samples'] += len(pixels)
        if audit['samples'] > POLICY['maximum_samples']:
            return rows, contacts, len(occupied), 'radial_section_sample_budget'
        if ((pixels < 0).any() or (pixels[:, 0] >= ink.shape[1]).any()
                or (pixels[:, 1] >= ink.shape[0]).any()):
            return rows, contacts, len(occupied), 'radial_section_clipped_by_roi'
        observed = ink[pixels[:, 1], pixels[:, 0]]
        changes = np.diff(np.r_[False, observed, False].astype(np.int8))
        runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        boundary = any(start == 0 or end == len(offsets) for start, end in runs)
        if len(runs) != 1 or boundary:
            contacts.append({'theta_deg': float(math.degrees(theta) % 360),
                'nominal_point_px': (center+radius*radial).tolist(),
                'radial_foreground_intervals_px': [
                    [float(offsets[start]-.25), float(offsets[end-1]+.25)] for start, end in runs],
                'reason': ('source_ink_missing' if not runs else
                           'radial_foreground_reaches_search_boundary' if boundary else
                           'multiple_radial_foreground_runs'),
                'foreground_owner': 'unresolved', 'used_for_circle_fit': False,
                'exclusive_arc_ownership_confirmed': False})
            continue
        start, end = runs[0]
        low, high = offsets[start]-.25, offsets[end-1]+.25
        rows.append({'theta': float(theta), 'point': center+(radius+(low+high)/2)*radial,
                     'width': float(high-low)})
    reason = None
    if len(rows) < POLICY['minimum_sections_per_side'] or len(rows) < POLICY['minimum_side_fraction']*len(occupied):
        reason = 'insufficient_unambiguous_arc_sections'
    return rows, contacts, len(occupied), reason


def _coverage(distance, center, radius, theta, stroke, half_angle_deg, audit):
    """Check the entire fitted arc between both selected outer flanks."""
    half_angle = math.radians(half_angle_deg)
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
    # Retain V1's centre-point test: a tangent must not cover an erased arc.
    allowance = POLICY['pixel_quantization_px']
    allowed_gap = stroke+2*allowance
    visible = distance[q[:, 1], q[:, 0]] <= allowance
    changes = np.diff(np.r_[False, ~visible, False].astype(np.int8))
    gaps = np.flatnonzero(changes == -1)-np.flatnonzero(changes == 1)
    maximum_gap = float(gaps.max(initial=0)*arc_length/(count-1))
    proof = {'fraction': float(visible.mean()), 'maximum_unexplained_gap_px': maximum_gap,
             'allowed_unexplained_gap_px': allowed_gap, 'pixel_radius_px': allowance,
             'source_curve_samples': count, 'half_angle_deg': half_angle_deg,
             'foreground_owner': 'unresolved', 'exclusive_arc_ownership_confirmed': False}
    if proof['fraction'] < POLICY['minimum_curve_coverage'] or maximum_gap > allowed_gap:
        return proof, 'source_arc_between_flanks_and_extremum_missing'
    return proof, None


def _agreement(accepted, theta):
    """Compare every accepted fit over the union of their observed extents.

    Input uncertainty is not used as a licence to choose a competing stroke.
    The agreement bound covers two fits' pixel quantization and residuals.
    """
    comparisons = []
    for index, first in enumerate(accepted):
        for second in accepted[index+1:]:
            extent = max(first['half_angle_deg'], second['half_angle_deg'])
            angles = theta+np.radians(np.linspace(-extent, extent, 97))
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
            a, b = first['evidence'], second['evidence']
            ac, bc = np.asarray(a['curve_center_px']), np.asarray(b['curve_center_px'])
            ar, br = a['curve_radius_px'], b['curve_radius_px']
            # Polar angles use each fitted centre. Corresponding numeric
            # angles therefore do not identify corresponding curve points:
            # their Euclidean difference includes tangential reparameterization.
            # Compare the two loci in both normal directions instead, and
            # separately keep the actual extremum-position disagreement.
            first_points, second_points = ac+ar*directions, bc+br*directions
            normal_shift = float(max(
                np.abs(np.linalg.norm(first_points-bc, axis=1)-br).max(),
                np.abs(np.linalg.norm(second_points-ac, axis=1)-ar).max()))
            axis = np.array([math.cos(theta), math.sin(theta)])
            anchor_shift = float(np.linalg.norm(ac+ar*axis-bc-br*axis))
            shift = max(normal_shift, anchor_shift)
            bound = (a['fit_residual95_px']+b['fit_residual95_px']
                     +2*POLICY['pixel_quantization_px'])
            comparisons.append({'flank_extents_deg': [first['half_angle_deg'], second['half_angle_deg']],
                'maximum_curve_disagreement_px': shift, 'allowed_disagreement_px': bound,
                'maximum_normal_curve_disagreement_px': normal_shift,
                'extremum_disagreement_px': anchor_shift,
                'metric': 'symmetric_point_to_circle_and_extremum_distance',
                'passed': shift <= bound})
    return comparisons


def observe(gray, curves, extension, dimension_axis):
    """Observe local original-pixel curves; same coordinate/input API as V1.

    The three declared angular extents are always evaluated, not searched
    until a favourable answer appears. Their domain, radial corridor,
    uncertainty, fit, and complete between-flank pixel support are checked.
    Contact profiles retain their denominator in the unchanged 80% side gate.
    Neither clean radial profiles nor overall ink coverage prove pixel owners.
    """
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError('Nonempty uint8 source ROI required')
    axis = frozen._point(dimension_axis, 'dimension_axis').copy()
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
    width = frozen._positive(extension['width_px'], 'extension.width_px')
    extension_uncertainty = frozen._positive(extension['uncertainty_px'], 'extension.uncertainty_px')
    ids = [curve.get('id') for curve in curves]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique curve IDs required')
    report = {'schema_version': POLICY['version'], 'anchors': [], 'events': [], 'complete': True,
              'source_pixels_sha256': None, 'source_size_px': list(gray.shape[::-1]),
              'curves_sha256': object_sha(curves), 'extension_sha256': object_sha(extension),
              'dimension_axis': axis.tolist(), 'policy': deepcopy(POLICY),
              'audit': {'samples': 0, 'limits': [], 'alternatives_evaluated': 0},
              'semantic_part_confirmed': False, 'physical_geometry_approved': False,
              'source_pixels_changed': 0, 'coordinate_space': 'original_roi_pixels',
              'exclusive_source_pixel_ownership_confirmed': False}
    extents = POLICY['flank_extents_deg']
    if gray.size > POLICY['maximum_roi_pixels'] or len(curves) > POLICY['maximum_curves']:
        report['audit']['limits'].append('roi_or_curve_budget')
    if len(extents) > POLICY['maximum_flank_alternatives']:
        report['audit']['limits'].append('flank_alternative_budget')
    if report['audit']['limits']:
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
        center = frozen._point(curve['center_px'], 'curve.center_px')
        radius = frozen._positive(curve['radius_px'], 'curve.radius_px')
        uncertainty = frozen._positive(curve['uncertainty_px'], 'curve.uncertainty_px')
        if curve['kind'] == 'arc':
            if not np.isfinite(curve['start_angle_deg']) or not 0 < frozen._positive(curve['sweep_angle_deg'], 'arc.sweep') <= 360:
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
                        or not 0 < frozen._positive(interval[1], 'input angular sweep') <= 360):
                    raise ValueError('Original angular intervals must be [start, positive sweep]')
        reach = math.ceil(uncertainty+3*width+2)
        for sign in (-1, 1):
            nominal = center+sign*radius*axis
            offset = float((nominal-endpoints[0]) @ unit)
            perpendicular = abs(frozen._cross(unit, nominal-endpoints[0]))
            candidate_bound = uncertainty+extension_uncertainty+POLICY['pixel_quantization_px']
            if perpendicular > candidate_bound or offset < -candidate_bound or offset > length+candidate_bound:
                continue
            event = {'curve_id': curve['id'], 'extremum_sign': sign, 'status': 'rejected',
                     'reasons': [], 'alternatives': [], 'alternative_agreement': []}
            report['events'].append(event)
            if reach > POLICY['maximum_radial_reach_px']:
                report['audit']['limits'].append('radial_reach_budget')
                break
            theta = math.atan2(sign*axis[1], sign*axis[0])
            minimum_angle = max(POLICY['minimum_angle_deg'],
                math.degrees(math.acos(radius/(radius+reach+2*width+extension_uncertainty)))+1.)
            accepted = []
            for half_angle in extents:
                report['audit']['alternatives_evaluated'] += 1
                trial = {'half_angle_deg': half_angle, 'minimum_angle_deg': minimum_angle,
                         'status': 'rejected', 'reasons': [], 'sides': []}
                event['alternatives'].append(trial)
                if minimum_angle >= half_angle:
                    trial['reasons'].append('curve_too_small_for_independent_tangent_flanks')
                    continue
                offsets = np.radians(np.linspace(minimum_angle, half_angle, POLICY['sections_per_side']))
                angles = [theta-offsets, theta+offsets]
                if not all(frozen._in_arc(t, curve) for group in angles for t in group) or not frozen._in_arc(theta, curve):
                    trial['reasons'].append('extremum_or_two_sided_neighbourhood_outside_arc_domain')
                    continue
                sides = []
                for group in angles:
                    rows, contacts, occupied, reason = _sections(ink, center, radius, group, reach, report['audit'])
                    trial['sides'].append({'single_run_sections': len(rows), 'observed_sections': occupied,
                        'ambiguous_sections': contacts, 'exclusive_arc_ownership_confirmed': False})
                    if reason:
                        trial['reasons'].append(reason)
                        if reason.endswith('_budget'):
                            report['audit']['limits'].append(reason)
                        break
                    sides.append(rows)
                if report['audit']['limits']:
                    break
                if len(sides) != 2:
                    continue
                rows = sides[0]+sides[1]
                points = np.asarray([row['point'] for row in rows])
                fitted = frozen._fit(points)
                if fitted is None:
                    trial['reasons'].append('source_sections_do_not_fit_one_circle')
                    continue
                actual_center, actual_radius, residual, maximum_residual = fitted
                support_angles = np.r_[angles[0], theta, angles[1]]
                directions = np.column_stack((np.cos(support_angles), np.sin(support_angles)))
                displacement = actual_center-center+(actual_radius-radius)*directions
                shift = float(np.linalg.norm(displacement, axis=1).max())
                if shift > uncertainty+POLICY['pixel_quantization_px']:
                    trial['reasons'].append('curve_refit_exceeds_input_uncertainty')
                    continue
                native_anchor = actual_center+sign*actual_radius*axis
                fitted_error = uncertainty+POLICY['pixel_quantization_px']
                gap = abs(frozen._cross(unit, native_anchor-endpoints[0]))
                along = float((native_anchor-endpoints[0]) @ unit)
                if gap > extension_uncertainty+fitted_error or not -fitted_error <= along <= length+fitted_error:
                    trial['reasons'].append('fitted_extremum_not_on_observed_extension')
                    continue
                measured_width = float(np.median([row['width'] for row in rows]))
                coverage, reason = _coverage(distance, actual_center, actual_radius, theta,
                                             measured_width, half_angle, report['audit'])
                trial['source_curve_coverage'] = coverage
                if reason:
                    trial['reasons'].append(reason)
                    if reason.endswith('_budget'):
                        report['audit']['limits'].append(reason)
                        break
                    continue
                q = np.rint(native_anchor).astype(int)
                if (q < 0).any() or q[0] >= gray.shape[1] or q[1] >= gray.shape[0] or not ink[q[1], q[0]]:
                    trial['reasons'].append('fitted_extremum_source_ink_missing')
                    continue
                proof = {'curve_center_px': actual_center.tolist(), 'curve_radius_px': actual_radius,
                    'input_curve_center_px': center.tolist(), 'input_curve_radius_px': radius,
                    'input_uncertainty_px': uncertainty, 'refit_shift_bound_px': shift,
                    'refit_center_shift_px': float(np.linalg.norm(actual_center-center)),
                    'refit_radius_shift_px': float(abs(actual_radius-radius)),
                    'measured_curve_width_px': measured_width, 'fit_residual95_px': residual,
                    'fit_max_residual_px': maximum_residual, 'localization_uncertainty_px': fitted_error,
                    'two_sided_source_sections': [len(side) for side in sides],
                    'source_curve_coverage': coverage, 'source_section_points_px': points.tolist(),
                    'arc_domain_checked': True, 'extension_perpendicular_gap_px': gap,
                    'extension_along_px': along, 'extension_endpoint_distance_px': min(along, length-along),
                    'extension_terminal_confirmed': False, 'through_crossings_resolved': False,
                    'extension_pixels_rechecked_here': False,
                    'source_curve_geometry_confirmed': True, 'semantic_part_confirmed': False,
                    'curve_geometry_confirmation_scope': 'two_sided_extremum_neighbourhood_only',
                    'input_angular_intervals_deg': deepcopy(intervals),
                    'input_curve_ids': deepcopy(curve.get('input_curve_ids', [curve['id']])),
                    'source_interval_merge_approved': False, 'physical_geometry_approved': False,
                    'coordinate_space': 'original_roi_pixels', 'flank_extent_deg': half_angle,
                    'exclusive_source_pixel_ownership_confirmed': False,
                    'ambiguous_sections': [c for side in trial['sides'] for c in side['ambiguous_sections']]}
                trial.update(status='source_curve_anchor_observed', fit_summary={
                    key: proof[key] for key in ('curve_center_px', 'curve_radius_px', 'fit_residual95_px')},
                    anchor_px=native_anchor.tolist())
                accepted.append({'half_angle_deg': half_angle, 'anchor_px': native_anchor.tolist(), 'evidence': proof})
            if report['audit']['limits']:
                break
            event['alternative_agreement'] = _agreement(accepted, theta)
            if any(not comparison['passed'] for comparison in event['alternative_agreement']):
                event['reasons'].append('conflicting_source_supported_arc_alternatives')
                continue
            if not accepted:
                event['reasons'] = sorted({reason for trial in event['alternatives'] for reason in trial['reasons']})
                continue
            chosen = max(accepted, key=lambda row: row['half_angle_deg'])
            chosen['evidence']['accepted_flank_extents_deg'] = [r['half_angle_deg'] for r in accepted]
            chosen['evidence']['alternative_agreement'] = deepcopy(event['alternative_agreement'])
            chosen['evidence']['contact_ambiguity_persisted'] = any(
                side['ambiguous_sections'] for trial in event['alternatives'] for side in trial['sides'])
            report['anchors'].append({'curve_id': curve['id'], 'curve_kind': curve['kind'],
                'anchor_px': chosen['anchor_px'], 'extremum_sign': sign, 'evidence': chosen['evidence'],
                'source_members': deepcopy(curve.get('source_members', []))})
            event.update(status='source_curve_anchor_observed', selected_flank_extent_deg=chosen['half_angle_deg'])
        if report['audit']['limits']:
            break
    if report['audit']['limits']:
        report.update(complete=False, anchors=[])
    return report
