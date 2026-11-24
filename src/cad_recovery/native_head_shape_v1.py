"""Measure visible ink flanks and their termination; never certify a CAD head.

The extrapolated intersection of exterior ink edges is not the logical apex of
a stroked CAD arrow.  The admitted contact, protections and roles stay intact.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from .native_dimension_transfer_v1 import _sha
from .native_head_membership_v1 import _checked_heads, _nonnegative, _supported_matrix, _unique
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix


POLICY = {'version': 'native-head-shape/1.0', 'maximum_roi_pixels': 4_000_000,
    'maximum_heads': 2, 'maximum_profile_samples': 500_000,
    'maximum_stations': 1025, 'maximum_offsets': 1025,
    'maximum_search_length_px': 512., 'maximum_half_width_px': 128.,
    'maximum_axis_uncertainty_px': 64., 'maximum_patterns': 64,
    'maximum_fit_attempts': 4096, 'maximum_shoulder_samples': 100_000,
    'station_step_px': 1., 'transverse_step_px': .5,
    'minimum_flank_stations': 12, 'minimum_flank_span_in_strokes': 2.,
    'minimum_flank_length_fraction': .35, 'minimum_edge_slope': .05,
    'maximum_edge_slope': 1.5, 'minimum_base_to_shaft_ratio': 3.,
    'minimum_shaft_run_px': 8., 'shoulder_probes_per_side': 3,
    'whole_triangle_observed': False, 'logical_apex_verified': False,
    'whole_triangle_ownership_confirmed': False,
    'semantic_exclusive_head_confirmed': False, 'independently_calibrated': False,
    'roles_changed': 0, 'cad_modification': False, 'physical_geometry_approved': False}

PIXEL_RADIUS = math.sqrt(.5)


class _Limit(ValueError):
    pass


def _point(value, label):
    p = np.asarray(value, float)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError('Finite '+label+' required')
    return p


def _runs(bits):
    changes = np.diff(np.r_[False, bits, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _cross_sections(q, hit, positions, offsets, reference, axis, normal, uncertainty):
    stations = []
    for i, s in enumerate(positions):
        center = reference+s*axis
        runs = []
        for lo, stop in _runs(hit[i]):
            hi = stop-1
            first, last = q[i, lo], q[i, hi]
            left = q[i, lo-1] if lo > 0 else None
            right = q[i, stop] if stop < len(offsets) else None
            li = sorted([float((left-center)@normal), float((first-center)@normal)]) if left is not None else None
            ri = sorted([float((last-center)@normal), float((right-center)@normal)]) if right is not None else None
            edge_left, edge_right = (sum(li)/2 if li is not None else None,
                                     sum(ri)/2 if ri is not None else None)
            runs.append({'lo_px': float(offsets[lo]), 'hi_px': float(offsets[hi]),
                'first_sample_index': int(lo), 'last_sample_index': int(hi),
                'first_cell': first.tolist(), 'last_cell': last.tolist(),
                'left_white_cell': None if left is None else left.tolist(),
                'right_white_cell': None if right is None else right.tolist(),
                'left_bracket_observed': left is not None, 'right_bracket_observed': right is not None,
                'left_boundary_interval_px': li, 'right_boundary_interval_px': ri,
                'left_edge_px': edge_left, 'right_edge_px': edge_right,
                'width_px': None if li is None or ri is None else edge_right-edge_left,
                'distinct_ink_cells': int(len(np.unique(q[i, lo:stop], axis=0)))})
        candidates = [j for j, run in enumerate(runs)
                      if run['lo_px'] <= uncertainty and run['hi_px'] >= -uncertainty]
        stations.append({'s_px': float(s), 'runs': runs,
            'central_run_index': candidates[0] if len(candidates) == 1 else None,
            'axis_run_ambiguous': len(candidates) > 1,
            'axis_candidate_run_indices': candidates})
    return stations


def _central(station):
    index = station['central_run_index']
    return None if index is None else station['runs'][index]


def _fit(stations, indices, sign, shaft, counter):
    """Retain the longest visible suffix meeting explicit ink-pattern checks.

    This chooses a maximal observed flank segment, not a nearest head owner or
    an apex fitted to its admitted contact.  Unused prefix stations remain in
    the complete profile and cannot authorize a whole triangle.
    """
    minimum = POLICY['minimum_flank_stations']
    for start in range(max(0, len(indices)-minimum+1)):
        counter['fits'] += 1
        if counter['fits'] > POLICY['maximum_fit_attempts']:
            raise _Limit('head_shape_fit_budget')
        selected = indices[start:]
        if len(selected) < minimum:
            break
        t = np.asarray([sign*stations[i]['s_px'] for i in selected])
        if t[-1]-t[0] < max(8., POLICY['minimum_flank_span_in_strokes']*shaft):
            continue
        samples = np.asarray([[_central(stations[i])['left_edge_px'],
                               _central(stations[i])['right_edge_px']] for i in selected])
        design = np.c_[t, np.ones(len(t))]
        params = np.linalg.lstsq(design, samples, rcond=None)[0]
        slopes = [-float(params[0, 0]), float(params[0, 1])]
        if not all(POLICY['minimum_edge_slope'] <= slope <= POLICY['maximum_edge_slope'] for slope in slopes):
            continue
        residuals = abs(design@params-samples)
        if float(residuals.max()) > max(1.5, .75*shaft):
            continue
        spread = float(params[0, 1]-params[0, 0])
        apex = float((params[1, 0]-params[1, 1])/spread)
        if t[-1] <= apex or (t[-1]-t[0])/(t[-1]-apex) < POLICY['minimum_flank_length_fraction']:
            continue
        sides = {}
        for j, label in enumerate(('left', 'right')):
            sides[label] = {'slope': float(params[0, j]), 'intercept': float(params[1, j]),
                'max_residual_px': float(residuals[:, j].max()),
                'edge_error_p95_px': float(np.quantile(residuals[:, j], .95)),
                'sample_count': len(selected), 'station_interval_px': [float(t[0]), float(t[-1])]}
        return {'sides': sides, 'flank_station_indices': list(map(int, selected)),
                'flank_coordinate_sign': sign, 'apex_along_axis_px': apex,
                'unused_prefix_station_indices': list(map(int, indices[:start]))}
    return None


def _shoulders(stations, q, hit, positions, offsets, indices, last_wide, first_thin, thin_stop,
               fit, sign, shaft, axis_uncertainty, reference, axis, normal, counter):
    last, first = indices[last_wide], indices[first_thin]
    t = float(sign*positions[last])
    left = fit['sides']['left']; right = fit['sides']['right']
    edge_values = [left['slope']*t+left['intercept'], right['slope']*t+right['intercept']]
    probes = []
    confirmation = max(2, math.ceil(shaft/2))
    end = min(thin_stop, first_thin+confirmation+1)
    if end-first_thin < confirmation:
        return None
    observed_wide = _central(stations[last])
    observed_edges = [observed_wide['left_edge_px'], observed_wide['right_edge_px']]
    shaft_runs = [_central(stations[i]) for i in indices[first_thin:end]]
    shaft_bounds = [min(run['left_edge_px'] for run in shaft_runs)-axis_uncertainty-PIXEL_RADIUS,
                    max(run['right_edge_px'] for run in shaft_runs)+axis_uncertainty+PIXEL_RADIUS]
    selection = {}
    for j, (side, edge) in enumerate(zip(('left', 'right'), edge_values)):
        # Rounded ink corners need not reach the extrapolated flank at the
        # final wide row. Probe only their measured intersection. The shaft's
        # observed transverse position may differ from the reference contact.
        inner = max(0., -shaft_bounds[0] if side == 'left' else shaft_bounds[1])+PIXEL_RADIUS
        outer = min(abs(edge), abs(observed_edges[j]))-2*PIXEL_RADIUS
        selection[side] = {'inner_distance_px': inner, 'outer_distance_px': outer}
        if outer <= inner+1:
            return None
        chosen = np.linspace(inner, outer, POLICY['shoulder_probes_per_side'])
        for amount in chosen:
            offset = -float(amount) if side == 'left' else float(amount)
            column = int(np.argmin(abs(offsets-offset)))
            sampled_indices = indices[last_wide:end]
            counter['shoulder_samples'] += len(sampled_indices)
            if counter['shoulder_samples'] > POLICY['maximum_shoulder_samples']:
                raise _Limit('head_shape_shoulder_budget')
            values = hit[sampled_indices, column]
            if not values[0]:
                return None
            changes = [j for j in range(len(values)-1) if values[j] and not values[j+1]]
            valid = [j for j in changes if last_wide+j < first_thin
                     and not values[j+1:].any() and len(values)-j-1 >= confirmation]
            if len(valid) != 1:
                return None
            j = valid[0]
            black_index, white_index = sampled_indices[j], sampled_indices[j+1]
            black, white = q[black_index, column], q[white_index, column]
            projected = [float((p-reference)@(sign*axis)) for p in (black, white)]
            probes.append({'side': side, 'offset_px': float(offsets[column]),
                'sample_column': column, 'last_ink_station_index': int(black_index),
                'first_white_station_index': int(white_index),
                'last_ink_cell': black.tolist(), 'first_white_cell': white.tolist(),
                'last_ink_observed': True, 'first_white_observed': True,
                'transition_interval_px': [min(projected)-PIXEL_RADIUS, max(projected)+PIXEL_RADIUS],
                'white_station_indices': list(map(int, sampled_indices[j+1:])),
                'white_cells': q[sampled_indices[j+1:], column].tolist()})
    interval = [min(p['transition_interval_px'][0] for p in probes),
                max(p['transition_interval_px'][1] for p in probes)]
    if interval[1]-interval[0] > max(3., shaft)+2*PIXEL_RADIUS:
        return None
    middle = sum(interval)/2
    return {'interval_along_axis_px': interval, 'center_along_axis_px': middle,
        'interval_px': [(reference+sign*axis*s).tolist() for s in interval],
        'shoulder_probes': probes, 'sustained_thin_shaft_observed': True,
        'thin_station_indices': list(map(int, indices[first_thin:thin_stop])),
        'last_wide_station_index': int(last), 'first_thin_station_index': int(first),
        'base_is_interval_not_exact': True, 'normal_unit': normal.tolist(),
        'shoulder_selection_bounds': selection,
        'observed_last_wide_edge_offsets_px': observed_edges,
        'fitted_last_wide_edge_offsets_px': edge_values,
        'observed_thin_shaft_envelope_px': shaft_bounds,
        'shoulder_selection_scope': 'observed_and_fitted_wide_run_intersection_outside_measured_thin_shaft_envelope',
        'rounded_corners_are_not_claimed_as_exact_base_vertices': True}


def _patterns(stations, q, hit, positions, offsets, sign, shaft, uncertainty,
              reference, axis, normal, counter):
    indices = [i for i in range(len(stations)) if sign*positions[i] >= 0]
    if sign < 0:
        indices.reverse()
    width = np.asarray([float('nan') if _central(stations[i]) is None
        or _central(stations[i])['width_px'] is None else _central(stations[i])['width_px'] for i in indices])
    thin_limit = max(shaft+2., 1.6*shaft)
    thin = np.isfinite(width) & (width > 0) & (width <= thin_limit)
    wide_limit = POLICY['minimum_base_to_shaft_ratio']*shaft
    patterns, rejected = [], []
    for start, stop in _runs(thin):
        if stop-start < max(POLICY['minimum_shaft_run_px'], 2*shaft):
            continue
        nearby = np.flatnonzero(np.isfinite(width[:start]) & (width[:start] >= wide_limit))
        if not len(nearby):
            continue
        last_wide = int(nearby[-1])
        if start-last_wide > max(3., shaft):
            continue
        block_start = last_wide
        while block_start > 0 and np.isfinite(width[block_start-1]) and width[block_start-1] > thin_limit:
            block_start -= 1
        fit = _fit(stations, indices[block_start:last_wide+1], sign, shaft, counter)
        if fit is None:
            rejected.append({'first_thin_station_index': int(indices[start]), 'reason': 'no_bilateral_visible_flanks'})
            continue
        base = _shoulders(stations, q, hit, positions, offsets, indices, last_wide, start, stop,
            fit, sign, shaft, uncertainty, reference, axis, normal, counter)
        if base is None:
            rejected.append({'first_thin_station_index': int(indices[start]), 'reason': 'base_shoulders_not_independently_visible'})
            continue
        center = base['center_along_axis_px']
        apex_t = fit['apex_along_axis_px']
        if not 6 <= center-apex_t <= POLICY['maximum_search_length_px']:
            continue
        left, right = fit['sides']['left'], fit['sides']['right']
        apex_r = left['slope']*apex_t+left['intercept']
        apex = reference+sign*axis*apex_t+normal*apex_r
        ends = [reference+sign*axis*center+normal*(side['slope']*center+side['intercept']) for side in (left, right)]
        conditional_edge = max(left['max_residual_px'], right['max_residual_px'])+PIXEL_RADIUS+uncertainty
        conditional_apex = 2*conditional_edge/(right['slope']-left['slope'])
        patterns.append({**fit, 'direction': 'forward' if sign > 0 else 'opposite',
            'base': base, 'measured_polygon_px': [apex.tolist(), *(p.tolist() for p in ends)],
            'raw_silhouette_extrapolated_apex_px': apex.tolist(), 'apex_extrapolated': True,
            'conditional_edge_envelope_px': conditional_edge,
            'conditional_apex_longitudinal_envelope_px': conditional_apex,
            'conditional_envelope_scope': 'empirical_visible_ink_edges_plus_pixel_and_axis_conditional_on_fitted_straight_flanks',
            'whole_triangle_observed': False, 'logical_apex_verified': False,
            'whole_triangle_ownership_confirmed': False, 'physical_geometry_approved': False,
            'independently_calibrated': False, 'semantic_exclusive_head_confirmed': False})
        if len(patterns) > POLICY['maximum_patterns']:
            raise _Limit('head_shape_pattern_budget')
    # An opposite taper need not terminate inside the observed window. Keep its
    # visible flanks independently of whether a full base is available.
    taper_blocks = []
    for start, stop in _runs(np.isfinite(width) & (width > thin_limit)):
        fit = _fit(stations, indices[start:stop], sign, shaft, counter)
        if fit is not None:
            taper_blocks.append(fit)
    return patterns, taper_blocks, rejected


def measure(gray, *, tip_px, toward_base, shaft_width_px, axis_uncertainty_px=0.,
            search_length_px=192., backward_length_px=None, half_width_px=64.,
            head_id=None, tip_uncertainty_px=None):
    """Sample all native transverse runs and measure finite ink patterns.

    ``tip_px`` is a reference contact, never a constraint on fitted flanks.
    ``complete`` means sampling completed; negative patterns can be complete.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError('Nonempty native uint8 gray window required')
    reference, axis = _point(tip_px, 'reference contact'), _point(toward_base, 'source axis')
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length <= 0:
        raise ValueError('Nondegenerate source axis required')
    axis = axis/length
    normal = np.array([-axis[1], axis[0]])
    shaft = float(shaft_width_px)
    if not math.isfinite(shaft) or not 1 <= shaft <= 24:
        raise ValueError('Source shaft width in [1,24] required')
    uncertainty = _nonnegative(axis_uncertainty_px)
    forward = _nonnegative(search_length_px)
    backward = forward if backward_length_px is None else _nonnegative(backward_length_px)
    half = _nonnegative(half_width_px)
    tip_bound = None if tip_uncertainty_px is None else _nonnegative(tip_uncertainty_px)
    report = {'schema_version': POLICY['version'], 'policy': deepcopy(POLICY), 'head_id': head_id,
        'source_window_sha256': _sha(gray), 'source_size_px': list(gray.shape[::-1]),
        'complete': False, 'sampling_complete': False, 'limits': [], 'reasons': [],
        'shape_pattern_observed': False, 'flanks_base_observed': False, 'opposing_taper_observed': False,
        'whole_triangle_observed': False, 'logical_apex_verified': False,
        'whole_triangle_ownership_confirmed': False,
        'semantic_exclusive_head_confirmed': False, 'independently_calibrated': False,
        'physical_geometry_approved': False,
        'head_uncertainty_reduced': False, 'roles_changed': 0, 'cad_modification': False,
        'patterns': [], 'opposing_patterns': [], 'opposing_tapers': [],
        'sides': None, 'base': None, 'measured_polygon_px': None,
        'raw_silhouette_extrapolated_apex_px': None, 'apex_extrapolated': True,
        'raw_silhouette_apex_native_ink_observed': None,
        'contact_consistency': None, 'profile': None, 'partial_observation_rolled_back': False,
        'shaft_width_px': shaft, 'axis_uncertainty_px': uncertainty,
        'search_length_px': forward, 'backward_length_px': backward, 'half_width_px': half,
        'preserved_tip_bound_px': tip_bound}
    before = _sha(gray)
    try:
        if gray.size > POLICY['maximum_roi_pixels']:
            raise _Limit('head_shape_roi_budget')
        if max(forward, backward) > POLICY['maximum_search_length_px'] or forward <= 0:
            raise _Limit('head_shape_search_length_budget')
        if half <= 0 or half > POLICY['maximum_half_width_px']:
            raise _Limit('head_shape_half_width_budget')
        if uncertainty > POLICY['maximum_axis_uncertainty_px']:
            raise _Limit('head_shape_axis_uncertainty_budget')
        positions = np.arange(-math.floor(backward), math.floor(forward)+1, dtype=float)
        step = POLICY['transverse_step_px']
        offsets = np.arange(-math.floor(half/step), math.floor(half/step)+1, dtype=float)*step
        if len(positions) > POLICY['maximum_stations'] or len(offsets) > POLICY['maximum_offsets']:
            raise _Limit('head_shape_station_or_offset_budget')
        samples = len(positions)*len(offsets)
        if samples > POLICY['maximum_profile_samples']:
            raise _Limit('head_shape_profile_sample_budget')
        q = np.rint(reference+positions[:, None, None]*axis+offsets[None, :, None]*normal).astype(int)
        if (q < 0).any() or (q >= np.asarray(gray.shape[::-1])).any():
            raise _Limit('head_shape_source_sampling_clipped')
        threshold, raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        ink = raw > 0
        hit = ink[q[..., 1], q[..., 0]]
        stations = _cross_sections(q, hit, positions, offsets, reference, axis, normal, uncertainty)
        report['profile'] = {'reference_tip_px': reference.tolist(), 'axis_unit': axis.tolist(),
            'normal_unit': normal.tolist(), 'station_positions_px': positions.tolist(),
            'transverse_offsets_px': offsets.tolist(), 'sampling_rule': 'rint(reference + s*axis + offset*normal)',
            'otsu_threshold': float(threshold), 'stations': stations, 'source_samples': samples,
            'distinct_source_cells': int(len(np.unique(q.reshape(-1, 2), axis=0))),
            'repeated_samples_are_not_independent_observations': True,
            'boundary_estimator': 'midpoint_of_projected_adjacent_ink_and_white_native_cell_centres',
            'sample_offsets_and_boundaries_are_distinct': True}
        counter = {'fits': 0, 'shoulder_samples': 0}
        patterns, _, rejected = _patterns(stations, q, hit, positions, offsets, 1, shaft,
            uncertainty, reference, axis, normal, counter)
        opposite, tapers, opposite_rejected = _patterns(stations, q, hit, positions, offsets, -1, shaft,
            uncertainty, reference, axis, normal, counter)
        report.update(complete=True, sampling_complete=True, patterns=patterns,
            opposing_patterns=opposite, opposing_tapers=tapers,
            opposing_taper_observed=bool(opposite or tapers), pattern_count=len(patterns),
            ambiguous_base_patterns=len(patterns) > 1, rejected_transitions=rejected,
            opposing_rejected_transitions=opposite_rejected,
            fit_attempts=counter['fits'], shoulder_samples=counter['shoulder_samples'])
        if len(patterns) == 1:
            pattern = patterns[0]
            apex = np.asarray(pattern['raw_silhouette_extrapolated_apex_px'])
            cell = np.rint(apex).astype(int)
            inside = bool((cell >= 0).all() and (cell < np.asarray(gray.shape[::-1])).all())
            distance = float(np.linalg.norm(apex-reference))
            flanks = pattern['sides']
            report.update(shape_pattern_observed=True, flanks_base_observed=True,
                sides=flanks, base=pattern['base'], measured_polygon_px=pattern['measured_polygon_px'],
                raw_silhouette_extrapolated_apex_px=apex.tolist(),
                raw_silhouette_apex_native_ink_observed=bool(ink[cell[1], cell[0]]) if inside else False,
                contact_consistency={'reference_tip_px': reference.tolist(),
                    'raw_silhouette_apex_distance_px': distance, 'preserved_tip_bound_px': tip_bound,
                    'fitted_point_within_preserved_tip_bound': None if tip_bound is None else distance <= tip_bound,
                    'exterior_flank_distances_px': [abs(s['intercept'])/math.sqrt(1+s['slope']**2)
                        for s in (flanks['left'], flanks['right'])],
                    'shaft_half_width_px': shaft/2, 'logical_contact_rejected': False,
                    'scope': 'raw_ink_silhouette_vs_admitted_contact_not_CAD_logical_apex_validation',
                    'possible_causes': ['stroke_outline_model', 'occlusion', 'nontriangular_ink_boundary']})
        else:
            report['reasons'] = ['multiple_measured_base_patterns' if patterns else 'no_visible_bilateral_flanks_and_base']
    except _Limit as exc:
        report.update(complete=False, sampling_complete=False, limits=[str(exc)],
            shape_pattern_observed=False, flanks_base_observed=False, opposing_taper_observed=False,
            patterns=[], opposing_patterns=[], opposing_tapers=[], profile=None,
            partial_observation_rolled_back=True)
    if _sha(gray) != before:
        raise ValueError('Head shape observation changed native source pixels')
    return {'report': report, 'complete': report['complete'],
        'shape_pattern_observed': report['shape_pattern_observed'],
        'opposing_taper_observed': report['opposing_taper_observed']}


def observe(gray, working_gray, geometry, roles, chain_row, context_row, matrix, *,
            head_refinement, hard_protection, source_heads, admitted_heads, membership):
    """Bind new measurements to the current source-only membership inventory."""
    matrix = checked_matrix(matrix)
    if (gray.ndim != 2 or gray.dtype != np.uint8
            or working_gray.ndim != 2 or working_gray.dtype != np.uint8
            or working_gray.shape != roles.shape or roles.dtype != np.uint8
            or hard_protection.shape != roles.shape or hard_protection.dtype != np.uint8):
        raise ValueError('Aligned native and working source, role and protection maps required')
    box = np.asarray(chain_row['source_box_px'], float)
    if (box.shape != (4,) or not np.isfinite(box).all()
            or tuple((box[2:]-box[:2])[::-1]) != gray.shape
            or chain_row['source_box_px'] != context_row['source_box_px']):
        raise ValueError('Current head shape source window differs')
    if not _supported_matrix(matrix):
        raise ValueError('Supported native source registration required')
    refinement = head_refinement.get('report', head_refinement)
    member = membership.get('report', membership)
    expected = {'source_window_sha256': _sha(gray), 'working_gray_sha256': _sha(working_gray),
        'roles_sha256': _sha(roles), 'hard_protection_sha256': _sha(hard_protection),
        'geometry_sha256': object_sha(geometry), 'source_heads_sha256': object_sha(source_heads),
        'admitted_heads_sha256': object_sha(admitted_heads), 'chain_sha256': object_sha(chain_row),
        'context_sha256': object_sha(context_row), 'head_refinement_sha256': object_sha(refinement)}
    if (member.get('complete') is not True or member.get('source_ink_partition_complete') is not True
            or member.get('source_context_id') != chain_row['id']
            or member.get('working_to_source') != matrix.tolist()
            or any(member.get(k) != v for k, v in expected.items())
            or member.get('semantic_exclusive_head_confirmed') is not False
            or member.get('roles_changed') != 0 or member.get('cad_modification') is not False):
        raise ValueError('Fresh unchanged source head membership required')
    structures = [geometry, chain_row, context_row, refinement, source_heads, admitted_heads, member]
    arrays = [gray, working_gray, roles, hard_protection, matrix]
    before = object_sha(structures), [_sha(a) for a in arrays]
    heads = _checked_heads(geometry, chain_row, context_row, matrix, refinement, source_heads, admitted_heads)
    lines = _unique(context_row['stroke_observation']['lines'], 'measured line')
    report = {'schema_version': POLICY['version'], 'policy': deepcopy(POLICY),
        'source_context_id': chain_row['id'], 'source_box_px': list(chain_row['source_box_px']),
        **expected, 'membership_sha256': object_sha(member), 'working_to_source': matrix.tolist(),
        'complete': True, 'sampling_complete': True, 'limits': [], 'heads': [],
        'shape_pattern_observed_count': 0, 'semantic_exclusive_head_confirmed': False,
        'whole_triangle_observed': False, 'logical_apex_verified': False,
        'whole_triangle_ownership_confirmed': False,
        'independently_calibrated': False, 'head_uncertainty_reduced': False,
        'physical_geometry_approved': False,
        'roles_changed': 0, 'cad_modification': False,
        'head_guard16_or_existing_bound_preserved': True,
        'chain_head_tolerance_px': chain_row['head_tolerance_px'],
        'native_head_protection_sha256': refinement['native_protection_sha256']}
    if len(heads) > POLICY['maximum_heads']:
        report.update(complete=False, sampling_complete=False, limits=['head_shape_head_budget'])
    else:
        for head in heads:
            line = lines[head['carrier_ids'][0]]
            p = np.asarray(line['points_px'], float)
            axis = p[1]-p[0]
            axis /= np.linalg.norm(axis)
            current = next(s for s in head['shapes'] if s['basis'] == 'current_admitted')
            polygon = current['polygon']
            reference = polygon[0]
            seed = polygon[1:].mean(0)-reference
            if axis@seed < 0:
                axis = -axis
            seed_length = float(np.linalg.norm(seed))
            guard = max(s['protection_bound'] for s in head['shapes'])
            width = float(line['width_px'])
            half = float(np.linalg.norm(polygon[1]-polygon[2]))/2+guard+2*width
            result = measure(gray, tip_px=reference, toward_base=axis, shaft_width_px=width,
                axis_uncertainty_px=line['uncertainty_px'],
                search_length_px=seed_length+guard+max(8., 4*width),
                backward_length_px=seed_length+guard, half_width_px=half,
                head_id=head['id'], tip_uncertainty_px=current['own_bound'])
            row = result['report']
            row['measured_carrier_points_px'] = p.tolist()
            row['profile_axis_scope'] = 'parallel_to_measured_carrier_through_unchanged_admitted_contact'
            row['profile_reference_offset_from_measured_carrier_px'] = abs(float(
                (reference-p[0])@np.array([-axis[1], axis[0]])))
            row['inherited_protection_bound_px'] = guard
            row['old_and_current_head_guards_unchanged'] = True
            report['heads'].append(row)
        if any(not row['complete'] for row in report['heads']):
            report.update(complete=False, sampling_complete=False,
                limits=sorted({reason for row in report['heads'] for reason in row['limits']}))
        report['shape_pattern_observed_count'] = sum(row['shape_pattern_observed'] for row in report['heads'])
    if before != (object_sha(structures), [_sha(a) for a in arrays]):
        raise ValueError('Head shape observation changed its source evidence')
    return {'report': report, 'complete': report['complete']}
