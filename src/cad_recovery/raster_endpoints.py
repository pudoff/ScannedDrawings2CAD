"""Local finite-stroke ends, with explicit assumptions about the ink cap.

An ink boundary is observable; the CAD center of a round cap is not that
boundary. Unknown caps therefore retain an interval and cannot return a
localized CAD endpoint. This module neither labels objects nor approves data.
"""
from __future__ import annotations

import math

import numpy as np

MAX_PIXELS = 4_194_304
STEP_PX = .5
DARK_THRESHOLD = 160


def _number(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def _vector(value, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (2,):
        raise ValueError(f'{name} must contain two finite coordinates')
    return np.array([_number(v, name, -32768, 32768) for v in value])


def _profile(rgb, origin, direction, positions, half_width):
    normal = np.array([-direction[1], direction[0]])
    offsets = np.linspace(-half_width, half_width, math.ceil(4*half_width)+1)
    step = float(offsets[1]-offsets[0])
    states, widths, centers = [], [], []
    for t in positions:
        center = origin+t*direction
        samples = center+offsets[:, None]*normal
        coords = np.floor(samples).astype(np.int64)
        if (np.any(coords[:, 0] < 0) or np.any(coords[:, 0]+1 >= rgb.shape[1])
                or np.any(coords[:, 1] < 0) or np.any(coords[:, 1]+1 >= rgb.shape[0])):
            states.append('outside'); widths.append(float('nan')); centers.append([float('nan')]*2)
            continue
        fraction = samples-coords
        x, y = coords[:, 0], coords[:, 1]
        fx, fy = fraction[:, 0, None], fraction[:, 1, None]
        intensity = ((1-fx)*(1-fy)*rgb[y, x] + fx*(1-fy)*rgb[y, x+1]
                     + (1-fx)*fy*rgb[y+1, x] + fx*fy*rgb[y+1, x+1])
        dark = np.max(intensity, axis=1) <= DARK_THRESHOLD
        changes = np.diff(np.r_[False, dark, False].astype(np.int8))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)-1
        state, width, point = 'missing', float('nan'), [float('nan')]*2
        if len(starts) > 1:
            state = 'multiple'
        elif len(starts) == 1:
            if starts[0] == 0 or ends[0] == len(offsets)-1:
                state = 'clipped'
            else:
                state = 'unique'
                width = float(offsets[ends[0]]-offsets[starts[0]]+step)
                point = center+(offsets[starts[0]]+offsets[ends[0]])/2*normal
        states.append(state); widths.append(width); centers.append(point)
    return np.array(states), np.array(widths), np.array(centers)


def _reject(report, reason, ambiguous=False):
    report['status'] = 'ambiguous' if ambiguous else 'not_supported'
    report['reasons'].append(reason)
    return report


def localize_endpoint(rgb, expected_xy, inward_xy, *, search_radius=12,
                      probe_length=64, half_width=12, cap_model='unknown'):
    """Find an ink end near ``expected_xy`` along a reviewer-supplied direction.

    ``inward_xy`` points from the expected end into the segment. ``butt`` and
    ``round`` are explicit cap assumptions, not automatically inferred styles.
    Thin continuation/extension lines are not silently treated as absent ink.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
            or max(rgb.shape[:2]) > 32768 or rgb.shape[0]*rgb.shape[1] > MAX_PIXELS):
        raise ValueError('Expected a nonempty RGB8 region within the 4-Mpixel limit')
    search_radius = _number(search_radius, 'search_radius', 2, 64)
    probe_length = _number(probe_length, 'probe_length', 24, 512)
    half_width = _number(half_width, 'half_width', 2, 64)
    if probe_length < 2*search_radius:
        raise ValueError('probe_length must be at least twice search_radius')
    if cap_model not in ('unknown', 'butt', 'round'):
        raise ValueError('cap_model must be unknown, butt or round')
    expected, direction = _vector(expected_xy, 'expected_xy'), _vector(inward_xy, 'inward_xy')
    if not 0 <= expected[0] < rgb.shape[1] or not 0 <= expected[1] < rgb.shape[0]:
        raise ValueError('expected_xy lies outside the region')
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        raise ValueError('inward_xy must be nonzero')
    direction /= length
    positions = np.linspace(-search_radius, probe_length, math.ceil(2*(probe_length+search_radius))+1)
    step = float(positions[1]-positions[0])
    report = {
        'schema_version': '1.0', 'method': 'unique-strip-ink-end-v1', 'status': 'not_supported',
        'observed_xy': None, 'boundary_xy': None, 'position_semantics': 'unresolved',
        'expected_xy': expected.tolist(), 'inward_xy': direction.tolist(), 'cap_model': cap_model,
        'search_radius_px': search_radius, 'probe_length_px': probe_length,
        'half_width_px': half_width, 'sampling': 'bilinear', 'axis_step_px': step,
        'normal_step_px': 2*half_width/math.ceil(4*half_width), 'dark_threshold': DARK_THRESHOLD,
        'training_eligible': False, 'annotation_approved': False, 'reasons': [],
        'limits': {'max_pixels': MAX_PIXELS, 'minimum_interior_unique_fraction': .8,
                   'maximum_fit_p95_px': 1.0, 'maximum_direction_change_deg': 10.,
                   'minimum_blank_before_tip_px': 2.},
        'uncertainty': {'coordinate_accuracy_bound_px': None,
                        'pixel_sampling_floor_px': 1., 'cap_model_verified_from_pixels': False},
        'limitations': ['Finite-grid bilinear sampling is not a calibrated coordinate error bound.',
                       'A localized endpoint depends on an explicitly supplied butt/round cap model.',
                       'This method does not separate a contour end from a continued extension line.'],
    }
    states, widths, centers = _profile(rgb, expected, direction, positions, half_width)
    interior = positions >= max(1.5*search_radius, .4*probe_length)
    valid = interior & (states == 'unique')
    report['initial_profile_counts'] = {str(v): int(np.sum(states == v)) for v in np.unique(states)}
    report['interior_unique_fraction'] = float(np.mean(states[interior] == 'unique'))
    if np.any(states == 'outside'):
        return _reject(report, 'search_strip_outside_region')
    if report['interior_unique_fraction'] < .8:
        return _reject(report, 'insufficient_unique_interior', bool(np.any(states[interior] == 'multiple')))
    points = centers[valid]
    mean = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points-mean, full_matrices=False)
    fitted = vectors[0]
    if fitted @ direction < 0:
        fitted = -fitted
    normal = np.array([-fitted[1], fitted[0]])
    residual = (points-mean) @ normal
    angle = math.degrees(math.acos(float(np.clip(fitted @ direction, -1, 1))))
    report['interior_fit'] = {'point_xy': mean.tolist(), 'direction_xy': fitted.tolist(),
                              'p95_residual_px': float(np.percentile(abs(residual), 95)),
                              'direction_difference_deg': angle}
    if report['interior_fit']['p95_residual_px'] > 1 or angle > 10:
        return _reject(report, 'unstable_interior_line', True)
    origin = mean+np.dot(expected-mean, fitted)*fitted
    if np.linalg.norm(origin-expected) > search_radius:
        return _reject(report, 'fitted_line_outside_search_radius')
    states, widths, _ = _profile(rgb, origin, fitted, positions, half_width)
    report['profile_counts'] = {str(v): int(np.sum(states == v)) for v in np.unique(states)}
    report['profile'] = [{'t_px': float(t), 'state': str(s), 'width_px': float(w) if math.isfinite(w) else None}
                         for t, s, w in zip(positions, states, widths)]
    if np.any(states == 'outside'):
        return _reject(report, 'fitted_strip_outside_region')
    plateau = widths[interior & (states == 'unique')]
    if len(plateau) < .8*np.sum(interior):
        return _reject(report, 'insufficient_fitted_interior')
    width = float(np.median(plateau))
    report['median_stroke_width_px'] = width
    if float(np.percentile(abs(plateau-width), 95)) > max(1.5, .2*width):
        return _reject(report, 'unstable_stroke_width', True)
    unique = np.flatnonzero(states == 'unique')
    if not len(unique):
        return _reject(report, 'no_unique_stroke')
    first = int(unique[0])
    # Every preceding cross section must be empty; otherwise the apparent
    # beginning can be an occlusion, another stroke or a window boundary.
    if first*step < 2 or not np.all(states[:first] == 'missing'):
        return _reject(report, 'no_blank_before_ink_end', bool(np.any(states[:first] == 'multiple')))
    if np.any(np.isin(states[first:], ['multiple', 'clipped'])):
        return _reject(report, 'merged_or_multiple_strokes_after_tip', True)
    missing = np.flatnonzero(states[first:] == 'missing')
    if len(missing):
        return _reject(report, 'stroke_gaps_after_tip')
    boundary_t = float((positions[first-1]+positions[first])/2)
    boundary = origin+boundary_t*fitted
    report['boundary_xy'] = boundary.tolist()
    report['position_semantics'] = 'thresholded_visible_ink_boundary'
    report['cap_center_interval_xy'] = [boundary.tolist(), (boundary+width/2*fitted).tolist()]
    # Width should be stable once past a possible round cap. Abrupt broadening
    # or persistent narrow continuation signals unmodeled geometry.
    stable = positions >= boundary_t+width/2+2
    if np.any(np.abs(widths[stable]-width) > max(2., .25*width)):
        return _reject(report, 'width_transition_or_occlusion', True)
    if np.any(widths[first:] > width+max(2., .25*width)):
        return _reject(report, 'wide_tip_or_crossing', True)
    if cap_model == 'unknown':
        return _reject(report, 'cap_model_unknown', True)
    if cap_model == 'butt':
        near = (positions >= boundary_t+1.5) & (positions <= boundary_t+width/2)
        if np.any(widths[near] < .7*width):
            return _reject(report, 'ink_cap_inconsistent_with_butt', True)
        endpoint_t = boundary_t
    else:
        endpoint_t = boundary_t+width/2
        cap = (positions >= boundary_t+.5) & (positions <= endpoint_t)
        expected_width = 2*np.sqrt(np.maximum(0., (width/2)**2-(positions[cap]-endpoint_t)**2))
        if len(expected_width) and np.mean(np.abs(widths[cap]-expected_width)) > max(1.5, .2*width):
            return _reject(report, 'ink_cap_inconsistent_with_round', True)
    observed = origin+endpoint_t*fitted
    if np.linalg.norm(observed-expected) > search_radius:
        return _reject(report, 'endpoint_outside_search_radius')
    report.update(status='localized', observed_xy=observed.tolist(),
                  position_semantics=f'ink_center_under_{cap_model}_cap_assumption',
                  hypothesis_offset_xy_px=(observed-expected).tolist(),
                  hypothesis_distance_px=float(np.linalg.norm(observed-expected)))
    return report
