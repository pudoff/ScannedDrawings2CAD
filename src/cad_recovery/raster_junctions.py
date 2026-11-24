"""Localize a junction from independently fitted and connected raster branches.

Directions describe the complete local branch hypothesis, not semantic roles.
The result is a raster stroke-center estimate, never an approved CAD anchor.
"""
from __future__ import annotations

import math

import numpy as np

MAX_PIXELS = 4_194_304
SAMPLE_COUNT = 128
DARK_THRESHOLD = 160


def _number(value, name, low, high):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite number') from error
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return result


def _point(value, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (2,):
        raise ValueError(f'{name} must contain two finite coordinates')
    return np.array([_number(v, name, -32768, 32768) for v in value])


def _sample(rgb, positions):
    coordinates = np.rint(positions).astype(np.int64)
    valid = ((coordinates[:, 0] >= 0) & (coordinates[:, 0] < rgb.shape[1])
             & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < rgb.shape[0]))
    dark = np.zeros(len(coordinates), dtype=bool)
    good = coordinates[valid]
    dark[valid] = np.max(rgb[good[:, 1], good[:, 0]], axis=1) <= DARK_THRESHOLD
    return dark, valid


def _fit_branch(rgb, expected, direction, search, inner, outer):
    normal = np.array([-direction[1], direction[0]])
    offsets = np.linspace(-search, search, math.ceil(4*search)+1)
    distances = np.linspace(inner, outer, SAMPLE_COUNT)
    counts = dict(unique=0, missing=0, ambiguous=0, clipped=0, outside=0, broad=0)
    points, widths, indices = [], [], []
    for index, distance in enumerate(distances):
        predicted = expected + distance*direction
        dark, valid = _sample(rgb, predicted + offsets[:, None]*normal)
        if not valid.all():
            counts['outside'] += 1
            continue
        changes = np.diff(np.r_[False, dark, False].astype(np.int8))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)-1
        if not len(starts):
            counts['missing'] += 1
        elif len(starts) > 1:
            counts['ambiguous'] += 1
        elif starts[0] == 0 or ends[0] == len(offsets)-1:
            counts['clipped'] += 1
        else:
            width = offsets[ends[0]]-offsets[starts[0]] + offsets[1]-offsets[0]
            if width > min(20., 1.25*search):
                counts['broad'] += 1
                continue
            counts['unique'] += 1
            points.append(predicted + (offsets[starts[0]]+offsets[ends[0]])/2*normal)
            widths.append(float(width))
            indices.append(index)
    report = {'requested_direction_xy': direction.tolist(), 'counts': counts,
              'unique_coverage': counts['unique']/SAMPLE_COUNT,
              'supported_bins_of_8': len(set(i*8//SAMPLE_COUNT for i in indices)),
              'median_stroke_width_px': float(np.median(widths)) if widths else None,
              'status': 'not_supported', 'fit': None, 'reasons': []}
    if report['unique_coverage'] < .65 or report['supported_bins_of_8'] < 6:
        report['reasons'].append('insufficient_distributed_unique_stroke')
        if counts['ambiguous'] >= SAMPLE_COUNT*.2 or counts['broad'] >= SAMPLE_COUNT*.2:
            report['status'] = 'ambiguous'
        return report
    observations = np.array(points)
    center = observations.mean(axis=0)
    _, _, vectors = np.linalg.svd(observations-center, full_matrices=False)
    fitted_direction = vectors[0]
    if np.dot(fitted_direction, direction) < 0:
        fitted_direction = -fitted_direction
    fitted_normal = np.array([-fitted_direction[1], fitted_direction[0]])
    residuals = (observations-center) @ fitted_normal
    p95 = float(np.percentile(np.abs(residuals), 95))
    angle = math.degrees(math.acos(float(np.clip(np.dot(direction, fitted_direction), -1, 1))))
    report['fit'] = {'point_xy': center.tolist(), 'direction_xy': fitted_direction.tolist(),
                     'normal_xy': fitted_normal.tolist(),
                     'residual_rms_px': float(np.sqrt(np.mean(residuals**2))),
                     'residual_p95_px': p95, 'direction_difference_deg': angle}
    if p95 > 1.5 or angle > 10:
        report['status'] = 'ambiguous'
        report['reasons'].append('inconsistent_straight_branch')
    else:
        report['status'] = 'supported'
    return report


def _longest_false_run(values):
    changes = np.diff(np.r_[False, ~values, False].astype(np.int8))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return int(max(ends-starts, default=0))


def _connectivity(rgb, point, branches, inner):
    reports = []
    for branch in branches:
        direction = np.array(branch['fit']['direction_xy'])
        distances = np.arange(0., inner+1., .5)
        dark, valid = _sample(rgb, point + distances[:, None]*direction)
        # Centerline continuity is deliberately stricter than interior stroke
        # coverage: two separate stubs are not an observed junction.
        coverage = float(np.mean(dark))
        longest_gap = _longest_false_run(dark)*.5
        reports.append({'dark_fraction': coverage, 'longest_gap_px': longest_gap,
                        'inside_image': bool(valid.all()),
                        'connected': bool(valid.all() and coverage >= .95 and longest_gap <= 1.)})
    return reports


def _circular_runs(dark):
    if dark.all():
        return [np.arange(len(dark))]
    starts = np.flatnonzero(dark & ~np.roll(dark, 1))
    runs = []
    for start in starts:
        indices = []
        for offset in range(len(dark)):
            index = (start+offset) % len(dark)
            if not dark[index]:
                break
            indices.append(index)
        runs.append(np.array(indices))
    return runs


def _topology(rgb, point, branches):
    """Observe all angular ink runs, refusing extra or merged local branches."""
    max_width = max(branch['median_stroke_width_px'] for branch in branches)
    base_radius = max(10., 2*max_width+4)
    directions = np.array([b['fit']['direction_xy'] for b in branches])
    angles = np.linspace(0, 2*math.pi, 720, endpoint=False)
    radial = np.column_stack((np.cos(angles), np.sin(angles)))
    rings = []
    for radius in (base_radius, base_radius+5):
        dark, valid = _sample(rgb, point + radius*radial)
        runs = _circular_runs(dark)
        centers, candidates, matched = [], [], []
        for run in runs:
            center = radial[run].mean(axis=0)
            norm = float(np.linalg.norm(center))
            if norm < 1e-6:
                candidates.append([])
                continue
            center /= norm
            centers.append(center.tolist())
            options = [i for i, direction in enumerate(directions)
                       if math.degrees(math.acos(float(np.clip(np.dot(center, direction), -1, 1)))) <= 12.]
            candidates.append(options)
            if len(options) == 1:
                matched.append(options[0])
        supported = bool(valid.all() and len(runs) == len(branches)
                         and all(len(c) == 1 for c in candidates) and len(set(matched)) == len(branches))
        rings.append({'radius_px': radius, 'ink_run_count': len(runs),
                      'run_direction_xy': centers, 'matching_branch_indices': candidates,
                      'inside_image': bool(valid.all()), 'matches_requested_topology': supported})
    return rings


def localize_junction(rgb, expected_xy, directions_xy, *, search_radius=12,
                      inner_radius=20, outer_radius=96):
    """Estimate a connected 2–6 branch junction in original-resolution ROI pixels.

    Every oriented ray must have an individually supported straight stroke.
    Extra angular branches are ambiguous, including annotation continuations.
    The intersection is derived from pixels; ``expected_xy`` only sets search
    bounds. Results do not prove CAD attachment or a registration error bound.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
            or max(rgb.shape[:2]) > 32768 or rgb.shape[0]*rgb.shape[1] > MAX_PIXELS):
        raise ValueError('ROI must be a nonempty RGB8 image within the 4-Mpixel limit')
    expected = _point(expected_xy, 'expected_xy')
    if not 0 <= expected[0] < rgb.shape[1] or not 0 <= expected[1] < rgb.shape[0]:
        raise ValueError('Expected junction lies outside the ROI')
    search = _number(search_radius, 'search_radius', 1, 64)
    inner = _number(inner_radius, 'inner_radius', 4, 128)
    outer = _number(outer_radius, 'outer_radius', 24, 256)
    if outer-inner < 20:
        raise ValueError('Branch support must span at least 20 pixels')
    if not isinstance(directions_xy, (list, tuple, np.ndarray)) or not 2 <= len(directions_xy) <= 6:
        raise ValueError('A junction requires 2..6 oriented branch directions')
    directions = [_point(d, 'branch direction') for d in directions_xy]
    for index, direction in enumerate(directions):
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            raise ValueError('Branch direction cannot be zero')
        directions[index] = direction/length
    report = {'schema_version': '1.0', 'method': 'connected-stroke-intersection-v1',
              'status': 'not_supported', 'expected_xy': expected.tolist(), 'observed_xy': None,
              'position_semantics': 'intersection of fitted raster stroke centerlines',
              'coordinate_system': 'ROI pixel centers; X right, Y down',
              'training_eligible': False, 'annotation_approved': False,
              'search_radius_px': search, 'inner_radius_px': inner, 'outer_radius_px': outer,
              'sample_count_per_branch': SAMPLE_COUNT, 'dark_threshold': DARK_THRESHOLD,
              'sampling': 'nearest_pixel', 'transverse_step_px': 2*search/math.ceil(4*search),
              'branches': [], 'connectivity': [], 'topology_rings': [], 'reasons': [],
              'connectivity_extent_px': inner, 'whole_finite_edge_connectivity_verified': False,
              'topology_sampling': 'two angular rings only; short extra stubs can remain unobserved',
              'uncertainty': {'accuracy_bound_px': None, 'fit_residual_is_accuracy_bound': False,
                              'sampling_pixel_quantization': True,
                              'independent_calibration_required': True},
              'limits': {'max_pixels': MAX_PIXELS, 'minimum_unique_coverage': .65,
                         'maximum_fit_p95_px': 1.5, 'maximum_condition_number': 6.,
                         'minimum_connection_fraction': .95, 'maximum_connection_gap_px': 1.},
              'limitations': ['Nearest-pixel and finite-grid sampling quantize line centers and residuals.',
                              'A localized raster junction is not an approved dimension anchor.',
                              'The requested rays describe a topology hypothesis; semantic roles are not inferred.',
                              'Connectivity is checked only up to inner_radius and topology on two rings; remote gaps or short extra branches can remain unobserved.']}
    branches = [_fit_branch(rgb, expected, direction, search, inner, outer) for direction in directions]
    report['branches'] = branches
    if any(branch['status'] != 'supported' for branch in branches):
        report['reasons'].append('not_all_branches_uniquely_supported')
        if any(branch['status'] == 'ambiguous' for branch in branches):
            report['status'] = 'ambiguous'
        return report
    normals = np.array([branch['fit']['normal_xy'] for branch in branches])
    centers = np.array([branch['fit']['point_xy'] for branch in branches])
    singular = np.linalg.svd(normals, compute_uv=False)
    condition = float(singular[0]/singular[-1]) if singular[-1] > 1e-12 else None
    report['intersection_condition_number'] = condition
    if condition is None or condition > 6:
        report['reasons'].append('parallel_or_ill_conditioned_branches')
        return report
    point, _, _, _ = np.linalg.lstsq(normals, np.sum(normals*centers, axis=1), rcond=None)
    line_distances = np.abs(np.sum(normals*(point-centers), axis=1))
    displacement = float(np.linalg.norm(point-expected))
    report['intersection_candidate_xy'] = point.tolist()
    report['candidate_distance_from_expected_px'] = displacement
    report['intersection_line_residuals_px'] = line_distances.tolist()
    if not np.isfinite(point).all() or displacement > search or max(line_distances) > 1.5:
        report['reasons'].append('inconsistent_or_outside_search_intersection')
        return report
    if not 0 <= point[0] < rgb.shape[1] or not 0 <= point[1] < rgb.shape[0]:
        report['reasons'].append('intersection_outside_image')
        return report
    connection = _connectivity(rgb, point, branches, inner)
    report['connectivity'] = connection
    if not all(item['connected'] for item in connection):
        report['reasons'].append('disconnected_or_obscured_junction')
        return report
    rings = _topology(rgb, point, branches)
    report['topology_rings'] = rings
    if not all(item['matches_requested_topology'] for item in rings):
        report['status'] = 'ambiguous'
        report['reasons'].append('extra_missing_or_merged_local_branches')
        return report
    report['status'] = 'localized'
    report['observed_xy'] = point.tolist()
    report['displacement_xy_px'] = (point-expected).tolist()
    return report
