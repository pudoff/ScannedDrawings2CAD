"""Bounded, conservative pixel evidence around an existing geometric hypothesis.

These probes do not find semantic objects or approve annotations. Each normal
cross-section must contain exactly one complete dark run; parallel strokes are
reported as ambiguous instead of selecting whichever is nearest the hypothesis.
Coordinates refer to pixel centers in the supplied, original-resolution ROI.
"""
from __future__ import annotations

import math

import numpy as np

MAX_PIXELS = 4_194_304
SAMPLE_COUNT = 128
DARK_THRESHOLD = 160
MIN_COVERAGE = 0.65
MAX_P95_RESIDUAL_PX = 1.5


def _number(value, name, low, high):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number in {low}..{high}')
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite number') from error
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f'{name} must be a finite number in {low}..{high}')
    return result


def _inputs(rgb, search_radius):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
            or max(rgb.shape[:2]) > 32768 or rgb.shape[0]*rgb.shape[1] > MAX_PIXELS):
        raise ValueError('ROI must be a nonempty RGB8 array within the 4-Mpixel limit')
    return _number(search_radius, 'search_radius', 1, 64)


def _point(value, name, rgb):
    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (2,):
        raise ValueError(f'{name} must contain two coordinates')
    return np.array([_number(value[0], name, 0, rgb.shape[1]-1),
                     _number(value[1], name, 0, rgb.shape[0]-1)])


def _sections(rgb, predicted, normals, search_radius):
    # At most 128 * 257 sample points, irrespective of ROI size.
    offsets = np.linspace(-search_radius, search_radius, int(math.ceil(4*search_radius))+1)
    observations, indices, counts = [], [], dict(unique=0, missing=0, ambiguous=0, clipped=0,
                                                outside=0, broad=0)
    widths = []
    for index, (position, normal) in enumerate(zip(predicted, normals)):
        points = position + offsets[:, None]*normal
        coords = np.rint(points).astype(np.int64)
        if (np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= rgb.shape[1])
                or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= rgb.shape[0])):
            counts['outside'] += 1
            continue
        dark = np.max(rgb[coords[:, 1], coords[:, 0]], axis=1) <= DARK_THRESHOLD
        changes = np.diff(np.r_[False, dark, False].astype(np.int8))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)-1
        if len(starts) == 0:
            counts['missing'] += 1
        elif len(starts) > 1:
            counts['ambiguous'] += 1
        elif starts[0] == 0 or ends[0] == len(offsets)-1:
            counts['clipped'] += 1
        else:
            width = float(offsets[ends[0]]-offsets[starts[0]] + offsets[1]-offsets[0])
            if width > min(20, search_radius*1.25):
                counts['broad'] += 1
                continue
            counts['unique'] += 1
            offset = (offsets[starts[0]]+offsets[ends[0]])/2
            observations.append(position + offset*normal)
            indices.append(index)
            widths.append(width)
    return np.asarray(observations), np.asarray(indices, dtype=int), counts, widths


def _report(kind, counts, indices, widths, search_radius):
    bins = len(set((indices*8//SAMPLE_COUNT).tolist()))
    return {'schema_version': '1.0', 'method': 'unique-dark-cross-sections-v1', 'kind': kind,
            'status': 'not_supported', 'training_eligible': False, 'annotation_approved': False,
            'coordinate_system': 'ROI pixel centers; X right, Y down',
            'sample_count': SAMPLE_COUNT, 'counts': counts, 'coverage': counts['unique']/SAMPLE_COUNT,
            'supported_bins_of_8': bins,
            'median_stroke_width_px': float(np.median(widths)) if widths else None,
            'search_radius_px': search_radius, 'dark_threshold': DARK_THRESHOLD,
            'sampling': 'nearest_pixel',
            'sampling_step_px': 2*search_radius/math.ceil(4*search_radius),
            'limits': {'max_pixels': MAX_PIXELS, 'minimum_coverage': MIN_COVERAGE,
                       'minimum_supported_bins': 6, 'maximum_p95_residual_px': MAX_P95_RESIDUAL_PX},
            'observed_geometry': None, 'reasons': [],
            'limitations': ['Cross-section centers are quantized by nearest-pixel samples on a finite offset grid.',
                            'A zero fit residual can arise from constant quantized offsets; it does not imply exact raster registration.'],
            'interpretation': ('Local pixel support only; geometry identity, dimensions and scale require '
                               'separate review. Fit residuals are not coordinate-accuracy bounds; '
                               'nearby cross-sections are not statistically independent.')}


def _coverage(report):
    if report['coverage'] < MIN_COVERAGE:
        report['reasons'].append('insufficient_unique_cross_sections')
    if report['supported_bins_of_8'] < 6:
        report['reasons'].append('insufficient_spatial_distribution')
    return not report['reasons']


def _residuals(report, residuals):
    report['fit_residual_px'] = {'rms': float(np.sqrt(np.mean(residuals**2))),
                                 'p95': float(np.percentile(np.abs(residuals), 95)),
                                 'maximum': float(np.max(np.abs(residuals)))}
    if report['fit_residual_px']['p95'] > MAX_P95_RESIDUAL_PX:
        report['reasons'].append('inconsistent_stroke_fit')
    if not report['reasons']:
        report['status'] = 'supported'


def probe_line(rgb, start_xy, end_xy, search_radius=12):
    """Fit uniquely supported ink along the central 90% of a proposed segment.

    Endpoint coordinates in the result are projections onto the fitted infinite
    line. They are not observed endpoints and must not be used as such.
    """
    search_radius = _inputs(rgb, search_radius)
    start, end = _point(start_xy, 'start_xy', rgb), _point(end_xy, 'end_xy', rgb)
    delta = end-start
    length = float(np.linalg.norm(delta))
    if length < 20:
        raise ValueError('Line hypothesis must span at least 20 pixels')
    direction = delta/length
    normal = np.array([-direction[1], direction[0]])
    predicted = start + np.linspace(.05, .95, SAMPLE_COUNT)[:, None]*delta
    observations, indices, counts, widths = _sections(
        rgb, predicted, np.tile(normal, (SAMPLE_COUNT, 1)), search_radius)
    report = _report('line', counts, indices, widths, search_radius)
    report['hypothesis'] = {'start_xy': start.tolist(), 'end_xy': end.tolist()}
    if not _coverage(report):
        return report
    center = np.mean(observations, axis=0)
    _, _, vectors = np.linalg.svd(observations-center, full_matrices=False)
    fitted_direction = vectors[0]
    if np.dot(fitted_direction, direction) < 0:
        fitted_direction = -fitted_direction
    fitted_normal = np.array([-fitted_direction[1], fitted_direction[0]])
    projected = np.array([center + np.dot(point-center, fitted_direction)*fitted_direction
                          for point in (start, end)])
    offsets = (projected - np.array([start, end])) @ normal
    report['observed_geometry'] = {'point_xy': center.tolist(), 'direction_xy': fitted_direction.tolist(),
                                    'normal_xy': fitted_normal.tolist(),
                                    'projected_start_xy': projected[0].tolist(),
                                    'projected_end_xy': projected[1].tolist(),
                                    'endpoints_observed': False}
    report['hypothesis_difference'] = {'signed_normal_offsets_at_ends_px': offsets.tolist(),
                                       'angle_deg': float(np.degrees(np.arccos(np.clip(
                                           np.dot(direction, fitted_direction), -1, 1))))}
    _residuals(report, (observations-center) @ fitted_normal)
    return report


def probe_circle(rgb, center_xy, radius, start_angle_deg=0, end_angle_deg=360, search_radius=12):
    """Fit radial stroke centers on a circle/arc; angles follow image XY axes.

    At least 60 degrees of arc and distributed unique samples are required.
    Supported means that a local stroke fits, not that it is the intended CAD arc.
    """
    search_radius = _inputs(rgb, search_radius)
    center = _point(center_xy, 'center_xy', rgb)
    radius = _number(radius, 'radius', 5, 32768)
    start_angle = _number(start_angle_deg, 'start_angle_deg', -360, 360)
    end_angle = _number(end_angle_deg, 'end_angle_deg', -360, 720)
    span = end_angle-start_angle
    if not 0 < span <= 360:
        raise ValueError('Arc span must be greater than zero and at most 360 degrees')
    angles = np.radians(np.linspace(start_angle, end_angle, SAMPLE_COUNT, endpoint=span < 360))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    predicted = center + radius*normals
    observations, indices, counts, widths = _sections(rgb, predicted, normals, search_radius)
    report = _report('circle', counts, indices, widths, search_radius)
    report['hypothesis'] = {'center_xy': center.tolist(), 'radius_px': radius,
                             'start_angle_deg': start_angle, 'end_angle_deg': end_angle}
    report['limits']['minimum_arc_span_deg'] = 60
    report['limits']['minimum_arc_length_px'] = 20
    if span < 60:
        report['reasons'].append('arc_span_too_short')
    if math.radians(span)*radius < 20:
        report['reasons'].append('arc_length_too_short')
    if not _coverage(report):
        return report
    # Normalize around the expected center/radius to avoid large-coordinate
    # conditioning; the matrix depends only on observed arc distribution.
    local = (observations-center)/radius
    design = np.column_stack((2*local, np.ones(len(local))))
    if np.linalg.cond(design) > 1000:
        report['reasons'].append('ill_conditioned_circle_fit')
        return report
    solution, _, _, _ = np.linalg.lstsq(design, np.sum(local**2, axis=1), rcond=None)
    squared_radius = solution[2]+np.dot(solution[:2], solution[:2])
    if squared_radius <= 0:
        report['reasons'].append('invalid_circle_fit')
        return report
    fitted_center = center + radius*solution[:2]
    fitted_radius = radius*math.sqrt(squared_radius)
    report['observed_geometry'] = {'center_xy': fitted_center.tolist(), 'radius_px': fitted_radius,
                                    'arc_endpoints_observed': False}
    report['hypothesis_difference'] = {'center_offset_xy_px': (fitted_center-center).tolist(),
                                       'center_distance_px': float(np.linalg.norm(fitted_center-center)),
                                       'radius_difference_px': fitted_radius-radius}
    if np.linalg.norm(fitted_center-center) > 2*search_radius or abs(fitted_radius-radius) > 2*search_radius:
        report['reasons'].append('unstable_circle_parameters')
    _residuals(report, np.linalg.norm(observations-fitted_center, axis=1)-fitted_radius)
    return report
