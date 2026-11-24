"""Replay locally reviewed ring profiles in source pixels, without CAD or scale.

This samples a previously selected visible outline; it does not discover a hole
or infer a semantic role from color. The four-pixel interval is the authored
reading assumption, never a statistical confidence or physical accuracy bound.
Source/evidence authentication belongs to the caller.
"""
import math

import numpy as np


MAX_CROP_PIXELS = 4_194_304
MINIMUM_OPPOSITE_PROFILES = 20
METHOD_NAME = 'midpoints_of_opposite_locally_reviewed_ring_stroke_centerlines'
METHOD_KEYS = {
    'method', 'ring_rgb', 'profile_rows_half_open', 'profile_columns_half_open',
    'left_x_half_open', 'right_x_half_open', 'top_y_half_open', 'bottom_y_half_open',
    'minimum_exact_ring_pixels_per_profile', 'reading_interval_half_width_px',
    'basis', 'no_scale_or_cad_read',
}


def _integers(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count or any(type(v) is not int for v in value):
        raise ValueError(name+' must contain explicit integer coordinates')
    return list(value)


def _window(value, limit, name):
    low, high = _integers(value, 2, name)
    if not 0 <= low < high <= limit:
        raise ValueError(name+' must be a nonempty half-open window within the crop')
    return low, high


def read_ring_center_v1(rgb, method, source_box):
    """Return deterministic pixel readouts matching the pinned rib review method.

    Input is an exact uint8 HxWx3 crop and its half-open original-source box.
    Neither a nominal nor a CAD coordinate/scale is accepted by this API.
    The result is a raster observation, not an approval of geometry or accuracy.
    """
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) <= 0
            or rgb.shape[0]*rgb.shape[1] > MAX_CROP_PIXELS):
        raise ValueError('Expected a bounded nonempty uint8 RGB crop')
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = _integers(source_box, 4, 'Source box')
    if not (0 <= x0 < x1 <= 10**12 and 0 <= y0 < y1 <= 10**12
            and x1-x0 == width and y1-y0 == height):
        raise ValueError('Source box must exactly match the crop extent')
    if not isinstance(method, dict) or set(method) != METHOD_KEYS:
        raise ValueError('Expected the complete, unambiguous reviewed method')
    if (method['method'] != METHOD_NAME or method['no_scale_or_cad_read'] is not True
            or not isinstance(method['basis'], str) or not method['basis'].strip()):
        raise ValueError('Method must retain its explicit raster-only review basis')
    ring_rgb = _integers(method['ring_rgb'], 3, 'Ring RGB')
    if any(not 0 <= v <= 255 for v in ring_rgb):
        raise ValueError('Ring RGB must use byte channel values')
    minimum = method['minimum_exact_ring_pixels_per_profile']
    if type(minimum) is not int or minimum < 1:
        raise ValueError('Minimum exact ring pixels must be a positive integer')
    # Version 1 reproduces the predeclared +/-4px review, not a tunable error bar.
    half = method['reading_interval_half_width_px']
    if type(half) is not int or half != 4:
        raise ValueError('Version 1 retains the authored four-pixel reading interval')
    windows = {}
    for key in ('profile_rows_half_open', 'top_y_half_open', 'bottom_y_half_open'):
        windows[key] = _window(method[key], height, key)
    for key in ('profile_columns_half_open', 'left_x_half_open', 'right_x_half_open'):
        windows[key] = _window(method[key], width, key)
    lx0, lx1 = windows['left_x_half_open']; rx0, rx1 = windows['right_x_half_open']
    ty0, ty1 = windows['top_y_half_open']; by0, by1 = windows['bottom_y_half_open']
    r0, r1 = windows['profile_rows_half_open']; c0, c1 = windows['profile_columns_half_open']
    if (not lx1 <= c0 < c1 <= rx0 or not ty1 <= r0 < r1 <= by0
            or min(r1-r0, c1-c0) < MINIMUM_OPPOSITE_PROFILES
            or minimum > min(lx1-lx0, rx1-rx0, ty1-ty0, by1-by0)):
        raise ValueError('Opposite profile windows must be ordered, separated and sufficiently large')

    mask = np.all(rgb == ring_rgb, axis=2)

    def midpoint(indices, offset):
        if len(indices) < minimum:
            return None
        return offset+(int(indices.min())+int(indices.max()))/2

    x_samples, y_samples = [], []
    for y in range(r0, r1):
        left = midpoint(np.flatnonzero(mask[y, lx0:lx1]), lx0)
        right = midpoint(np.flatnonzero(mask[y, rx0:rx1]), rx0)
        if left is not None and right is not None:
            x_samples.append({'y_local': y, 'left_stroke_center_x': left,
                              'right_stroke_center_x': right, 'center_x': (left+right)/2})
    for x in range(c0, c1):
        top = midpoint(np.flatnonzero(mask[ty0:ty1, x]), ty0)
        bottom = midpoint(np.flatnonzero(mask[by0:by1, x]), by0)
        if top is not None and bottom is not None:
            y_samples.append({'x_local': x, 'top_stroke_center_y': top,
                              'bottom_stroke_center_y': bottom, 'center_y': (top+bottom)/2})
    if min(len(x_samples), len(y_samples)) < MINIMUM_OPPOSITE_PROFILES:
        raise ValueError('At least twenty opposite outline profiles are required on each axis')
    cx = float(np.median([r['center_x'] for r in x_samples]))
    cy = float(np.median([r['center_y'] for r in y_samples]))
    point = [x0+cx, y0+cy]
    if not all(math.isfinite(v) for v in point):
        raise ValueError('Nonfinite source center')
    return {'center_local_xy': [cx, cy], 'center_source_xy': point,
        'interval_source_xyxy': [point[0]-half, point[1]-half, point[0]+half, point[1]+half],
        'x_readouts': x_samples, 'y_readouts': y_samples,
        'sample_ranges_local': {
            'x': [min(r['center_x'] for r in x_samples), max(r['center_x'] for r in x_samples)],
            'y': [min(r['center_y'] for r in y_samples), max(r['center_y'] for r in y_samples)]}}
