"""Subtract native replacement cells from fixed straight/polyline CAD models.

Source ownership and model position are distinct: a fitted LINE may be outside
the replacement cell although its observed source walk lies inside. Both
domains are subtracted. Retained pieces stay on the original model exactly.
"""
from copy import deepcopy

import numpy as np

from .cad_object_assembly_v1 import compact


EPSILON = 1e-12
MAXIMUM_RECTANGLES = 4096
MAXIMUM_POINTS = 120000
MAXIMUM_TESTS = 2_000_000


def _points(value, *, minimum=2):
    p = np.asarray(value, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < minimum or not np.isfinite(p).all():
        raise ValueError('Finite ordered model/source points required')
    if len(p) > MAXIMUM_POINTS:
        raise ValueError('Native clipping point budget exceeded')
    if np.any(np.linalg.norm(np.diff(p, axis=0), axis=1) <= EPSILON):
        raise ValueError('Repeated adjacent model/source point')
    return p


def _rectangles(value):
    rows = np.asarray(value, float)
    if rows.size == 0:
        return np.empty((0, 4), float)
    if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all():
        raise ValueError('Finite replacement rectangles required')
    if len(rows) > MAXIMUM_RECTANGLES or np.any(rows[:, 2:] <= rows[:, :2]):
        raise ValueError('Invalid or excessive replacement rectangles')
    return rows


def _merge(spans):
    result = []
    for low, high in sorted(spans):
        if high-low <= EPSILON:
            continue
        low, high = max(0., float(low)), min(1., float(high))
        if high-low <= EPSILON:
            continue
        if result and low <= result[-1][1]+EPSILON:
            result[-1][1] = max(result[-1][1], high)
        else:
            result.append([low, high])
    return result


def _complement(spans):
    result = []; cursor = 0.
    for low, high in spans:
        if low-cursor > EPSILON:
            result.append([cursor, low])
        cursor = max(cursor, high)
    if 1.-cursor > EPSILON:
        result.append([cursor, 1.])
    return result


def _inside_span(a, b, rectangle):
    """Liang--Barsky interval; a boundary-only touch has no interior area."""
    low, high = 0., 1.
    delta = b-a
    for axis in (0, 1):
        left, right = rectangle[axis], rectangle[axis+2]
        if abs(delta[axis]) <= EPSILON:
            if not left < a[axis] < right:
                return None
            continue
        start, end = sorted(((left-a[axis])/delta[axis], (right-a[axis])/delta[axis]))
        low, high = max(low, start), min(high, end)
        if high-low <= EPSILON:
            return None
    return [low, high]


def clip_segment(a, b, rectangles):
    """Return normalized kept/removed intervals of one fixed model segment."""
    p = _points([a, b]); boxes = _rectangles(rectangles)
    spans = [span for box in boxes if (span := _inside_span(p[0], p[1], box)) is not None]
    removed = _merge(spans)
    return {'removed': removed, 'kept': _complement(removed)}


def _walk_spans(points, boxes):
    if (len(points)-1)*len(boxes) > MAXIMUM_TESTS:
        raise ValueError('Native clipping comparison budget exceeded')
    spans = []
    for i, (a, b) in enumerate(zip(points, points[1:])):
        for low, high in clip_segment(a, b, boxes)['removed']:
            spans.append([float(i+low), float(i+high)])
    return spans


def _line(model, source, boxes):
    points = _points(model['points_px'])
    if len(points) != 2:
        raise ValueError('Two fixed LINE endpoints required')
    delta = points[1]-points[0]
    t = (source-points[0])@delta/float(delta@delta)
    # Source membership may be recorded in the opposite walk direction.
    reverse = bool(t[-1] < t[0])
    if reverse:
        source = source[::-1].copy(); t = t[::-1].copy()
    if np.any(np.diff(t) <= EPSILON):
        raise ValueError('Source walk folds back on frozen LINE')
    t[0], t[-1] = 0., 1.
    if (t < -EPSILON).any() or (t > 1+EPSILON).any() or np.any(np.diff(t) <= EPSILON):
        raise ValueError('Source samples do not monotonically partition frozen LINE')
    source_spans = _walk_spans(source, boxes)
    removed = clip_segment(points[0], points[1], boxes)['removed']
    for low, high in source_spans:
        def parameter(value):
            index = min(int(np.floor(value)), len(source)-2)
            fraction = value-index
            return float(t[index]+fraction*(t[index+1]-t[index]))
        removed.append([parameter(low), parameter(high)])
    removed = _merge(removed)
    def fixed_position(value):
        if value == 0.: return points[0].tolist()
        if value == 1.: return points[1].tolist()
        return (points[0]+value*delta).tolist()
    pieces = [{'kind': 'line', 'points_px': [fixed_position(a), fixed_position(b)],
               'closed': False, 'model_span': [a, b]} for a, b in _complement(removed)]
    return {'changed': bool(removed), 'pieces': pieces,
        'removed_model_spans': removed, 'source_patch_spans': source_spans,
        'model_parameter_basis': 'monotone_source_projection_and_fixed_LINE_geometry',
        'source_walk_reversed': reverse}


def _polyline(model, source, boxes):
    if model.get('closed'):
        raise ValueError('Closed polyline replacement requires a separate certificate')
    points = _points(model['points_px'])
    if np.array_equal(points[0], points[-1]) or np.array_equal(source[0], source[-1]):
        raise ValueError('Closed source walk replacement requires a separate certificate')
    source_compact = np.asarray(compact(source), float)
    reverse = False
    if not np.array_equal(source_compact, points):
        if not np.array_equal(source_compact[::-1], points):
            raise ValueError('Polyline differs from original source walk')
        source = source[::-1].copy(); reverse = True
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    accumulated = np.r_[0., np.cumsum(lengths)]; total = float(accumulated[-1])
    removed = []
    for i, (a, b) in enumerate(zip(points, points[1:])):
        for low, high in clip_segment(a, b, boxes)['removed']:
            removed.append([(accumulated[i]+low*lengths[i])/total,
                            (accumulated[i]+high*lengths[i])/total])
    removed = _merge(removed)

    def position(value):
        if value == 0.: return points[0].copy()
        if value == 1.: return points[-1].copy()
        distance = value*total
        i = min(int(np.searchsorted(accumulated, distance, side='right'))-1, len(points)-2)
        i = max(0, i)
        return points[i]+(distance-accumulated[i])/lengths[i]*(points[i+1]-points[i])

    pieces = []
    for low, high in _complement(removed):
        inner = [point.tolist() for distance, point in zip(accumulated, points)
                 if low*total+EPSILON < distance < high*total-EPSILON]
        vertices = [position(low).tolist(), *inner, position(high).tolist()]
        pieces.append({'kind': 'polyline', 'points_px': vertices, 'closed': False,
                       'model_span': [low, high]})
    return {'changed': bool(removed), 'pieces': pieces,
        'removed_model_spans': removed, 'source_patch_spans': _walk_spans(source, boxes),
        'model_parameter_basis': 'fixed_polyline_arc_length', 'source_walk_reversed': reverse}


def trim_model(model, support_points, rectangles):
    """Return exact retained model pieces plus a fractional removal ledger.

    Coordinates are working raster pixel centres, y down. This function does
    not infer semantic roles or create new fitted geometry. Unsupported curves,
    closed walks, displaced polylines and nonmonotone LINE support are rejected.
    """
    original = deepcopy(model)
    source = _points(support_points); boxes = _rectangles(rectangles)
    if (len(source)-1)*len(boxes) > MAXIMUM_TESTS:
        raise ValueError('Native clipping comparison budget exceeded')
    if model['kind'] == 'line':
        result = _line(model, source, boxes)
    elif model['kind'] == 'polyline':
        result = _polyline(model, source, boxes)
    else:
        raise ValueError('Native patch supports LINE/open LWPOLYLINE only')
    if model != original:
        raise ValueError('Native clipping mutated its fixed model')
    result.update(schema_version='native-patch-clip/1.0', geometry_refits=0,
                  fractional_model_partition_exact=True,
                  outside_retained_model_geometry_exact=True,
                  source_patch_span_basis='orientation_normalized_source_walk_segment_index_plus_fraction')
    return result
