"""Parallel ink makes a new local head ambiguous unless its tail leaves the field."""
import math
import cv2
import numpy as np


def check(mask, head, audit, *, maximum_pixels=4000000):
    p = np.asarray(head['polygon_px'], float); base = p[1:].mean(0); tip = p[0]
    length = float(np.linalg.norm(tip-base)); stroke = float(head['evidence']['local_shaft_width_px'])
    margin = min(96., 2*length)
    lo = np.maximum([0, 0], np.floor(p.min(0)-margin)).astype(int)
    hi = np.minimum(mask.shape[::-1], np.ceil(p.max(0)+margin+1)).astype(int)
    cost = int(np.prod(hi-lo))
    if audit['context_pixels']+cost > maximum_pixels:
        return {'available': False, 'incomplete': True, 'reason': 'context_pixel_budget'}
    audit['context_pixels'] += cost
    roi = mask[lo[1]:hi[1], lo[0]:hi[0]].copy(); original = roi.copy()
    cv2.fillConvexPoly(roi, np.rint(p-lo).astype(np.int32), 0)
    minimum = max(8, round(length*1.25))
    raw = cv2.HoughLinesP(roi, 1, np.pi/180, threshold=max(6, minimum//2), minLineLength=minimum, maxLineGap=1)
    if raw is None: return {'available': True, 'incomplete': False, 'reason': 'no_parallel_source_field'}
    if len(raw) > 256: return {'available': False, 'incomplete': True, 'reason': 'context_segment_budget'}
    distance = cv2.distanceTransform(original, cv2.DIST_L2, 5); lines = []
    for a, b in raw.reshape(-1, 2, 2).astype(float):
        delta = b-a; extent = float(np.linalg.norm(delta))
        if extent < minimum: continue
        q = np.rint(a+np.linspace(.1, .9, 16)[:, None]*delta).astype(int)
        widths = 2*distance[q[:, 1], q[:, 0]]-1
        if np.median(widths) <= 0 or np.median(widths) > max(2., 1.5*stroke): continue
        lines.append({'a': a+lo, 'b': b+lo, 'u': delta/extent, 'length': extent})
    audit['context_segments'] += len(lines)
    for seed in sorted(lines, key=lambda v: -v['length']):
        u = seed['u']; normal = np.array([-u[1], u[0]]); offsets, group = [], []
        for line in lines:
            if abs(u@line['u']) < math.cos(math.radians(8)): continue
            offset = float(((line['a']+line['b'])/2)@normal)
            if any(abs(offset-v) < max(2., 1.5*stroke) for v in offsets): continue
            offsets.append(offset); group.append(line)
        if len(group) < 3: continue
        points = np.array([q for line in group for q in (line['a'], line['b'])], np.float32)
        hull = cv2.convexHull(points)
        if cv2.pointPolygonTest(hull, tuple(map(float, p.mean(0))), True) < -stroke: continue
        direction = -np.asarray(head['direction'], float)
        along = np.arange(0, min(96., max(40., 2*length))+.5, .5)
        tail = base+along[:, None]*direction
        q = np.rint(tail).astype(int)
        inside = (q >= lo+math.ceil(stroke)).all(1) & (q < hi-math.ceil(stroke)).all(1)
        supported = np.zeros(len(q), bool)
        supported[inside] = mask[q[:, 1][inside], q[:, 0][inside]] > 0
        outside = np.array([cv2.pointPolygonTest(hull, tuple(map(float, point)), True) < -stroke for point in tail])
        connected = np.logical_and.accumulate(supported)
        run = outside & supported & connected & inside
        changes = np.diff(np.r_[False, run, False].astype(int))
        protected = any((b-a)*.5 >= max(6., 2*stroke) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        evidence = {'parallel_offsets': len(group), 'source_segments': [[v['a'].tolist(), v['b'].tolist()] for v in group],
            'hull_px': hull.reshape(-1, 2).tolist(), 'own_tail_outside_parallel_field': protected}
        if not protected:
            return {'available': False, 'incomplete': False, 'reason': 'ambiguous_parallel_ink_field', 'support': evidence}
        return {'available': True, 'incomplete': False, 'reason': 'observed_own_tail_leaves_parallel_field', 'support': evidence}
    return {'available': True, 'incomplete': False, 'reason': 'no_parallel_field_at_head'}
