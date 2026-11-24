"""Spatial crop fragmentation and degree coverage; no segmentation code imported."""
from itertools import combinations
import cv2
import numpy as np
from .angular_link_metrics_v1 import overlap
from .angular_text_components_v1 import polygon


def coverage(box, observed):
    p = np.asarray(polygon(box), np.float32)
    area, _ = cv2.intersectConvexConvex(p, np.asarray(observed, np.float32))
    return float(area/cv2.contourArea(p))


def evaluate(case, rows):
    degrees = case.get('degree_boxes', [])
    covered = sum(any(coverage(d['box_px'], r['polygon_px']) >= .8 for r in rows) for d in degrees)
    fragmented = 0
    for t in case['components']:
        candidates = [(coverage(t['box_px'], r['polygon_px']), r) for r in rows]
        candidates = [(c, r) for c, r in candidates if c >= .2]
        fragmented += any(a+b >= .7 and overlap(x['polygon_px'], y['polygon_px']) <= .1
                          for (a, x), (b, y) in combinations(candidates, 2))
    return {'degree_box_targets': len(degrees), 'degree_boxes_covered_at_80_percent': covered,
            'targets_fragmented_across_disjoint_crops': int(fragmented),
            'exact_degree_truth_available': bool(degrees),
            'degree_coverage_is_not_glyph_recognition': True}
