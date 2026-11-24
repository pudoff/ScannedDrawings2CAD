"""Evaluation only: distinguish source edge coverage from native CAD type recall."""
import cv2
import numpy as np
from .curve_graph_metrics_v1 import score, native, target_candidate
from .curve_graph_v1 import sample


def measure(case, geometry, matrix):
    if case['scope'] != 'primary_contour_control':
        return None
    typed = score({**case, 'scope': 'analytical_control'}, geometry, matrix)
    shape = tuple(case['size_px'][::-1]); target = np.zeros(shape, np.uint8); found = np.zeros(shape, np.uint8)
    def draw(mask, candidate):
        points = np.rint(sample(candidate)).astype(np.int32)
        if len(points) > 1:
            cv2.polylines(mask, [points], candidate['kind'] == 'circle' or candidate.get('closed', False), 255, 1, cv2.LINE_8)
    for item in case['labels']['geometry']:
        if item['role'] == 1:
            draw(target, target_candidate(item))
    for item in geometry:
        draw(found, native(item, matrix))
    count = int(np.count_nonzero(target))
    distances = cv2.distanceTransform(255-found, cv2.DIST_L2, 5)[target > 0] if found.any() else np.full(count, np.inf)
    return {'typed': typed, 'known_part_centerline_pixels': count,
            'source_edge_coverage_within_2_native_px': float(np.mean(distances <= 2)) if count else None,
            'source_edge_p95_native_px': float(np.percentile(distances, 95)) if count and np.isfinite(distances).all() else None,
            'closed_polyline_observations': sum(c['kind'] == 'polyline' and c.get('closed', False) for c in geometry),
            'geometry_unchanged': True, 'units': 'native_px', 'physical_geometry_approved': False,
            'CAD_topology_approval_implied': False}
