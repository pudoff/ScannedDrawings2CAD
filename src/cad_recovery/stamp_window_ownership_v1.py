"""Build source-path ownership for matching stamp rules without fitting CAD."""

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .curve_graph_v1 import graph_paths
from .interval_roles_v1 import distance_to_polyline, local_width
from .stroke_roles_v1 import foreground
from .supported_role_overlay_v1 import source_ownership


def observe(gray, *, scale):
    ink = foreground(gray)
    paths, topology = graph_paths(skeletonize(ink > 0, method='zhang'))
    intervals = []
    if topology['search_incomplete']:
        return None, topology
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    for index, path in enumerate(paths):
        points = path['points']
        width = local_width(points, distance) or 1.
        length = float(np.linalg.norm(points[-1]-points[0]))
        linear = False
        if not path['closed'] and length >= max(6/scale, 2*width):
            errors = distance_to_polyline(points, points[[0, -1]])
            linear = (np.percentile(errors, 95) <= max(.75/scale, .35*width)
                      and errors.max() <= max(1.25/scale, .5*width))
        intervals.append({'id': f'observed-stroke-{index}', 'points_px': points.tolist(),
                          'linear_context': bool(linear)})
    ownership, audit = source_ownership(ink, intervals)
    if audit['incomplete']:
        return None, {**topology, **audit}
    nearest = ownership['owner'] if ownership is not None else np.zeros_like(gray, dtype=np.int32)
    return {'graph': {'proposals': intervals, 'search_incomplete': False},
            'source_owner': np.maximum(nearest, 0).astype(np.uint16),
            'ambiguous_owner_mask': (nearest < 0).astype(np.uint8)*255,
            'foreground': ink}, {**topology, 'ownership': audit}
