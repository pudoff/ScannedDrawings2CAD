"""Rebuild existing source graph features from sealed intervals, without fitting."""
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked
from cad_recovery.interval_roles_v1 import tracks, hatch_groups
from cad_recovery.ownership_contours_v1 import boundaries
from cad_recovery.stroke_roles_v1 import foreground


def reconstruct(gray, intervals, matrix):
    scale = float(matrix[0][0]); ink = foreground(gray)
    lines, incomplete = tracks(intervals, ink, scale)
    widths = [i['width_px'] for i in intervals if i['linear_context']]
    stroke = max(1., min(24., float(np.quantile(widths, .25)))) if widths else 2.
    bundles, bundle_incomplete = hatch_groups(lines, stroke)
    geometry = [{'id': i['id'], 'path_id': i['id'], 'kind': 'line' if i['linear_context'] else 'polyline',
                 'points_px': [i['points_px'][0], i['points_px'][-1]] if i['linear_context'] else i['points_px'],
                 'closed': len(i['points_px']) > 1 and i['points_px'][0] == i['points_px'][-1]} for i in intervals]
    loops = boundaries(geometry, gray.shape, stroke)
    return {'proposals': intervals, 'tracks': [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in t.items()} for t in lines],
            'hatch_bundles': bundles, 'boundaries': loops, 'stroke_px': stroke,
            'search_incomplete': incomplete or bundle_incomplete or loops['search_incomplete']}
