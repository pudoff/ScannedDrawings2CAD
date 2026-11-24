"""Keep legacy native joins, adding pixel support for continuous CAD lines."""
import math
from unittest.mock import patch
import numpy as np
from . import native_sheet_stitch_v1 as previous


def supported_line(row, page):
    a, b = np.asarray(row['points_px'], float)
    n = max(2, math.ceil(np.linalg.norm(b-a)*2)+1)
    if n > 200000:
        return False
    points = np.linspace(a, b, n)
    near = np.full(n, 255, np.uint8)
    for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        near = np.minimum(near, page.samples(points+[dx, dy]))
    return bool(np.all(near < 240))


def stitch(rows, page, *, CENTER_pattern=False):
    original = previous.union

    def union(a, b, source):
        joined = original(a, b, source)
        if joined is not None and joined['kind'] == 'line' and not CENTER_pattern:
            if not supported_line(joined, source):
                return None
        return joined

    # Only one conversion job per process, as required by the frozen core.
    with patch.object(previous, 'union', union):
        result, proof = previous.stitch(rows, page)
    proof.update(continuous_line_native_support_required=not CENTER_pattern,
                 CENTER_pattern_retained=CENTER_pattern,
                 no_geometric_gap_extrapolation=True)
    return result, proof
