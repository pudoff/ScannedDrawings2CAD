"""Turn whole-page stamp hypotheses into bounded, overlapped original windows."""

from math import ceil

import cv2
import numpy as np

from .attached_grid_context_io_v1 import MAX_PIXELS
from .frame_view_guard_v1 import detect
from .page_grid_attachment_v1 import analyze


WINDOW_SIDE = 2600
WINDOW_OVERLAP = 256
PADDING = WINDOW_OVERLAP // 2
MAX_WINDOWS = 8


def edge_band(points, size):
    """Project a group toward the closest page edge parallel to its long axis."""
    low = np.maximum(np.floor(points.min(0)).astype(int) - PADDING, 0)
    high = np.minimum(np.ceil(points.max(0)).astype(int) + PADDING + 1, size)
    tangent = int(np.argmax(high-low))
    normal = 1-tangent
    at_end = low[normal] > size[normal]-high[normal]
    if at_end:
        high[normal] = size[normal]
    else:
        low[normal] = 0
    return (normal, bool(at_end)), np.r_[low, high]


def merge_bands(bands):
    pending = sorted((edge, tuple(box)) for edge, box in bands)
    merged = []
    while pending:
        edge, coords = pending.pop(0)
        box = np.array(coords)
        changed = True
        while changed:
            changed = False
            for i, (other_edge, other) in enumerate(pending):
                if (edge == other_edge and np.all(np.maximum(box[:2], other[:2]) <= np.minimum(box[2:], other[2:]))):
                    box = np.r_[np.minimum(box[:2], other[:2]), np.maximum(box[2:], other[2:])]
                    pending.pop(i)
                    changed = True
                    break
        merged.append((edge, box))
    return merged


def bounded_windows(bands):
    windows = set()
    reasons = []
    for (normal, _), band in merge_bands(bands):
        tangent = 1-normal
        depth = int(band[normal+2]-band[normal])
        if not 0 < depth <= WINDOW_SIDE:
            reasons.append('edge_band_depth_budget')
            continue
        side = min(WINDOW_SIDE, MAX_PIXELS // depth)
        if side <= WINDOW_OVERLAP:
            reasons.append('edge_band_window_budget')
            continue
        start, end = int(band[tangent]), int(band[tangent+2])
        span = end-start
        count = 1 + max(0, ceil((span-side)/(side-WINDOW_OVERLAP)))
        if count > MAX_WINDOWS:
            return [], ['page_window_budget']
        starts = np.linspace(start, max(start, end-side), count).astype(int)
        for offset in starts:
            window = band.copy()
            window[tangent] = offset
            window[tangent+2] = min(offset + side, end)
            if len(windows) >= MAX_WINDOWS and tuple(window) not in windows:
                return [], ['page_window_budget']
            windows.add(tuple(map(int, window)))
    return [list(box) for box in sorted(windows)], sorted(set(reasons))


def plan(overview, preparation):
    metadata = preparation['overview']
    if list(overview.shape[::-1]) != metadata['size_px'] or overview.size > MAX_PIXELS:
        raise ValueError('Bounded preparation overview required')
    cv2.setRNGSeed(0)
    frame = detect(overview)
    attached = analyze(overview, frame)
    if not attached['complete']:
        return frame, {'complete': False, 'windows': [], 'reasons': ['incomplete_overview_attachment'], 'groups': 0}
    matrix = np.asarray(metadata['overview_to_source'], float)
    rules = {r['id']: r['points_source_px'] for r in attached['rules']}
    bands = []
    for group in attached['groups']:
        points = np.concatenate([rules[identity] for identity in group['rule_ids']])
        projected = points @ matrix[:2, :2].T + matrix[:2, 2]
        bands.append(edge_band(projected, preparation['source_size_px']))
    windows, reasons = bounded_windows(bands)
    return frame, {'complete': not reasons, 'windows': windows, 'reasons': reasons,
                   'groups': len(attached['groups']), 'overview_is_locator_only': True}
