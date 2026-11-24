"""Propose bounded stamp windows using observed sides of a partial frame."""

import cv2
import numpy as np

from .grid_role_revision_v1 import sha
from .native_section_roles_v1 import axis_strokes
from .native_table_roles_v1 import analyze as table_roles
from .stamp_window_plan_v1 import PADDING, bounded_windows, plan as attached_plan


CORNER_FRACTION = .48
MINIMUM_GRID_SPAN = .045
MINIMUM_SIDE_SPAN = .55
MINIMUM_SIDE_EXTENSION = .12
EDGE_BAND = .10


def edge_strokes(gray):
    """Find long observed side candidates, excluding the pixel canvas rim."""
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    size = gray.shape[::-1]
    sides = {}
    for axis, length in enumerate(size):
        across = size[1-axis]
        for _, span, center, pen in axis_strokes(
                ink, axis, opening=max(3, round(length/5) | 1),
                minimum_length=round(length*MINIMUM_SIDE_SPAN), maximum_band=max(3, round(across/80))):
            far = center > (across-1)/2
            margin = across-1-center if far else center
            if max(2*pen, 1) < margin <= EDGE_BAND*across:
                sides.setdefault((axis, far), []).append((span, center, pen))
    return sides


def adjacent_sides(box, corner, size, sides):
    result = []
    for axis, length in enumerate(size):
        normal = 1-axis
        low, high = box[axis], box[axis+2]
        far = corner[normal]
        matches = []
        for (start, end), center, pen in sides.get((axis, far), ()):
            normal_gap = center-box[normal+2] if far else box[normal]-center
            coverage = min(end, high)-max(start, low)
            extension = max(low-start, end-high)
            if (normal_gap >= 0 and normal_gap <= EDGE_BAND*size[normal]
                    and coverage >= .8*(high-low) and extension >= MINIMUM_SIDE_EXTENSION*length):
                matches.append((normal_gap, start, end, center, pen))
        if matches:
            _, start, end, center, pen = min(matches)
            points = [[start, center], [end, center]]
            if axis:
                points = [point[::-1] for point in points]
            result.append({'edge': (normal, far), 'points_overview': points,
                           'stroke_width_overview_px': pen, 'position_uncertainty_overview_px': 1.})
    return result


def corner_candidates(overview):
    height, width = overview.shape
    size = np.array([width, height])
    crop_w, crop_h = np.maximum(1, np.rint(size * CORNER_FRACTION)).astype(int)
    sides = edge_strokes(overview)
    candidates = []
    for right in (False, True):
        for bottom in (False, True):
            offset = np.array([width-crop_w if right else 0, height-crop_h if bottom else 0])
            x, y = offset
            table = table_roles(overview[y:y+crop_h, x:x+crop_w], original_edge=False)
            if not table['complete']:
                return [], 'incomplete_corner_table_search'
            for network in table['networks']:
                l, t, r, b = network['box_px']
                yy, xx = np.nonzero(table['grid'][t:b, l:r])
                if not len(xx):
                    continue
                low = offset + [l+xx.min(), t+yy.min()]
                high = offset + [l+xx.max()+1, t+yy.max()+1]
                if np.any(high-low < MINIMUM_GRID_SPAN*size):
                    continue
                box = np.r_[low, high]
                supports = adjacent_sides(box, (right, bottom), size, sides)
                if supports:
                    candidates.append({'corner': [right, bottom], 'box_overview_px': box.tolist(),
                                       'sides': supports, 'cells': network['cells'],
                                       'text_cells': network['text_cells']})
    return candidates, None


def plan(overview, preparation):
    frame, primary = attached_plan(overview, preparation)
    if primary['windows'] or not primary['complete']:
        return frame, primary
    options, failure = corner_candidates(overview)
    reasons = [failure] if failure else []
    windows = []
    if len(options) == 1:
        option = options[0]
        matrix = np.asarray(preparation['overview']['overview_to_source'], float)
        size = np.asarray(preparation['source_size_px'])
        low = np.asarray(option['box_overview_px'][:2], float)
        high = np.asarray(option['box_overview_px'][2:], float)
        # Use the table and the nearby side positions, never the full side lengths.
        for side in option['sides']:
            normal, far = side['edge']
            (high if far else low)[normal] = side['points_overview'][0][normal]
        major = int(np.argmax(high-low))
        side = next((s for s in option['sides'] if s['edge'][0] == 1-major), option['sides'][0])
        normal, far = side['edge']
        corners = np.array([[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])])
        corners = corners @ matrix[:2, :2].T + matrix[:2, 2]
        lo = np.maximum(0, np.floor(corners.min(0)).astype(int)-PADDING)
        hi = np.minimum(size, np.ceil(corners.max(0)).astype(int)+PADDING+1)
        if far:
            hi[normal] = size[normal]
        else:
            lo[normal] = 0
        windows, reasons = bounded_windows([((normal, far), np.r_[lo, hi])])
        if windows:
            frame = {**frame, 'source_sha256': sha(overview), 'frames': [
                {'frame_semantics_confirmed': False, 'sides': option['sides']}]}
    elif len(options) > 1:
        reasons = ['ambiguous_edge_tables']
    complete = not failure and not (len(options) == 1 and reasons)
    return frame, {**primary, 'complete': complete, 'windows': windows, 'reasons': reasons,
                   'route': 'partial_frame_fallback', 'candidates': options,
                   'groups': int(bool(windows))}
