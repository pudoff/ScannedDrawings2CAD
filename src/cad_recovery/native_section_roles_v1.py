"""Confirm an observed contour segment from repeat hatch contacts."""
import math

import cv2
import numpy as np

from .native_frame_v1 import side_masks
from .source_stamp_context_v1 import continuous
from .stroke_roles_v1 import checked_gray, foreground


def axis_strokes(ink, major, *, opening, minimum_length, maximum_band):
    kernel = (1, opening) if major == 0 else (opening, 1)
    opened = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, np.ones(kernel, np.uint8))
    _, labels, bounds, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    for index, box in enumerate(bounds[1:], 1):
        if box[2+major] < minimum_length or box[3-major] > maximum_band:
            continue
        x, y, width, height, _ = box
        rows, columns = np.nonzero(labels[y:y+height, x:x+width] == index)
        points = np.c_[columns+x, rows+y]
        axial = points[:, major]
        pen = float(np.median(np.unique(axial, return_counts=True)[1]))
        yield points, (int(axial.min()), int(axial.max())), float(np.median(points[:, 1-major])), pen


def _longest_hatch_group(hits, pen):
    best = []
    for _, direction, side in hits:
        same = sorted(axis for axis, orient, sign in hits
                      if side == sign and direction @ orient >= math.cos(math.radians(8)))
        spaced = []
        for axis in same:
            if not spaced or axis-spaced[-1] > 1.5*pen:
                spaced.append(axis)
        if len(spaced) > len(best):
            best = spaced
    return best if len(best) >= 3 and best[-1]-best[0] > 3*pen else []


def hatch_supported_contours(gray, graph):
    checked_gray(gray)
    ink = foreground(gray) > 0
    result = np.zeros_like(ink)
    if graph.get('search_incomplete') or len(graph['tracks']) > 1024:
        return result
    audit = {'source_samples': 0, 'raster_pixels': 0, 'limits': []}
    checks = 0
    for major in (0, 1):
        minor = 1-major
        for _, (lo, hi), center, pen in axis_strokes(ink, major, opening=57, minimum_length=57, maximum_band=28):
            if pen < 5:
                continue
            hits = []
            for track in graph['tracks']:
                checks += 1
                if checks > 250_000:
                    return np.zeros_like(ink)
                ends = np.asarray(track['points'], float)
                delta = ends[1]-ends[0]
                length = float(np.linalg.norm(delta))
                if length < 3*pen or track['width']*1.25 > pen or not .38 <= abs(delta[major])/length <= .90:
                    continue
                for endpoint, other in (ends, ends[::-1]):
                    if (not lo <= endpoint[major] <= hi or abs(other[minor]-center) <= pen
                            or abs(endpoint[minor]-center) > .5*(pen+track['width'])+2):
                        continue
                    contact = endpoint.copy()
                    contact[minor] = center
                    if np.linalg.norm(endpoint-contact) > 1 and not continuous(ink, [endpoint, contact], track['width'], audit):
                        continue
                    direction = (other-endpoint)/length
                    hits.append((float(endpoint[major]), direction, np.sign(direction[minor])))
                    break
            group = _longest_hatch_group(hits, pen)
            if audit['limits']:
                return np.zeros_like(ink)
            if not group:
                continue
            ends = np.array([[group[0], center], [group[-1], center]])
            if major == 1:
                ends = ends[:, ::-1]
            fits, reason = side_masks(ink, ends, pen, math.sqrt(.5), audit)
            if reason and reason.endswith('_budget'):
                return np.zeros_like(ink)
            if fits is not None:
                x = np.arange(gray.shape[1])[None, :] if major == 0 else np.arange(gray.shape[0])[:, None]
                result |= fits['safe'] & (x >= group[0]) & (x <= group[-1])
    return result


def anchored_straight_contours(gray, accepted):
    """Fit an observed thick stroke where the same line already carries PART."""
    checked_gray(gray)
    if accepted.dtype != bool or accepted.shape != gray.shape:
        raise ValueError('Aligned contour anchors required')
    result = np.zeros_like(accepted)
    if not accepted.any():
        return result
    ink = foreground(gray) > 0
    audit = {'source_samples': 0}
    sampled_pixels = 0
    for major in (0, 1):
        along = np.arange(gray.shape[1-major])
        along = along[None, :] if major == 0 else along[:, None]
        for points, (lo, hi), center, pen in axis_strokes(ink, major, opening=35, minimum_length=43, maximum_band=20):
            x, y = points.T
            if pen < 4 or accepted[y, x].sum() < .55*len(points):
                continue
            sampled_pixels += ink.size
            if sampled_pixels > 64_000_000:
                return np.zeros_like(result)
            ends = np.array([[lo, center], [hi, center]])
            if major:
                ends = ends[:, ::-1]
            fitted, reason = side_masks(ink, ends, pen, .7, audit)
            if reason and reason.endswith('_budget'):
                return np.zeros_like(result)
            if fitted is not None:
                result |= fitted['safe'] & (along >= lo) & (along <= hi)
    return result
