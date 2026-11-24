"""Corroborate ambiguous contour roles with long, thick original strokes."""
import math

import cv2
import numpy as np

from .native_frame_v1 import side_masks
from .source_stamp_context_v1 import continuous
from .stroke_roles_v1 import checked_gray, foreground
from .supported_role_overlay_v1 import source_ownership


def _track_members(ink, graph, widths, stable, material, thin):
    indexes = {item['id']: i for i, item in enumerate(graph['proposals'], 1)}
    accepted = np.zeros_like(material)
    audit = {'limits': [], 'raster_pixels': 0}
    for track in graph.get('tracks', []):
        if len(track['interval_ids']) < 2:
            continue
        indices = np.asarray([indexes[i] for i in track['interval_ids']])
        thick = stable[indices] & material[indices] & (widths[indices] >= 1.55*thin)
        if np.count_nonzero(thick) < 2:
            continue
        pen = float(np.median(widths[indices[thick]]))
        ends = np.asarray(track['points'], float)
        delta = ends[1]-ends[0]
        length = np.linalg.norm(delta)
        if length < 12*pen or not continuous(ink, ends, pen, audit):
            continue
        normal = np.array([-delta[1], delta[0]])/length
        for index in indices[thick]:
            points = np.asarray(graph['proposals'][index-1]['points_px'])
            if np.quantile(np.abs((points-ends[0]) @ normal), .95) <= pen/4:
                accepted[index] = True
    if audit['limits']:
        accepted[:] = False
    return accepted, not audit['limits']


def analyze(gray, graph, proposed, *, join_tracks=False):
    """Check the common ink of the two role routes against observed profiles."""
    checked_gray(gray)
    if proposed.shape != gray.shape or proposed.dtype != bool:
        raise ValueError('Aligned boolean contour proposal required')
    blank = np.zeros_like(proposed)
    if graph.get('search_incomplete') or not proposed.any():
        return {'mask': blank, 'complete': not graph.get('search_incomplete', False), 'supported_intervals': 0}
    ink = foreground(gray)
    intervals = graph['proposals']
    ownership, work = source_ownership(ink, intervals)
    if ownership is None or work['incomplete']:
        return {'mask': blank, 'complete': not work['incomplete'], 'supported_intervals': 0}
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    owners = ownership['owner']
    widths = np.zeros(len(intervals)+1, float)
    references = np.zeros(len(intervals)+1, bool)
    stable = np.zeros(len(intervals)+1, bool)
    for i, interval in enumerate(intervals, 1):
        points = np.rint(interval['points_px']).astype(int)
        if len(points) < 8:
            continue
        trimmed = max(1, len(points)//8)
        x, y = points[trimmed:-trimmed].T
        samples = distance[y, x]
        samples = samples[samples > 0]
        if len(samples) < .8*len(x):
            continue
        low, high = np.quantile(samples, [.20, .85])
        width = max(1., 2*low - 1)
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        widths[i] = width
        well_observed = (2*high - 1) <= 1.5*width
        references[i] = well_observed and length >= 3*width
        stable[i] = well_observed and length >= 6*width and interval['parent_kind'] in ('line', 'arc', 'circle')
    comparison = widths[references]
    if len(comparison) < 4:
        return {'mask': blank, 'complete': True, 'supported_intervals': 0}
    thin = float(np.quantile(comparison, .25))
    material = np.array([False]+[i['role'] in ('part_contour', 'unknown') for i in intervals])
    accepted = material & stable & (widths >= 1.55*thin)
    track_complete = True
    if join_tracks:
        track_members, track_complete = _track_members(ink, graph, widths, references, material, thin)
        accepted |= track_members
    numbers = np.maximum(owners, 0)
    mask = proposed & (owners > 0) & accepted[numbers] & (ownership['distance'] <= widths[numbers]/2+np.sqrt(.5))
    return {'mask': mask, 'complete': track_complete, 'supported_intervals': int(accepted.sum()), 'thin_pen_px': thin}


def supported_track_footprints(gray, graph, accepted):
    """Fit both digital edges of a track with independently accepted ink."""
    checked_gray(gray)
    if accepted.shape != gray.shape or accepted.dtype != bool:
        raise ValueError('Aligned accepted original contour samples required')
    ink = foreground(gray) > 0
    result = np.zeros_like(ink)
    if not accepted.any() or graph.get('search_incomplete', False):
        return result
    y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
    audit = {'source_samples': 0}
    sampled_pixels = 0
    for track in graph['tracks']:
        ends = np.asarray(track['points'], float)
        delta = ends[1]-ends[0]
        length = float(np.linalg.norm(delta))
        pen = float(track['width'])
        if not 3 <= pen <= 24 or length < 6*pen:
            continue
        sampled_pixels += ink.size
        if sampled_pixels > 128_000_000:
            return np.zeros_like(ink)
        u = delta/length
        dx, dy = x-ends[0, 0], y-ends[0, 1]
        distance_along = dx*u[0]+dy*u[1]
        between = (distance_along >= 0) & (distance_along <= length)
        distance_across = np.abs(-dx*u[1]+dy*u[0])
        envelope = ink & between & (distance_across <= pen/2+1)
        if (envelope & accepted).sum() < max(60, .55*envelope.sum()):
            continue
        fitted, reason = side_masks(ink, ends, pen, math.sqrt(.5), audit)
        if reason and reason.endswith('_budget'):
            return np.zeros_like(ink)
        if fitted is not None and fitted['observed_width_across_px'] > 2:
            result |= fitted['safe'] & between
    return result
