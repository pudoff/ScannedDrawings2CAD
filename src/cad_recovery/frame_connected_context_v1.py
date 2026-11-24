"""Additional source frame evidence from a connected, closed view outside a grid.

Only an analysis copy loses candidate ruling ink. Original pixels, role maps
and primitives are never edited. Existing detached-view decisions stay intact.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from . import page_frame_context_v1 as quad_tools
from .page_frame_context_v2 import recentered_sides
from .frame_view_guard_v1 import detect as previous_detect, POLICY as VIEW_POLICY
from .grid_role_revision_v1 import sha
from .native_frame_v1 import side_masks
from .page_grid_attachment_v1 import analyze as attach
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'frame-connected-context/1.0', 'maximum_pixels': 4194304,
    'maximum_quads': 24, 'maximum_components': 4096, 'maximum_contours': 4096,
    'maximum_component_pixels': 16000000, 'maximum_raster_pixels': 64000000,
    'minimum_view_extent_fraction': VIEW_POLICY['minimum_view_extent_fraction'],
    'minimum_fraction_outside_grid': .8, 'maximum_view_bbox_fraction': .5,
    'fallback_only_without_accepted_frames': True, 'exact_fit_all_cut_strokes': True,
    'source_semantics_confirmed': False, 'source_pixels_erased': 0,
    'physical_geometry_approved': False, 'production_admitted': False}


def candidates(gray, audit):
    ink = foreground(gray); yy, xx = np.nonzero(ink)
    if not len(xx): return []
    content = max(1, int(np.ptp(xx)+1)*int(np.ptp(yy)+1))
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > POLICY['maximum_contours']:
        audit['limits'].append('candidate_contour_budget'); return []
    quads = []
    def add(q):
        if not any(np.max(np.linalg.norm(q-old, axis=1)) < 2 for old in quads): quads.append(q)
    for contour in sorted(contours, key=lambda c: -abs(cv2.contourArea(c))):
        if abs(cv2.contourArea(contour)) < quad_tools.POLICY['minimum_content_area_fraction']*content: continue
        p = cv2.approxPolyDP(contour, .015*cv2.arcLength(contour, True), True)
        if len(p) != 4 or not cv2.isContourConvex(p): continue
        q = quad_tools.ordered_quad(p); edges = np.roll(q, -1, axis=0)-q
        lengths = np.linalg.norm(edges, axis=1)
        if (lengths <= 0).any(): continue
        u = edges/lengths[:, None]
        if np.max(np.abs(np.sum(u*np.roll(u, -1, axis=0), axis=1))) > math.sin(math.radians(8)): continue
        add(q)
    for q in list(quads):
        for inner in quad_tools.inset_quads(ink, q, audit): add(inner)
        if audit['limits']: break
    if len(quads) > POLICY['maximum_quads']:
        audit['limits'].append('candidate_quad_budget'); return []
    return quads


def connected_view(ink, quad, rules, exclusion, audit):
    """A ruling cell, open cross or large stroke extent alone is not a view."""
    audit['raster_pixels'] += ink.size
    if audit['raster_pixels'] > POLICY['maximum_raster_pixels']:
        audit['limits'].append('analysis_raster_budget'); return None
    if exclusion.shape != ink.shape or exclusion.dtype != bool or np.any(exclusion & ~ink):
        raise ValueError('Analysis exclusion must contain only observed source ink')
    remaining = (ink & ~exclusion).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(remaining, connectivity=8)
    if count-1 > POLICY['maximum_components']:
        audit['limits'].append('analysis_component_budget'); return None
    hull = cv2.convexHull(np.concatenate([np.asarray(r['points_source_px'], float) for r in rules]).astype(np.float32))
    if len(hull) < 3: return None
    grid = np.zeros_like(remaining); cv2.fillConvexPoly(grid, np.rint(hull).astype(np.int32), 1)
    inside = np.zeros_like(remaining); cv2.fillConvexPoly(inside, np.rint(quad).astype(np.int32), 1)
    span = min(np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[3]-quad[0]))
    area = abs(cv2.contourArea(quad.astype(np.float32)))
    minimum = POLICY['minimum_view_extent_fraction']*span
    for number in range(1, count):
        x, y, width, height, pixels = map(int, stats[number])
        if min(width, height) < minimum or width*height > POLICY['maximum_view_bbox_fraction']*area: continue
        audit['component_pixels'] += pixels
        if audit['component_pixels'] > POLICY['maximum_component_pixels']:
            audit['limits'].append('view_component_pixel_budget'); return None
        yy, xx = np.nonzero(labels[y:y+height, x:x+width] == number); xx += x; yy += y
        points = np.c_[xx, yy].astype(np.float32)
        extent = float(min(cv2.minAreaRect(points)[1]))
        outside = float(np.mean(grid[yy, xx] == 0))
        if extent < minimum or outside < POLICY['minimum_fraction_outside_grid'] or not inside[yy, xx].all(): continue
        local = (labels[y:y+height, x:x+width] == number).astype(np.uint8)
        contours, hierarchy = cv2.findContours(local, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > POLICY['maximum_contours']:
            audit['limits'].append('view_contour_budget'); return None
        enclosed = []
        for index, contour in enumerate(contours):
            if hierarchy[0, index, 3] < 0: continue
            hole_extent = float(min(cv2.minAreaRect(contour.astype(np.float32))[1]))
            if hole_extent < minimum: continue
            # Do not count a cavity made solely by excluding originally dark
            # pixels. A large connected white interior must exist in raw source.
            audit['raster_pixels'] += width*height
            if audit['raster_pixels'] > POLICY['maximum_raster_pixels']:
                audit['limits'].append('analysis_raster_budget'); return None
            hole = np.zeros_like(local); cv2.drawContours(hole, [contour], -1, 1, cv2.FILLED)
            white = ((hole > 0) & ~ink[y:y+height, x:x+width]
                     & (grid[y:y+height, x:x+width] == 0)).astype(np.uint8)
            nwhite, white_labels, white_stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
            if nwhite-1 > POLICY['maximum_components']:
                audit['limits'].append('source_white_component_budget'); return None
            for white_id in range(1, nwhite):
                if min(white_stats[white_id, 2:4]) < minimum: continue
                wy, wx = np.nonzero(white_labels == white_id)
                original_extent = float(min(cv2.minAreaRect(np.c_[wx, wy].astype(np.float32))[1]))
                if original_extent >= minimum:
                    enclosed.append(original_extent); break
        if not enclosed: continue
        return {'component': number, 'source_bbox': [x, y, x+width, y+height],
            'minimum_oriented_extent_px': extent, 'minimum_enclosed_extent_px': max(enclosed),
            'frame_span_px': float(span), 'outside_grid_fraction': outside,
            'analysis_exclusion_pixels': int(exclusion.sum()), 'source_pixels_erased': 0,
            'semantic_view_confirmed': False}
    return None


def extend(gray, legacy):
    checked_gray(gray)
    if (legacy['source_sha256'] != sha(gray) or legacy['size_px'] != list(gray.shape[::-1])
            or any(f.get('frame_semantics_confirmed') is not False for f in legacy['frames'])):
        raise ValueError('Unconfirmed pixel-bound legacy frame packet required')
    if gray.size > POLICY['maximum_pixels']: raise ValueError('Bounded original page required')
    before, pixels_before = object_sha(legacy), sha(gray)
    result = deepcopy(legacy)
    audit = {'limits': [], 'warp_pixels': 0, 'raster_pixels': 0, 'component_pixels': 0,
             'source_samples': 0, 'discarded_frames': 0}
    decisions = []; accepted = []
    if not legacy['complete']: audit['limits'].append('incomplete_legacy_frame_search')
    if not legacy['frames'] and not audit['limits']:
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
        for quad in candidates(gray, audit):
            sides = quad_tools.observed_sides(ink.astype(np.uint8)*255, quad) or recentered_sides(ink.astype(np.uint8)*255, quad)
            if not sides:
                decisions.append({'reason': 'source_sides_unobserved'}); continue
            frame = {'quad_overview': quad.tolist(), 'sides': sides, 'view_components': [],
                     'corner_grid': {'rows': [], 'columns': []}, 'frame_semantics_confirmed': False}
            packet = {'source_sha256': pixels_before, 'size_px': list(gray.shape[::-1]), 'complete': True, 'frames': [frame]}
            cv2.setRNGSeed(0); observed = attach(gray, packet)
            if not observed['complete']:
                audit['limits'].append('incomplete_source_grid_search'); break
            rules = [r for r in observed['rules'] if r['grounded_group_member']]
            if not rules:
                decisions.append({'reason': 'no_source_attached_grid'}); continue
            strokes = [(np.asarray(s['points_overview'], float), s['stroke_width_overview_px'], s['position_uncertainty_overview_px']) for s in sides]
            strokes += [(np.asarray(r['points_source_px'], float), r['width_px'], r['uncertainty_px']) for r in rules]
            exclusion = np.zeros_like(ink); failed = None
            for points, width, error in strokes:
                fit, reason = side_masks(ink, points, float(width), float(error), audit)
                if fit is None:
                    failed = reason
                    if reason.endswith('_budget'): audit['limits'].append(reason)
                    break
                exclusion |= fit['possible']
            if failed:
                decisions.append({'reason': 'unfitted_cut_stroke', 'fit_reason': failed})
                if audit['limits']: break
                continue
            proof = connected_view(ink, quad, rules, exclusion, audit)
            decisions.append({'reason': None if proof else 'no_closed_view_outside_grid', 'proof': proof,
                              'qualified_rules': len(rules), 'quad_overview': quad.tolist()})
            if proof:
                frame.update(connected_view_witness=proof, evidence='exact_source_connected_view_outside_grid')
                accepted.append(frame)
            if audit['limits']: break
    complete = not audit['limits']
    if not complete:
        audit['discarded_frames'] = len(accepted); accepted = []
    result['frames'] += accepted
    result.update(complete=complete, connected_view_context={'policy': deepcopy(POLICY), 'audit': audit,
        'decisions': decisions, 'additional_frames': len(accepted), 'legacy_frames': len(legacy['frames']),
        'source_immutable': True, 'complete': complete})
    if object_sha(legacy) != before or sha(gray) != pixels_before:
        raise ValueError('Connected view observer mutated source observations')
    return result


def detect(gray):
    return extend(gray, previous_detect(gray))
