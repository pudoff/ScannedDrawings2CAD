"""Recover thick corner rules using their observed component width."""
import math
import cv2
import numpy as np
from . import page_frame_context_v1 as previous
from .stroke_roles_v1 import checked_gray, foreground
from .grid_role_revision_v1 import sha

POLICY = {**previous.POLICY, 'version': 'page-frame-context/2.0',
          'maximum_minor_to_major': .1, 'maximum_bbox_to_mean_width': 2.}


def straight_component(row, horizontal):
    """A broad rule is valid; a thin squiggle with a broad bbox is not."""
    x, y, width, height, area = map(int, row)
    major, minor = (width, height) if horizontal else (height, width)
    observed = area/max(1, major)
    return minor <= POLICY['maximum_minor_to_major']*major and minor <= POLICY['maximum_bbox_to_mean_width']*observed+math.sqrt(.5)


def corner_grid(ink, quad, audit):
    options = []
    for turn in range(4):
        p = np.roll(quad, -turn, axis=0)
        w = max(16, round(np.linalg.norm(p[1]-p[0]))); h = max(16, round(np.linalg.norm(p[3]-p[0])))
        scale = min(1., POLICY['maximum_side']/max(w, h)); w, h = round(w*scale), round(h*scale)
        audit['warp_pixels'] += w*h
        if audit['warp_pixels'] > POLICY['maximum_warp_pixels']:
            audit['limits'].append('warp_pixel_budget'); return None
        matrix = cv2.getPerspectiveTransform(p.astype(np.float32), np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32))
        local = cv2.warpPerspective(ink, matrix, (w, h), flags=cv2.INTER_NEAREST)
        hh = cv2.morphologyEx(local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w//8), 1)))
        vv = cv2.morphologyEx(local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h//18))))
        hs = cv2.connectedComponentsWithStats(hh, connectivity=8)[2][1:]
        vs = cv2.connectedComponentsWithStats(vv, connectivity=8)[2][1:]
        rows = [r.tolist() for r in hs if .15*w <= r[2] <= .75*w and r[1]+(r[3]-1)/2 >= .55*h and r[0]+r[2] >= .95*w and straight_component(r, True)]
        columns = [r.tolist() for r in vs if r[0]+(r[2]-1)/2 >= .25*w and r[1]+r[3] >= .95*h and r[3] >= .05*h and straight_component(r, False)]
        if len(rows) >= POLICY['corner_grid_rows'] and len(columns) >= POLICY['corner_grid_columns']:
            options.append({'quarter_turn': turn, 'rows': rows, 'columns': columns, 'supported': True,
                            'width_evidence': 'component_area_divided_by_observed_length'})
    return max(options, key=lambda o: (len(o['rows'])+len(o['columns']), -o['quarter_turn'])) if options else None


def recentered_sides(ink, quad):
    """Contour edges are not stroke centres; move by observed cross-sections."""
    span = min(np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[3]-quad[0]))
    reach = min(64., max(4., .04*span)); offsets = np.arange(-reach, reach+.25, .5)
    lines = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        u = (b-a)/np.linalg.norm(b-a); normal = np.array([-u[1], u[0]]); samples = []; widths = []
        for t in np.linspace(.08, .92, 21):
            point = a+t*(b-a); q = np.rint(point+offsets[:, None]*normal).astype(int)
            if (q < 0).any() or (q[:, 0] >= ink.shape[1]).any() or (q[:, 1] >= ink.shape[0]).any(): continue
            dark = ink[q[:, 1], q[:, 0]] > 0; change = np.diff(np.r_[False, dark, False].astype(int))
            runs = [(lo, hi) for lo, hi in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)) if lo > 0 and hi < len(offsets)]
            if not runs: continue
            lo, hi = min(runs, key=lambda r: abs((offsets[r[0]]+offsets[r[1]-1])/2))
            shift = (offsets[lo]+offsets[hi-1])/2; width = (hi-lo)*.5
            if abs(shift) > width/2+math.sqrt(.5): continue
            samples.append(point+shift*normal); widths.append(width)
        if len(samples) < 18: return []
        points = np.asarray(samples); center = points.mean(0)
        _, _, v = np.linalg.svd(points-center, full_matrices=False)
        fitted_normal = np.array([-v[0, 1], v[0, 0]])
        residual = float(np.quantile(np.abs((points-center) @ fitted_normal), .95))
        if residual > max(1., .2*np.median(widths)): return []
        lines.append((center, v[0], float(np.median(widths)), residual, len(samples)))
    corners = []
    for i in range(4):
        a, u = lines[i-1][:2]; b, v = lines[i][:2]
        if abs(np.linalg.det(np.c_[u, -v])) < 1e-6: return []
        t, _ = np.linalg.solve(np.c_[u, -v], b-a); corners.append(a+t*u)
    output = []
    for i, (center, u, width, error, count) in enumerate(lines):
        ends = center+((np.asarray([corners[i], corners[(i+1) % 4]])-center) @ u)[:, None]*u
        output.append({'points_overview': ends.tolist(), 'stroke_width_overview_px': width,
                       'position_uncertainty_overview_px': error+math.sqrt(.5),
                       'source_sample_count': count, 'observed_fraction': count/21})
    return output


def detect(gray):
    checked_gray(gray)
    legacy = previous.detect(gray)
    if legacy['frames'] or not legacy['complete']:
        return {**legacy, 'legacy_frames_retained': True, 'fallback_used': False}
    ink = foreground(gray); yy, xx = np.nonzero(ink)
    audit = {'warp_pixels': legacy['audit']['warp_pixels'], 'quad_checks': 0, 'limits': []}
    if not len(xx): return {**legacy, 'legacy_frames_retained': True, 'fallback_used': False}
    content = max(1, int(np.ptp(xx)+1)*int(np.ptp(yy)+1))
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if len(contours) > POLICY['maximum_contours'] or count-1 > POLICY['maximum_components']:
        audit['limits'].append('component_or_contour_budget')
    quads = []
    for contour in sorted(contours, key=lambda c: -abs(cv2.contourArea(c))):
        if abs(cv2.contourArea(contour)) < POLICY['minimum_content_area_fraction']*content: continue
        poly = cv2.approxPolyDP(contour, .015*cv2.arcLength(contour, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly): continue
        q = previous.ordered_quad(poly); e = np.roll(q, -1, axis=0)-q; u = e/np.linalg.norm(e, axis=1)[:, None]
        if np.max(np.abs(np.sum(u*np.roll(u, -1, axis=0), axis=1))) > math.sin(math.radians(8)): continue
        if not any(np.max(np.linalg.norm(q-old, axis=1)) < 2 for old in quads): quads.append(q)
    for outer in list(quads):
        if audit['limits']: break
        for inner in previous.inset_quads(ink, outer, audit):
            if not any(np.max(np.linalg.norm(inner-old, axis=1)) < 2 for old in quads): quads.append(inner)
    if len(quads) > POLICY['maximum_quads']: audit['limits'].append('quad_budget')
    frames = []; rejected = []
    for q in quads[:POLICY['maximum_quads']]:
        if audit['limits']: break
        audit['quad_checks'] += 1; w = np.linalg.norm(q[1]-q[0]); h = np.linalg.norm(q[3]-q[0])
        corners = np.clip(np.rint(q).astype(int), [0, 0], np.array(ink.shape[::-1])-1)
        attached = set(labels[corners[:, 1], corners[:, 0]].tolist()); views = []
        for i in range(1, count):
            if i in attached: continue
            x, y, ww, hh, pixels = map(int, stats[i])
            if min(ww, hh) < POLICY['minimum_view_extent_fraction']*min(w, h) or ww*hh > .5*w*h: continue
            if cv2.pointPolygonTest(q.astype(np.float32), (x+ww/2, y+hh/2), False) < 0: continue
            views.append({'component': i, 'bbox_overview': [x, y, x+ww, y+hh], 'ink_pixels': pixels})
        if not views: rejected.append('no_detached_view'); continue
        grid = corner_grid(ink, q, audit)
        if not grid: rejected.append('corner_grid'); continue
        sides = previous.observed_sides(ink, q) or recentered_sides(ink, q)
        if not sides: rejected.append('side_sampling'); continue
        frames.append({'quad_overview': q.tolist(), 'sides': sides, 'view_components': views,
                       'corner_grid': grid, 'frame_semantics_confirmed': False,
                       'evidence': 'observed_frame_with_width_normalized_corner_rules'})
    complete = not audit['limits']
    return {'source_sha256': sha(gray), 'size_px': list(gray.shape[::-1]), 'frames': frames if complete else [],
            'audit': audit, 'complete': complete, 'legacy_frames_retained': True, 'fallback_used': True,
            'fallback_rejections': rejected, 'production_admitted': False, 'physical_geometry_approved': False}
