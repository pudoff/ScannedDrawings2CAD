"""Pixel-derived page-frame hypotheses; no annotation or physical authority."""
import math
import cv2
import numpy as np
from .stroke_roles_v1 import checked_gray, foreground
from .grid_role_revision_v1 import sha

POLICY = {'version': 'page-frame-context/1.0', 'maximum_contours': 4096, 'maximum_quads': 24,
          'maximum_warp_pixels': 400000000, 'maximum_side': 1800,
          'maximum_components': 4096, 'minimum_content_area_fraction': .4,
          'minimum_view_extent_fraction': .08, 'minimum_view_components': 1,
          'corner_grid_rows': 3, 'corner_grid_columns': 3,
          'source_semantics_confirmed': False, 'physical_geometry_approved': False}


def ordered_quad(points):
    p = np.asarray(points, float).reshape(4, 2)
    center = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1]-center[1], p[:, 0]-center[0]))]
    return np.roll(p, -int(np.argmin(p.sum(1))), axis=0)


def corner_grid(ink, quad, audit):
    options = []
    for turn in range(4):
        p = np.roll(quad, -turn, axis=0)
        width = max(16, round(np.linalg.norm(p[1]-p[0])))
        height = max(16, round(np.linalg.norm(p[3]-p[0])))
        scale = min(1., POLICY['maximum_side']/max(width, height))
        width, height = round(width*scale), round(height*scale)
        audit['warp_pixels'] += width*height
        if audit['warp_pixels'] > POLICY['maximum_warp_pixels']:
            audit['limits'].append('warp_pixel_budget'); return None
        matrix = cv2.getPerspectiveTransform(p.astype(np.float32), np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32))
        local = cv2.warpPerspective(ink, matrix, (width, height), flags=cv2.INTER_NEAREST)
        horizontal = cv2.morphologyEx(local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, width//8), 1)))
        vertical = cv2.morphologyEx(local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, height//18))))
        hs = cv2.connectedComponentsWithStats(horizontal, connectivity=8)[2][1:]
        vs = cv2.connectedComponentsWithStats(vertical, connectivity=8)[2][1:]
        rows = [r.tolist() for r in hs if .15*width <= r[2] <= .75*width and r[1] >= .55*height
                and r[0]+r[2] >= .95*width and r[3] <= max(12, .02*height)]
        columns = [r.tolist() for r in vs if r[0] >= .25*width and r[1]+r[3] >= .95*height
                   and r[3] >= .05*height and r[2] <= max(12, .02*width)]
        supported = len(rows) >= POLICY['corner_grid_rows'] and len(columns) >= POLICY['corner_grid_columns']
        options.append({'quarter_turn': turn, 'rows': rows, 'columns': columns, 'supported': supported})
    good = [o for o in options if o['supported']]
    return max(good, key=lambda o: (len(o['rows'])+len(o['columns']), -o['quarter_turn'])) if good else None


def observed_sides(ink, quad):
    center = quad.mean(0); result = []
    span = min(np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[3]-quad[0]))
    radius = min(32., max(4., .015*span))
    offsets = np.arange(-radius, radius+.25, .5)
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        axis = (b-a)/np.linalg.norm(b-a); normal = np.array([-axis[1], axis[0]])
        if normal @ (center-(a+b)/2) < 0: normal = -normal
        samples, widths = [], []
        for t in np.linspace(.08, .92, 21):
            point = a+t*(b-a); q = np.rint(point+offsets[:, None]*normal).astype(int)
            if (q < 0).any() or (q[:, 0] >= ink.shape[1]).any() or (q[:, 1] >= ink.shape[0]).any(): continue
            dark = ink[q[:, 1], q[:, 0]] > 0
            change = np.diff(np.r_[False, dark, False].astype(int))
            runs = list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))
            runs = [(lo, hi) for lo, hi in runs if lo > 0 and hi < len(offsets)]
            if not runs: continue
            lo, hi = min(runs, key=lambda r: abs((offsets[r[0]]+offsets[r[1]-1])/2))
            shift = (offsets[lo]+offsets[hi-1])/2
            if abs(shift) > radius/2: continue
            samples.append(point+shift*normal); widths.append((hi-lo)*.5)
        if len(samples) < 18: return []
        samples = np.asarray(samples); location = samples.mean(0)
        _, _, vectors = np.linalg.svd(samples-location, full_matrices=False); u = vectors[0]
        if u @ axis < 0: u = -u
        n = np.array([-u[1], u[0]])
        residual = float(np.quantile(np.abs((samples-location) @ n), .95))
        if residual > max(1., .2*np.median(widths)): return []
        ends = location+((np.stack([a, b])-location) @ u)[:, None]*u
        result.append({'points_overview': ends.tolist(), 'stroke_width_overview_px': float(np.median(widths)),
                       'position_uncertainty_overview_px': residual+math.sqrt(.5),
                       'source_sample_count': len(samples), 'observed_fraction': len(samples)/21})
    return result


def inset_quads(ink, outer, audit):
    """Recover inset working frames even if a footer connects them to paper ink."""
    width = max(16, round(np.linalg.norm(outer[1]-outer[0])))
    height = max(16, round(np.linalg.norm(outer[3]-outer[0])))
    scale = min(1., POLICY['maximum_side']/max(width, height))
    width, height = round(width*scale), round(height*scale)
    audit['warp_pixels'] += width*height
    if audit['warp_pixels'] > POLICY['maximum_warp_pixels']:
        audit['limits'].append('warp_pixel_budget'); return []
    matrix = cv2.getPerspectiveTransform(outer.astype(np.float32), np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32))
    local = cv2.warpPerspective(ink, matrix, (width, height), flags=cv2.INTER_NEAREST)
    horizontal = cv2.morphologyEx(local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, width//2), 1)))
    vertical = cv2.morphologyEx(local, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, height//2))))
    hs = cv2.connectedComponentsWithStats(horizontal, connectivity=8)[2][1:]
    vs = cv2.connectedComponentsWithStats(vertical, connectivity=8)[2][1:]
    tops = [r for r in hs if r[2] >= .7*width and r[1] <= .12*height]
    bottoms = [r for r in hs if r[2] >= .7*width and r[1] >= .88*height]
    lefts = [r for r in vs if r[3] >= .7*height and r[0] <= .12*width]
    rights = [r for r in vs if r[3] >= .7*height and r[0] >= .88*width]
    # These are long-side hypotheses only, not arbitrary cell rectangles.
    if any(len(group) > 6 for group in (tops, bottoms, lefts, rights)): return []
    output = []; inverse = np.linalg.inv(matrix)
    for top in tops:
        for bottom in bottoms:
            y0, y1 = top[1]+(top[3]-1)/2, bottom[1]+(bottom[3]-1)/2
            for left in lefts:
                for right in rights:
                    x0, x1 = left[0]+(left[2]-1)/2, right[0]+(right[2]-1)/2
                    if x1-x0 < .75*width or y1-y0 < .75*height: continue
                    if any(r[0] > x0+max(2, 2*r[3]) or r[0]+r[2]-1 < x1-max(2, 2*r[3]) for r in (top, bottom)): continue
                    if any(r[1] > y0+max(2, 2*r[2]) or r[1]+r[3]-1 < y1-max(2, 2*r[2]) for r in (left, right)): continue
                    q = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]) @ inverse.T
                    output.append(q[:, :2]/q[:, 2:])
    return output


def detect(gray):
    checked_gray(gray); ink = foreground(gray)
    audit = {'warp_pixels': 0, 'quad_checks': 0, 'limits': []}
    y, x = np.nonzero(ink)
    if not len(x): return {'source_sha256': sha(gray), 'size_px': list(gray.shape[::-1]), 'frames': [], 'audit': audit, 'complete': True}
    content_area = max(1, int(np.ptp(x)+1)*int(np.ptp(y)+1))
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if len(contours) > POLICY['maximum_contours'] or count-1 > POLICY['maximum_components']:
        audit['limits'].append('component_or_contour_budget')
    quads = []
    for contour in sorted(contours, key=lambda c: -abs(cv2.contourArea(c))):
        area = abs(cv2.contourArea(contour))
        if area < POLICY['minimum_content_area_fraction']*content_area: continue
        polygon = cv2.approxPolyDP(contour, .015*cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon): continue
        quad = ordered_quad(polygon)
        edges = np.roll(quad, -1, axis=0)-quad
        units = edges/np.linalg.norm(edges, axis=1)[:, None]
        if np.max(np.abs(np.sum(units*np.roll(units, -1, axis=0), axis=1))) > math.sin(math.radians(8)): continue
        # Opposite stroke edges produce near-identical contour quads. Deduplicate
        # within observed edge width, without merging different inset frames.
        if any(np.max(np.linalg.norm(quad-old, axis=1)) < 2 for old in quads): continue
        quads.append(quad)
    for outer in list(quads):
        for inner in inset_quads(ink, outer, audit):
            if not any(np.max(np.linalg.norm(inner-old, axis=1)) < 2 for old in quads): quads.append(inner)
    if len(quads) > POLICY['maximum_quads']: audit['limits'].append('quad_budget')
    frames = []
    for quad in quads[:POLICY['maximum_quads']]:
        if audit['limits']: break
        audit['quad_checks'] += 1
        w, h = np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[3]-quad[0]); area = w*h
        corner = np.rint(quad).astype(int); corner = np.clip(corner, [0, 0], np.array(ink.shape[::-1])-1)
        attached = set(labels[corner[:, 1], corner[:, 0]].tolist())
        views = []
        for i in range(1, count):
            if i in attached: continue
            xx, yy, ww, hh, pixels = map(int, stats[i])
            if min(ww, hh) < POLICY['minimum_view_extent_fraction']*min(w, h) or ww*hh > .5*area: continue
            if cv2.pointPolygonTest(quad.astype(np.float32), (xx+ww/2, yy+hh/2), False) < 0: continue
            views.append({'component': i, 'bbox_overview': [xx, yy, xx+ww, yy+hh], 'ink_pixels': pixels})
        if len(views) < POLICY['minimum_view_components']: continue
        grid = corner_grid(ink, quad, audit)
        if not grid: continue
        sides = observed_sides(ink, quad)
        if not sides: continue
        frames.append({'quad_overview': quad.tolist(), 'sides': sides, 'view_components': views,
                       'corner_grid': grid, 'frame_semantics_confirmed': False,
                       'evidence': 'observed_enclosing_quad_plus_detached_view_and_local_corner_grid'})
    complete = not audit['limits']
    if not complete: frames = []
    return {'source_sha256': sha(gray), 'size_px': list(gray.shape[::-1]), 'frames': frames,
            'audit': audit, 'complete': complete, 'production_admitted': False, 'physical_geometry_approved': False}
