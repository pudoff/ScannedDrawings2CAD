"""Bounded regional proposals with source-pixel validation and explicit rejections.

Masks affect proposal generation only. Neither masks nor a Hough vote can
approve a curve. The unmodified raster and the original full-ring gate decide.
"""
import math

import cv2
import numpy as np

POLICY = {'version': 'bounded-regional-geometry/2.0', 'profiles': ['ranked_global', 'regional_masked'],
    'circle_validation_budget': 300, 'cheap_screen_budget': 4096, 'cheap_angles': 72,
    'cheap_support_min': .72, 'proposal_max_side': 1200, 'regional_overlap_fraction': .2,
    'maximum_regions': 5, 'new_short_line_budget': 250,
    'mask_authority': 'automatic proposal hints only; not semantic labels or acceptance evidence',
    'source_pixel_validation': True, 'physical_geometry_approved': False}


def proposal_exclusions(foreground):
    h, w = foreground.shape; mask = np.zeros_like(foreground)
    _, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
    max_height = min(40, max(18, int(min(h, w)*.035)))
    text_boxes = []
    for x, y, cw, ch, area in stats[1:]:
        x, y, cw, ch, area = map(int, (x, y, cw, ch, area))
        possible_ring = min(cw, ch) >= 12 and .75 <= cw/ch <= 1.33 and area/(cw*ch) < .6
        if (3 <= area <= max_height**2 and 2 <= ch <= max_height and 2 <= cw <= max_height*2
                and .12 <= cw/ch <= 5 and not possible_ring):
            mask[max(0, y-1):min(h, y+ch+1), max(0, x-1):min(w, x+cw+1)] = 255
            text_boxes.append([x, y, x+cw, y+ch])
    horizontal = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((1, max(30, int(w*.6))), np.uint8))
    vertical = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((max(30, int(h*.6)), 1), np.uint8))
    yy, xx = np.indices((h, w))
    border = ((xx < w*.15) | (xx > w*.85) | (yy < h*.15) | (yy > h*.85))
    frame = cv2.bitwise_or(horizontal, vertical); frame[~border] = 0
    frame = cv2.dilate(frame, np.ones((3, 3), np.uint8)); mask = cv2.bitwise_or(mask, frame)
    return mask, {'text_like_component_boxes_analysis_px': text_boxes[:1000],
        'text_like_components': len(text_boxes), 'mask_records_truncated': len(text_boxes) > 1000,
        'masked_pixels': int(np.count_nonzero(mask)), 'frame_hint_pixels': int(np.count_nonzero(frame)),
        'semantics_approved': False, 'acceptance_raster_modified': False}


def regions_for(shape, profile):
    if profile not in POLICY['profiles']: raise ValueError('Unknown geometry search profile')
    h, w = shape; regions = [[0, 0, w, h]]
    if profile == 'regional_masked' and min(h, w) >= 80:
        for top, bottom in ((0, math.ceil(h*.6)), (math.floor(h*.4), h)):
            for left, right in ((0, math.ceil(w*.6)), (math.floor(w*.4), w)):
                regions.append([left, top, right, bottom])
    return regions


def build_circle_queue(analysis, foreground, stroke, profile):
    regions = regions_for(analysis.shape, profile)
    mask, mask_stats = proposal_exclusions(foreground) if profile == 'regional_masked' else (np.zeros_like(foreground), {})
    proposal_image = analysis.copy(); proposal_image[mask != 0] = 255
    calls = []; queues = []
    for index, box in enumerate(regions):
        x, y, right, bottom = box; tile = proposal_image[y:bottom, x:right]
        scale = min(1., POLICY['proposal_max_side']/max(tile.shape))
        if scale < 1:
            tile = cv2.warpAffine(tile, np.array([[scale, 0, (scale-1)/2], [0, scale, (scale-1)/2]]),
                (math.ceil(tile.shape[1]*scale), math.ceil(tile.shape[0]*scale)), flags=cv2.INTER_AREA, borderValue=255)
        circles = None
        if min(tile.shape) >= 30 and cv2.countNonZero(255-tile):
            circles = cv2.HoughCircles(cv2.GaussianBlur(tile, (3, 3), .7), cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=max(10., stroke*scale*3), param1=100, param2=24,
                minRadius=max(6, int(10*scale)), maxRadius=max(10, int(min(tile.shape)*.45)))
        proposals = [] if circles is None else circles.reshape(-1, 3).astype(float)
        queue = []
        for cx, cy, radius in proposals:
            queue.append({'center': [(cx+.5)/scale-.5+x, (cy+.5)/scale-.5+y],
                          'radius': radius/scale, 'region': index})
        queues.append(queue); calls.append({'id': index, 'box_analysis_px': box, 'proposal_scale': scale, 'raw_proposals': len(queue)})
    raw_total = sum(map(len, queues))
    distance = cv2.distanceTransform(255-foreground, cv2.DIST_L2, 3)
    directions = np.column_stack((np.cos(np.linspace(0, 2*np.pi, 72, endpoint=False)), np.sin(np.linspace(0, 2*np.pi, 72, endpoint=False))))
    eligible = []; screened = 0; rejected = {'insufficient_angular_ink': 0, 'duplicate_proposal': 0}; records = []
    # Round robin ensures that a global proposal flood cannot starve all regions.
    for position in range(max(map(len, queues), default=0)):
        for queue in queues:
            if position >= len(queue) or screened >= POLICY['cheap_screen_budget']: continue
            item = queue[position]; screened += 1
            center = np.array(item['center']); probes = np.rint(center+item['radius']*directions).astype(int)
            valid = (probes[:, 0] >= 0)&(probes[:, 0] < analysis.shape[1])&(probes[:, 1] >= 0)&(probes[:, 1] < analysis.shape[0])
            support = np.zeros(72, bool); support[valid] = distance[probes[valid, 1], probes[valid, 0]] <= max(1.25, stroke*.65)
            item['cheap_support'] = float(support.mean())
            if item['cheap_support'] < POLICY['cheap_support_min']:
                rejected['insufficient_angular_ink'] += 1; reason = 'insufficient_angular_ink'
            else: eligible.append(item); reason = 'eligible_for_ranking'
            records.append({**item, 'reason': reason})
    eligible.sort(key=lambda p: (-p['cheap_support'], p['region'], p['center'][0], p['center'][1], p['radius']))
    unique = []
    for item in eligible:
        if any(np.linalg.norm(np.array(item['center'])-old[:2]) <= max(2., stroke)
               and abs(item['radius']-old[2]) <= max(2., stroke) for old in unique):
            rejected['duplicate_proposal'] += 1; continue
        unique.append(np.array([*item['center'], item['radius']]))
    retained = unique[:POLICY['circle_validation_budget']]
    return [x.tolist() for x in retained], {'search_profile': profile, 'policy': POLICY,
        'proposal_regions': calls, 'proposal_exclusions': mask_stats, 'hough_circle_proposals': raw_total,
        'cheap_screen_examined': screened, 'cheap_screen_rejections': rejected,
        'cheap_screen_limit_reached': screened < raw_total, 'eligible_unique_proposals': len(unique),
        'validation_queue_retained': len(retained), 'validation_queue_dropped': max(0, len(unique)-len(retained)),
        'proposal_records': records, 'scope': 'finite proposal search; not a claim of exhaustive geometric recovery'}


def validate_circle_proposal(center, radius, foreground, distance, stroke, curves, fit_circle):
    if not np.isfinite(center).all() or not math.isfinite(radius): return None, 'nonfinite_proposal'
    if radius < 10: return None, 'radius_below_minimum'
    for curve in curves:
        if np.linalg.norm(center-curve['center']) <= max(3., stroke) and abs(radius-curve['radius']) <= max(3., stroke*1.5):
            return None, 'existing_contour_circle' if curve['kind'] == 'circle' else 'conflicts_with_observed_open_arc'
    margin = max(3., stroke*1.5)
    left, top = np.maximum(np.floor(center-radius-margin).astype(int), 0)
    right, bottom = np.minimum(np.ceil(center+radius+margin).astype(int)+1, [foreground.shape[1], foreground.shape[0]])
    yy, xx = np.nonzero(foreground[top:bottom, left:right]); points = np.column_stack((xx+left, yy+top)).astype(float)
    sampled = np.empty((0, 2))
    for _ in range(3):
        near = points[np.abs(np.linalg.norm(points-center, axis=1)-radius) <= margin]
        if len(near) < 60: return None, 'insufficient_nearby_ink'
        angles = np.arctan2(near[:, 1]-center[1], near[:, 0]-center[0]) % (2*np.pi)
        bins = np.minimum(71, (angles*72/(2*np.pi)).astype(int)); occupied = np.unique(bins)
        if len(occupied) < 68: return None, 'insufficient_angular_bins'
        sampled = np.asarray([np.median(near[bins == bucket], axis=0) for bucket in occupied])
        fitted = fit_circle(sampled)
        if fitted is None: return None, 'circle_fit_degenerate'
        center, radius, residual = fitted
    rmse = float(np.sqrt(np.mean(residual**2)))
    if rmse > max(1.1, stroke*.5) or rmse/radius > .035: return None, 'radial_residual'
    angles = np.linspace(0, 2*np.pi, 720, endpoint=False)
    probes = np.rint(center+radius*np.column_stack((np.cos(angles), np.sin(angles)))).astype(int)
    inside = (probes[:, 0] >= 0)&(probes[:, 0] < foreground.shape[1])&(probes[:, 1] >= 0)&(probes[:, 1] < foreground.shape[0])
    support = np.zeros(720, bool); support[inside] = distance[probes[inside, 1], probes[inside, 0]] <= max(.75, stroke*.55)
    gap = run = 0
    for present in np.tile(support, 2):
        run = 0 if present else run+1; gap = max(gap, run)
    if support.mean() < .95: return None, 'full_ring_support'
    if gap > 16: return None, 'missing_arc_above_eight_degrees'
    if any(c['kind'] == 'circle' and np.linalg.norm(center-c['center']) <= max(2., stroke)
           and abs(radius-c['radius']) <= max(2., stroke) for c in curves): return None, 'duplicate_validated_circle'
    return {'kind': 'circle', 'center': center, 'radius': radius, 'start': 0., 'sweep': 2*np.pi,
        'rmse': rmse, 'support': float(support.mean()), 'contour_count': 0, 'point_count': len(sampled),
        'method': 'ranked_hough_angular_centerline_fit', 'maximum_missing_arc_deg': gap/2}, 'accepted_pixel_circle_candidate'


def propose_short_lines(skeleton, foreground, minimum_length, remaining_budget, profile):
    stats = {'regional_short_line_proposals': 0, 'regional_short_lines_retained': 0, 'regional_short_line_limit_reached': False}
    if profile != 'regional_masked' or remaining_budget <= 0: return [], stats
    mask, _ = proposal_exclusions(foreground); evidence = skeleton.copy(); evidence[mask != 0] = 0
    lower = max(10., minimum_length*.55)
    lines = cv2.HoughLinesP(evidence, 1, np.pi/720, threshold=max(8, int(lower*.55)), minLineLength=lower, maxLineGap=1)
    rows = [] if lines is None else [x.astype(float) for x in lines.reshape(-1, 4)]
    rows = [x for x in rows if lower <= np.linalg.norm(x[2:]-x[:2]) < minimum_length]
    rows.sort(key=lambda x: (-float(np.linalg.norm(x[2:]-x[:2])), *x.tolist()))
    limit = min(remaining_budget, POLICY['new_short_line_budget'])
    stats.update(regional_short_line_proposals=len(rows), regional_short_lines_retained=min(len(rows), limit),
                 regional_short_line_limit_reached=len(rows) > limit)
    return rows[:limit], stats
