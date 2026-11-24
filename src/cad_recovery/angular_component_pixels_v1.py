"""Bounded component crops from pixels and OCR polygons, without text/targets.

The route proposes crops, not recognized glyphs or accepted dimensions. Native
pixels are never closed/dilated; quarter-turns are exact array permutations.
"""
import math
from itertools import combinations

import cv2
import numpy as np

from .family_catalog_v2 import require

POLICY = {'id': 'angular-component-pixels/1.0', 'max_image_pixels': 4000000,
          'max_regions': 64, 'max_windows': 16, 'max_window_pixels': 262144,
          'max_page_window_pixels': 1048576, 'max_components_per_window': 256,
          'max_pair_checks': 8192, 'max_crops': 32, 'max_crop_pixels': 100000,
          'window_padding_height': .2, 'axis_snap_tolerance_deg': 15,
          'ink_threshold': 200, 'minimum_component_area': 3, 'body_height_fraction': .45,
          'body_height_ratio_max': 1.8, 'body_center_offset_fraction': .3,
          'body_overlap_fraction': .45, 'body_gap_height_fraction': .65,
          'satellite_height_fraction': .5, 'satellite_gap_height_fraction': .6,
          'crop_padding_px': 2, 'morphological_closing': False,
          'physical_geometry_approved': False, 'production_admitted': False, 'training_eligible': False}


def _overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _native_polygon(box, window, quarter):
    a, b, c, d = box; x0, y0, x1, y1 = window; width, height = x1-x0, y1-y0
    points = np.array([[a, b], [c, b], [c, d], [a, d]], float)
    if quarter == 1: points = np.column_stack((width-points[:, 1], points[:, 0]))
    elif quarter == 2: points = np.column_stack((width-points[:, 0], height-points[:, 1]))
    elif quarter == 3: points = np.column_stack((points[:, 1], height-points[:, 0]))
    return (points+[x0, y0]).tolist()


def propose(rgb, regions):
    """Only ID/polygon keys are admitted; raw strings and expected boxes cannot enter."""
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
            and 0 < rgb.shape[0]*rgb.shape[1] <= POLICY['max_image_pixels'], 'Invalid native RGB image')
    require(isinstance(regions, list) and all(set(r) == {'id', 'polygon_px'} for r in regions), 'Only OCR IDs and polygons may enter segmentation')
    require(all(isinstance(r['id'], str) and r['id'] for r in regions), 'Invalid seed ID')
    require(len({r['id'] for r in regions}) == len(regions), 'Repeated seed region ID')
    height, width = rgb.shape[:2]; costs = {'window_pixels': 0, 'pair_checks': 0, 'connected_components': 0, 'border_components_discarded': 0}
    issues = []; crops = []; windows = []; alternatives = []
    def result():
        return {'algorithm_version': 'angular-component-pixels-v1', 'policy': POLICY, 'crops': crops,
                'windows': windows, 'alternatives': alternatives, 'costs': costs, 'search_incomplete': bool(issues),
                'issues': issues, 'scope': 'Supplied OCR-seeded windows only; an empty seed set is not full-page recall.',
                'regions_received': len(regions), 'raw_strings_received': False}
    def issue(reason):
        if reason not in issues: issues.append(reason)
    def check():
        costs['pair_checks'] += 1
        if costs['pair_checks'] > POLICY['max_pair_checks']:
            issue('pair_check_limit'); return False
        return True
    if len(regions) > POLICY['max_regions']:
        issue('region_limit'); return result()
    seeds = []
    for row in regions:
        p = np.asarray(row['polygon_px'], float)
        require(p.shape == (4, 2) and np.isfinite(p).all() and (p >= 0).all() and (p <= [width, height]).all()
                and cv2.contourArea(p.astype(np.float32)) > 0 and cv2.isContourConvex(p.astype(np.float32)), 'Invalid OCR seed polygon')
        delta = p[1]-p[0]; require(np.linalg.norm(delta) > 0, 'Invalid seed axis')
        angle = math.degrees(math.atan2(delta[1], delta[0])); quarter = round(angle/90) % 4
        error = abs((angle-quarter*90+180) % 360-180)
        if error > POLICY['axis_snap_tolerance_deg']:
            issue('unsupported_seed_orientation'); continue
        padding = POLICY['window_padding_height']*max(1, np.linalg.norm(p[3]-p[0]))
        low = np.maximum([0, 0], np.floor(p.min(axis=0)-padding)).astype(int)
        high = np.minimum([width, height], np.ceil(p.max(axis=0)+padding)).astype(int)
        seeds.append({'box_px': [*low.tolist(), *high.tolist()], 'quarter_turn': quarter, 'source_region_ids': [row['id']]})
    # Union overlapping windows so repeated OCR cannot produce independent votes.
    changed = True
    while changed:
        changed = False
        for i, j in combinations(range(len(seeds)), 2):
            if not check(): return result()
            a, b = seeds[i], seeds[j]
            if _overlap(a['box_px'], b['box_px']):
                if a['quarter_turn'] != b['quarter_turn']:
                    issue('overlapping_seed_orientation_conflict'); return result()
                seeds[i] = dict(a, box_px=_union([a['box_px'], b['box_px']]),
                                source_region_ids=sorted(a['source_region_ids']+b['source_region_ids']))
                seeds.pop(j); changed = True; break
    for seed in seeds:
        if len(windows) >= POLICY['max_windows']:
            issue('window_limit'); break
        x0, y0, x1, y1 = seed['box_px']; pixels = (x1-x0)*(y1-y0)
        if pixels > POLICY['max_window_pixels'] or costs['window_pixels']+pixels > POLICY['max_page_window_pixels']:
            issue('window_pixel_limit'); continue
        costs['window_pixels'] += pixels
        gray = cv2.cvtColor(np.ascontiguousarray(np.rot90(rgb[y0:y1, x0:x1], seed['quarter_turn'])), cv2.COLOR_RGB2GRAY)
        count, _, stats, _ = cv2.connectedComponentsWithStats((gray < POLICY['ink_threshold']).astype(np.uint8), connectivity=8)
        costs['connected_components'] += count-1
        window = dict(seed, id='window-'+str(len(windows)), connected_components=count-1,
                      pixel_work_units=8*pixels)
        windows.append(window)
        if count-1 > POLICY['max_components_per_window']:
            issue('connected_component_limit'); continue
        glyphs = []
        for identifier, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if x == 0 or y == 0 or x+w == gray.shape[1] or y+h == gray.shape[0]:
                costs['border_components_discarded'] += 1; continue
            if area >= POLICY['minimum_component_area']:
                glyphs.append({'id': identifier, 'box': [int(x), int(y), int(x+w), int(y+h)], 'height': int(h), 'area': int(area)})
        if not glyphs: continue
        maximum = max(g['height'] for g in glyphs)
        bodies = [g for g in glyphs if g['height'] >= POLICY['body_height_fraction']*maximum]
        small = [g for g in glyphs if g not in bodies]
        parents = list(range(len(bodies)))
        def root(i):
            while parents[i] != i: i = parents[i]
            return i
        for i, j in combinations(range(len(bodies)), 2):
            if not check(): return result()
            a, b = bodies[i], bodies[j]; ba, bb = a['box'], b['box']; scale = max(a['height'], b['height'])
            overlap = max(0, min(ba[3], bb[3])-max(ba[1], bb[1]))
            gap = max(ba[0], bb[0])-min(ba[2], bb[2])
            center_offset = abs(ba[1]+ba[3]-bb[1]-bb[3])/2
            if (scale/min(a['height'], b['height']) <= POLICY['body_height_ratio_max'] and
                    overlap >= POLICY['body_overlap_fraction']*min(a['height'], b['height']) and
                    center_offset <= POLICY['body_center_offset_fraction']*scale and
                    gap <= POLICY['body_gap_height_fraction']*scale):
                parents[root(j)] = root(i)
        clusters = []
        for r in sorted({root(i) for i in range(len(bodies))}):
            members = [bodies[i] for i in range(len(bodies)) if root(i) == r]
            clusters.append({'body_ids': [g['id'] for g in members], 'box': _union([g['box'] for g in members]),
                             'satellite_ids': [], 'ambiguous_satellite_ids': []})
        for satellite in small:
            eligible = []
            a, b, c, d = satellite['box']; h = d-b
            for i, cluster in enumerate(clusters):
                if not check(): return result()
                left, top, right, bottom = cluster['box']; scale = bottom-top
                degree_like = (h <= POLICY['satellite_height_fraction']*scale and .4 <= (c-a)/h <= 2.5
                    and -.1*scale <= a-right <= POLICY['satellite_gap_height_fraction']*scale
                    and top-.4*scale <= (b+d)/2 < (top+bottom)/2)
                left_modifier = (h <= .25*scale and (c-a)/h >= 3 and 0 <= left-c <= .35*scale
                                 and abs((b+d-top-bottom)/2) <= .25*scale)
                if degree_like or left_modifier: eligible.append(i)
            alternatives.append({'window_id': window['id'], 'satellite_id': satellite['id'], 'candidate_body_clusters': eligible})
            for i in eligible:
                clusters[i]['satellite_ids' if len(eligible) == 1 else 'ambiguous_satellite_ids'].append(satellite['id'])
        by = {g['id']: g for g in glyphs}
        for cluster in clusters:
            ids = cluster['body_ids']+cluster['satellite_ids']; box = _union([by[i]['box'] for i in ids])
            padding = POLICY['crop_padding_px']; a, b, c, d = box
            box = [max(0, a-padding), max(0, b-padding), min(gray.shape[1], c+padding), min(gray.shape[0], d+padding)]
            if (box[2]-box[0])*(box[3]-box[1]) > POLICY['max_crop_pixels']:
                issue('crop_pixel_limit'); continue
            if len(crops) >= POLICY['max_crops']:
                issue('crop_count_limit'); return result()
            polygon = _native_polygon(box, seed['box_px'], seed['quarter_turn'])
            p = np.asarray(polygon); native_box = [*p.min(axis=0).astype(int).tolist(), *p.max(axis=0).astype(int).tolist()]
            crops.append({'id': 'crop-'+str(len(crops)), 'window_id': window['id'], 'box_px': native_box,
                'polygon_px': polygon, 'quarter_turn': seed['quarter_turn'], 'body_component_ids': cluster['body_ids'],
                'satellite_component_ids': cluster['satellite_ids'], 'structure_ambiguous': bool(cluster['ambiguous_satellite_ids']),
                'ambiguous_satellite_ids': cluster['ambiguous_satellite_ids'], 'source_region_ids': seed['source_region_ids'],
                'native_pixels_unscaled': True, 'accepted': False})
    return result()
