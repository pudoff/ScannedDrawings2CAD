"""Bounded component crops from pixels and OCR polygons, without text/targets.

The route proposes crops, not recognized glyphs or accepted dimensions. Native
pixels are never closed/dilated; quarter-turns are exact array permutations.
"""
import math
from itertools import combinations

import cv2
import numpy as np

from .family_catalog_v2 import require
from . import component_row_partition_v2 as rows
from . import angular_component_pixels_v1 as original

POLICY = {'id': 'angular-component-pixels/3.0', 'row_partition': rows.POLICY, 'max_window_expansion_rounds': 1, 'border_extension_height_fraction': .75, 'max_image_pixels': 4000000,
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


_overlap = original._overlap
_union = original._union
_native_polygon = original._native_polygon


def propose(rgb, regions):
    """Only ID/polygon keys are admitted; raw strings and expected boxes cannot enter."""
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
            and 0 < rgb.shape[0]*rgb.shape[1] <= POLICY['max_image_pixels'], 'Invalid native RGB image')
    require(isinstance(regions, list) and all(set(r) == {'id', 'polygon_px'} for r in regions), 'Only OCR IDs and polygons may enter segmentation')
    require(all(isinstance(r['id'], str) and r['id'] for r in regions), 'Invalid seed ID')
    require(len({r['id'] for r in regions}) == len(regions), 'Repeated seed region ID')
    height, width = rgb.shape[:2]; costs = {'window_pixels': 0, 'pair_checks': 0, 'connected_components': 0, 'border_components_discarded': 0, 'feature_pixels': 0, 'expanded_windows': 0}
    issues = []; crops = []; windows = []; alternatives = []; finalized = False
    def result():
        nonlocal finalized
        if not finalized:
            finalized = True; unique = {}
            for crop in crops:
                key = (tuple(crop['box_px']), crop['quarter_turn'])
                if key not in unique: unique[key] = crop
                else:
                    other = unique[key]
                    other['source_region_ids'] = sorted(set(other['source_region_ids']+crop['source_region_ids']))
                    other['structure_ambiguous'] |= crop['structure_ambiguous']
            crops[:] = list(unique.values())
            for a, b in combinations(crops, 2):
                if not check(): break
                x, y = a['box_px'], b['box_px']
                intersection = max(0, min(x[2], y[2])-max(x[0], y[0]))*max(0, min(x[3], y[3])-max(x[1], y[1]))
                area = (x[2]-x[0])*(x[3]-x[1])+(y[2]-y[0])*(y[3]-y[1])-intersection
                if intersection/max(1, area) >= .65:
                    a['structure_ambiguous'] = b['structure_ambiguous'] = True
                    alternatives.append({'kind': 'overlapping_candidate_crops', 'crop_ids': [a['id'], b['id']]})
        return {'algorithm_version': 'angular-component-pixels-v3', 'policy': POLICY, 'crops': crops,
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
        count, labels, stats, _ = cv2.connectedComponentsWithStats((gray < POLICY['ink_threshold']).astype(np.uint8), connectivity=8)
        costs['connected_components'] += count-1
        window = dict(seed, id='window-'+str(len(windows)), connected_components=count-1,
                      pixel_work_units=8*pixels)
        windows.append(window)
        if count-1 > POLICY['max_components_per_window']:
            issue('connected_component_limit'); continue
        glyphs = []; border_glyphs = []
        for identifier, (x, y, w, h, area) in enumerate(stats[1:], 1):
            if x == 0 or y == 0 or x+w == gray.shape[1] or y+h == gray.shape[0]:
                costs['border_components_discarded'] += 1
                border_glyphs.append({'box': [int(x), int(y), int(x+w), int(y+h)], 'height': int(h), 'area': int(area)})
                continue
            if area >= POLICY['minimum_component_area']:
                glyphs.append({'id': identifier, 'box': [int(x), int(y), int(x+w), int(y+h)], 'height': int(h), 'area': int(area)})
        if not glyphs: continue
        maximum_height = max(g['height'] for g in glyphs)
        clipped = [g for g in border_glyphs if g['area'] >= 3 and
                   max(g['box'][2]-g['box'][0], g['height']) <= 1.25*maximum_height]
        window['potential_clipped_glyphs'] = clipped
        if clipped and not seed.get('expansion_round'):
            extension = math.ceil(POLICY['border_extension_height_fraction']*maximum_height)
            expanded = [x0, y0, x1, y1]
            for g in clipped:
                a, b, c, d = g['box']
                # Edge coordinates are in the canonical quarter-turned window.
                padbox = [a-extension if a == 0 else a, b-extension if b == 0 else b,
                          c+extension if c == gray.shape[1] else c, d+extension if d == gray.shape[0] else d]
                native = np.asarray(_native_polygon(padbox, seed['box_px'], seed['quarter_turn']))
                expanded = _union([expanded, [*native.min(axis=0).tolist(), *native.max(axis=0).tolist()]])
            expanded = [max(0, int(expanded[0])), max(0, int(expanded[1])), min(width, int(expanded[2])), min(height, int(expanded[3]))]
            if expanded != seed['box_px']:
                if POLICY['max_window_expansion_rounds'] == 0:
                    issue('window_expansion_limit')
                else:
                    seeds.append(dict(seed, box_px=expanded, expansion_round=1, expanded_from_window=window['id']))
                    costs['expanded_windows'] += 1; window['replaced_by_expansion'] = True
                    continue
        if clipped: issue('window_has_unresolved_clipped_ink')
        feature_start = costs['feature_pixels']
        def charge(area):
            costs['feature_pixels'] += area
            if (costs['feature_pixels'] > rows.POLICY['max_feature_pixels'] or
                    costs['feature_pixels']-feature_start > rows.POLICY['max_features_per_window']):
                raise rows.PartitionLimit('glyph_feature_pixel_limit')
        try:
            clusters, proposals = rows.partition(glyphs, gray < POLICY['ink_threshold'], check, charge, labels)
        except rows.PartitionLimit as error:
            issue(str(error)); return result()
        window['feature_pixels'] = costs['feature_pixels']-feature_start
        window['pixel_work_units'] += 4*window['feature_pixels']
        alternatives += [dict(p, window_id=window['id']) for p in proposals]
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
