"""Whole-row partitions with loop protection and conservative short-bar retention."""
import math
import cv2
import numpy as np

POLICY = {'id': 'component-row-partition/2.0', 'baseline_delta_fraction': .3,
          'height_ratio_max': 1.8, 'center_delta_fraction': .3, 'minimum_vertical_overlap': .45,
          'horizontal_gap_fraction': .65, 'cross_peak_min': .7, 'cross_center_min': .3,
          'cross_center_max': .7, 'cross_band_fraction': .16, 'cross_ink_fraction': .8,
          'modifier_stroke_ratio_min': .65, 'modifier_stroke_ratio_max': 1.6,
          'max_feature_pixels': 1048576, 'max_features_per_window': 262144,
          'minimum_ink_area': 3, 'horizontal_bar_aspect_min': 3.,
          'left_bar_height_fraction': .25, 'left_bar_gap_fraction': .35,
          'left_bar_width_fraction_max': .8,
          'left_bar_center_fraction': .25, 'loop_topology_connectivity': 4, 'physical_geometry_approved': False, 'production_admitted': False}


class PartitionLimit(Exception): pass


def union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def feature(glyph, binary, charge, labels=None):
    a, b, c, d = glyph['box']; charge((c-a)*(d-b))
    ink = binary[b:d, a:c] if labels is None else labels[b:d, a:c] == glyph['id']
    h, w = ink.shape
    rows = ink.sum(axis=1); cols = ink.sum(axis=0)
    ry, cx = int(np.argmax(rows)), int(np.argmax(cols))
    yy, xx = np.mgrid[:h, :w]; band = max(1, math.ceil(min(w, h)*POLICY['cross_band_fraction']))
    on_cross = (np.abs(yy-ry) <= band) | (np.abs(xx-cx) <= band)
    crossed = (.5 <= w/h <= 1.8 and rows[ry]/w >= POLICY['cross_peak_min'] and cols[cx]/h >= POLICY['cross_peak_min']
        and POLICY['cross_center_min'] <= ry/max(1, h-1) <= POLICY['cross_center_max']
        and POLICY['cross_center_min'] <= cx/max(1, w-1) <= POLICY['cross_center_max']
        and np.count_nonzero(ink & on_cross)/max(1, np.count_nonzero(ink)) >= POLICY['cross_ink_fraction'])
    padded = np.pad(ink.astype(np.uint8), 1)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    stroke = 2*float(np.quantile(distance[ink], .9))
    # Count enclosed background regions in this label only, including one-pixel holes.
    charge(w*h)
    background = np.pad(~ink, 1, constant_values=True).astype(np.uint8)
    loops = max(0, cv2.connectedComponents(background, connectivity=POLICY['loop_topology_connectivity'])[0]-2)
    horizontal_bar = w/h >= POLICY['horizontal_bar_aspect_min'] and loops == 0
    return dict(glyph, compact_modifier=bool((crossed and loops == 0) or horizontal_bar),
                horizontal_bar=bool(horizontal_bar), loop_count=int(loops), stroke_width=stroke)


def vertical(a, b):
    ba, bb = a['box'], b['box']; scale = max(a['height'], b['height'])
    overlap = max(0, min(ba[3], bb[3])-max(ba[1], bb[1]))
    return (scale/min(a['height'], b['height']) <= POLICY['height_ratio_max']
        and overlap >= POLICY['minimum_vertical_overlap']*min(a['height'], b['height'])
        and abs(ba[1]+ba[3]-bb[1]-bb[3])/2 <= POLICY['center_delta_fraction']*scale
        and abs(ba[3]-bb[3]) <= POLICY['baseline_delta_fraction']*scale)


def partition(glyphs, binary, check, charge, labels=None):
    """Complete vertical agreement prevents a small bridge from joining two rows."""
    if not glyphs: return [], []
    features = [feature(g, binary, charge, labels) for g in glyphs]
    maximum = max(g['height'] for g in features)
    bodies = [g for g in features if not g['compact_modifier'] and g['height'] >= .45*maximum]
    satellites = [g for g in features if g not in bodies]
    clusters = []; alternatives = []
    for body in sorted(bodies, key=lambda g: (-g['height'], g['box'][0], g['id'])):
        eligible = []
        for i, cluster in enumerate(clusters):
            if not check(): raise PartitionLimit('pair_check_limit')
            box = cluster['box']; gap = max(box[0], body['box'][0])-min(box[2], body['box'][2])
            if gap <= POLICY['horizontal_gap_fraction']*max(body['height'], box[3]-box[1]) and all(vertical(body, g) for g in cluster['members']):
                eligible.append(i)
        if len(eligible) == 1:
            c = clusters[eligible[0]]; c['members'].append(body); c['body_ids'].append(body['id']); c['box'] = union([c['box'], body['box']])
        else:
            if eligible:
                alternatives.append({'kind': 'competing_row_memberships', 'component_id': body['id'], 'candidate_clusters': eligible})
                for i in eligible: clusters[i]['ambiguous_satellite_ids'].append(body['id'])
            clusters.append({'members': [body], 'body_ids': [body['id']], 'box': body['box'], 'satellite_ids': [],
                             'ambiguous_satellite_ids': [body['id']] if eligible else []})
    for satellite in satellites:
        eligible = []; rejected = []; a, b, c, d = satellite['box']; h, w = d-b, c-a
        for i, cluster in enumerate(clusters):
            if not check(): raise PartitionLimit('pair_check_limit')
            if not cluster['members']: continue
            left, top, right, bottom = cluster['box']; scale = bottom-top
            degree = (not satellite['compact_modifier'] and h <= .5*scale and .4 <= w/h <= 2.5
                      and -.1*scale <= a-right <= .6*scale and top-.4*scale <= (b+d)/2 < (top+bottom)/2)
            left_bar = (satellite['horizontal_bar'] and h <= POLICY['left_bar_height_fraction']*scale
                        and w <= POLICY['left_bar_width_fraction_max']*scale
                        and 0 <= left-c <= POLICY['left_bar_gap_fraction']*scale
                        and abs((b+d-top-bottom)/2) <= POLICY['left_bar_center_fraction']*scale)
            modifier = (satellite['compact_modifier'] and not satellite['horizontal_bar'] and 0 <= left-c <= .65*scale
                        and abs((b+d-top-bottom)/2) <= .3*scale and h <= 1.1*scale)
            decimal = (not satellite['compact_modifier'] and len(cluster['members']) >= 2 and h <= .25*scale
                       and left <= a <= c <= right and bottom-.25*scale <= (b+d)/2 <= bottom+.25*scale)
            if modifier:
                stroke = float(np.median([g['stroke_width'] for g in cluster['members']]))
                ratio = satellite['stroke_width']/max(.1, stroke)
                if not POLICY['modifier_stroke_ratio_min'] <= ratio <= POLICY['modifier_stroke_ratio_max']:
                    rejected.append({'cluster': i, 'reason': 'stroke_width_incompatible', 'stroke_ratio': ratio}); modifier = False
            if degree or left_bar or modifier or decimal: eligible.append(i)
        alternatives.append({'kind': 'satellite_attachment', 'component_id': satellite['id'], 'candidate_clusters': eligible, 'rejected': rejected,
                             'box_px': list(satellite['box']), 'loop_count': satellite['loop_count'],
                             'horizontal_bar': satellite['horizontal_bar'],
                             'meaning': 'shape/position hypothesis only; no glyph identity or text correction'})
        for i in eligible:
            clusters[i]['satellite_ids' if len(eligible) == 1 else 'ambiguous_satellite_ids'].append(satellite['id'])
        if not eligible and satellite['compact_modifier']:
            clusters.append({'members': [], 'body_ids': [satellite['id']], 'box': satellite['box'], 'satellite_ids': [],
                             'ambiguous_satellite_ids': [satellite['id']] if rejected else []})
    return clusters, alternatives
