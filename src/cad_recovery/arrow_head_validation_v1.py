"""Reject geometric wedges; abstain on nonunique attached-head vertices."""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from . import arrow_head_completion_v1 as previous
from .arrow_detection import _profile
from .family_catalog_v2 import require

POLICY = {'version': 'native-head-validation/1.0', 'minimum_base_to_shaft_ratio': 3,
    'profile_head_fraction': [.2, .7], 'maximum_median_relative_width_error': .25,
    'maximum_shape_profile_pixels': 2000000, 'maximum_junction_pairs': 4096,
    'maximum_nearby_lines': 16, 'maximum_attachment_seeds': 64,
    'maximum_attachment_profile_pixels': 2000000,
    'maximum_attachment_native_checks': 4096,
    'junction_maximum_axis_dot': .866, 'minimum_junction_line_length_px': 20,
    'maximum_junction_distance_in_head_lengths': .6,
    'maximum_junction_extension_in_head_lengths': .5,
    'junction_width_fractions': [1, .75, .6, .45],
    'minimum_unoccluded_flank_probes': 12, 'minimum_unoccluded_probes_per_side': 4,
    'occluder_maximum_axis_dot': .97, 'minimum_occluder_native_support': .8,
    'unique_vertex_cluster_radius_in_strokes': .5,
    'manual_references_used': False, 'production_admitted': False,
    'physical_geometry_approved': False, 'confidence_calibrated': False,
    'preserve_baseline': True, 'uncertain_vertices_are_not_arrow_candidates': True}


def profile_cost(candidate):
    return 81 * (math.ceil(candidate['evidence']['head_length_px']) + 97)


def observed_line_support(mask, point, direction, radius):
    """A fitted or locally extended segment needs native ink at the junction."""
    offsets = np.linspace(0, max(12, radius*3), 16)
    fractions = []
    for sign in (-1, 1):
        p = np.rint(point+sign*offsets[:, None]*direction).astype(int)
        inside = (p >= 1).all(1) & (p[:, 0] < mask.shape[1]-1) & (p[:, 1] < mask.shape[0]-1)
        fractions.append(float(np.mean([bool(mask[y-1:y+2, x-1:x+2].any()) if valid else False
            for (x, y), valid in zip(p, inside)])))
    return max(fractions)


def attached_head_check(candidate, mask, stroke, text_polygons, lines):
    if not previous.tip_has_native_ink(candidate, mask, stroke):
        return False, {'reason': 'apex_without_native_ink'}
    valid, check = previous.previous.native_head_check(candidate, mask, stroke, text_polygons)
    if valid or check['reason'] != 'native_flanks_not_background': return valid, check
    p = np.asarray(candidate['polygon_px']); tip = p[0]; base = p[1:].mean(0)
    unit = np.asarray(candidate['direction']); normal = np.array([-unit[1], unit[0]])
    half = float(np.linalg.norm(p[1]-p[2])/2); t = np.linspace(.2, .8, 16)
    centers = tip+t[:, None]*(base-tip); radius = half*t+max(2, stroke)
    probes = np.rint(np.r_[centers+radius[:, None]*normal, centers-radius[:, None]*normal]).astype(int)
    occluded = np.zeros(32, bool); ids = []
    probe_ink = mask[probes[:, 1], probes[:, 0]] > 0
    for identity, points, direction, extent in lines:
        if abs(direction@unit) >= POLICY['occluder_maximum_axis_dot']: continue
        position = (probes-points[0])@direction
        across = np.abs((probes-points[0])@np.array([-direction[1], direction[0]]))
        near = (position >= 0) & (position <= extent) & (across <= max(2, stroke*.75))
        if not near.any(): continue
        sample = np.rint(points[0]+np.linspace(0, extent, 32)[:, None]*direction).astype(int)
        inside = (sample >= 1).all(1) & (sample[:, 0] < mask.shape[1]-1) & (sample[:, 1] < mask.shape[0]-1)
        support = np.mean([bool(mask[y-1:y+2, x-1:x+2].any()) if good else False
                          for (x, y), good in zip(sample, inside)])
        if support < POLICY['minimum_occluder_native_support']: continue
        occluded |= near & probe_ink; ids.append(identity)
    visible = ~occluded
    if (int(visible.sum()) < POLICY['minimum_unoccluded_flank_probes'] or
        min(int(visible[:16].sum()), int(visible[16:].sum())) < POLICY['minimum_unoccluded_probes_per_side']):
        return False, {**check, 'reason': 'insufficient_unoccluded_native_flanks', 'visible_probes': int(visible.sum())}
    flank = float(np.mean(mask[probes[visible, 1], probes[visible, 0]] > 0))
    if flank > previous.previous.POLICY['native_flank_ink_max']: return False, check
    # The old checker returns before its OCR gate on flank failure.
    for polygon in text_polygons:
        contour = np.asarray(polygon, np.float32)
        if cv2.pointPolygonTest(contour, tuple(tip), False) >= 0 or cv2.pointPolygonTest(contour, tuple(p.mean(0)), False) >= 0:
            return False, {'reason': 'native_head_inside_saved_text'}
    return True, {**check, 'reason': 'native_filled_triangle_with_observed_crossing_strokes',
        'unoccluded_flank_ink': flank, 'visible_probes': int(visible.sum()),
        'visible_probes_per_side': [int(visible[:16].sum()), int(visible[16:].sum())],
        'occluding_line_ids': ids, 'raw_flank_check_passed': False}


def shape_check(candidate, mask):
    p = np.asarray(candidate['polygon_px'], float)
    base = p[1:].mean(axis=0); length = float(np.linalg.norm(p[0] - base))
    width = float(np.linalg.norm(p[1] - p[2]))
    along, widths, _ = _profile(mask, base, np.asarray(candidate['direction']), length)
    low, high = POLICY['profile_head_fraction']; chosen = (along >= length*low) & (along <= length*high)
    expected = width * (1 - along[chosen]/length)
    error = float(np.median(np.abs(widths[chosen] - expected) / np.maximum(2., expected)))
    return error <= POLICY['maximum_median_relative_width_error'], {
        'reason': 'native_triangle_width_agreement' if error <= POLICY['maximum_median_relative_width_error']
                  else 'native_width_disagrees_with_filled_triangle',
        'median_relative_width_error': error, 'stations': int(chosen.sum())}


def vertex_clusters(hypotheses, stroke):
    clusters = []
    for c in hypotheses:
        for cluster in clusters:
            if all(np.linalg.norm(np.asarray(c['tip_px'])-v['tip_px']) <= max(2, stroke*.5) for v in cluster):
                cluster.append(c); break
        else: clusters.append([c])
    return clusters


def validate_new(candidates, gray):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kept, rejected = [], []; pixels = 0; limits = []
    for original in candidates:
        c = deepcopy(original)
        if c['evidence']['native_shaft']['base_to_shaft_ratio'] < POLICY['minimum_base_to_shaft_ratio']:
            rejected.append({'candidate': c, 'reason': 'insufficient_head_to_native_shaft_contrast'}); continue
        cost = profile_cost(c)
        if pixels + cost > POLICY['maximum_shape_profile_pixels']:
            limits.append('shape_profile_budget'); break
        pixels += cost; valid, check = shape_check(c, mask)
        if valid: c['evidence']['native_shape_check'] = check; kept.append(c)
        else: rejected.append({'candidate': c, **check})
    return {'candidates': kept, 'rejected': rejected, 'shape_profile_pixels': pixels,
            'limits': limits, 'search_incomplete': bool(limits)}


def attachment_search(gray, *, lines, stroke_px, text_polygons=()):
    """Bounded hypotheses from source-observed intersections, without GT vertices.

    A rejected triangle touching geometry is not evidence of an apex. Every
    intersection must independently pass fill/flank/shaft/shape checks, and
    competing vertices remain explicit abstentions rather than an average.
    """
    require(isinstance(gray, np.ndarray) and gray.ndim == 2 and gray.dtype == np.uint8
            and 0 < gray.size <= previous.POLICY['maximum_pixels'], 'Invalid native junction raster')
    require(np.isfinite(stroke_px) and 1 <= stroke_px <= 24, 'Invalid native junction stroke')
    for polygon in text_polygons:
        p = np.asarray(polygon, float)
        require(p.ndim == 2 and p.shape[0] >= 3 and p.shape[1] == 2 and np.isfinite(p).all()
                and np.abs(p).max() <= 1e7, 'Invalid junction text polygon')
    require(len(lines) <= 2500, 'Too many junction support lines')
    parsed = []
    for line in lines:
        if line.get('kind') != 'line': continue
        p = np.asarray(line['points_px'], float)
        require(p.shape == (2, 2) and np.isfinite(p).all() and np.abs(p).max() <= 1e7,
                'Invalid junction support line')
        d = p[1] - p[0]; length = float(np.linalg.norm(d))
        require(length > 0, 'Zero junction support line')
        if length >= max(POLICY['minimum_junction_line_length_px'], 3*stroke_px):
            parsed.append((line['id'], p, d/length, length))
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seeds, limits = [], set()
    for co in sorted(contours, key=lambda c: -cv2.contourArea(c)):
        _, _, w, h = cv2.boundingRect(co)
        if cv2.contourArea(co) < 8 or max(w, h) > 192: continue
        seed = previous.triangle_proposal(co, 'attachment-seed-' + str(len(seeds)+1))
        if seed is None or seed['evidence']['base_width_px'] < max(8, 3*stroke_px): continue
        valid, check = previous.previous.native_head_check(seed, mask, stroke_px, text_polygons)
        if valid or check['reason'] != 'native_triangle_not_filled' or check.get('triangle_ink', 0) < .6: continue
        if len(seeds) >= POLICY['maximum_attachment_seeds']: limits.add('attachment_seed_budget'); break
        seeds.append(seed)
    observations, accepted = [], []; examined = pixels = native_checks = 0
    for seed in seeds:
        tip = np.asarray(seed['tip_px']); p = np.asarray(seed['polygon_px']); base = p[1:].mean(0)
        width = seed['evidence']['base_width_px']; length = seed['evidence']['head_length_px']; near = []
        for line in parsed:
            _, points, u, extent = line
            position = np.clip((tip-points[0])@u, 0, extent)
            if np.linalg.norm(points[0]+position*u-tip) <= length*.6: near.append(line)
        if len(near) > POLICY['maximum_nearby_lines']: limits.add('attachment_line_budget')
        near = near[:POLICY['maximum_nearby_lines']]; hypotheses = []; rejections = Counter(); attempted = []
        for i, (one, a, u, length_a) in enumerate(near):
            for two, b, v, length_b in near[i+1:]:
                if examined >= POLICY['maximum_junction_pairs']: limits.add('attachment_pair_budget'); break
                examined += 1
                if abs(u@v) > POLICY['junction_maximum_axis_dot']: continue
                t, s = np.linalg.solve(np.column_stack((u, -v)), b[0]-a[0])
                extension = length*POLICY['maximum_junction_extension_in_head_lengths']
                if not (-extension <= t <= length_a+extension and -extension <= s <= length_b+extension): continue
                q = a[0]+t*u
                if np.linalg.norm(q-tip) > length*POLICY['maximum_junction_distance_in_head_lengths']: continue
                support = [observed_line_support(mask, q, axis, stroke_px) for axis in (u, v)]
                if min(support) < .8: rejections['junction_without_native_line_support'] += 1; continue
                delta = q-base; head_length = float(np.linalg.norm(delta))
                if not 6 <= head_length <= 192: continue
                direction = delta/head_length; normal = np.array([-direction[1], direction[0]])
                for fraction in POLICY['junction_width_fractions']:
                    w = width*fraction
                    if not 1.8 <= head_length/w <= 8: continue
                    c = deepcopy(seed); c.update(tip_px=q.tolist(), direction=direction.tolist(),
                        polygon_px=[q.tolist(), (base+normal*w/2).tolist(), (base-normal*w/2).tolist()])
                    c['evidence'].update(head_length_px=head_length, base_width_px=w, observed_junction_lines=[one, two],
                                         junction_native_line_support=support)
                    if native_checks >= POLICY['maximum_attachment_native_checks']:
                        limits.add('attachment_native_check_budget'); break
                    native_checks += 1
                    valid, check = attached_head_check(c, mask, stroke_px, text_polygons, near)
                    if not valid: rejections[check['reason']] += 1; continue
                    cost = 81*(math.ceil(max(12, min(40, head_length*.6)))+97) + profile_cost(c)
                    if pixels+cost > POLICY['maximum_attachment_profile_pixels']:
                        limits.add('attachment_profile_budget'); break
                    pixels += cost; valid, shaft = previous.shaft_evidence(c, mask, stroke_px)
                    if not valid: rejections[shaft['reason']] += 1; continue
                    c['evidence']['native_shaft'] = shaft
                    valid, shape = shape_check(c, mask)
                    if shaft['base_to_shaft_ratio'] < POLICY['minimum_base_to_shaft_ratio']:
                        valid = False; shape['reason'] = 'insufficient_head_to_native_shaft_contrast'
                    attempted.append({'tip_px': q.tolist(), 'support_ids': [one, two], 'shape': shape,
                                      'accepted': valid})
                    if not valid: rejections[shape['reason']] += 1; continue
                    c['evidence'].update(method=POLICY['version'], path_kind='source_validated_attached_head',
                        native_head_check=check, native_shape_check=shape, native_tip_ink=True)
                    hypotheses.append(c); break
            if limits & {'attachment_pair_budget', 'attachment_profile_budget', 'attachment_native_check_budget'}: break
        clusters = vertex_clusters(hypotheses, stroke_px)
        resolved = len(clusters) == 1 and not limits
        if resolved: accepted.append(clusters[0][0])
        observations.append({'seed': seed, 'state': 'unique_validated_junction' if resolved else 'abstained',
            'valid_vertex_clusters': len(clusters), 'attempted_native_vertices': attempted, 'rejections': dict(rejections),
            'reason': 'one_source_validated_vertex' if resolved else 'no_unique_source_validated_vertex',
            'search_incomplete': bool(limits)})
        if limits & {'attachment_pair_budget', 'attachment_profile_budget', 'attachment_native_check_budget'}: break
    return {'candidates': previous.previous.combine(accepted), 'observations': observations,
        'analysis': {'attachment_seeds': len(seeds), 'junction_pairs': examined, 'profile_pixels': pixels,
                     'native_checks': native_checks, 'morphology_pixels': int(gray.size),
                     'limits': sorted(limits)}, 'search_incomplete': bool(limits)}


def search(gray, *, baseline, lines=(), stroke_px=3, text_polygons=()):
    # Frozen producer validates native image/stroke/OCR before any new search.
    raw = previous.proposals(gray, stroke_px=stroke_px, text_polygons=text_polygons)
    validated = validate_new(raw['candidates'], gray)
    attached = attachment_search(gray, lines=lines, stroke_px=stroke_px, text_polygons=text_polygons)
    return {'schema_version': 'arrow-head-validation/1.0', 'policy': POLICY,
        'candidates': previous.append_preserving_baseline(baseline, validated['candidates']+attached['candidates']),
        'raw_auxiliary': raw, 'validated_auxiliary': validated, 'attached': attached,
        'search_incomplete': raw['search_incomplete'] or validated['search_incomplete'] or attached['search_incomplete'],
        'production_admitted': False, 'physical_geometry_approved': False, 'CAD_promotions': 0}
