"""A dimension overlay abstains at the source-supported boundary contact.

This guard can only remove additions made by the sealed overlay. It neither
labels the ambiguous contact as part geometry nor changes existing roles.
"""
from collections import Counter, defaultdict

import cv2
import numpy as np

from .supported_role_overlay_v1 import analyze as previous_analyze
from .stroke_roles_v1 import foreground

POLICY = {'version': 'contact-role-overlay/2.0',
          'maximum_competitor_pairs': 2000000,
          'maximum_combined_source_points': 2000000,
          'maximum_combined_raster_pixels': 40000000,
          'boundary_nearby_width_lengths': 4.,
          'pixel_footprint_radius_px': 2**.5/2,
          'ambiguous_context': 'abstain_from_new_extension_pixels',
          'baseline_known_roles_immutable': True, 'source_erasure_allowed': False,
          'production_admitted': False, 'physical_geometry_approved': False}


def context_for(previous, intervals, chain, foot, distance, audit):
    boundaries = previous.get('boundaries', {})
    boundaries = boundaries.get('boundaries', []) if isinstance(boundaries, dict) else boundaries
    boundary = next((b for b in boundaries if b['id'] == chain['boundary_id']), None)
    track = next((t for t in previous.get('tracks', []) if t['id'] == chain['shaft_track']), None)
    if boundary is None or boundary.get('frame_hypothesis') or track is None:
        return None, 'missing_boundary_or_shaft_context'
    points = np.asarray(track['points'], float)
    foot = np.asarray(foot, float)
    if points.shape != (2, 2) or foot.shape != (2,) or not np.isfinite(np.r_[points.ravel(), foot]).all():
        return None, 'invalid_joint_geometry'
    delta = points[1]-points[0]
    norm2 = float(delta@delta)
    if norm2 <= 0:
        return None, 'degenerate_shaft_track'
    nearest = points[0]+np.clip((foot-points[0])@delta/norm2, 0., 1.)*delta
    outward = nearest-foot
    extent = float(np.linalg.norm(outward))
    if extent <= 1e-6:
        return None, 'unresolved_outward_side'
    outward /= extent
    excluded = set(chain['shaft_interval_ids']) | {i for group in chain['extension_interval_ids'] for i in group}
    widths, witnesses, clouds, cloud_radii = [], [], [], []
    for identifier in boundary['owners']:
        if identifier in excluded or identifier not in intervals:
            continue
        interval = intervals[identifier]
        p = np.asarray(interval['points_px'], float)
        if audit['inherited_source_points']+audit['width_sample_points']+len(p) > POLICY['maximum_combined_source_points']:
            audit['limits'].add('joint_source_point_budget')
            return None, 'joint_source_point_budget'
        audit['width_sample_points'] += len(p)
        hint = float(interval.get('width_px', 1.))
        if not np.isfinite(hint) or hint <= 0:
            continue
        all_q = np.rint(p).astype(int)
        all_r = distance[all_q[:, 1], all_q[:, 0]]
        if (all_r > 0).any():
            clouds.append(p[all_r > 0]); cloud_radii.append(2*all_r[all_r > 0])
        selected = np.linalg.norm(p-foot, axis=1) <= max(2., POLICY['boundary_nearby_width_lengths']*hint)
        q = np.rint(p[selected]).astype(int)
        if not len(q):
            continue
        radii = distance[q[:, 1], q[:, 0]]
        radii = radii[radii > 0]
        if len(radii) < 3:
            continue
        observed = 2*float(np.median(radii))
        widths.append(max(hint, observed))
        witnesses.append({'source_interval_id': identifier, 'positive_samples': len(radii),
                          'source_width_hint_px': hint, 'observed_width_px': observed})
    if not widths:
        return None, 'no_independent_boundary_width_witness'
    return {'boundary_id': boundary['id'], 'shaft_track': track['id'],
            'foot_px': foot.tolist(), 'outward_unit': outward.tolist(),
            'boundary_support_width_px': max(widths), 'boundary_width_witnesses': witnesses,
            '_competitor_points': np.concatenate(clouds), '_competitor_radii': np.concatenate(cloud_radii),
            'competitor_source_points': sum(len(p) for p in clouds),
            'geometric_hypothesis_only': True}, None


def outside_competitor_support(points, context, paint_radius, audit):
    source, radii = context['_competitor_points'], context['_competitor_radii']
    margin = paint_radius+POLICY['pixel_footprint_radius_px']
    maximum = float(radii.max())+margin
    candidate_min, candidate_max = points.min(0), points.max(0)
    selected = ((source >= candidate_min-maximum) & (source <= candidate_max+maximum)).all(1)
    source, radii = source[selected], radii[selected]+margin
    result = np.ones(len(points), bool)
    if not len(source): return result
    nearby = ((points >= source.min(0)-maximum) & (points <= source.max(0)+maximum)).all(1)
    indices = np.flatnonzero(nearby)
    cost = int(len(indices)*len(source))
    if audit['competitor_pair_checks']+cost > POLICY['maximum_competitor_pairs']:
        audit['limits'].add('joint_competitor_pair_budget'); return np.zeros(len(points), bool)
    audit['competitor_pair_checks'] += cost
    for start in range(0,len(indices),64):
        block=indices[start:start+64]
        squared=((points[block,None,:]-source[None,:,:])**2).sum(2)
        result[block]=~(squared <= radii[None,:]**2).any(1)
    return result


def analyze(gray, previous, baseline_pixel_roles, local_result, *, heads=(), diagnostic_heads=()):
    result = previous_analyze(gray, previous, baseline_pixel_roles, local_result,
                              heads=heads, diagnostic_heads=diagnostic_heads)
    output = result['pixel_roles']
    before = output.copy()
    assignment = result['pixel_interval_owner']
    audit = {'inherited_source_points': result['source_ownership_audit'].get('source_points', 0),
             'inherited_raster_pixels': result['source_ownership_audit'].get('raster_pixels', 0),
             'width_sample_points': 0, 'distance_preparation_pixels': int(gray.size),
             'competitor_pair_checks': 0, 'contexts_examined': 0, 'guarded_intervals': 0, 'removed_input_pixels': 0, 'limits': set()}
    if audit['inherited_raster_pixels']+gray.size > POLICY['maximum_combined_raster_pixels']:
        audit['limits'].add('joint_raster_pixel_budget')
    ink = foreground(gray)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5) if not audit['limits'] else None
    lookup = {p['id']: p for p in previous['proposals']}
    contexts = defaultdict(list)
    cache = {}
    for chain in local_result['hybrid']['dimension_chains']:
        if len(chain.get('feet_px', [])) != len(chain['extension_interval_ids']):
            continue
        for foot, group in zip(chain['feet_px'], chain['extension_interval_ids']):
            key = chain['boundary_id'], chain['shaft_track'], *map(float, foot)
            if key not in cache and not audit['limits']:
                cache[key] = context_for(previous, lookup, chain, foot, distance, audit)
                audit['contexts_examined'] += 1
            for identifier in group:
                if key not in contexts[identifier]:
                    contexts[identifier].append(key)
    for number, (proposal, record) in enumerate(zip(result['proposals'], result['interval_admission']), 1):
        selected = (assignment == number) & (output == 3) & (baseline_pixel_roles == 255)
        if not selected.any():
            continue
        audit['guarded_intervals'] += 1
        keys = contexts[proposal['id']]
        context = cache[keys[0]][0] if len(keys) == 1 and keys[0] in cache else None
        if context is None:
            keep = np.zeros(int(selected.sum()), bool)
            reason = 'ambiguous_or_unobserved_contact_context'
            evidence = {'context_count': len(keys)}
        else:
            points = np.argwhere(selected)[:, ::-1]
            displacement = points-np.asarray(context['foot_px'])
            radius = (context['boundary_support_width_px']+float(record['paint_radius_px'])
                      +POLICY['pixel_footprint_radius_px'])
            keep = displacement@np.asarray(context['outward_unit']) > radius
            keep &= outside_competitor_support(points, context, float(record['paint_radius_px']), audit)
            reason = 'outward_side_and_no_competing_boundary_support'
            evidence = {**{k:v for k,v in context.items() if not k.startswith('_')}, 'contact_abstention_extent_px': radius}
        y, x = np.nonzero(selected)
        output[y[~keep], x[~keep]] = 255
        assignment[y[~keep], x[~keep]] = 0
        removed = int((~keep).sum())
        audit['removed_input_pixels'] += removed
        record.update(contact_guard={'reason': reason, 'removed_pixels': removed, **evidence})
        record['added_pixels'] -= removed
        if record['added_pixels'] == 0:
            proposal['role'] = 'unknown'
            proposal['evidence'] = 'contact_pixels_require_review'
    if audit['limits']:
        output[:] = baseline_pixel_roles
        assignment[:] = 0
        for p, r in zip(result['proposals'], result['interval_admission']):
            p['role'] = 'unknown'
            r['added_pixels'] = 0
        result['search_incomplete'] = True
        audit['removed_input_pixels'] = int(np.count_nonzero(before != baseline_pixel_roles))
    audit['limits'] = sorted(audit['limits'])
    audit['combined_source_points'] = audit['inherited_source_points']+audit['width_sample_points']
    audit['combined_raster_pixels'] = audit['inherited_raster_pixels']+gray.size
    result.update(schema=POLICY['version'], contact_policy=dict(POLICY), contact_guard_audit=audit,
                  added_pixels=int(np.count_nonzero(output != baseline_pixel_roles)),
                  baseline_pixel_changes=int(np.count_nonzero(output != baseline_pixel_roles)),
                  role_counts=dict(Counter(p['role'] for p in result['proposals'])),
                  old_overlay_additions=int(np.count_nonzero(before != baseline_pixel_roles)),
                  additions_are_subset_of_previous_overlay=bool(np.all(
                      (output == baseline_pixel_roles) | (output == before))))
    return result
