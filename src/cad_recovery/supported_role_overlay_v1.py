"""Unknown-pixel dimension roles require an observed source owner and chain proof."""
from collections import Counter, defaultdict
from copy import deepcopy

import cv2
import numpy as np

from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'supported-role-overlay/1.0', 'maximum_intervals': 4096,
    'maximum_source_points': 2000000, 'maximum_interval_raster_pixels': 40000000,
    'maximum_pixels': 25000000, 'new_pixel_roles': ['dimension_line', 'extension_line'],
    'existing_known_roles_immutable': True, 'maximum_width_q95_to_q25': 1.5,
    'source_ownership': 'nearest observed source seed, same ink component; shared seeds abstain',
    'new_hatch_contour_leader_text_pixels': False, 'source_erasure_allowed': False,
    'semantic_CAD_promotions': 0, 'production_admitted': False, 'physical_geometry_approved': False}


def source_ownership(ink, intervals):
    """All intervals compete, including unknown/noise paths; no GT or head targets."""
    owners = np.zeros(ink.shape, np.int32); raster_pixels = 0; points_used = 0
    for i, interval in enumerate(intervals, 1):
        p = np.asarray(interval['points_px'], float); points_used += len(p)
        if (p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all()
                or (p < 0).any() or (p[:, 0] > ink.shape[1]-1).any() or (p[:, 1] > ink.shape[0]-1).any()):
            raise ValueError('Source intervals must be finite points inside the source raster')
        if points_used > POLICY['maximum_source_points']:
            return None, {'incomplete': True, 'reason': 'source_point_budget'}
        lo = np.floor(p.min(0)).astype(int); hi = np.ceil(p.max(0)).astype(int)+1
        raster_pixels += int(np.prod(hi-lo))
        if raster_pixels > POLICY['maximum_interval_raster_pixels']:
            return None, {'incomplete': True, 'reason': 'source_interval_raster_budget'}
        mask = np.zeros(tuple((hi-lo)[::-1]), np.uint8)
        cv2.polylines(mask, [np.rint(p-lo).astype(np.int32)], False, 255, 1)
        view = owners[lo[1]:hi[1], lo[0]:hi[0]]
        selected = (mask > 0) & (ink[lo[1]:hi[1], lo[0]:hi[0]] > 0)
        shared = selected & (view != 0) & (view != i)
        view[selected & (view == 0)] = i; view[shared] = -1
    seeds = owners != 0
    if not seeds.any():
        return None, {'incomplete': False, 'reason': 'no_observed_source_seeds'}
    distance, labels = cv2.distanceTransformWithLabels((~seeds).astype(np.uint8), cv2.DIST_L2, 5,
                                                      labelType=cv2.DIST_LABEL_PIXEL)
    _, components = cv2.connectedComponents((ink > 0).astype(np.uint8), connectivity=8)
    lookup = np.zeros(int(labels.max())+1, np.int32)
    component_lookup = np.zeros_like(lookup)
    lookup[labels[seeds]] = owners[seeds]; component_lookup[labels[seeds]] = components[seeds]
    nearest = lookup[labels]; same_component = component_lookup[labels] == components
    nearest[~same_component | (ink == 0)] = 0
    return {'owner': nearest, 'distance': distance, 'components': components}, {
        'incomplete': False, 'source_points': points_used, 'raster_pixels': raster_pixels,
        'source_seed_pixels': int(seeds.sum()), 'shared_seed_pixels': int((owners == -1).sum()),
        'assigned_ink_owner_pixels': int(((nearest > 0) & (ink > 0)).sum()),
        'unresolved_ink_owner_pixels': int(((nearest <= 0) & (ink > 0)).sum())}


def analyze(gray, previous, baseline_pixel_roles, local_result, *, heads=(), diagnostic_heads=()):
    """Consume frozen local evidence; never convert a bundle/boundary into pixel authority."""
    checked_gray(gray)
    if gray.size > POLICY['maximum_pixels'] or len(previous['proposals']) > POLICY['maximum_intervals']:
        raise ValueError('Bounded source raster and intervals required')
    if baseline_pixel_roles.shape != gray.shape or baseline_pixel_roles.dtype != np.uint8:
        raise ValueError('Aligned baseline uint8 pixel-role map required')
    local = local_result['hybrid']
    intervals = deepcopy(previous['proposals']); ink = foreground(gray)
    ownership, audit = source_ownership(ink, intervals)
    incomplete = bool(local['search_incomplete'] or audit['incomplete'])
    proof = defaultdict(set)
    for chain in local['dimension_chains']:
        for i in chain['shaft_interval_ids']: proof[i].add(2)
        for ext in chain['extension_interval_ids']:
            for i in ext: proof[i].add(3)
    output = baseline_pixel_roles.copy(); evidence_role = local_result['local']['pixel_roles']
    head_mask = np.zeros_like(ink)
    for head in list(heads)+list(diagnostic_heads):
        cv2.fillConvexPoly(head_mask, np.rint(head['polygon_px']).astype(np.int32), 255)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    rows = []; assignment = np.zeros(ink.shape, np.uint16)
    for number, p in enumerate(intervals, 1):
        hints = proof[p['id']]; p['role'] = 'unknown'
        record = {'id': p['id'], 'source_index_interval': p['source_index_interval'],
            'path_id': p['path_id'], 'chain_role_candidates': sorted(hints), 'added_pixels': 0}
        if len(hints) != 1:
            record['reason'] = 'no_unique_dimension_chain_role'
        elif incomplete or ownership is None:
            record['reason'] = 'source_ownership_or_local_search_unavailable'
        else:
            q = np.rint(p['points_px']).astype(int)
            valid = (ink[q[:, 1], q[:, 0]] > 0) & (head_mask[q[:, 1], q[:, 0]] == 0)
            samples = distance[q[:, 1][valid], q[:, 0][valid]]
            if len(samples) < 3:
                record['reason'] = 'insufficient_observed_body_width_samples'
            else:
                low, high = np.quantile(samples, [.25, .95])
                record['body_radius_q25_px'], record['body_radius_q95_px'] = float(low), float(high)
                if high > POLICY['maximum_width_q95_to_q25']*low:
                    record['reason'] = 'source_interval_width_not_stable'
                else:
                    cls = next(iter(hints)); radius = max(.5, float(low)-1.)
                    eligible = ((baseline_pixel_roles == 255) & (evidence_role == cls)
                        & (ownership['owner'] == number) & (ownership['distance'] <= radius)
                        & (head_mask == 0) & (ink > 0))
                    output[eligible] = cls; assignment[eligible] = number
                    record['reason'] = 'observed_interval_owner_and_dimension_chain'
                    record['paint_radius_px'] = radius; record['added_pixels'] = int(eligible.sum())
                    p['role'] = 'dimension_line' if cls == 2 else 'extension_line'
        p['alternatives'] = ['dimension_line' if i == 2 else 'extension_line' for i in sorted(hints)]
        p['evidence'] = record['reason']; p['semantic_ownership_confirmed'] = False; p['status'] = 'proposed_only'
        rows.append(record)
    known = baseline_pixel_roles != 255
    # Keep head exclusions/chains exactly as local guard v1; this stage cannot admit new DT heads.
    common = {k: deepcopy(v) for k, v in local.items() if k not in ('pixel_roles', 'text_mask', 'proposals', 'role_counts')}
    common.update(schema=POLICY['version'], policy=deepcopy(POLICY), proposals=intervals, pixel_roles=output,
        pixel_interval_owner=assignment, text_mask=local['text_mask'].copy(), interval_admission=rows,
        source_ownership_audit=audit, search_incomplete=incomplete,
        baseline_known_roles_unchanged=bool(np.array_equal(output[known], baseline_pixel_roles[known])),
        baseline_pixel_changes=int(np.count_nonzero(output != baseline_pixel_roles)),
        added_pixels=int(np.count_nonzero(output != baseline_pixel_roles)),
        role_counts=dict(Counter(p['role'] for p in intervals)),
        pixel_map_geometry_authority='partial_dimension_interval_overlay_requires_review',
        source_pixels_erased=0, semantic_CAD_promotions=0, production_admitted=False, physical_geometry_approved=False)
    return common
