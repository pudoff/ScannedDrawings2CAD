"""Source-supported part-contour proposals; never semantic CAD approval.

Only unknown pixels may be added, on a closed non-frame boundary supported
by an existing dimension chain and independent local-role evidence. All source
intervals compete for ownership, including text, unknown and noise intervals.
"""
from collections import defaultdict
from copy import deepcopy
import math

import cv2
import numpy as np

from .stroke_roles_v1 import checked_gray, foreground
from .supported_role_overlay_v1 import source_ownership

POLICY = {'version': 'boundary-role-overlay/1.0', 'maximum_intervals': 4096,
          'maximum_boundaries': 512, 'maximum_chains': 4096, 'maximum_boundary_samples': 250000,
          'maximum_source_points': 2000000, 'maximum_raster_pixels': 40000000,
          'maximum_width_q95_to_q25': 1.5, 'minimum_foot_separation_in_strokes': 6.,
          'maximum_foot_boundary_distance_in_strokes': 2.,
          'new_roles': ['part_contour'], 'inherited_known_roles_immutable': True,
          'semantic_ownership_confirmed': False, 'source_erasure_allowed': False,
          'production_admitted': False, 'physical_geometry_approved': False}


def boundary_witnesses(previous, local):
    """A chain is evidence for a proposal, not proof that a loop is a part."""
    intervals = previous['proposals']; by_id = {p['id']: p for p in intervals}
    if len(by_id) != len(intervals): raise ValueError('Duplicate source interval IDs')
    boundaries = previous['boundaries']['boundaries']; chains = local['dimension_chains']
    if len(boundaries) > 512 or len(chains) > 4096:
        return {}, [{'reason': 'boundary_or_chain_budget'}], True
    lookup = {b['id']: b for b in boundaries}
    if len(lookup) != len(boundaries): raise ValueError('Duplicate boundary IDs')
    stroke = float(previous['stroke_px'])
    if not math.isfinite(stroke) or stroke <= 0: raise ValueError('Invalid observed stroke')
    claims = defaultdict(list); rejected = []; samples = 0
    for index, chain in enumerate(chains):
        boundary = lookup.get(chain['boundary_id'])
        if boundary is None or boundary.get('frame_hypothesis'):
            rejected.append({'chain': index, 'reason': 'missing_or_frame_boundary'}); continue
        p = np.asarray(boundary['polygon'], float); feet = np.asarray(chain.get('feet_px', []), float)
        samples += len(p)
        if samples > 250000: return {}, [{'reason': 'boundary_sample_budget'}], True
        # The existing half-edge boundary walker emits implicitly closed polygons;
        # its final vertex need not repeat the first. Open source paths never
        # acquire a boundary record from that walker.
        if (p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all()
                or boundary.get('orientation') not in ('bounded_face', 'component_outline')
                or abs(cv2.contourArea(p.astype(np.float32))) <= 0):
            rejected.append({'chain': index, 'reason': 'not_closed_finite_boundary'}); continue
        owners = set(boundary['owners'])
        excluded = set(chain['shaft_interval_ids']) | {i for group in chain['extension_interval_ids'] for i in group}
        if not owners or not owners <= by_id.keys() or owners & excluded:
            rejected.append({'chain': index, 'reason': 'missing_or_dimension_boundary_owner'}); continue
        if (feet.shape != (2, 2) or not np.isfinite(feet).all() or len(chain['extension_interval_ids']) != 2
                or not all(chain['extension_interval_ids']) or not excluded <= by_id.keys()
                or np.linalg.norm(feet[0]-feet[1]) < 6*stroke):
            rejected.append({'chain': index, 'reason': 'unresolved_two_foot_context'}); continue
        # Existing geometric verifier has already checked the chains; independently
        # reject stale boundary/foot references before using their proposed role.
        polygon = p.astype(np.float32)
        distance = [abs(cv2.pointPolygonTest(polygon, tuple(q), True)) for q in feet]
        if max(distance) > 2*stroke+1:
            rejected.append({'chain': index, 'reason': 'feet_do_not_support_this_boundary'}); continue
        for key in owners:
            claims[key].append({'boundary_id': boundary['id'], 'chain_index': index,
                                'feet_px': feet.tolist(), 'source_boundary_owner_ids': sorted(owners)})
    return dict(claims), rejected, False


def analyze(gray, previous, inherited_roles, local_result, *, heads=(), diagnostic_heads=()):
    checked_gray(gray)
    if inherited_roles.shape != gray.shape or inherited_roles.dtype != np.uint8:
        raise ValueError('Aligned uint8 role map required')
    if not np.isin(inherited_roles, list(range(10))+[255]).all(): raise ValueError('Unknown role code')
    if len(previous['proposals']) > 4096: raise ValueError('Interval budget')
    local = local_result['local']
    evidence = local['pixel_roles']; text = local['text_mask']
    if evidence.shape != gray.shape or text.shape != gray.shape: raise ValueError('Aligned local evidence required')
    claims, rejections, limit = boundary_witnesses(previous, local_result['hybrid'])
    ink = foreground(gray); output = inherited_roles.copy()
    ownership, owner_audit = source_ownership(ink, previous['proposals'])
    work = owner_audit.get('raster_pixels', 0)+2*gray.size
    incomplete = bool(previous['search_incomplete'] or previous['boundaries']['search_incomplete']
                      or local['search_incomplete'] or local_result['hybrid']['search_incomplete']
                      or limit or owner_audit['incomplete'] or work > 40000000)
    assignment = np.zeros(gray.shape, np.uint16)
    protect = text > 0
    for head in list(heads)+list(diagnostic_heads):
        polygon = np.asarray(head['polygon_px'], float)
        if polygon.shape != (3, 2) or not np.isfinite(polygon).all(): raise ValueError('Finite triangular head required')
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
        protect |= mask > 0
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5) if not incomplete else None
    records = []
    for number, interval in enumerate(previous['proposals'], 1):
        witnesses = claims.get(interval['id'], [])
        record = {'id': interval['id'], 'path_id': interval['path_id'],
                  'source_index_interval': deepcopy(interval['source_index_interval']), 'added_pixels': 0,
                  'status': 'proposed_only', 'semantic_ownership_confirmed': False}
        if not witnesses: record['reason'] = 'no_dimension_supported_boundary'
        elif incomplete or ownership is None: record['reason'] = 'search_or_ownership_incomplete'
        elif len({w['boundary_id'] for w in witnesses}) != 1:
            record['reason'] = 'ambiguous_boundary_membership'
        else:
            q = np.rint(interval['points_px']).astype(int)
            valid = (ink[q[:, 1], q[:, 0]] > 0) & ~protect[q[:, 1], q[:, 0]]
            values = distance[q[valid, 1], q[valid, 0]]
            if len(values) < 3: record['reason'] = 'insufficient_observed_body_samples'
            else:
                low, high = np.quantile(values, [.25, .95])
                if low <= 0 or high > 1.5*low: record['reason'] = 'unstable_source_width'
                else:
                    radius = max(.5, float(low)-1.)
                    eligible = ((inherited_roles == 255) & (evidence == 1) & (ink > 0) & ~protect
                                & (ownership['owner'] == number) & (ownership['distance'] <= radius))
                    output[eligible] = 1; assignment[eligible] = number
                    record.update(reason='source_interval_local_role_and_dimension_boundary', witnesses=witnesses,
                                  paint_radius_px=radius, body_radius_q25_px=float(low), body_radius_q95_px=float(high),
                                  added_pixels=int(eligible.sum()))
        records.append(record)
    known = inherited_roles != 255
    return {'schema': POLICY['version'], 'policy': deepcopy(POLICY), 'pixel_roles': output,
            'pixel_interval_owner': assignment, 'interval_admission': records, 'boundary_rejections': rejections,
            'source_ownership_audit': owner_audit, 'raster_work_pixels': work,
            'search_incomplete': incomplete, 'added_pixels': int(np.count_nonzero(output != inherited_roles)),
            'inherited_known_roles_unchanged': bool(np.array_equal(output[known], inherited_roles[known])),
            'source_pixels_erased': 0, 'semantic_CAD_promotions': 0, 'production_admitted': False,
            'physical_geometry_approved': False, 'semantic_ownership_confirmed': False}
