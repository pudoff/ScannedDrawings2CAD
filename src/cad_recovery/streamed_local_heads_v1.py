"""Bounded head search with recoverable shaft axes and local repeat fields.

The sealed v1 search loop is retained in a versioned module; numeric helpers
remain shared. No runtime monkey-patching of frozen experiment modules.
"""
from collections import Counter
from copy import deepcopy

import cv2
import numpy as np

from .family_catalog_v2 import require
from .sparse_head_regions_v1 import regions
from .sparse_local_heads_v1 import axes
from .local_width_heads_v1 import append_preserving
from .streamed_head_profiles_v1 import Profiles, prepare, finish
from .local_head_parallel_field_v2 import check as context_check

POLICY = {'version': 'streamed-local-heads/1.0', 'radii_px': [7, 5, 4, 3, 2, 1],
    'profile_interpolation': 'float32_maps_explicit_float64_bilinear_and_fsum',
    'row_stopping': 'unseen_spans_cannot_win_nearest_center_selection',
    'body_stopping': 'measured_false_band_beyond_every_permitted_start',
    'scalar_head_thresholds_unchanged': True,
    'maximum_pixels': 25000000, 'maximum_active_pixels': 2000000, 'maximum_union_checks': 8000000,
    'maximum_component_memberships': 8000000, 'maximum_morphology_pixels': 40000000,
    'maximum_profile_samples': 8000000, 'maximum_context_pixels': 4000000,
    'maximum_raw_regions': 4096, 'maximum_unique_contours': 8192,
    'maximum_head_hypotheses': 512, 'maximum_heads_per_level': 256, 'maximum_candidates': 512,
    'proposal_counter_domain': 'unique source-validated heads; raw regions are independently bounded and fully accounted',
    'existing_observations_immutable': True, 'DT_head_proposals_used': False,
    'source_erasure_allowed': False, 'manual_references_used': False,
    'production_admitted': False, 'physical_geometry_approved': False, 'confidence_calibrated': False}


def search(gray, *, baseline=(), text_polygons=(), eager_reference=False):
    require(isinstance(gray, np.ndarray) and gray.ndim == 2 and gray.dtype == np.uint8
            and 0 < gray.size <= POLICY['maximum_pixels'], 'Expected bounded uint8 grayscale')
    require(len(baseline) <= 512 and len(text_polygons) <= 256, 'Bounded baseline/text required')
    for polygon in text_polygons:
        q = np.asarray(polygon, float)
        require(q.ndim == 2 and q.shape[1] == 2 and 3 <= len(q) <= 4096
                and np.isfinite(q).all() and np.abs(q).max() <= 1e7, 'Invalid text polygon')
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    audit = {'distance_preparation_pixels': 0, 'sparse_active_pixels': 0, 'dense_threshold_pixels': 0,
        'union_neighbor_checks': 0, 'component_memberships': 0, 'morphology_pixels': 0,
        'raw_regions_generated': 0, 'raw_regions_processed': 0, 'directional_proposals': 0,
        'unique_head_hypotheses': 0, 'duplicate_contours': 0, 'context_pixels': 0, 'context_segments': 0,
        'levels_completed': [], 'limits': set()}
    profiles = Profiles(mask, maximum_samples=POLICY['maximum_profile_samples'], eager=eager_reference)
    proposals, kept, exclusions, signatures = [], [], [], set()
    reasons, region_status = Counter(), Counter()
    for radius, components in regions(mask, POLICY['radii_px'], audit, POLICY):
        audit['raw_regions_generated'] += len(components); level_heads = 0; level_complete = True
        for region in components:
            if audit['raw_regions_processed'] >= POLICY['maximum_raw_regions']:
                audit['limits'].add('raw_region_budget'); level_complete = False; break
            audit['raw_regions_processed'] += 1
            x, y, w, h = region['box']; left, top = max(0, x-radius), max(0, y-radius)
            right, bottom = min(mask.shape[1], x+w+radius), min(mask.shape[0], y+h+radius)
            cost = (right-left)*(bottom-top)
            if audit['morphology_pixels']+cost > POLICY['maximum_morphology_pixels']:
                audit['limits'].add('morphology_pixel_budget'); region_status['budget_aborted'] += 1
                level_complete = False; break
            audit['morphology_pixels'] += cost
            seed = np.zeros((bottom-top, right-left), np.uint8); q = region['points']-[left, top]
            seed[q[:, 1], q[:, 0]] = 255
            opened = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
            opened &= mask[top:bottom, left:right]
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            status = 'source_axes_rejected'
            for contour in contours:
                if cv2.contourArea(contour) < 4: continue
                contour = contour+np.array([left, top], np.int32)
                signature = (contour.shape, contour.tobytes())
                if signature in signatures:
                    audit['duplicate_contours'] += 1; continue
                if len(signatures) >= POLICY['maximum_unique_contours']:
                    audit['limits'].add('contour_cache_budget'); level_complete = False; break
                signatures.add(signature)
                for base, unit, length, width in axes(contour):
                    audit['directional_proposals'] += 1
                    prepared, reason = prepare(base, unit, length, width, profiles)
                    if reason is None: head, reason = finish(prepared, profiles, text_polygons)
                    else: head = None
                    if reason: reasons[reason] += 1
                    if head is not None:
                        _, novel = append_preserving(list(baseline)+proposals, [head])
                        if novel:
                            if len(proposals) >= POLICY['maximum_head_hypotheses'] or level_heads >= POLICY['maximum_heads_per_level']:
                                audit['limits'].add('head_hypothesis_budget'); level_complete = False; break
                            identity = 'context-head-'+str(len(proposals)+1).zfill(5)
                            while identity in {h['id'] for h in list(baseline)+proposals}: identity += '-new'
                            head['id'] = identity
                            head['evidence']['source_level_radius_px'] = radius
                            decision = context_check(mask, head, audit, maximum_pixels=POLICY['maximum_context_pixels'])
                            head['evidence']['parallel_context'] = decision
                            proposals.append(deepcopy(head)); level_heads += 1
                            if decision['incomplete']:
                                audit['limits'].add(decision['reason']); level_complete = False
                            elif decision['available']: kept.append(deepcopy(head))
                            else: exclusions.append({'candidate_id': head['id'], **deepcopy(decision), 'observation_retained': True})
                            status = 'source_head_hypothesis'
                    if profiles.incomplete or not level_complete: break
                if profiles.incomplete or not level_complete: break
            region_status[status] += 1
            if profiles.incomplete or not level_complete: break
        if level_complete and not profiles.incomplete: audit['levels_completed'].append(radius)
        else: break
    if profiles.incomplete: audit['limits'].add('profile_sample_budget')
    candidates, added = append_preserving(baseline, kept)
    if len(candidates) > 512:
        audit['limits'].add('candidate_budget'); candidates = candidates[:512]; added = candidates[len(baseline):]
    # Append helper preserves prefixes but allocates legacy local-width IDs; use
    # the retained native hypothesis identity for new experimental observations.
    for i, head in enumerate(added):
        source = next(h for h in kept if h['tip_px'] == head['tip_px'] and h['direction'] == head['direction'])
        head['id'] = source['id']; candidates[len(baseline)+i] = deepcopy(head)
    audit.update(profile_streaming=profiles.statistics, axis_recovery_proposals=profiles.axis_proposals, recovered_axes=len(profiles.recovered_axes),
        unique_head_hypotheses=len(proposals), profile_samples=profiles.samples,
        profile_stage_samples=profiles.stage_samples, profile_row_reuse=profiles.hits, profile_cache_evictions=profiles.evictions,
        contour_cache_entries=len(signatures), region_status=dict(region_status), rejections=dict(reasons),
        source_observations_retained=len(proposals), accepted_additions=len(added), context_abstentions=len(exclusions))
    audit['coverage_complete'] = (not audit['limits'] and audit['levels_completed'] == POLICY['radii_px']
        and audit['raw_regions_processed'] == audit['raw_regions_generated'])
    audit['limits'] = sorted(audit['limits'])
    return {'schema': POLICY['version'], 'policy': deepcopy(POLICY), 'candidates': candidates, 'added': added,
        'proposals': proposals, 'context_exclusions': exclusions, 'analysis': audit,
        'search_incomplete': bool(audit['limits']) or not audit['coverage_complete'], 'eager_reference': eager_reference,
        'source_modified': False, 'production_admitted': False, 'physical_geometry_approved': False, 'CAD_promotions': 0}


def guarded_search(gray, *, baseline=(), text_polygons=(), eager_reference=False):
    from .head_support_ownership_v1 import apply
    raw=search(gray,baseline=baseline,text_polygons=text_polygons,eager_reference=eager_reference)
    return apply(gray,raw,baseline=baseline)
