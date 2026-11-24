"""Versioned role experiment with observed short source routes at contacts."""
from collections import defaultdict,Counter
import copy
import numpy as np
import cv2
from .stroke_roles_v1 import CLASSES,checked_gray,foreground
from .local_role_guard_v1 import hard_head_gate,local_parallel_conflict,supported_bundles,paint,POLICY as LEGACY_POLICY
from .local_chain_evidence_v2 import verify
POLICY={**LEGACY_POLICY,'version':'local-role-guard/2.0'}


def analyze(gray, previous, baseline_pixel_roles, text_mask, heads, *, diagnostic_heads=()):
    checked_gray(gray)
    if len(previous['proposals']) > 4096 or len(heads)+len(diagnostic_heads) > 2048:
        raise ValueError('Bounded source intervals/heads required')
    if baseline_pixel_roles.shape != gray.shape or baseline_pixel_roles.dtype != np.uint8:
        raise ValueError('Aligned uint8 baseline role map required')
    intervals = copy.deepcopy(previous['proposals'])
    for p in intervals:
        p['role'], p['alternatives'] = 'unknown', []
    ink = foreground(gray)
    stroke = previous['stroke_px']
    hard_available, hard_events = hard_head_gate(heads, text_mask)
    boundaries = [b for b in previous['boundaries']['boundaries'] if max(b['box'][2]-b['box'][0], b['box'][3]-b['box'][1]) >= .15*min(gray.shape)]
    chains = verify(intervals, previous['tracks'], boundaries, hard_available, ink)
    bundles, bundle_rejections, bundle_incomplete = supported_bundles(intervals, previous['hatch_bundles'], chains['eligible_boundaries'], stroke)
    independently_supported = {i for c in chains['chains']+chains['leaders'] for i in c['head_ids']}
    head_events = list(hard_events)
    available = []
    support_events = []
    local_incomplete = False
    for head in hard_available:
        local = local_parallel_conflict(head, previous['tracks'], intervals, stroke)
        local_incomplete |= local['incomplete']
        midpoint = (np.asarray(head['tip_px'])+np.asarray(head['polygon_px'])[1:].mean(axis=0))/2
        in_supported_bundle = any(cv2.pointPolygonTest(np.asarray(b['polygon'], np.float32), tuple(midpoint), True) >= -stroke for b in bundles)
        if local['supported'] and in_supported_bundle and head['id'] not in independently_supported:
            head_events.append({'id': head['id'], 'reason': 'locally_supported_interior_hatch_conflict', 'observation_retained': True, 'support': local})
        else:
            available.append(head)
            if head['id'] in independently_supported:
                support_events.append({'id': head['id'], 'reason': 'independent_dimension_or_leader_evidence_before_hatch_gate'})
    dt_events = []
    for head in diagnostic_heads:
        local = local_parallel_conflict(head, previous['tracks'], intervals, stroke, minimum_offsets=3)
        local_incomplete |= local['incomplete']
        if local['supported']:
            dt_events.append({'id': head['id'], 'reason': 'ambiguous_local_parallel_ink', 'observation_retained': True, 'support': local})
    roles = defaultdict(set)
    boundaries_by_id = {b['id']: b for b in boundaries}
    available_ids = {h['id'] for h in available}
    accepted_chains = [c for c in chains['chains'] if set(c['head_ids']) <= available_ids]
    accepted_leaders = [c for c in chains['leaders'] if set(c['head_ids']) <= available_ids]
    supported_boundaries = set()
    for chain in accepted_chains:
        for i in chain['shaft_interval_ids']:
            roles[i].add('dimension_line')
        for ext in chain['extension_interval_ids']:
            for i in ext:
                roles[i].add('extension_line')
        supported_boundaries.add(chain['boundary_id'])
    for leader in accepted_leaders:
        for i in leader['interval_ids']:
            roles[i].add('leader_line')
        supported_boundaries.add(leader['boundary_id'])
    hatch_widths = {}
    for bundle in bundles:
        for i in bundle['interval_ids']:
            if not roles[i]:
                roles[i].add('hatching')
                hatch_widths[i] = bundle['thin_width_px']
        supported_boundaries.add(bundle['boundary_id'])
    for boundary_id in supported_boundaries:
        for i in boundaries_by_id[boundary_id]['owners']:
            if not roles[i]:
                roles[i].add('part_contour')
    for p in intervals:
        q = np.rint(p['points_px']).astype(int)
        if not roles[p['id']] and np.mean(text_mask[q[:, 1], q[:, 0]] > 0) >= .5:
            roles[p['id']].add('text')
    incomplete = previous['search_incomplete'] or chains['incomplete'] or bundle_incomplete or local_incomplete
    if incomplete:
        roles = defaultdict(set)
    local = paint(intervals, roles, ink, text_mask, heads, diagnostic_heads, hatch_widths)
    hybrid = baseline_pixel_roles.copy()
    local_classes = (local > 0) & (local < 10)
    preserved_baseline = np.isin(hybrid, [5, 6, 7, 8])
    specific = np.isin(local, [2, 3, 4]) & ~preserved_baseline
    additions = local_classes & ((hybrid == 255) | (hybrid == local))
    eligible = (specific | additions) & (ink > 0) if not incomplete else np.zeros(gray.shape, bool)
    hybrid[eligible] = local[eligible]
    hybrid_proposals = copy.deepcopy(intervals)
    for p in hybrid_proposals:
        q = np.rint(p['points_px']).astype(int)
        values = hybrid[q[:, 1], q[:, 0]]
        known = values[(values > 0) & (values < 10)]
        if p['role'] == 'unknown' and len(known) >= .8*len(values):
            counts = np.bincount(known, minlength=10)
            cls = int(np.argmax(counts))
            if counts[cls] >= .8*len(known):
                p['role'] = CLASSES[cls]
                p['evidence'] = 'baseline_compatible_pixel_consensus_proposed_only'
    common = {'source_route_audit':chains.get('source_route_audit',{}), 'recovered_chains':chains.get('recovered_chains',0), 'dimension_chains': accepted_chains, 'leader_hypotheses': accepted_leaders,
        'head_events': head_events, 'head_support_events': support_events, 'head_ids_used': sorted(available_ids),
        'head_observations_retained': len(heads), 'DT_context_events': dt_events, 'DT_admitted_for_roles': False,
        'supported_hatch_bundles': bundles, 'rejected_hatch_bundles': bundle_rejections,
        'search_incomplete': bool(incomplete), 'source_pixels_erased': 0, 'semantic_CAD_promotions': 0,
        'semantic_ownership_confirmed': False, 'physical_geometry_approved': False, 'production_admitted': False}
    return {'local': {**common, 'proposals': intervals, 'pixel_roles': local, 'text_mask': text_mask.copy(),
                      'role_counts': dict(Counter(p['role'] for p in intervals))},
        'hybrid': {**common, 'proposals': hybrid_proposals, 'pixel_roles': hybrid, 'text_mask': text_mask.copy(),
                   'role_counts': dict(Counter(p['role'] for p in hybrid_proposals)),
                   'baseline_pixel_changes': int(np.count_nonzero(hybrid != baseline_pixel_roles)),
                   'protected_baseline_classes_unchanged': bool(np.array_equal(hybrid[preserved_baseline], baseline_pixel_roles[preserved_baseline])),
                   'pixel_map_geometry_authority': 'partial_interval_proposals_require_review'}}
