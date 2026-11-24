"""Local hatch evidence and baseline-compatible role proposals, with source retained."""
from collections import Counter, defaultdict
import copy
import math

import cv2
import numpy as np

from .interval_roles_v1 import BoundaryQueries
from .local_chain_evidence_v1 import verify
from .stroke_roles_v1 import CLASSES, checked_gray, foreground

POLICY = {'version': 'local-role-guard/1.0', 'maximum_intervals': 4096, 'maximum_heads': 2048,
    'maximum_local_pair_checks': 250000, 'minimum_parallel_offsets': 4, 'minimum_diagnostic_parallel_offsets': 3,
    'parallel_angle_deg': 8., 'local_hatch_offset_min_in_strokes': 1.5,
    'maximum_hatch_width_ratio': 1.5, 'head_uncertainty_fraction_max': .5,
    'hybrid_preserves_baseline_classes': ['axis_construction', 'text', 'sheet_frame', 'title_grid'],
    'source_erasure': 0, 'semantic_CAD_promotions': 0, 'production_admitted': False}


def hard_head_gate(heads, text_mask):
    available, events = [], []
    for head in heads:
        tip = np.asarray(head['tip_px'], float)
        base = np.asarray(head['polygon_px'][1:], float).mean(axis=0)
        reason = None
        if head.get('uncertainty_px', 0) > .5*np.linalg.norm(base-tip):
            reason = 'unresolved_head_uncertainty'
        x, y = np.rint((tip+base)/2).astype(int)
        if 0 <= x < text_mask.shape[1] and 0 <= y < text_mask.shape[0] and text_mask[y, x]:
            reason = 'head_in_text_component'
        if reason:
            events.append({'id': head['id'], 'reason': reason, 'observation_retained': True})
        else:
            available.append(head)
    return available, events


def local_parallel_conflict(head, raw_tracks, intervals, stroke, *, minimum_offsets=4):
    """Count observed nearby parallel fragments, independently of head direction/hull."""
    polygon = np.asarray(head['polygon_px'], float)
    tip = np.asarray(head['tip_px'], float)
    base = polygon[1:].mean(axis=0)
    center = (tip+base)/2
    radius = np.linalg.norm(base-tip)+float(np.linalg.norm(polygon[1]-polygon[2]))/2
    lookup = {p['id']: p for p in intervals}
    selected = []
    for track in raw_tracks:
        points = np.concatenate([np.asarray(lookup[i]['points_px']) for i in track['interval_ids']])
        if not np.any(np.linalg.norm(points-center, axis=1) <= radius+stroke):
            continue
        # The fragment must reach the head's local box, not merely share a broad global hull.
        lo, hi = polygon.min(axis=0)-stroke, polygon.max(axis=0)+stroke
        if not np.any(np.all((points >= lo) & (points <= hi), axis=1)):
            continue
        selected.append(track)
    checks = 0
    for seed in selected:
        u = np.asarray(seed['u'], float)
        normal = np.array([-u[1], u[0]])
        offsets = []
        ids = []
        for track in selected:
            checks += 1
            if checks > 250000:
                return {'supported': False, 'incomplete': True, 'track_ids': []}
            if abs(u@track['u']) < math.cos(math.radians(8)):
                continue
            offset = float((np.asarray(track['points']).mean(axis=0)-center)@normal)
            if not any(abs(offset-value) < 1.5*stroke for value in offsets):
                offsets.append(offset)
                ids.append(track['id'])
        if len(offsets) >= minimum_offsets:
            return {'supported': True, 'incomplete': False, 'track_ids': ids, 'offset_count': len(offsets),
                    'evidence': 'observed_parallel_fragments_in_head_local_box'}
    return {'supported': False, 'incomplete': False, 'track_ids': []}


def supported_bundles(intervals, bundles, eligible, stroke):
    lookup = {p['id']: p for p in intervals}
    query = BoundaryQueries()
    supported, rejected = [], []
    for bundle in bundles:
        members = [lookup[i] for i in bundle['interval_ids']]
        found = None
        for boundary in eligible:
            if set(boundary['owners']) & set(bundle['interval_ids']):
                continue
            good = True
            for member in members:
                p = np.asarray(member['points_px'])
                for point in (p[0], p[len(p)//2], p[-1]):
                    d = query.distance(point, boundary)
                    if d is None or d <= max(stroke, boundary['chord_error_px']):
                        good = False
                        break
                if not good:
                    break
            if good:
                found = boundary
                break
        if found:
            supported.append({**bundle, 'boundary_id': found['id'], 'thin_width_px': float(np.quantile([p['width_px'] for p in members], .25)),
                              'local_support': 'member_endpoints_and_midpoints_inside_non_shaft_boundary'})
        else:
            rejected.append({'id': bundle['id'], 'reason': 'no_independent_interior_boundary_support'})
    return supported, rejected, query.incomplete


def paint(intervals, roles, ink, text_mask, heads, diagnostic_heads, hatch_widths):
    pixel = np.full(ink.shape, 255, np.uint8)
    conflicts, head_mask = np.zeros_like(ink, bool), np.zeros_like(ink)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    for head in list(heads)+list(diagnostic_heads):
        cv2.fillConvexPoly(head_mask, np.rint(head['polygon_px']).astype(np.int32), 255)
    for interval in intervals:
        hints = roles[interval['id']]
        role = next(iter(hints)) if len(hints) == 1 else 'unknown'
        interval['role'], interval['alternatives'] = role, sorted(hints)
        if role == 'unknown':
            continue
        width = interval['width_px']
        if role == 'hatching':
            reference = hatch_widths[interval['id']]
            if width > 1.5*reference:
                interval['role'] = 'unknown'
                interval['width_conflict'] = True
                continue
            width = min(width, reference)
        mask = np.zeros_like(ink)
        cv2.polylines(mask, [np.rint(interval['points_px']).astype(np.int32)], False, 255, max(1, min(32, round(width))))
        selected = (mask > 0) & (ink > 0) & (head_mask == 0)
        if role == 'hatching':
            selected &= 2*distance <= 1.5*hatch_widths[interval['id']]
        cls = CLASSES.index(role)
        conflicts |= selected & (pixel != 255) & (pixel != cls)
        pixel[selected & ~conflicts] = cls
    selected = (text_mask > 0) & (ink > 0) & (head_mask == 0)
    conflicts |= selected & (pixel != 255) & (pixel != 6)
    pixel[selected & ~conflicts] = 6
    pixel[conflicts] = 255
    return pixel


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
    common = {'dimension_chains': accepted_chains, 'leader_hypotheses': accepted_leaders,
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
