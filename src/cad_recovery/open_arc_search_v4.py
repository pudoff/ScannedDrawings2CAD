"""Event priority and contour diversity within the original finite arc budgets."""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path

import cv2
import numpy as np

from . import open_arc_search_v3 as previous
from .arc_final_validation_v1 import validate
from .family_catalog_v2 import require

POLICY = deepcopy(previous.POLICY)
POLICY.update(version='event-diverse-open-arcs/4.0', queue_profile='events_diverse',
    maximum_screen_boundary_samples=512*2*180, maximum_screen_line_tests=512*2500,
    maximum_discovery_radial_samples=(512+4*96)*720*65)


def isolated(name, sha):
    path = Path(__file__).with_name(name+'.py')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == sha, 'Frozen arc dependency changed')
    spec = importlib.util.spec_from_file_location('cad_recovery._'+name+'_queue_v4', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def ordered(rows, profile):
    """Within each evidence tier, visit source contours round-robin."""
    ranked = sorted(rows, key=lambda r: (-r['event_count'], -r['priority'], r['seed_index']))
    if profile == 'events_only': return ranked
    output = []
    for tier in (2, 1, 0):
        buckets = defaultdict(deque)
        for row in ranked:
            if row['event_count'] == tier: buckets[row['seed']['contour_id']].append(row)
        contours = sorted(buckets, key=lambda k: (-buckets[k][0]['priority'], k))
        while contours:
            for key in contours: output.append(buckets[key].popleft())
            contours = [key for key in contours if buckets[key]]
    return output


def search(gray, *, stroke_px, lines=(), text_polygons=(), queue_profile='events_diverse'):
    require(queue_profile in ('events_only', 'events_diverse'), 'Unknown arc queue profile')
    require(isinstance(gray, np.ndarray) and gray.ndim == 2 and gray.dtype == np.uint8 and
        0 < gray.size <= POLICY['maximum_pixels'], 'Expected bounded uint8 gray raster')
    require(np.isfinite(stroke_px) and 1 <= stroke_px <= 24 and len(lines) <= 2500, 'Invalid stroke or line budget')
    for line in lines:
        if line.get('kind') == 'line':
            points = np.asarray(line['points_px'])
            require(points.shape == (2, 2) and np.isfinite(points).all() and
                np.linalg.norm(points[1]-points[0]) > 0 and line.get('id'), 'Invalid tangent line')
    for polygon in text_polygons:
        p = np.asarray(polygon)
        require(p.ndim == 2 and p.shape[1] == 2 and len(p) >= 3 and np.isfinite(p).all(), 'Invalid OCR polygon')
    kernel = isolated('open_arc_search_v1', '37db4562213098669e4a52f3aa68a2b9af16854e47679b79eda41ab14a9f3308')
    tangent = isolated('open_arc_search_v2', '98a2b8c341cc33a785ec1ba7882e7061dd08bbac38578a49face8018a97d9370')
    kernel.POLICY = deepcopy(POLICY); tangent.POLICY = deepcopy(POLICY)
    threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    seeds, stats = kernel.contour_seeds(mask); queue = []
    costs = {'screen_radial_samples': 0, 'refinement_radial_samples': 0,
        'screen_boundary_samples': 0, 'screen_line_tests': 0}
    limits = stats['limits']; phase = ['screen']; original_radial = kernel.radial_profile

    def radial(mask, center, radius, stroke):
        band = min(POLICY['maximum_radial_band_px'], max(8, math.ceil(stroke*2), math.ceil(radius*.03)))
        cost = POLICY['angular_samples']*(2*band+1)
        # Same theoretical maximum as 512 old screens plus 96*4 old refinement profiles.
        require(sum(costs[k] for k in ('screen_radial_samples', 'refinement_radial_samples'))+cost
            <= POLICY['maximum_discovery_radial_samples'], 'Discovery radial sample bound exceeded')
        costs[phase[0]+'_radial_samples'] += cost
        return original_radial(mask, center, radius, stroke)
    kernel.radial_profile = radial
    for index, seed in enumerate(seeds):
        profile = radial(mask, np.asarray(seed['center']), seed['radius'], stroke_px)
        eligible = not profile['closed_or_nearly_closed'] and bool(profile['runs'])
        row = {'seed_index': index, 'seed': seed, 'state': 'awaiting_refinement' if eligible else 'screen_rejected',
            'priority': None, 'event_count': 0}
        queue.append(row)
        if not eligible: continue
        first, length = max(profile['runs'], key=lambda r: r[1]); step = 2*np.pi/POLICY['angular_samples']
        row['priority'] = float(length*seed['radius'])
        # These are ranking hints at an unrefined circle, never a hard rejection.
        line_cost = sum(l.get('kind') == 'line' for l in lines)
        if costs['screen_line_tests']+line_cost > POLICY['maximum_screen_line_tests']:
            if 'screen_line_budget' not in limits: limits.append('screen_line_budget')
            row['screen_events'] = ['unexamined', 'unexamined']; continue
        costs['screen_line_tests'] += line_cost
        start, sweep, tangents = kernel.tangent_endpoints(np.asarray(seed['center']), seed['radius'],
            first*step, (length-1)*step, lines, stroke_px, profile['band'])
        events = []
        for endpoint, pos, outward in ((0, first, -1), (1, (first+length-1) % POLICY['angular_samples'], 1)):
            if tangents[endpoint]: events.append('tentative_tangent'); continue
            cost = POLICY['endpoint_boundary_probes']
            if costs['screen_boundary_samples']+cost > POLICY['maximum_screen_boundary_samples']:
                if 'screen_boundary_budget' not in limits: limits.append('screen_boundary_budget')
                events.append('unexamined'); continue
            costs['screen_boundary_samples'] += cost  # Upper bound: width jumps may avoid the disk entirely.
            angle = start+(sweep if endpoint else 0)
            point = np.asarray(seed['center'])+seed['radius']*np.array([math.cos(angle), math.sin(angle)])
            events.append(kernel.endpoint_event(mask, point, stroke_px, profile, pos, outward) or 'unsupported')
        row.update(screen_events=events, event_count=sum(e not in ('unsupported', 'unexamined') for e in events))
    ranked = ordered([r for r in queue if r['priority'] is not None], queue_profile)
    for rank, row in enumerate(ranked, 1): row['rank'] = rank
    distance = cv2.distanceTransform(255-mask, cv2.DIST_L2, 3)
    candidates = []; tangent_budget = {'used': 0, 'exhausted': False}; final_budget = {'used': 0, 'limited': False}
    phase[0] = 'refinement'
    selected_rows = ranked[:POLICY['maximum_refinements']] if POLICY['maximum_candidates'] > 0 else []
    if POLICY['maximum_candidates'] == 0: limits.append('candidate_budget')
    for row in selected_rows:
        row['state'] = 'refined'
        curve, reason = kernel.refine_seed(mask, row['seed'], stroke_px, lines, distance, text_polygons)
        if curve is not None:
            witnesses = [tangent.tangent_witness(mask, curve, i, stroke_px, tangent_budget)
                for i, method in enumerate(curve['evidence']['endpoint_methods']) if method == 'straight_tangent']
            row['tangent_witnesses'] = witnesses
            if not all(w['accepted'] for w in witnesses):
                curve = None; reason = 'tangent_profile_budget' if tangent_budget['exhausted'] else 'unsupported_native_tangent'
            else:
                curve['evidence']['native_tangent_witnesses'] = witnesses
        if curve is not None:
            valid, reason, evidence = validate(mask, curve, stroke_px, POLICY, final_budget, original_radial)
            row.update(evidence)
            if not valid: curve = None
            else: curve['evidence'].update(evidence)
        if curve is not None:
            duplicate = any(np.linalg.norm(np.asarray(curve['center_px'])-c['center_px']) <= max(3, stroke_px)
                and abs(curve['radius_px']-c['radius_px']) <= max(3, stroke_px)
                and abs(curve['sweep_angle_deg']-c['sweep_angle_deg']) < 10
                and abs((curve['start_angle_deg']-c['start_angle_deg']+180) % 360-180) < 10 for c in candidates)
            if duplicate: curve = None; reason = 'duplicate_arc'
        if curve is not None:
            curve['id'] = f'open-arc-{len(candidates)+1:05d}'
            row['candidate_id'] = curve['id']; reason = 'accepted'; candidates.append(curve)
        row['refinement_reason'] = reason
        if len(candidates) >= POLICY['maximum_candidates']: limits.append('candidate_budget'); break
    if any(r['state'] == 'awaiting_refinement' for r in ranked): limits.append('refinement_budget')
    if tangent_budget['exhausted']: limits.append('tangent_profile_budget')
    if final_budget['limited']: limits.append('final_centerline_budget')
    stats.update(queue=queue, screened_seeds=len(queue), eligible_seeds=len(ranked),
        refinements=sum(r['state'] == 'refined' for r in queue),
        pending_refinements=sum(r['state'] == 'awaiting_refinement' for r in queue),
        refinement_rejections=dict(Counter(r['refinement_reason'] for r in queue
            if r['state'] == 'refined' and r['refinement_reason'] != 'accepted')),
        costs=costs, tangent_profile_samples=tangent_budget['used'], final_centerline_samples=final_budget['used'])
    policy = deepcopy(POLICY); policy['queue_profile'] = queue_profile
    return {'schema_version': 'open-arc-search/3.0', 'policy': policy, 'source_size_px': [gray.shape[1], gray.shape[0]],
        'otsu_threshold': float(threshold), 'source_stroke_width_px': float(stroke_px), 'candidates': candidates,
        'analysis': stats, 'search_incomplete': bool(limits), 'production_admitted': False,
        'physical_geometry_approved': False, 'CAD_promotions': 0}
