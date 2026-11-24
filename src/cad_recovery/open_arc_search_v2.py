"""Native tangent continuation and bounded queue witnesses for open arcs.

The historical search is loaded into a private namespace on every invocation.
Its proposal order and budgets remain unchanged. No reference ROI enters search.
"""
from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

from . import open_arc_search_v1 as base
from .arrow_detection import _profile, MARGIN, PROFILE_RADIUS
from .family_catalog_v2 import require

BASE_SHA256 = '37db4562213098669e4a52f3aa68a2b9af16854e47679b79eda41ab14a9f3308'
POLICY = deepcopy(base.POLICY)
POLICY.update(version='native-tangent-open-arcs/2.0',
    minimum_tangent_native_support=.9, tangent_span_min_px=24,
    tangent_span_max_px=96, tangent_span_in_radii=.75,
    maximum_tangent_profile_samples=3000000)


def tangent_witness(mask, curve, endpoint, stroke, budget):
    """Check the outward ray, beyond the junction, against thin centred ink."""
    radius = curve['radius_px']
    angle = math.radians(curve['start_angle_deg'] + endpoint*curve['sweep_angle_deg'])
    unit = np.array([-math.sin(angle), math.cos(angle)]) * (-1 if endpoint == 0 else 1)
    point = np.asarray(curve['center_px']) + radius*np.array([math.cos(angle), math.sin(angle)])
    span = max(POLICY['tangent_span_min_px'], min(POLICY['tangent_span_max_px'],
               radius*POLICY['tangent_span_in_radii']))
    cost = (2*PROFILE_RADIUS+1)*(math.ceil(span)+2*MARGIN+1)
    if budget['used']+cost > POLICY['maximum_tangent_profile_samples']:
        budget['exhausted'] = True
        return {'endpoint': endpoint, 'accepted': False, 'reason': 'tangent_profile_budget'}
    budget['used'] += cost
    along, widths, centers = _profile(mask, point, unit, span)
    selected = (along >= max(6, 2*stroke)) & (along <= span)
    positions = point + along[selected, None]*unit
    # A ray clipped by the native raster never counts as negative evidence
    # hidden behind remap padding, nor as a supported tangent.
    in_bounds = ((positions >= 0).all(axis=1) &
                 (positions <= [mask.shape[1]-1, mask.shape[0]-1]).all(axis=1))
    good = ((widths[selected] > 0) & (widths[selected] <= max(6, 1.8*stroke)) &
            (np.abs(centers[selected]) <= max(1.5, .4*stroke)) & in_bounds)
    fraction = float(good.mean()) if len(good) else 0.
    return {'endpoint': endpoint, 'point_px': point.tolist(), 'outward_unit': unit.tolist(),
        'span_px': span, 'junction_excluded_px': max(6, 2*stroke),
        'sample_count': int(selected.sum()), 'supported_samples': int(good.sum()),
        'native_support': fraction, 'all_samples_in_bounds': bool(in_bounds.all()),
        'accepted': bool(len(good) and fraction >= POLICY['minimum_tangent_native_support']),
        'reason': 'native_straight_continuation' if fraction >= POLICY['minimum_tangent_native_support']
                  else 'unsupported_native_tangent'}


def search(gray, *, stroke_px, lines=(), text_polygons=()):
    path = Path(__file__).with_name('open_arc_search_v1.py')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == BASE_SHA256, 'Frozen arc producer changed')
    spec = importlib.util.spec_from_file_location('cad_recovery._native_arc_kernel_v2', path)
    kernel = importlib.util.module_from_spec(spec); spec.loader.exec_module(kernel)
    kernel.POLICY = deepcopy(POLICY)
    queue = []; active = [None]
    budget = {'used': 0, 'exhausted': False}
    original_seeds, original_radial, original_refine = kernel.contour_seeds, kernel.radial_profile, kernel.refine_seed
    by_identity = {}

    def seeds(mask):
        proposed, stats = original_seeds(mask)
        for index, seed in enumerate(proposed):
            row = {'seed_index': index, 'seed': deepcopy(seed), 'state': 'not_screened'}
            queue.append(row); by_identity[id(seed)] = row
        return proposed, stats

    def radial(mask, center, radius, stroke):
        result = original_radial(mask, center, radius, stroke)
        if active[0] is None:
            row = queue[sum(r['state'] != 'not_screened' for r in queue)]
            eligible = not result['closed_or_nearly_closed'] and bool(result['runs'])
            row.update(state='awaiting_refinement' if eligible else 'screen_rejected',
                screen_reason='eligible' if eligible else ('closed_ring' if result['closed_or_nearly_closed'] else 'no_open_run'),
                priority=float(max((n for _, n in result['runs']), default=0)*radius) if eligible else None)
        return result

    def refine(mask, seed, stroke, supporting_lines, distance, polygons):
        row = by_identity[id(seed)]; active[0] = row
        try:
            curve, reason = original_refine(mask, seed, stroke, supporting_lines, distance, polygons)
            if curve is not None:
                witnesses = [tangent_witness(mask, curve, endpoint, stroke, budget)
                    for endpoint, method in enumerate(curve['evidence']['endpoint_methods']) if method == 'straight_tangent']
                row['tangent_witnesses'] = witnesses
                if not all(w['accepted'] for w in witnesses):
                    curve = None
                    reason = 'tangent_profile_budget' if budget['exhausted'] else 'unsupported_native_tangent'
                else:
                    curve['evidence']['native_tangent_witnesses'] = witnesses
            row.update(state='refined', refinement_reason=reason)
            return curve, reason
        finally:
            active[0] = None

    kernel.contour_seeds, kernel.radial_profile, kernel.refine_seed = seeds, radial, refine
    result = kernel.search(gray, stroke_px=stroke_px, lines=lines, text_polygons=text_polygons)
    ranked = sorted((r for r in queue if r['priority'] is not None), key=lambda r: -r['priority'])
    for rank, row in enumerate(ranked, 1): row['rank'] = rank
    for row in queue:
        if row['state'] == 'refined' and row['refinement_reason'] == 'proposed_open_arc':
            candidate = next((c for c in result['candidates'] if c['evidence']['seed'] == row['seed']), None)
            row['refinement_reason'] = 'accepted' if candidate else 'duplicate_arc'
            row['candidate_id'] = candidate['id'] if candidate else None
    if budget['exhausted']: result['analysis']['limits'].append('tangent_profile_budget')
    result['analysis'].update(queue=queue,
        pending_refinements=sum(r['state'] == 'awaiting_refinement' for r in queue),
        unexamined_window_proposals=result['analysis']['window_proposals']-result['analysis']['window_fits'],
        tangent_profile_samples=budget['used'], base_code_sha256=BASE_SHA256)
    result['schema_version'] = 'open-arc-search/2.0'
    result['search_incomplete'] = bool(result['analysis']['limits'])
    return result
