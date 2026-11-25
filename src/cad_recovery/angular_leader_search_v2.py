"""Versioned leader search with native connected junction evidence.

Arc/head/ray checks are the frozen v1 primitives; only the local junction
strategy changes. Keeping v1 intact preserves reproducible historical runs.
"""
from itertools import product
import math
import numpy as np
from . import angular_association_v1 as core
from . import angular_leader_search_v1 as base
from . import dimension_association_v3 as linear
from . import leader_junction_v1 as junction

POLICY = dict(base.POLICY, id='angular-leader-search/2.0', junction=junction.POLICY)
Budget = junction.Budget
endpoint_options = base.endpoint_options
continuous_arc = base.continuous_arc
segment_proof = base.segment_proof

def paths(text, lines, curve, raster, budget):
    """At most one finite shelf and one finite leader; junctions need actual ink."""
    found = []
    for shelf in lines:
        budget.check()
        if abs(float(shelf['u']@text['u'])) < math.cos(math.radians(POLICY['shelf_parallel_tolerance_deg'])):
            continue
        if linear._segment_distance(text['center'], shelf) > POLICY['shelf_text_height_factor']*text['height']:
            continue
        a, b = sorted(shelf['points']@text['u'])
        ta, tb = float(np.min(text['polygon']@text['u'])), float(np.max(text['polygon']@text['u']))
        if min(b, tb)-max(a, ta) < .5*text['width']:
            continue
        for leader in lines:
            budget.check()
            if leader['id'] == shelf['id'] or abs(float(leader['u']@shelf['u'])) > math.cos(math.radians(15)):
                continue
            for si, li in product(range(2), repeat=2):
                shelf_tip, near, far = shelf['points'][si], leader['points'][li], leader['points'][1-li]
                if np.linalg.norm(shelf_tip-near) > POLICY['max_junction_gap_px']:
                    continue
                delta = far-curve['center']
                if abs(float(np.linalg.norm(delta))-curve['radius']) > POLICY['contact_tolerance_px']:
                    continue
                angle = math.degrees(math.atan2(delta[1], delta[0]))
                offset = (angle-curve['start_angle_deg']) % 360
                if not 1 < offset < curve['sweep_angle_deg']-1:
                    continue
                proofs = [segment_proof(raster, *shelf['points']), segment_proof(raster, near, far)]
                if not all(proofs):
                    continue
                local = junction.prove(shelf, leader, si, li, raster, budget)
                for proof in local:
                    found.append({'shelf_candidate_id': shelf['id'], 'leader_candidate_ids': [leader['id']],
                                  'contact_px': far.tolist(), 'junction_px': [shelf_tip.tolist(), near.tolist()],
                                  'pixel_proofs': proofs, 'junction_proof': proof})
                    if len(found) > POLICY['max_paths']:
                        raise core.SearchLimit('leader_path_limit')
    return found


def hypotheses(text, lines, curves, arrows, leader_lines, raster, budget):
    """Local derived arcs carry proof; the caller's geometry remains immutable."""
    if max(len(lines), len(leader_lines)) > POLICY['max_lines'] or len(curves) > POLICY['max_arcs'] or len(arrows) > POLICY['max_heads']:
        raise core.SearchLimit('leader_input_limit')
    results = []
    for curve in curves:
        budget.check()
        if curve['kind'] != 'arc' or not 3 < curve['sweep_angle_deg'] < 357:
            continue
        leader_paths = paths(text, leader_lines, curve, raster, budget)
        direct = core._arc_context(curve, text)
        if not direct and not leader_paths:
            continue
        first = endpoint_options(curve, arrows, 0, raster, budget)
        second = endpoint_options(curve, arrows, 1, raster, budget)
        variants = []
        # An observed leader is still an angular context when its ends are unresolved.
        if leader_paths:
            variants.append((curve, None))
        for left, right in product(first, second):
            if left[1]['arrow_candidate_id'] == right[1]['arrow_candidate_id']:
                continue
            sweep = right[0]-left[0]
            if not 3 < sweep < 357:
                continue
            variants.append((dict(curve, start_angle_deg=left[0], sweep_angle_deg=sweep), [left[1], right[1]]))
        for variant, proof in variants:
            continuity = continuous_arc(variant, proof, arrows, raster) if proof is not None else None
            if proof is not None and continuity is None:
                continue
            contexts = [(text, None)] if direct else []
            contexts += [(dict(text, center=np.asarray(path['contact_px'])), path) for path in leader_paths]
            for context, path in contexts:
                choices = core.hypotheses(context, lines, [variant], arrows, raster, budget)
                for choice in choices:
                    if proof is None:
                        # No filled, endpoint-verified occluder means this context is review-only.
                        choice['complete_support'] = False
                    if path and not path['junction_proof']['unique_ink_support']:
                        choice['complete_support'] = False
                    if path and set(choice['measured_candidate_ids']) & {path['shelf_candidate_id'], *path['leader_candidate_ids']}:
                        continue
                    if proof is None and not choice['complete_support'] and first and second:
                        # Resolved variants retain the relevant measured alternatives.
                        continue
                    choice['evidence'].update(context_basis='observed_shelf_and_leader' if path else 'direct_text_near_arc',
                                              leader_path=path, endpoint_occlusion=proof,
                                              arc_continuity=continuity,
                                              original_arc_range_deg=[curve['start_angle_deg'], curve['sweep_angle_deg']],
                                              text_inside_observed_sector=bool(direct),
                                              leader_contact_inside_observed_sector=bool(path))
                    # The same observation with and without a zero-length repair is one hypothesis.
                    key = (tuple(choice['measured_candidate_ids']), tuple(choice['arrow_candidate_ids']),
                           choice['dimension_arc_candidate_id'],
                           path['shelf_candidate_id'] if path else None,
                           tuple(path['leader_candidate_ids']) if path else (),
                           path['junction_proof']['component_id'] if path else None)
                    previous = next((r for k, r in results if k == key), None)
                    if previous is None:
                        results.append((key, choice))
                    elif choice['complete_support'] and not previous['complete_support']:
                        previous.clear(); previous.update(choice)
                    if len(results) > POLICY['max_alternatives']:
                        raise core.SearchLimit('leader_alternative_limit')
    return [r for _, r in results]
