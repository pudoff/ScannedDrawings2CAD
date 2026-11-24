"""Bounded pixel evidence for shelf/leader paths and arrow-covered arc ends.

Only geometry enters this search. The text polygon supplies placement and masks;
neither the raw string nor its numerical value is an argument.
"""
from itertools import product
import math

import cv2
import numpy as np

from . import angular_association_v1 as core
from . import dimension_association_v3 as linear

POLICY = {
    'id': 'angular-leader-search/1.0', 'max_lines': 256, 'max_arcs': 64,
    'max_heads': 64, 'max_paths': 8, 'max_alternatives': 16,
    'max_checks_per_notation': 30000, 'max_checks_per_page': 300000,
    'max_junction_gap_px': 12, 'contact_tolerance_px': 6,
    'shelf_parallel_tolerance_deg': 12, 'shelf_text_height_factor': 2,
    'minimum_path_ink_fraction': .85, 'max_path_white_samples': 2,
    'max_extension_deg': 18, 'max_extension_px': 128,
    'max_head_length_px': 256, 'polygon_tolerance_px': 2,
    'minimum_head_fill_fraction': .85, 'max_samples_per_segment': 256,
    'max_arc_continuity_samples': 2048, 'max_arc_white_run_samples': 3,
    'nominal_used_for_selection': False, 'physical_geometry_approved': False,
    'production_admitted': False, 'training_eligible': False,
}


class Budget(core.Budget):
    def check(self, count=1):
        self.page += count
        self.notation += count
        if (self.page > POLICY['max_checks_per_page'] or
                self.notation > POLICY['max_checks_per_notation']):
            raise core.SearchLimit('leader_candidate_check_limit')


def _ink(raster, point, *, allow_head=None):
    """One native 3x3 probe; a head never licenses text or out-of-frame ink."""
    raster.budget.check()
    x, y = np.rint(point).astype(int)
    h, w = raster.gray.shape
    if not (1 <= x < w-1 and 1 <= y < h-1):
        return False
    if raster.text_excluded[y, x] or (raster.excluded[y, x] and allow_head is None):
        return False
    if raster.excluded[y, x] and allow_head is not None:
        if cv2.pointPolygonTest(allow_head, (float(point[0]), float(point[1])), True) < -2:
            return False
    raster.budget.cells += 9
    if raster.budget.cells > core.POLICY['max_pixel_cells']:
        raise core.SearchLimit('pixel_probe_limit')
    return bool((raster.gray[y-1:y+2, x-1:x+2] < 200).any())


def segment_proof(raster, first, second):
    length = float(np.linalg.norm(second-first))
    count = max(2, math.ceil(length/2)+1)
    # Do not silently undersample a long, potentially interrupted leader.
    if count > POLICY['max_samples_per_segment']:
        return None
    flags = [_ink(raster, p) for p in np.linspace(first, second, count)]
    run = longest = 0
    for value in flags:
        run = 0 if value else run+1
        longest = max(longest, run)
    if sum(flags)/count < POLICY['minimum_path_ink_fraction'] or longest > POLICY['max_path_white_samples']:
        return None
    return {'samples': count, 'ink_samples': sum(flags), 'maximum_white_run_samples': longest,
            'length_px': length, 'native_pixels': True}


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
                junction, near, far = shelf['points'][si], leader['points'][li], leader['points'][1-li]
                if np.linalg.norm(junction-near) > POLICY['max_junction_gap_px']:
                    continue
                delta = far-curve['center']
                if abs(float(np.linalg.norm(delta))-curve['radius']) > POLICY['contact_tolerance_px']:
                    continue
                angle = math.degrees(math.atan2(delta[1], delta[0]))
                offset = (angle-curve['start_angle_deg']) % 360
                if not 1 < offset < curve['sweep_angle_deg']-1:
                    continue
                proofs = [segment_proof(raster, *shelf['points']),
                          segment_proof(raster, near, far), segment_proof(raster, junction, near)]
                if not all(proofs):
                    continue
                found.append({'shelf_candidate_id': shelf['id'], 'leader_candidate_ids': [leader['id']],
                              'contact_px': far.tolist(), 'junction_px': [junction.tolist(), near.tolist()],
                              'pixel_proofs': proofs})
                if len(found) > POLICY['max_paths']:
                    raise core.SearchLimit('leader_path_limit')
    return found


def endpoint_options(curve, arrows, side, raster, budget):
    start = curve['start_angle_deg'] % 360
    end = start+curve['sweep_angle_deg']
    original = start if side == 0 else end
    result = []
    for arrow in arrows:
        budget.check()
        if arrow['confidence'] < core.POLICY['minimum_confidence']:
            continue
        delta = arrow['tip']-curve['center']
        if abs(float(np.linalg.norm(delta))-curve['radius']) > POLICY['contact_tolerance_px']:
            continue
        angle = math.degrees(math.atan2(delta[1], delta[0]))
        # Signed shortest displacement avoids turning a full circle into a tiny gap.
        difference = (original-angle+180) % 360-180 if side == 0 else (angle-original+180) % 360-180
        contact = curve['center']+curve['radius']*core.direction(original)
        if np.linalg.norm(arrow['tip']-contact) <= POLICY['contact_tolerance_px']:
            difference = 0.
        if not 0 <= difference <= POLICY['max_extension_deg']:
            continue
        resolved = original-difference if side == 0 else original+difference
        radial = core.direction(resolved)
        if abs(float(arrow['u']@np.array([-radial[1], radial[0]]))) < math.cos(math.radians(15)):
            continue
        polygon = np.asarray(arrow['polygon_px'], np.float32)
        # The admitted occluder is a filled triangle containing its reported tip.
        if len(polygon) != 3 or cv2.contourArea(polygon) < 4 or cv2.pointPolygonTest(polygon, tuple(map(float, arrow['tip'])), True) < -2:
            continue
        head_length = float(np.max(np.linalg.norm(polygon-arrow['tip'], axis=1)))
        length = curve['radius']*math.radians(difference)
        if head_length > POLICY['max_head_length_px'] or length > min(POLICY['max_extension_px'], head_length+6):
            continue
        proof = {'extension_deg': difference, 'extension_px': length, 'arrow_candidate_id': arrow['id'],
                 'covered_samples': 0, 'visible_ink_samples': 0, 'head_fill_fraction': None}
        interior = [a*polygon[0]+b*polygon[1]+(1-a-b)*polygon[2]
                    for a in np.linspace(.15, .7, 6) for b in np.linspace(.15, .7, 6) if a+b <= .85]
        fill = sum(_ink(raster, p, allow_head=polygon) for p in interior)/len(interior)
        proof['head_fill_fraction'] = fill
        if fill < POLICY['minimum_head_fill_fraction']:
            continue
        if difference:
            valid = True
            for angle in np.linspace(original, resolved, max(3, math.ceil(length)+1)):
                p = curve['center']+curve['radius']*core.direction(angle)
                covered = cv2.pointPolygonTest(polygon, tuple(map(float, p)), True) >= -POLICY['polygon_tolerance_px']
                if not _ink(raster, p, allow_head=polygon if covered else None):
                    valid = False
                    break
                proof['covered_samples' if covered else 'visible_ink_samples'] += 1
            if not valid or not proof['covered_samples']:
                continue
        result.append((resolved, proof))
        if len(result) > core.POLICY['max_arrows_per_endpoint']:
            raise core.SearchLimit('leader_arrow_support_limit')
    return result


def continuous_arc(curve, proofs, arrows, raster):
    """A mean ink score must not hide an unsupported interior gap."""
    count = max(32, math.ceil(curve['radius']*math.radians(curve['sweep_angle_deg']))+1)
    if count > POLICY['max_arc_continuity_samples']:
        raise core.SearchLimit('arc_continuity_probe_limit')
    ids = {p['arrow_candidate_id'] for p in (proofs or [])}
    polygons = [np.asarray(a['polygon_px'], np.float32) for a in arrows if a['id'] in ids]
    run = longest = 0
    for angle in np.linspace(curve['start_angle_deg'], curve['start_angle_deg']+curve['sweep_angle_deg'], count):
        p = curve['center']+curve['radius']*core.direction(angle)
        head = next((poly for poly in polygons if cv2.pointPolygonTest(poly, tuple(map(float, p)), True) >= -2), None)
        valid = _ink(raster, p, allow_head=head)
        run = 0 if valid else run+1
        longest = max(longest, run)
        if longest > POLICY['max_arc_white_run_samples']:
            return None
    return {'samples': count, 'maximum_white_run_samples': longest, 'native_pixels': True}


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
                           tuple(path['leader_candidate_ids']) if path else ())
                    previous = next((r for k, r in results if k == key), None)
                    if previous is None:
                        results.append((key, choice))
                    elif choice['complete_support'] and not previous['complete_support']:
                        previous.clear(); previous.update(choice)
                    if len(results) > POLICY['max_alternatives']:
                        raise core.SearchLimit('leader_alternative_limit')
    return [r for _, r in results]
