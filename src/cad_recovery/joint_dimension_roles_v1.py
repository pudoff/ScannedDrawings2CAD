"""Observe linear dimension sets without inventing a text reading or a scale."""

from collections import Counter
from copy import deepcopy
import itertools
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import dimension_association_v3 as shared
from .arrow_shaft_support_v1 import observe
from .contour_topology_v3 import ContourTopology
from .part_admission_v1 import object_sha
from .supported_role_overlay_v1 import source_ownership


POLICY = {'version': 'joint-dimension-roles/1.0', 'maximum_pairs': 4096,
          'maximum_samples': 500000, 'minimum_carrier_fraction': .95,
          'maximum_ownership_intervals': 4096, 'maximum_source_points': 2000000,
          'maximum_unexplained_gap': 'measured_stroke_plus_pixel_diagonal',
          'maximum_role_conflict_strokes': 2., 'minimum_anchor_support_points': 3,
          'source_radius_quantization_px': float(np.sqrt(.5)),
          'anchor_context_scale': 'source_verified_head_base_width',
          'existing_roles_immutable': True, 'physical_geometry_approved': False,
          'nominal_value_inference': False, 'supported_family': 'linear_two_directed_heads'}


class _Limit(Exception):
    pass


def _points(candidate):
    points = candidate.get('support_points', candidate.get('points_px'))
    return np.asarray(points, float) if points is not None else None


def _observed_text_mask(gray, ocr, first, axis, length, tolerance):
    mask = np.zeros_like(gray, np.uint8)
    records = []
    for text in ocr.get('lines', []):
        polygon = np.asarray(text['polygon_px'], float)
        along = (polygon-first) @ axis
        normal = np.array([-axis[1], axis[0]])
        across = (polygon-first) @ normal
        # A text observation may explain an interior interruption only. Its
        # actual ink and footprint are retained independently of OCR confidence.
        if (along.min() <= tolerance or along.max() >= length-tolerance
                or np.ptp(along) > length*.5 or across.min() > tolerance or across.max() < -tolerance):
            continue
        local = np.zeros_like(mask)
        cv2.fillConvexPoly(local, np.rint(polygon).astype(np.int32), 1)
        if np.count_nonzero(local & (gray < 128)) < 3:
            continue
        mask |= local
        records.append({'source_line_id': text['id'], 'raw': text.get('raw', ''),
                        'confidence': text.get('confidence'), 'reading_used_for_geometry': False})
    return mask > 0, records


def _span_support(a, b, distance, radius, budget, text_mask=None):
    count = max(2, int(np.ceil(np.linalg.norm(b-a)*2))+1)
    budget['samples'] += count
    if budget['samples'] > POLICY['maximum_samples']:
        raise _Limit()
    points = np.rint(np.linspace(a, b, count)).astype(int)
    height, width = distance.shape
    valid = ((points >= 0).all(1) & (points[:, 0] < width) & (points[:, 1] < height))
    observed = np.zeros(count, bool)
    exempt = np.zeros(count, bool)
    x, y = points[valid].T
    observed[valid] = distance[y, x] <= radius
    if text_mask is not None:
        exempt[valid] = text_mask[y, x]
    denominator = np.count_nonzero(~exempt)
    bad = ~observed & ~exempt
    changes = np.diff(np.r_[False, bad, False].astype(np.int8))
    lengths = np.flatnonzero(changes == -1)-np.flatnonzero(changes == 1)
    gap = float(lengths.max(initial=0)*np.linalg.norm(b-a)/(count-1))
    return {'fraction': float(np.count_nonzero(observed & ~exempt)/denominator) if denominator else 0.,
            'maximum_unexplained_gap_px': gap, 'allowed_unexplained_gap_px': float(2*radius)}


def _anchor_is_part(point, candidate, roles, ink, tolerance):
    points = _points(candidate)
    if points is None or not len(points):
        return False
    q = np.rint(points).astype(int)
    inside = ((q >= 0).all(1) & (q[:, 0] < roles.shape[1]) & (q[:, 1] < roles.shape[0]))
    q, points = q[inside], points[inside]
    selected = np.linalg.norm(points-point, axis=1) <= tolerance*2
    if not selected.any():
        return False
    observed = roles[q[selected, 1], q[selected, 0]]
    visible = ink[q[selected, 1], q[selected, 0]] > 0
    return (np.count_nonzero((observed == 1) & visible) >= POLICY['minimum_anchor_support_points']
            and not np.isin(observed, [2, 3, 4, 5, 6, 7, 8, 9]).any())


def _long_conflict(candidate, start, end, roles, role, tolerance):
    points = _points(candidate)
    if points is None or not len(points):
        return True
    delta = end-start
    length = float(np.linalg.norm(delta))
    along = (points-start) @ (delta/length)
    q = np.rint(points).astype(int)
    valid = ((q >= 0).all(1) & (q[:, 0] < roles.shape[1]) & (q[:, 1] < roles.shape[0])
             & (along >= 0) & (along <= length))
    values = roles[q[valid, 1], q[valid, 0]]
    bad = ~np.isin(values, (0, 255, role))
    positions = np.sort(along[valid][bad])
    if not len(positions):
        return False
    # Disconnected short intersections are not a longitudinal competing role.
    groups = np.split(positions, np.flatnonzero(np.diff(positions) > 2)+1)
    return any(np.ptp(group) > POLICY['maximum_role_conflict_strokes']*tolerance for group in groups)


def _source_intervals(geometry, gray):
    intervals, numbers = [], {}
    observations = geometry.get('ownership_intervals', geometry['candidates'])
    if len(observations) > POLICY['maximum_ownership_intervals']:
        return None, {}, 'source_interval_count_limit'
    total_points = 0
    seen = set()
    for candidate in observations:
        if not isinstance(candidate.get('id'), str) or candidate['id'] in seen:
            raise ValueError('Source ownership intervals require unique IDs')
        seen.add(candidate['id'])
        raw = candidate.get('support_points', candidate.get('points_px', []))
        total_points += len(raw)
        if total_points > POLICY['maximum_source_points']:
            return None, {}, 'source_point_count_limit'
        points = _points(candidate)
        if points is None or not len(points):
            continue
        if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                or (points < 0).any() or (points[:, 0] >= gray.shape[1]).any()
                or (points[:, 1] >= gray.shape[0]).any()):
            raise ValueError('Observed support must stay inside its raster')
        intervals.append({'id': candidate['id'], 'points_px': points.tolist()})
        numbers[candidate['id']] = len(intervals)
    if 'ownership_intervals' in geometry and not {g['id'] for g in geometry['candidates']}.issubset(numbers):
        raise ValueError('Complete source ownership must contain every typed candidate')
    return intervals, numbers, None


def _local_mask(owner, number, distance, source_radius, a, b, tolerance):
    selected = owner == number
    y, x = np.nonzero(selected)
    if not len(x):
        return selected
    axis = b-a
    length = float(np.linalg.norm(axis))
    axis /= length
    offset = np.column_stack((x, y))-a
    along = offset @ axis
    across = abs(offset @ np.array([-axis[1], axis[0]]))
    valid = ((along >= 0) & (along <= length) & (across <= tolerance)
             & (distance[y, x] <= source_radius))
    selected[y[~valid], x[~valid]] = False
    return selected


def propose(source_image, *, geometry, ocr, arrows, roles):
    """Only source-bound observations are accepted; reference labels are not inputs."""
    source_image = Path(source_image)
    digest, size, lines, curves, heads = shared._validate(source_image, geometry, ocr, arrows)
    with Image.open(source_image) as image:
        gray = np.asarray(image.convert('L'))
    if (roles.shape != gray.shape or roles.dtype != np.uint8
            or not np.isin(roles, list(range(10))+[255]).all()):
        raise ValueError('Aligned uint8 roles required')
    original = roles.copy()
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    distance = cv2.distanceTransform(255-ink, cv2.DIST_L2, 5)
    radius = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    head_support = observe(source_image, geometry=geometry, ocr=ocr, arrows=arrows)
    supported = {s['arrow_id']: s for s in head_support['spans']}
    head_evidence = {event['arrow_id']: event for event in head_support['events']
                     if event.get('status') == 'supported'}
    heads = sorted((h for h in heads if h['id'] in supported), key=lambda h: h['id'])
    by_id = {g['id']: g for g in geometry['candidates']}
    shaft_ids = {s['shaft_id'] for s in supported.values()}
    topology = ContourTopology(source_image, lines, excluded_ids=shaft_ids,
                               text_polygons=[r['polygon_px'] for r in ocr.get('lines', [])])
    intervals, numbers, interval_failure = _source_intervals(geometry, gray)
    if intervals is None:
        ownership, ownership_work = None, {'incomplete': True, 'reason': interval_failure}
    else:
        ownership, ownership_work = source_ownership(ink, intervals)
        if ownership is not None:
            # The shared owner builder does not seed one-vertex paths. Their
            # pixels must remain competitive instead of becoming a nearby line.
            for interval in intervals:
                if len(interval['points_px']) == 1:
                    x, y = np.rint(interval['points_px'][0]).astype(int)
                    ownership['owner'][y, x] = -1
    complete = bool(head_support['complete'] and topology.complete and not ownership_work['incomplete'])
    # Do not turn a capped upstream detector into a complete new relationship search.
    geometry_analysis = geometry.get('analysis', {})
    geometry_flags = ('contour_limit_reached', 'hough_circle_limit_reached',
                      'short_connector_limit_reached', 'regional_short_line_limit_reached',
                      'cheap_screen_limit_reached')
    if (arrows.get('analysis', {}).get('search_limited') is True
            or geometry.get('search_complete') is False or geometry.get('complete') is False
            or any(geometry_analysis.get(flag) is True for flag in geometry_flags)
            or geometry_analysis.get('validation_queue_dropped', 0) > 0
            or ocr.get('coverage', {}).get('complete') is False):
        complete = False
    budget = {'notation': 0, 'page': 0, 'samples': 0, 'pairs': 0}
    links, events = [], []
    if ownership is None:
        complete = False
    try:
        for first, second in itertools.combinations(heads, 2):
            budget['pairs'] += 1
            if budget['pairs'] > POLICY['maximum_pairs']:
                raise _Limit()
            delta = second['tip']-first['tip']
            span = float(np.linalg.norm(delta))
            if span <= 0:
                continue
            axis = delta/span
            if (float(first['u'] @ second['u']) > -shared._COS
                    or abs(float(first['u'] @ axis)) < shared._COS
                    or abs(float(second['u'] @ axis)) < shared._COS):
                continue
            stroke = max(float(first['evidence']['shaft_width_px']), float(second['evidence']['shaft_width_px']))
            tolerance = stroke + max(float(first['uncertainty_px']), float(second['uncertainty_px'])) + np.sqrt(.5)
            if span <= max(8*stroke, 4*tolerance):
                continue
            row = {'id': 'joint-dimension/'+object_sha(sorted([first['id'], second['id']]))[:20],
                   'status': 'unresolved', 'kind': 'linear', 'nominal_mm': None,
                   'arrow_candidate_ids': sorted([first['id'], second['id']]),
                   'dimension_line_candidate_ids': [], 'extension_line_candidate_ids': [],
                   'measured_candidate_ids': [], 'anchors_px': [], 'alternatives': [], 'reasons': [],
                   'evidence': {'pixel_tolerance': float(tolerance), 'measured_stroke_px': stroke,
                                'text_required_for_geometric_role': False, 'physical_geometry_approved': False}}
            links.append(row)
            baseline = {'u': axis, 'points': np.array([first['tip'], second['tip']]), 'length': span}
            intervening = [h['id'] for h in heads if h['id'] not in row['arrow_candidate_ids']
                           and 2*tolerance < float((h['tip']-first['tip']) @ axis) < span-2*tolerance
                           and shared._perp(h['tip'], baseline) <= tolerance
                           and abs(float(h['u'] @ axis)) >= shared._COS]
            if intervening:
                row['evidence']['intervening_head_ids'] = intervening
                row['reasons'].append('intervening_head_dimension_domain')
                continue
            members = [line for line in shared._budgeted(lines, budget)
                       if abs(float(line['u'] @ axis)) >= shared._COS
                       and max(shared._perp(p, baseline) for p in line['points']) <= tolerance
                       and max((line['points']-first['tip']) @ axis) >= 0
                       and min((line['points']-first['tip']) @ axis) <= span]
            owners = {supported[h['id']]['shaft_id'] for h in (first, second)}
            if not owners.issubset({line['id'] for line in members}):
                row['reasons'].append('opposed_heads_do_not_share_observed_carrier')
                continue
            text_mask, text_records = _observed_text_mask(gray, ocr, first['tip'], axis, span, tolerance)
            carrier = _span_support(first['tip'], second['tip'], distance,
                                    stroke/2+np.sqrt(.5), budget, text_mask)
            row['evidence'].update(carrier_ink_fraction=carrier['fraction'], carrier_pixel_support=carrier,
                                   observed_text=text_records)
            if (carrier['fraction'] < POLICY['minimum_carrier_fraction']
                    or carrier['maximum_unexplained_gap_px'] > carrier['allowed_unexplained_gap_px']):
                row['reasons'].append('carrier_source_ink_missing')
                continue
            if any(_long_conflict(by_id[m['id']], first['tip'], second['tip'], roles, 2, tolerance) for m in members):
                row['reasons'].append('carrier_has_confirmed_competing_role')
                continue
            excluded = {line['id'] for line in members} | shaft_ids
            # The shared helper's historical "height" argument is an
            # annotation-context scale controlling carrier endpoint tolerance.
            # Use each source-verified head footprint, not the much thinner
            # shaft or a fabricated OCR polygon. Doubling the raster then
            # doubles the supported witness gap as well as the arrow shape.
            first_scale = float(head_evidence[first['id']]['base_width_px'])
            second_scale = float(head_evidence[second['id']]['base_width_px'])
            row['evidence']['anchor_context_scale_px'] = {first['id']: first_scale, second['id']: second_scale}
            row['evidence']['anchor_context_scale_source'] = POLICY['anchor_context_scale']
            left, cut_left = shared._anchor_options(first['tip'], baseline, lines, excluded,
                {'tol': tolerance, 'height': first_scale}, budget, topology)
            right, cut_right = shared._anchor_options(second['tip'], baseline, lines, excluded,
                {'tol': tolerance, 'height': second_scale}, budget, topology)
            if cut_left or cut_right:
                complete = False
                row['reasons'].append('anchor_search_truncated')
                continue
            choices, competing_choices = [], []
            for a, b in shared._budgeted(itertools.product(left, right), budget):
                pa, ma, ea, fa = a
                pb, mb, eb, fb = b
                if ea == eb or ma == eb or mb == ea:
                    continue
                common = sorted(set(fa) & set(fb))
                if fa and fb and not common:
                    continue
                if shared._cross(pa-first['tip'], axis)*shared._cross(pb-second['tip'], axis) <= 0:
                    continue
                if not (_anchor_is_part(pa, by_id[ma], roles, ink, tolerance)
                        and _anchor_is_part(pb, by_id[mb], roles, ink, tolerance)):
                    continue
                extensions = [(first['tip'], pa, ea), (second['tip'], pb, eb)]
                competing_role = any(_long_conflict(by_id[e], tip, anchor, roles, 3, tolerance)
                                     for tip, anchor, e in extensions)
                extension_proofs = [_span_support(tip, anchor, distance, stroke/2+np.sqrt(.5), budget)
                                    for tip, anchor, _ in extensions]
                if any(p['fraction'] < POLICY['minimum_carrier_fraction']
                       or p['maximum_unexplained_gap_px'] > p['allowed_unexplained_gap_px'] for p in extension_proofs):
                    continue
                choice = shared._candidate([pa, pb], [ma, mb], [m['id'] for m in members], [ea, eb],
                                           row['arrow_candidate_ids'],
                    {'relation': 'opposed_directed_heads_observed_carrier_extensions_confirmed_part',
                     'contour_face_ids': common, 'extension_ink_fractions': [p['fraction'] for p in extension_proofs],
                     'extension_pixel_support': extension_proofs,
                     'dimension_source_members': [{'candidate_id': m['id'],
                         'source_members': deepcopy(by_id[m['id']].get('source_members', [])),
                         'claim_scope': 'observed_source_points_between_directed_tips'} for m in members],
                     'anchor_bindings': [{'arrow_id': first['id'], 'extension_id': ea, 'measured_id': ma,
                                          'tip_px': first['tip'].tolist(), 'anchor_px': pa.tolist(),
                                          'extension_source_members': deepcopy(by_id[ea].get('source_members', [])),
                                          'measured_source_members': deepcopy(by_id[ma].get('source_members', [])),
                                          'claim_scope': 'observed_source_points_between_tip_and_anchor'},
                                         {'arrow_id': second['id'], 'extension_id': eb, 'measured_id': mb,
                                          'tip_px': second['tip'].tolist(), 'anchor_px': pb.tolist(),
                                          'extension_source_members': deepcopy(by_id[eb].get('source_members', [])),
                                          'measured_source_members': deepcopy(by_id[mb].get('source_members', [])),
                                          'claim_scope': 'observed_source_points_between_tip_and_anchor'}]})
                if competing_role:
                    choice['evidence']['competing_role_on_extension'] = True
                    shared._add_linear(competing_choices, choice)
                else:
                    shared._add_linear(choices, choice)
            if competing_choices:
                row.update(status='ambiguous', alternatives=choices+competing_choices)
                row['reasons'].append('confirmed_competing_role_on_anchor_carrier')
            elif len(choices) == 1:
                evidence = row['evidence']
                row.update(choices[0], status='confirmed')
                row['evidence'].update(evidence)
            else:
                row['status'] = 'ambiguous' if choices else 'unresolved'
                row['alternatives'] = choices
                row['reasons'].append('multiple_observed_anchor_bindings' if choices else 'confirmed_part_anchor_chain_missing')
    except (_Limit, shared._SearchLimit):
        complete = False
        events.append({'reason': 'joint_search_budget_exhausted'})

    # A single head or source interval cannot silently select two different sets.
    consumers = {}
    for row in links:
        alternatives = [row] if row['status'] == 'confirmed' else row['alternatives']
        for choice in alternatives:
            for identity in choice['arrow_candidate_ids']+choice['dimension_line_candidate_ids']+choice['extension_line_candidate_ids']:
                consumers.setdefault(identity, set()).add(row['id'])
    conflicts = {key for key, ids in consumers.items() if len(ids) > 1}
    masks = []
    output = roles.copy()
    head_mask = np.zeros_like(gray, np.uint8)
    for h in arrows['candidates']:
        cv2.fillConvexPoly(head_mask, np.rint(h['polygon_px']).astype(np.int32), 1)
    for text in ocr.get('lines', []):
        cv2.fillConvexPoly(head_mask, np.rint(text['polygon_px']).astype(np.int32), 1)
    for row in links:
        if row['status'] != 'confirmed':
            continue
        identities = row['arrow_candidate_ids']+row['dimension_line_candidate_ids']+row['extension_line_candidate_ids']
        if not complete or conflicts.intersection(identities):
            row['status'] = 'ambiguous'
            row['reasons'].append('joint_search_incomplete' if not complete else 'shared_head_or_stroke_competes_between_sets')
            continue
        evidence = row['evidence']
        bindings = evidence['anchor_bindings']
        first, second = [np.asarray(b['tip_px']) for b in bindings]
        targets = [(gid, 2, first, second) for gid in row['dimension_line_candidate_ids']]
        targets += [(b['extension_id'], 3, np.asarray(b['tip_px']), np.asarray(b['anchor_px'])) for b in bindings]
        added = {2: 0, 3: 0}
        for identity, role, a, b in targets:
            number = numbers.get(identity)
            if number is None:
                continue
            points = np.rint(_points(by_id[identity])).astype(int)
            source_radius = max(.5, float(np.median(radius[points[:, 1], points[:, 0]]))) + np.sqrt(.5)
            mask = _local_mask(ownership['owner'], number, ownership['distance'], source_radius,
                               a, b, evidence['pixel_tolerance'])
            mask &= (original == 255) & (ink > 0) & (head_mask == 0)
            masks.append((row['id'], identity, role, mask))
            added[role] += int(mask.sum())
        row['role_pixels'] = {'dimension': added[2], 'extension': added[3]}
    choices = np.zeros_like(roles, np.uint16)
    for _, _, role, mask in masks:
        choices[mask] |= 1 << role
    for role in (2, 3):
        output[(original == 255) & (choices == 1 << role)] = role
    unchanged = np.array_equal(output[original != 255], original[original != 255])
    if not unchanged:
        raise ValueError('Joint role assignment changed an accepted input role')
    summary = dict(Counter(row['status'] for row in links))
    for status in ('confirmed', 'ambiguous', 'unresolved', 'unsupported'):
        summary.setdefault(status, 0)
    summary['role_pixels'] = {'dimension': int(((original == 255) & (output == 2)).sum()),
                              'extension': int(((original == 255) & (output == 3)).sum())}
    report = {'schema_version': POLICY['version'], 'source_sha256': digest, 'processed_size_px': size,
              'policy': deepcopy(POLICY), 'complete': complete, 'links': links, 'events': events,
              'summary': summary, 'head_support': head_support, 'search': budget,
              'contour_topology': topology.report(), 'source_ownership': ownership_work,
              'unpaired_supported_heads': sorted(set(supported)-{h for r in links if r['status'] == 'confirmed'
                                                               for h in r['arrow_candidate_ids']}),
              'unsupported_relations': ['single_head_leader', 'radial', 'angular', 'curved_dimension_carrier'],
              'input_known_roles_preserved': unchanged, 'input_geometry_preserved': True,
              'physical_geometry_approved': False, 'production_changed': False,
              'pixel_role_semantics': 'source_observation_candidate_requires_independent_acceptance'}
    return {'roles': output, 'report': report}
