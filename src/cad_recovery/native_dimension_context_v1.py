"""Observe dimension carriers and real anchor context in original PNG windows.

Working labels are priors, never enlarged into original-pixel proof. This stage
does not rewrite accepted roles: a complete geometric relation is still a
proposal until its ownership and any role revision are independently admitted.
"""
from copy import deepcopy
import hashlib
import itertools
import math
from pathlib import Path

import cv2
import numpy as np

from .native_head_bridge_v1 import build
from .native_dimension_strokes_v1 import observe as observe_lines
from .native_dimension_arc_v1 import observe as observe_arcs
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import (checked_matrix, digest, local_lines,
    source_windows, transform, verify)


POLICY = {'version': 'native-dimension-context/1.0', 'maximum_pairs': 4096,
          'maximum_windows': 32, 'maximum_roi_pixels': 4_000_000,
          'maximum_context_observations': 32, 'maximum_extension_rechecks': 128,
          'maximum_total_roi_pixels': 32_000_000, 'maximum_window_side': 4096,
          'minimum_direction_dot': .98, 'minimum_visible_fraction': .95,
          'roles_changed': False, 'physical_geometry_approved': False,
          'complete_dimension_acceptance': False}


def _line(points):
    p = np.asarray(points, float)
    delta = p[1]-p[0]
    length = float(np.linalg.norm(delta))
    return p, delta/length, length


def _distance(point, ends):
    p, u, length = _line(ends)
    along = np.clip((point-p[0]) @ u, 0, length)
    return float(np.linalg.norm(point-p[0]-along*u))


def _intersection(first, second):
    a, u, _ = _line(first)
    b, v, _ = _line(second)
    if abs(float(np.linalg.det(np.column_stack((u, v))))) < .1:
        return None
    t, _ = np.linalg.solve(np.column_stack((u, -v)), b[0]-a[0])
    return a[0]+t*u


def _native_geometry(geometry, matrix):
    scale = float(np.linalg.svd(matrix[:2, :2], compute_uv=False).max())
    result = []
    for c in geometry['candidates']:
        row = deepcopy(c)
        row['prior_working_role'] = c.get('role', 255)
        row['uncertainty_px'] = float(c.get('uncertainty_px', 1.))*scale + math.sqrt(.5)*scale
        row['source_projection_quantization_px'] = math.sqrt(.5)*scale
        row['width_px'] = float(c.get('width_px', 1.))*scale
        if c['kind'] == 'line':
            row['points_px'] = transform(c['points_px'], matrix).tolist()
        elif c['kind'] in ('arc', 'circle'):
            row['center_px'] = transform(c['center_px'], matrix).tolist()
            row['radius_px'] = float(c['radius_px'])*math.sqrt(abs(np.linalg.det(matrix[:2, :2])))
            if c['kind'] == 'arc':
                start = c['start_angle_deg']
                if np.linalg.det(matrix[:2, :2]) < 0:
                    start += c['sweep_angle_deg']
                u = matrix[:2, :2] @ np.array([math.cos(math.radians(start)), math.sin(math.radians(start))])
                row['start_angle_deg'] = math.degrees(math.atan2(u[1], u[0])) % 360
        else:
            continue
        if c.get('support_points'):
            row['support_points'] = transform(c['support_points'], matrix).tolist()
        result.append(row)
    return result


def _curve_groups(curves):
    """Join only explicitly common fitted circle families, retaining gaps."""
    groups = {}
    for curve in curves:
        family = curve.get('circle_family_support')
        key = object_sha([curve['center_px'],curve['radius_px'],family]) if family else curve['id']
        groups.setdefault(key,[]).append(curve)
    output=[]
    for members in groups.values():
        if len(members)==1 or any(c['kind']=='circle' for c in members):
            output.extend(deepcopy(members)); continue
        row=deepcopy(members[0]); intervals=[]
        for c in members:
            start=c['start_angle_deg']%360
            intervals.append([start,c['sweep_angle_deg']])
        # Start the envelope after the largest uncovered angular gap.
        angles=sorted({v%360 for a,b in intervals for v in (a,a+b)})
        gaps=[]
        for a,b in zip(angles,angles[1:]+[angles[0]+360]):
            middle=(a+b)/2%360
            if not any((middle-s)%360 <= sweep for s,sweep in intervals): gaps.append((b-a,b%360))
        gap,start=max(gaps,default=(0.,0.))
        row.update(id='source-arc-family/'+object_sha(sorted(c['id'] for c in members))[:20],
            start_angle_deg=start,sweep_angle_deg=360-gap,
            input_angular_intervals_deg=intervals,
            input_curve_ids=[c['id'] for c in members],
            source_curve_ids=[c['id'] for c in members],
            uncertainty_px=max(c['uncertainty_px'] for c in members),
            source_members=[m for c in members for m in c.get('source_members',[])])
        output.append(row)
    return output


def _pairs(heads, lines, size, curves=()):
    rows, boxes, limits = [], [], []
    for index, (first, second) in enumerate(itertools.combinations(heads, 2)):
        if index >= POLICY['maximum_pairs']:
            limits.append('head_pair_budget')
            break
        a, b = np.asarray(first['tip_px']), np.asarray(second['tip_px'])
        span = float(np.linalg.norm(b-a))
        if span <= 0:
            continue
        axis = (b-a)/span
        if (np.dot(first['direction'], second['direction']) > -POLICY['minimum_direction_dot']
                or any(abs(np.dot(h['direction'], axis)) < POLICY['minimum_direction_dot'] for h in (first, second))):
            continue
        stroke = max(h['evidence']['shaft_width_px'] for h in (first, second))
        tolerance = stroke+max(h['uncertainty_px'] for h in (first, second))+math.sqrt(.5)
        row = {'id': 'source-dimension/'+object_sha([first['id'], second['id']])[:20],
               'head_ids': [first['id'], second['id']], 'status': 'unresolved', 'reasons': [],
               'native_tip_span_px': span, 'initial_tolerance_px': tolerance,
               'nominal_value': None, 'reading_used_for_geometry': False}
        rows.append(row)
        inside = [h['id'] for h in heads if h['id'] not in row['head_ids']
                  and 2*tolerance < float((np.asarray(h['tip_px'])-a) @ axis) < span-2*tolerance
                  and abs(float((np.asarray(h['tip_px'])-a) @ np.array([-axis[1], axis[0]]))) <= tolerance
                  and abs(np.dot(h['direction'], axis)) >= POLICY['minimum_direction_dot']]
        if inside:
            row.update(reasons=['intervening_source_head'], intervening_head_ids=inside)
            endpoint_context=max(np.linalg.norm(np.asarray(h['tip_px'])-np.asarray(h['polygon_px'])[1:].mean(0))
                                 for h in (first,second))+2*tolerance
            if any(endpoint_context < float((np.asarray(h['tip_px'])-a)@axis) < span-endpoint_context
                   for h in heads if h['id'] in inside):
                continue
        seeds = []
        for head in (first, second):
            length = np.linalg.norm(np.asarray(head['tip_px'])-np.asarray(head['polygon_px'])[1:].mean(0))
            reach = length+2*head['uncertainty_px']+2*stroke
            candidates = [line for line in lines
                if abs(np.dot(_line(line['points_px'])[1], axis)) < .2
                and _distance(np.asarray(head['tip_px']), line['points_px']) <= reach]
            seeds.append(candidates)
        row['extension_seed_ids'] = [[c['id'] for c in group] for group in seeds]
        if not all(seeds):
            row['reasons'].append('source_extension_seed_missing')
            continue
        cloud = [a, b]+[p for h in (first,second) for p in h['polygon_px']]
        cloud += [p for group in seeds for c in group for p in c['points_px']]
        # Context extends beyond projected endpoints, so a crop boundary cannot
        # masquerade as the first physical termination of a witness line.
        margin = max(16*stroke, 4*tolerance)
        lo = np.maximum(0, np.floor(np.min(cloud, axis=0)-margin)).astype(int)
        hi = np.minimum(size, np.ceil(np.max(cloud, axis=0)+margin+1)).astype(int)
        seed_lo,seed_hi=lo.copy(),hi.copy()
        for curve in curves:
            center=np.asarray(curve['center_px']); radius=curve['radius_px']
            if (center+radius >= seed_lo).all() and (center-radius < seed_hi).all():
                # Include both sides of a possible tangent, not just the near
                # half of a curve clipped by a projected witness endpoint.
                lo=np.maximum(0,np.minimum(lo,np.floor(center-radius-margin))).astype(int)
                hi=np.minimum(size,np.maximum(hi,np.ceil(center+radius+margin+1))).astype(int)
        box = [*lo.tolist(), *hi.tolist()]
        pixels = int(np.prod(hi-lo))
        if pixels > POLICY['maximum_roi_pixels'] or max(hi-lo) > POLICY['maximum_window_side']:
            row['reasons'].append('native_context_window_budget')
            limits.append('native_context_window_budget')
            continue
        if box not in boxes:
            if len(boxes) >= POLICY['maximum_windows'] or sum((r-l)*(b-t) for l,t,r,b in boxes)+pixels > POLICY['maximum_total_roi_pixels']:
                row['reasons'].append('native_context_total_budget')
                limits.append('native_context_total_budget')
                continue
            boxes.append(box)
        row['source_box_px'] = box
    return rows, boxes, limits


def _span(ink, a, b, radius, *, distance=None):
    if distance is None:
        distance = cv2.distanceTransform(255-ink, cv2.DIST_L2, 5)
    extent = float(np.linalg.norm(b-a))
    q = np.rint(np.linspace(a, b, max(2, math.ceil(extent)+1))).astype(int)
    inside = ((q >= 0).all(1) & (q[:,0] < ink.shape[1]) & (q[:,1] < ink.shape[0]))
    seen = np.zeros(len(q), bool)
    seen[inside] = distance[q[inside,1],q[inside,0]] <= radius
    changes = np.diff(np.r_[False, ~seen, False].astype(np.int8))
    gaps = np.flatnonzero(changes == -1)-np.flatnonzero(changes == 1)
    gap = float(gaps.max(initial=0)*extent/(len(q)-1))
    return {'fraction': float(seen.mean()), 'maximum_gap_px': gap,
            'allowed_gap_px': float(2*radius), 'passed': bool(seen.mean() >= POLICY['minimum_visible_fraction'] and gap <= 2*radius)}


def _head_on_extension(head, extension, gray, lines, text):
    ends = np.asarray(extension['points_px'])
    old_tip = np.asarray(head['tip_px'])
    axis_line = [old_tip, old_tip+np.asarray(head['direction'])]
    point = _intersection(axis_line, ends)
    if point is None or _distance(point, ends) > extension['width_px']:
        return None
    base = np.asarray(head['polygon_px'])[1:].mean(0)
    delta = point-base
    length = float(np.linalg.norm(delta))
    shift = float(np.linalg.norm(point-old_tip))
    old_length = float(np.linalg.norm(old_tip-base))
    if length <= 0 or shift > old_length or np.dot(delta/length, head['direction']) < .98:
        return None
    candidate = deepcopy(head)
    candidate.update(tip_px=point.tolist(), direction=(delta/length).tolist())
    candidate['polygon_px'][0] = point.tolist()
    checked, reason = verify(candidate, gray, lines, text, head['evidence']['shaft_width_px'])
    if checked is not None:
        owners=checked['evidence']['geometry_ids']
        if len(owners)!=1:
            checked,reason=None,'multiple_refitted_source_shaft_owners'
        elif set(owners)!=set(head['evidence']['geometry_ids']):
            checked,reason=None,'refitted_source_shaft_owner_changed'
        else:
            checked['evidence']['tip_revision_not_admitted_to_detector']=True
            checked['evidence']['tip_uncertainty_independently_calibrated']=False
    result = {'extension_id': extension['id'], 'intersection_px': point.tolist(),
              'old_tip_px': old_tip.tolist(), 'tip_shift_px': shift,
              'old_uncertainty_px': head['uncertainty_px'],
              'old_tip_consistent': bool(shift <= head['uncertainty_px']+extension['width_px']/2+math.sqrt(.5)),
              'source_shape_rechecked': checked is not None, 'recheck_reason': reason,
              'extension_source_profile': deepcopy(extension['source_profile']),
              'candidate': checked, 'published_to_head_detector': False}
    return result


def _crossings(extension, lines, tip):
    result = []
    for line in lines:
        if line['id'] == extension['id']:
            continue
        point = _intersection(extension['points_px'], line['points_px'])
        if point is None:
            continue
        tolerance = max(line['width_px'], extension['width_px'])+math.sqrt(.5)
        if _distance(point, line['points_px']) > tolerance or _distance(point, extension['points_px']) > tolerance:
            continue
        e, eu, length = _line(extension['points_px'])
        along = float((point-e[0]) @ eu)
        # Measured continuation on both sides excludes this crossing as a
        # terminal witness, even when its prior role says PART.
        through = tolerance < along < length-tolerance
        result.append({'candidate_id': line['id'], 'point_px': point.tolist(),
            'distance_from_tip_px': float(np.linalg.norm(point-tip)),
            'witness_continues_both_sides': through,
            'same_thin_stroke_continuation_confirmed': False,
            'continuation_scope': 'observed_geometry_only_not_semantic_witness_ownership',
            'terminal_anchor': False, 'reason': 'observed_through_crossing' if through else 'line_terminal_requires_part_witness',
            'source_profile': deepcopy(line['source_profile'])})
    return sorted(result, key=lambda r:(r['distance_from_tip_px'],r['candidate_id']))


def inspect(original_source, working_image, *, geometry, ocr, roles, source_heads, working_to_source):
    """Return original-frame evidence and unresolved revisions; never mutate inputs."""
    original_source, working_image = Path(original_source), Path(working_image)
    matrix = checked_matrix(working_to_source)
    bridge = build(original_source, working_image, geometry=geometry, ocr=ocr,
                   source_heads=source_heads, working_to_source=matrix)
    if roles.dtype != np.uint8 or list(roles.shape[::-1]) != geometry['processed_size_px']:
        raise ValueError('Aligned working role map required')
    if not np.isin(roles, list(range(10))+[255]).all():
        raise ValueError('Unknown role code')
    supported_ids = {r['id'].removeprefix('native/') for r in bridge['arrows']['candidates']}
    heads = [deepcopy(h) for h in source_heads['verified_candidates'] if h['id'] in supported_ids]
    native = _native_geometry(geometry, matrix)
    lines = [c for c in native if c['kind'] == 'line']
    rows, boxes, limits = _pairs(heads, lines, source_heads['processed_size_px'],
                                [c for c in native if c['kind'] in ('arc','circle')])
    windows = source_windows(original_source, source_heads['source_sha256'], source_heads['processed_size_px'], boxes) if boxes else []
    windows = dict(zip(map(tuple, boxes), windows))
    head_by_id = {h['id']:h for h in heads}
    native_by_id = {c['id']:c for c in native}
    working_by_id = {c['id']:c for c in geometry['candidates']}
    processed,rechecks=0,0
    for row in rows:
        if 'source_box_px' not in row:
            continue
        if processed>=POLICY['maximum_context_observations']:
            row['reasons'].append('context_observation_budget'); limits.append('context_observation_budget')
            continue
        processed+=1
        box = row['source_box_px']; origin = np.asarray(box[:2]); gray = windows[tuple(box)]
        row['source_window_sha256'] = hashlib.sha256(gray.tobytes()).hexdigest()
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        background_distance=cv2.distanceTransform(255-ink, cv2.DIST_L2, 5)
        local = local_lines(lines, box)
        seeds = deepcopy(local)
        for head_id, group in zip(row['head_ids'], row['extension_seed_ids']):
            head=head_by_id[head_id]; tip=np.asarray(head['tip_px'])-origin
            for candidate in seeds:
                if candidate['id'] not in group:
                    continue
                p,u,length=_line(candidate['points_px'])
                along=float(np.clip((tip-p[0])@u,0,length))
                reach=max(24.,32*head['evidence']['shaft_width_px'])
                ends=np.array([max(0.,along-reach),min(length,along+reach)])
                candidate['points_px']=(p[0]+ends[:,None]*u).tolist()
        observed = observe_lines(gray, seeds)
        row['stroke_observation'] = observed
        if not observed['complete']:
            row['reasons'].append('native_stroke_observation_incomplete')
            limits.extend(observed['audit']['limits'])
            continue
        observed_by_id = {c['id']:c for c in observed['lines']}
        owner_lines=deepcopy(local)
        for candidate in owner_lines:
            fitted=observed_by_id.get(candidate['id'])
            if fitted is None: continue
            p,u,_=_line(fitted['points_px'])
            # Use the measured transverse location only. The original seed's
            # longitudinal interval still limits ownership of a rechecked head.
            source=np.asarray(candidate['points_px'])
            along=(source-p[0])@u
            candidate['points_px']=(p[0]+along[:,None]*u).tolist()
            candidate['uncertainty_px']=fitted['uncertainty_px']
        curves=[]
        for c in native:
            if c['kind'] not in ('arc','circle'): continue
            center=np.asarray(c['center_px'])-origin
            if (center+c['radius_px']<0).any() or (center-c['radius_px']>=gray.shape[::-1]).any(): continue
            current=deepcopy(c); current['center_px']=center.tolist(); curves.append(current)
        curves=_curve_groups(curves)
        text = [(transform(t['polygon_px'],matrix)-origin).tolist() for t in ocr.get('lines',[])]
        bindings = []
        local_heads = []
        for hid, seeds in zip(row['head_ids'], row['extension_seed_ids']):
            head = deepcopy(head_by_id[hid])
            head['tip_px'] = (np.asarray(head['tip_px'])-origin).tolist()
            head['polygon_px'] = (np.asarray(head['polygon_px'])-origin).tolist()
            previous_evidence=head['evidence']
            head['evidence']={k:deepcopy(previous_evidence[k]) for k in
                ('shaft_width_px','geometry_ids','source_members') if k in previous_evidence}
            head['evidence']['measurement_coordinate_space']='source_window_pixels'
            head['source_ref']={'sha256':source_heads['source_sha256'],
                'coordinate_space':'source_window_pixels','box_original_px':box}
            local_heads.append(head)
            options = []
            for eid in seeds:
                if eid not in observed_by_id:
                    continue
                if rechecks>=POLICY['maximum_extension_rechecks']:
                    row['reasons'].append('extension_recheck_budget'); limits.append('extension_recheck_budget')
                    break
                rechecks+=1
                extension = observed_by_id[eid]
                # Refitting a line may observe a longer stroke, but does not
                # grant its original owner all newly observed intervals.
                option = _head_on_extension(head, extension, gray, owner_lines, text)
                if option is None:
                    continue
                tip = np.asarray(option['intersection_px'])
                option['crossings'] = _crossings(extension, observed['lines'], tip)
                axis=np.asarray(head['direction'])
                arc_result=observe_arcs(gray,curves,extension,axis)
                option['arc_observation']=arc_result
                option['arc_anchors']=arc_result['anchors']
                for anchor in option['arc_anchors']:
                    anchor['source_curve_ids']=next(c.get('source_curve_ids',[c['id']]) for c in curves if c['id']==anchor['curve_id'])
                    anchor['prior_working_roles']={gid:native_by_id[gid]['prior_working_role'] for gid in anchor['source_curve_ids']}
                    anchor['extension_pixel_support']=_span(ink,tip,np.asarray(anchor['anchor_px']),extension['width_px']/2+math.sqrt(.5),distance=background_distance)
                    anchor['semantic_part_confirmed']=False
                if not arc_result['complete']:
                    limits.extend(arc_result['audit']['limits'])
                options.append(option)
            bindings.append({'head_id':hid, 'extension_options':options})
        row['head_extension_bindings'] = bindings
        choices=[[o for o in b['extension_options'] if o['source_shape_rechecked']] for b in bindings]
        if all(len(group)==1 for group in choices):
            revised=[np.asarray(group[0]['intersection_px']) for group in choices]
            separation=float(np.linalg.norm(revised[1]-revised[0]))
            widths=[head['evidence']['shaft_width_px'] for head in local_heads]
            row['refined_tip_pair']={'tips_original_px':[(p+origin).tolist() for p in revised],
                'span_px':separation,'source_shape_rechecked':True,'confidence_recalibrated':False,
                'published_to_head_detector':False}
            if separation<=max(widths)+math.sqrt(.5):
                row['reasons'].append('refined_heads_share_one_transverse_junction')
            else:
                row['carrier_pixel_support_after_tip_revision']=_span(ink,*revised,max(widths)/2+math.sqrt(.5),distance=background_distance)
        a,b = [np.asarray(h['tip_px']) for h in local_heads]
        row['carrier_pixel_support_before_tip_revision'] = _span(ink,a,b,
            max(h['evidence']['shaft_width_px'] for h in local_heads)/2+math.sqrt(.5),distance=background_distance)
        conflict = []
        owner_ids = sorted({gid for h in local_heads for gid in h['evidence']['geometry_ids']})
        for gid in owner_ids:
            points = np.rint(working_by_id[gid].get('support_points',working_by_id[gid]['points_px'])).astype(int)
            q = transform(points,matrix)-origin
            axis=(b-a)/np.linalg.norm(b-a); along=(q-a)@axis
            chosen=points[(along>=0)&(along<=np.linalg.norm(b-a))]
            selected=chosen[roles[chosen[:,1],chosen[:,0]]==1]
            if len(selected):
                conflict.append({'candidate_id':gid,'working_PART_points':selected.tolist(),
                                 'prior_is_native_semantic_proof':False})
        row['carrier_prior_conflicts'] = conflict
        row['status'] = 'source_context_observed'
        row['reasons'].append('native_anchor_and_role_revision_not_yet_admitted')
    if digest(original_source) != source_heads['source_sha256']:
        raise ValueError('Original changed during dimension context observation')
    return {'schema_version':POLICY['version'], 'policy':deepcopy(POLICY),
        'source_sha256':source_heads['source_sha256'], 'working_source_sha256':geometry['source_sha256'],
        'geometry_sha256':object_sha(geometry), 'ocr_sha256':object_sha(ocr),
        'source_heads_sha256':object_sha(source_heads), 'roles_sha256':hashlib.sha256(roles.tobytes()).hexdigest(),
        'working_to_source':matrix.tolist(), 'coordinate_space':'source_window_pixels_with_explicit_box_original_px',
        'native_bridge':bridge['report'], 'contexts':rows, 'limits':sorted(set(limits)),
        'work':{'processed_contexts':processed,'extension_rechecks':rechecks},
        'complete':not limits and bridge['support']['complete'], 'roles_changed':0,
        'physical_geometry_approved':False, 'production_changed':False,
        'working_role_map_used_as_native_evidence':False}
