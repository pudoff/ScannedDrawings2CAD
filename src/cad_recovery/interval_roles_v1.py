"""Role hypotheses on source-index intervals; never promote or erase CAD geometry."""
from collections import Counter, defaultdict
import math

import cv2
import numpy as np

from .stroke_roles_v1 import CLASSES, checked_gray, foreground, stroke_floor, line_distance
from .stroke_ownership_v1 import strict_text
from .ownership_contours_v1 import boundaries
from .stroke_curve_control_v1 import distance_to_polyline

POLICY = {'version': 'interval-roles/1.0', 'maximum_intervals': 4096, 'maximum_tracks': 1024,
          'maximum_pair_checks_per_stage': 250000, 'maximum_boundary_edge_tests': 2000000,
          'maximum_role_split_depth': 8, 'minimum_linear_native_px': 6.,
          'maximum_collinear_angle_deg': 8., 'head_angle_deg': 20.,
          'maximum_head_uncertainty_fraction': .5, 'maximum_observed_join_in_strokes': 3.,
          'minimum_hatch_offsets': 4, 'hatch_gap_CV_max': .5,
          'maximum_extension_overshoot_in_strokes': 4.,
          'role_status': 'proposed_only', 'semantic_ownership_confirmed': False,
          'source_pixels_erased': 0, 'production_admitted': False}


def local_width(points, distance):
    q = np.rint(points).astype(int)
    valid = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < distance.shape[1]) & (q[:, 1] < distance.shape[0])
    values = distance[q[valid, 1], q[valid, 0]]
    values = values[values > 0]
    return float(2*np.median(values)) if len(values) else None


def make_intervals(source, geometry, ink, scale, *, subdivide=True):
    originals = {c['path_id']: c for c in source}
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    output = []

    def split(points, low, high, width, depth=0):
        p = points[low:high+1]
        if len(p) < 4 or depth >= 8:
            return [(low, high)]
        if np.linalg.norm(p[-1]-p[0]) < 1e-9:
            middle = (low+high)//2
        else:
            errors = distance_to_polyline(p, p[[0, -1]])
            if float(errors.max()) <= max(1./scale, .35*width):
                return [(low, high)]
            middle = low+int(np.argmax(errors))
        if middle <= low or middle >= high:
            return [(low, high)]
        return split(points, low, middle, width, depth+1)+split(points, middle, high, width, depth+1)

    for c in geometry:
        original = originals[c['path_id']]
        lo, hi = c['source_index_interval']
        points = np.asarray(original['points_px'], float)
        width = local_width(points[lo:hi+1], distance) or 1.
        spans = split(points, lo, hi, width) if subdivide else [(lo, hi)]
        for low, high in spans:
            selected = points[low:high+1]
            width_here = local_width(selected, distance) or width
            a, b = selected[[0, -1]]
            length = float(np.linalg.norm(b-a))
            errors = distance_to_polyline(selected, [a, b]) if length > 0 else np.array([np.inf])
            linear = (length >= max(6/scale, 2*width_here)
                      and np.percentile(errors, 95) <= max(.75/scale, .35*width_here)
                      and errors.max() <= max(1.25/scale, .5*width_here))
            interval = {'id': f'{c["path_id"]}:{low}-{high}', 'path_id': c['path_id'],
                'source_observation_id': original['id'], 'parent_primitive_id': c['id'],
                'parent_kind': c['kind'], 'source_index_interval': [low, high],
                'points_px': selected.tolist(), 'width_px': width_here,
                'linear_context': bool(linear), 'role': 'unknown', 'status': 'proposed_only',
                'semantic_ownership_confirmed': False, 'physical_geometry_approved': False}
            if not subdivide:
                interval['interval_budget_unknown'] = True
            output.append(interval)
            if subdivide and len(output) > 4096:
                return make_intervals(source, geometry, ink, scale, subdivide=False)
    return output


def ink_segment(ink, a, b):
    a, b = np.asarray(a), np.asarray(b)
    count = max(2, int(math.ceil(np.linalg.norm(b-a)*2))+1)
    q = np.rint(a+np.linspace(0, 1, count)[:, None]*(b-a)).astype(int)
    valid = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < ink.shape[1]) & (q[:, 1] < ink.shape[0])
    return bool(valid.all() and np.all(ink[q[:, 1], q[:, 0]] > 0))


def tracks(intervals, ink, scale):
    lines = []
    for item in intervals:
        if not item['linear_context']:
            continue
        p = np.asarray(item['points_px'])[[0, -1]]
        u = (p[1]-p[0])/np.linalg.norm(p[1]-p[0])
        lines.append({'points': p, 'u': u, 'width': item['width_px'], 'interval_ids': [item['id']]})
    if len(lines) > 1024:
        return [], True
    parent = list(range(len(lines)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    checks = 0
    for i, a in enumerate(lines):
        for j in range(i+1, len(lines)):
            checks += 1
            if checks > 250000:
                return [], True
            b = lines[j]
            if abs(a['u']@b['u']) < math.cos(math.radians(8)):
                continue
            normal = np.array([-a['u'][1], a['u'][0]])
            width = min(a['width'], b['width'])
            if abs((b['points'].mean(axis=0)-a['points'][0])@normal) > max(1/scale, .6*width):
                continue
            options = [(np.linalg.norm(p-q), p, q) for p in a['points'] for q in b['points']]
            gap, p, q = min(options, key=lambda o: o[0])
            if gap <= 3*width and ink_segment(ink, p, q):
                parent[root(j)] = root(i)
    groups = defaultdict(list)
    for i, item in enumerate(lines):
        groups[root(i)].append(item)
    result = []
    for group in groups.values():
        u = group[0]['u']
        origin = group[0]['points'][0]
        points = np.concatenate([g['points'] for g in group])
        projections = (points-origin)@u
        span = origin+np.array([projections.min(), projections.max()])[:, None]*u
        result.append({'id': f'track-{len(result):04d}', 'points': span, 'u': u,
            'width': float(np.median([g['width'] for g in group])), 'length': float(np.ptp(projections)),
            'interval_ids': sorted({v for g in group for v in g['interval_ids']})})
    return result, False


def hatch_groups(lines, stroke):
    # Cluster fragments by orientation and perpendicular offset, not by primitive identity.
    used, output, checks = set(), [], 0
    for i, seed in enumerate(lines):
        if i in used or not 3*stroke <= seed['length'] <= 90*stroke:
            continue
        u = seed['u']
        normal = np.array([-u[1], u[0]])
        parallel = []
        for j, other in enumerate(lines):
            checks += 1
            if checks > 250000:
                return [], True
            if not 3*stroke <= other['length'] <= 90*stroke or abs(u@other['u']) < math.cos(math.radians(8)):
                continue
            along = sorted(other['points']@u)
            base = sorted(seed['points']@u)
            if min(along[1], base[1])-max(along[0], base[0]) < -4*stroke:
                continue
            offset = float(other['points'].mean(axis=0)@normal)
            if abs(offset-float(seed['points'].mean(axis=0)@normal)) <= 80*stroke:
                parallel.append((offset, j))
        parallel.sort()
        offsets = []
        for offset, j in parallel:
            if offsets and abs(offset-offsets[-1][0]) <= stroke:
                offsets[-1][1].append(j)
            else:
                offsets.append([offset, [j]])
        if len(offsets) < 4:
            continue
        gaps = np.diff([o[0] for o in offsets])
        if np.min(gaps) < 1.5*stroke or np.max(gaps) > 16*stroke or np.std(gaps)/np.mean(gaps) > .5:
            continue
        indices = sorted({j for _, js in offsets for j in js})
        hull = cv2.convexHull(np.concatenate([lines[j]['points'] for j in indices]).astype(np.float32))[:, 0, :]
        output.append({'id': f'fragment-bundle-{len(output):03d}', 'track_ids': [lines[j]['id'] for j in indices],
            'interval_ids': sorted({v for j in indices for v in lines[j]['interval_ids']}),
            'polygon': hull.tolist(), 'offset_count': len(offsets), 'direction': u.tolist(),
            'semantic_ownership_confirmed': False})
        used.update(indices)
    return output, False


class BoundaryQueries:
    def __init__(self):
        self.cache, self.work, self.incomplete = {}, 0, False

    def distance(self, point, boundary):
        key = (boundary['id'], *map(float, point))
        if key in self.cache:
            return self.cache[key]
        p = np.asarray(boundary['polygon'], float)
        self.work += len(p)
        if self.work > 2000000:
            self.incomplete = True
            return None
        vertices = np.r_[p, p[:1]]
        d = float(distance_to_polyline(np.asarray([point]), vertices)[0])
        a, b = p-np.asarray(point), np.roll(p, -1, axis=0)-point
        cross = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
        inside = np.count_nonzero((a[:, 1] <= 0) & (b[:, 1] > 0) & (cross > 0)) != np.count_nonzero((b[:, 1] <= 0) & (a[:, 1] > 0) & (cross < 0))
        value = d if inside else -d
        self.cache[key] = value
        return value


def finite_hit(a, b, tolerance, ink):
    p, q = np.asarray(a), np.asarray(b)
    u, v = p[1]-p[0], q[1]-q[0]
    denominator = float(np.linalg.det(np.stack([u, v])))
    if abs(denominator) < 1e-9:
        return None
    delta = q[0]-p[0]
    t = float(np.linalg.det(np.stack([delta, v]))/denominator)
    s = float(np.linalg.det(np.stack([delta, u]))/denominator)
    hit = p[0]+t*u
    if min(t, 1-t)*np.linalg.norm(u) < -tolerance or min(s, 1-s)*np.linalg.norm(v) < -tolerance:
        return None
    near_a, near_b = p[0]+np.clip(t, 0, 1)*u, q[0]+np.clip(s, 0, 1)*v
    return hit if ink_segment(ink, near_a, hit) and ink_segment(ink, near_b, hit) else None


def head_context(heads, mask, bundles, stroke):
    events, available = [], []
    for head in heads:
        tip = np.asarray(head['tip_px'], float)
        base = np.asarray(head['polygon_px'][1:], float).mean(axis=0)
        middle = (tip+base)/2
        uncertainty = float(head.get('uncertainty_px', 0))
        reason = None
        if uncertainty > .5*np.linalg.norm(base-tip):
            reason = 'unresolved_head_uncertainty'
        for group in bundles:
            if cv2.pointPolygonTest(np.asarray(group['polygon'], np.float32), tuple(middle), True) >= -stroke:
                direction = np.asarray(head['direction'], float).copy()
                direction /= np.linalg.norm(direction)
                if abs(direction@group['direction']) >= math.cos(math.radians(20)):
                    reason = 'head_in_fragmented_parallel_bundle'
        x, y = np.rint(middle).astype(int)
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
            reason = 'head_in_text_component'
        if reason:
            events.append({'id': head['id'], 'reason': reason, 'observation_retained': True})
        else:
            available.append(head)
    return available, events


def analyze(gray, geometry, source, heads, *, input_to_native, diagnostic_heads=()):
    checked_gray(gray)
    if len(geometry) > 4096 or len(heads)+len(diagnostic_heads) > 2048:
        raise ValueError('Bounded geometry/head observations required')
    for head in list(heads)+list(diagnostic_heads):
        if (np.asarray(head['tip_px']).shape != (2,) or np.asarray(head['polygon_px']).shape != (3, 2)
                or np.asarray(head['direction']).shape != (2,) or np.linalg.norm(head['direction']) <= 0
                or not np.isfinite([*head['tip_px'], *np.ravel(head['polygon_px']), *head['direction'], head.get('uncertainty_px', 0)]).all()):
            raise ValueError('Finite observed triangular head required')
    ink = foreground(gray)
    scale = float(input_to_native[0][0])
    intervals = make_intervals(source, geometry, ink, scale)
    lookup = {i['id']: i for i in intervals}
    lines, line_incomplete = tracks(intervals, ink, scale) if len(intervals) <= 4096 else ([], True)
    widths = [i['width_px'] for i in intervals if i['linear_context']]
    stroke = max(1., min(24., float(np.quantile(widths, .25)))) if widths else 2.
    role_geometry = []
    for item in intervals:
        p = item['points_px']
        role_geometry.append({'id': item['id'], 'path_id': item['id'],
            'kind': 'line' if item['linear_context'] else 'polyline',
            'points_px': [p[0], p[-1]] if item['linear_context'] else p,
            'closed': len(p) > 1 and p[0] == p[-1]})
    text_geometry = [g for g in role_geometry if g['kind'] == 'line']
    text = strict_text(gray, text_geometry, stroke)
    text_mask = text['component_mask'].copy()
    bundles, bundle_incomplete = hatch_groups(lines, stroke)
    usable, head_events = head_context(heads, text_mask, bundles, stroke)
    _, dt_events = head_context(list(diagnostic_heads), text_mask, bundles, stroke)
    topology = boundaries(role_geometry, gray.shape, stroke) if len(role_geometry) <= 4096 else {'boundaries': [], 'rejected': ['interval_budget'], 'search_incomplete': True}
    loops = [b for b in topology['boundaries'] if max(b['box'][2]-b['box'][0], b['box'][3]-b['box'][1]) >= .15*min(gray.shape)]
    query = BoundaryQueries()
    contacts = defaultdict(list)
    for track in lines:
        for head in usable:
            tip = np.asarray(head['tip_px'], float)
            base = np.asarray(head['polygon_px'][1:], float).mean(axis=0)
            direction = np.asarray(head['direction'], float).copy()
            direction /= np.linalg.norm(direction)
            uncertainty = float(head.get('uncertainty_px', 0))
            if abs(track['u']@direction) < math.cos(math.radians(20)):
                continue
            if line_distance(base, track['points'])[0] > 1.5*track['width']+uncertainty:
                continue
            if float(np.min((track['points']-tip)@direction)) >= -.5*np.linalg.norm(base-tip):
                continue
            projections = (base-track['points'][0])@track['u']
            near = track['points'][0]+np.clip(projections, 0, track['length'])*track['u']
            if ink_segment(ink, base, near):
                contacts[track['id']].append(head)
    headed = {v for t in lines if contacts[t['id']] for v in t['interval_ids']}
    eligible = [b for b in loops if not b['frame_hypothesis'] and not set(b['owners']) & headed]
    roles = defaultdict(set)
    chains, chain_signatures, confirmed = [], set(), set()
    checks = 0
    for shaft in lines:
        attached = contacts[shaft['id']]
        if not attached:
            continue
        u = shaft['u']
        origin = shaft['points'][0]
        projections = [0., shaft['length'], *[float((np.asarray(h['tip_px'])-origin)@u) for h in attached]]
        semantic = origin+np.array([min(projections), max(projections)])[:, None]*u
        options = []
        for extension in lines:
            if extension is shaft or abs(extension['u']@u) > .25:
                continue
            checks += 1
            if checks > 250000:
                break
            tolerance = max(shaft['width'], extension['width']) + max(float(h.get('uncertainty_px', 0)) for h in attached)
            hit = finite_hit(semantic, extension['points'], tolerance, ink)
            if hit is None or min(np.linalg.norm(hit-p) for p in semantic) > 3*shaft['width']+tolerance:
                continue
            for boundary in eligible:
                outside = query.distance(shaft['points'].mean(axis=0), boundary)
                if outside is None or outside >= -shaft['width']:
                    continue
                x, y, r, b = boundary['box']
                feet = []
                for identifier in extension['interval_ids']:
                    for endpoint in (lookup[identifier]['points_px'][0], lookup[identifier]['points_px'][-1]):
                        p = np.asarray(endpoint)
                        if not (x-tolerance <= p[0] <= r+tolerance and y-tolerance <= p[1] <= b+tolerance):
                            continue
                        value = query.distance(p, boundary)
                        if value is not None and abs(value) <= tolerance and np.linalg.norm(p-hit) >= 3*extension['width']:
                            feet.append(p)
                if feet:
                    foot = min(feet, key=lambda p: np.linalg.norm(p-hit))
                    direction = (foot-hit)/np.linalg.norm(foot-hit)
                    chosen = []
                    for identifier in extension['interval_ids']:
                        midpoint = np.asarray(lookup[identifier]['points_px']).mean(axis=0)
                        value = query.distance(midpoint, boundary)
                        projection = float((midpoint-hit)@direction)
                        if value is not None and value < -.25*extension['width'] and -4*extension['width'] <= projection <= np.linalg.norm(foot-hit):
                            chosen.append(identifier)
                    if chosen:
                        options.append((extension, boundary, hit, foot, chosen))
        for i, first in enumerate(options):
            for second in options[i+1:]:
                if first[0]['id'] == second[0]['id'] or first[1]['id'] != second[1]['id']:
                    continue
                if np.linalg.norm(first[2]-second[2]) < .6*shaft['length']:
                    continue
                signature = (shaft['id'], tuple(sorted((first[0]['id'], second[0]['id']))), tuple(first[1]['owners']))
                if signature in chain_signatures:
                    continue
                chain_signatures.add(signature)
                for identifier in shaft['interval_ids']:
                    roles[identifier].add('dimension_line')
                for identifier in first[4]+second[4]:
                    roles[identifier].add('extension_line')
                confirmed.add(first[1]['id'])
                chains.append({'shaft_track': shaft['id'], 'shaft_interval_ids': shaft['interval_ids'],
                    'extension_interval_ids': [first[4], second[4]], 'boundary_id': first[1]['id'],
                    'head_ids': [h['id'] for h in attached], 'visible_heads': len(attached),
                    'feet_px': [first[3].tolist(), second[3].tolist()], 'status': 'geometric_hypothesis_only'})
        if not any(roles[i] for i in shaft['interval_ids']):
            for head in attached:
                tip = np.asarray(head['tip_px'])
                back = shaft['points'][np.argmax(np.linalg.norm(shaft['points']-tip, axis=1))]
                for boundary in eligible:
                    near, far = query.distance(tip, boundary), query.distance(back, boundary)
                    if near is not None and far is not None and abs(near) <= 2*shaft['width'] and far < -2*shaft['width']:
                        for identifier in shaft['interval_ids']:
                            roles[identifier].add('leader_line')
                        confirmed.add(boundary['id'])
    for bundle in bundles:
        for boundary in eligible:
            values = [query.distance(np.asarray(lookup[i]['points_px']).mean(axis=0), boundary) for i in bundle['interval_ids']]
            if values and all(v is not None and v > stroke for v in values):
                for identifier in bundle['interval_ids']:
                    roles[identifier].add('hatching')
                confirmed.add(boundary['id'])
                break
    for boundary in loops:
        if set(boundary['owners']) & headed:
            continue
        members = [lookup[i] for i in boundary['owners'] if i in lookup]
        thick_closed = members and np.median([i['width_px'] for i in members]) >= 1.35*stroke
        if boundary['frame_hypothesis'] or boundary['id'] in confirmed or thick_closed:
            role = 'sheet_frame' if boundary['frame_hypothesis'] else 'part_contour'
            for identifier in boundary['owners']:
                if not roles[identifier]:
                    roles[identifier].add(role)
    # Reject whole component groups if a proposed text component touches structural ownership.
    _, labels = cv2.connectedComponents(ink, connectivity=8)
    protected_components = set()
    for interval in intervals:
        if not roles[interval['id']]:
            continue
        p = np.rint(interval['points_px']).astype(int)
        protected_components.update(int(v) for v in labels[p[:, 1], p[:, 0]] if v)
    rejected_text = []
    for group in text['groups']:
        if protected_components & set(group['component_ids']):
            text_mask[np.isin(labels, group['component_ids'])] = 0
            rejected_text.append({'component_ids': group['component_ids'], 'reason': 'touches_structural_role_interval'})
    pixel = np.full(gray.shape, 255, np.uint8)
    conflicts = np.zeros_like(gray, bool)
    head_mask = np.zeros_like(gray)
    for head in list(heads)+list(diagnostic_heads):
        cv2.fillConvexPoly(head_mask, np.rint(head['polygon_px']).astype(np.int32), 255)
    incomplete = line_incomplete or bundle_incomplete or topology['search_incomplete'] or query.incomplete or checks > 250000 or len(intervals) > 4096 or any(i.get('interval_budget_unknown') for i in intervals)
    for interval in intervals:
        hints = roles[interval['id']]
        p = np.rint(interval['points_px']).astype(int)
        text_fraction = float(np.mean(text_mask[p[:, 1], p[:, 0]] > 0))
        if text_fraction >= .5:
            hints.add('text')
        role = next(iter(hints)) if len(hints) == 1 and not incomplete else 'unknown'
        interval['role'], interval['alternatives'] = role, sorted(hints)
        if role == 'unknown':
            continue
        mask = np.zeros_like(gray)
        cv2.polylines(mask, [p], False, 255, max(1, min(32, round(interval['width_px']))))
        selected = (mask > 0) & (ink > 0) & (head_mask == 0)
        cls = CLASSES.index(role)
        conflicts |= selected & (pixel != 255) & (pixel != cls)
        pixel[selected & ~conflicts] = cls
    selected = (text_mask > 0) & (ink > 0) & (head_mask == 0)
    conflicts |= selected & (pixel != 255) & (pixel != 6)
    pixel[selected & ~conflicts] = 6
    pixel[conflicts] = 255
    return {'proposals': intervals, 'pixel_roles': pixel, 'text_mask': text_mask,
        'role_counts': dict(Counter(i['role'] for i in intervals)), 'dimension_chains': chains,
        'boundaries': topology, 'tracks': [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in t.items()} for t in lines],
        'hatch_bundles': bundles, 'head_events': head_events, 'DT_context_events': dt_events,
        'head_ids_used': [h['id'] for h in usable], 'head_observations_retained': len(heads),
        'DT_admitted_for_roles': False, 'text': {'groups': text['groups'], 'rejected': text['rejected']+rejected_text},
        'stroke_px': stroke, 'boundary_edge_tests': min(query.work, 2000000), 'boundary_budget_exhausted': query.incomplete,
        'search_incomplete': bool(incomplete), 'source_pixels_erased': 0, 'semantic_ownership_confirmed': False,
        'physical_geometry_approved': False, 'production_admitted': False}
