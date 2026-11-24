"""Recover source-bound ruling evidence on pages with crossing strokes."""
from copy import deepcopy
from itertools import combinations
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .curve_graph_v1 import graph_paths
from .frame_connected_context_v1 import candidates, connected_view, detect
from .grid_role_revision_v1 import sha
from .native_frame_v1 import side_masks
from .page_grid_attachment_v1 import analyze as attach, contact, line
from .part_admission_v1 import object_sha
from .source_line_extension_v1 import recover
from .source_observations_v2 import source_fragments
from .stroke_roles_v1 import checked_gray


POLICY = {'maximum_checks': 250_000, 'maximum_recoveries': 256, 'maximum_rules_per_corner': 64,
          'angle_deg': 8., 'maximum_frame_shift_fraction': .075, 'minimum_frame_seed_fraction': .28,
          'minimum_rule_span_fraction': .05, 'maximum_rule_span_fraction': .62,
          'minimum_exact_frame_fraction': .65}


def axes(quad):
    delta = np.roll(quad, -1, axis=0)-quad
    lengths = np.linalg.norm(delta, axis=1)
    directions = delta/lengths[:, None]
    return lengths, directions, np.c_[-directions[:, 1], directions[:, 0]]


def intersection(a, u, b, v):
    return a+u*np.linalg.solve(np.column_stack((u, -v)), b-a)[0]


def recover_line(ink, distance, fragment, audit, cache):
    token = (fragment['path_number'], *fragment['source_index_interval'])
    if token not in cache:
        if len(cache) >= POLICY['maximum_recoveries']:
            audit['limits'].append('recovery_budget')
            return None
        points = fragment['points']
        ends = points[[0, -1]].astype(float)
        delta = ends[1]-ends[0]
        major_span = float(np.linalg.norm(delta, ord=np.inf))
        if major_span < 26:
            # The short observed segment locates a sampling window. The fit
            # still reads original cross-sections and has to cover that seed.
            half_window = 20 * delta / major_span
            ends = ends.mean(0)+np.stack((-half_window, half_window))
        cache[token] = recover(ink, distance, ends, fragment['pen_px'], audit)
        if cache[token] is not None:
            candidate = cache[token]
            a, b = candidate['points']
            length = np.linalg.norm(b-a)
            u = (b-a)/length
            shift = points-a
            along = shift @ u
            offsets = abs(shift @ np.array([-u[1], u[0]]))
            if (along.min() < -math.sqrt(2) or along.max() > length+math.sqrt(2)
                    or np.quantile(offsets, .95) > candidate['width']/2+math.sqrt(.5)):
                cache[token] = None
            else:
                candidate['source_fragment'] = [int(v) for v in token]
    return cache[token]


def continuous(ink, points, width, audit):
    """Require a connected original stroke near both terminals of the line."""
    start, end = np.asarray(points, float)
    length = float(np.linalg.norm(end-start))
    if not length:
        return False
    radius = width/2+math.sqrt(2)
    low = np.maximum(np.floor(np.minimum(start, end)-radius).astype(int), 0)
    high = np.minimum(np.ceil(np.maximum(start, end)+radius).astype(int)+1, ink.shape[::-1])
    if np.any(high <= low):
        return False
    pixels = int(np.prod(high-low))
    audit['raster_pixels'] += pixels
    if audit['raster_pixels'] > 64_000_000:
        audit['limits'].append('stroke_connectivity_budget')
        return False
    y, x = np.ogrid[low[1]:high[1], low[0]:high[0]]
    dx, dy = x-start[0], y-start[1]
    u = (end-start)/length
    t = dx*u[0]+dy*u[1]
    normal = abs(-dx*u[1]+dy*u[0])
    local = (ink[low[1]:high[1], low[0]:high[0]] > 0) & (normal <= radius) & (t >= -radius) & (t <= length+radius)
    _, labels = cv2.connectedComponents(local.astype(np.uint8), connectivity=8)
    return bool((set(labels[t <= radius]) & set(labels[t >= length-radius])) - {0})


def refit_frame(ink, distance, rough, fragments, audit, cache):
    lengths, directions, normals = axes(rough)
    shift = POLICY['maximum_frame_shift_fraction']*min(lengths)
    recovered = []
    cosine = math.cos(math.radians(POLICY['angle_deg']))
    for start, unit, normal, length in zip(rough, directions, normals, lengths):
        options = []
        for fragment in fragments:
            audit['checks'] += 1
            if audit['checks'] > POLICY['maximum_checks']:
                audit['limits'].append('frame_fragment_budget')
                return None
            if abs(unit @ fragment['direction']) < cosine:
                continue
            points = fragment['points'][[0, -1]]-start
            along = points @ unit
            overlap = min(length, along.max())-max(0., along.min())
            if overlap >= POLICY['minimum_frame_seed_fraction']*length and np.max(abs(points @ normal)) <= shift:
                options.append((float(overlap), fragment))
        if not options:
            return None
        # This ordering chooses a seed for each observed side. No candidate after
        # an exhausted search budget may enter the result.
        evidence = None
        for _, fragment in sorted(options, key=lambda item: -item[0]):
            evidence = recover_line(ink, distance, fragment, audit, cache)
            if audit['limits']:
                return None
            if evidence is not None:
                break
        if evidence is None:
            return None
        recovered.append(evidence)
    points = [r['points'] for r in recovered]
    unit = [(q[1]-q[0])/np.linalg.norm(q[1]-q[0]) for q in points]
    if max(abs(unit[i-1] @ unit[i]) for i in range(4)) > math.sin(math.radians(POLICY['angle_deg'])):
        return None
    quad = np.array([intersection(points[i-1][0], unit[i-1], points[i][0], unit[i]) for i in range(4)])
    if np.max(np.linalg.norm(quad-rough, axis=1)) > shift or not cv2.isContourConvex(quad.astype(np.float32)):
        return None
    sides = []
    for i, support in enumerate(recovered):
        ends = quad[[i, (i+1)%4]]
        direction = unit[i]
        observed_span = np.sort((support['points']-ends[0]) @ direction)
        side_span = np.sort((ends-ends[0]) @ direction)
        width = support['width']
        if observed_span[0] > side_span[0]+width+math.sqrt(2) or observed_span[1] < side_span[1]-width-math.sqrt(2):
            return None
        sides.append({'points_overview': ends.tolist(), 'stroke_width_overview_px': width,
                      'position_uncertainty_overview_px': math.sqrt(.5), 'source_seed': support['source_fragment'],
                      'source_sample_count': support['source_sections']})
    return {'quad_overview': quad.tolist(), 'sides': sides, 'view_components': [],
            'frame_semantics_confirmed': False, 'evidence': 'refit_observed_straight_side_intervals'}


def side_line(side):
    return line(side['points_overview'], side['stroke_width_overview_px'], side['position_uncertainty_overview_px'])


def trace_rule(ink, fragment, origin, direction, span, audit):
    """Follow original ink through a subdivision, retaining the exact seed."""
    q = fragment['points']
    normal = np.array([-direction[1], direction[0]])
    width = fragment['pen_px']
    radius = width/2+math.sqrt(2)
    start = origin + normal * np.median((q-origin) @ normal)
    stop = POLICY['maximum_rule_span_fraction']*span + 2*radius
    ends = np.stack((start-2*radius*direction, start+stop*direction))
    low = np.maximum(np.floor(ends.min(0)-radius).astype(int), 0)
    high = np.minimum(np.ceil(ends.max(0)+radius).astype(int)+1, ink.shape[::-1])
    if np.any(high <= low) or np.any(q < low) or np.any(q >= high):
        return None
    audit['raster_pixels'] += int(np.prod(high-low))
    if audit['raster_pixels'] > 64_000_000:
        audit['limits'].append('rule_trace_raster_budget')
        return None
    y, x = np.ogrid[low[1]:high[1], low[0]:high[0]]
    dx, dy = x-start[0], y-start[1]
    along = dx*direction[0]+dy*direction[1]
    lateral = abs(dx*normal[0]+dy*normal[1])
    local = (ink[low[1]:high[1], low[0]:high[0]] > 0) & (lateral <= radius) & (along >= -2*radius) & (along <= stop)
    _, labels = cv2.connectedComponents(local.astype(np.uint8), connectivity=8)
    sx, sy = (q-low).T
    votes = np.bincount(labels[sy, sx]); votes[0] = 0
    selected = int(votes.argmax())
    if votes[selected] < .8*len(q):
        return None
    ts = along[labels == selected]
    # Stop at an observed terminal. A stroke continuing through the end of
    # the search window cannot be made into a short corner rule.
    if ts.max() >= stop-1 or np.ptp(ts) < POLICY['minimum_rule_span_fraction']*span:
        return None
    return {'points': start+np.array([ts.min(), ts.max()])[:, None]*direction, 'width': width,
            'source_fragment': [fragment['path_number'], *fragment['source_index_interval']],
            'observed_corridor_pixels': int(len(ts))}


def anchored_rule(ink, observed, origin, local_directions, local_spans, major, border, border_stroke, audit):
    if observed is None:
        return None
    ends = observed['points']
    local = (ends-origin) @ local_directions.T
    width = observed['width']
    if (np.ptp(local[:, major]) < POLICY['minimum_rule_span_fraction']*local_spans[major]
            or local[:, major].max() > POLICY['maximum_rule_span_fraction']*local_spans[major]
            or local[:, 1-major].min() <= max(2., width/2+2)):
        return None
    candidate = line(ends, width, math.sqrt(.5))
    anchor = contact(ink, candidate, border_stroke, audit, require_first_endpoint=True)
    if not anchor['connected'] or not continuous(ink, ends, width, audit):
        return None
    delta = ends[1]-ends[0]
    normal = np.array([-delta[1], delta[0]])/np.linalg.norm(delta)
    return {**candidate, 'normal': normal.tolist(), 'anchor': anchor, 'anchor_side': border,
            'source_fragment': observed['source_fragment'], 'basis_family': major,
            'corridor_observation': observed.get('observed_corridor_pixels')}


def projected_rules(ink, distance, frame, fragments, audit, cache):
    quad = np.asarray(frame['quad_overview'], float)
    lengths, directions, _ = axes(quad)
    side_strokes = [side_line(s) for s in frame['sides']]
    cosine = math.cos(math.radians(POLICY['angle_deg']))
    output = {}
    for corner in range(4):
        origin = quad[corner]
        first, second = corner, (corner-1)%4
        local_directions = np.stack((directions[first], -directions[second]))
        local_spans = np.array([lengths[first], lengths[second]])
        for major, border in ((0, second), (1, first)):
            minor = 1-major
            entries = []
            for fragment in fragments:
                audit['checks'] += 1
                if audit['checks'] > POLICY['maximum_checks']:
                    audit['limits'].append('rule_fragment_budget')
                    return {}
                if abs(fragment['direction'] @ local_directions[major]) < cosine:
                    continue
                points = fragment['points'][[0, -1]]
                p = (points-origin) @ local_directions.T
                if (p[:, minor].min() <= 2 or p[:, minor].max() >= .5*local_spans[minor]
                        or p[:, major].min() < -4 or p[:, major].max() > POLICY['maximum_rule_span_fraction']*local_spans[major]
                        or np.linalg.norm(points[1]-points[0], ord=np.inf) < 8):
                    continue
                if any(np.max(abs((points-np.asarray(e['points_source_px'])[0]) @ np.asarray(e['normal']))) <= 1.5
                       for e in entries):
                    continue
                observed = recover_line(ink, distance, fragment, audit, cache)
                if audit['limits']:
                    return {}
                rule = anchored_rule(ink, observed, origin, local_directions, local_spans, major, border,
                                     side_strokes[border], audit)
                if rule is None and not audit['limits']:
                    tracked = trace_rule(ink, fragment, origin, local_directions[major], local_spans[major], audit)
                    rule = anchored_rule(ink, tracked, origin, local_directions, local_spans, major, border,
                                         side_strokes[border], audit)
                if rule is None:
                    continue
                entries.append(rule)
                if len(entries) > POLICY['maximum_rules_per_corner']:
                    audit['limits'].append('corner_rule_budget')
                    return {}
            output[corner, major] = entries
    return output


def witnessed_span(rule, crossings):
    """Expose only the span from the frame anchor to a proved grid crossing."""
    a, b = np.asarray(rule['points_source_px'], float)
    length = float(np.linalg.norm(b-a))
    u = (b-a)/length
    anchor = float((np.asarray(rule['anchor']['intersection_source_px'])-a) @ u)
    intersections = [(np.asarray(proof['intersection_source_px'])-a) @ u for proof in crossings]
    terminal = max(intersections, key=lambda point: abs(point-anchor))
    low, high = np.clip(sorted([anchor, terminal]), 0., length)
    return {**rule, 'contact_bounded_points_source_px': (a+np.array([low, high])[:, None]*u).tolist()}


def connected_rules(ink, by_corner, audit):
    for corner in range(4):
        rows = by_corner.get((corner, 0), [])
        columns = by_corner.get((corner, 1), [])
        crossings = {}
        for i, row in enumerate(rows):
            for j, column in enumerate(columns):
                proof = contact(ink, row, column, audit, require_first_endpoint=False)
                if audit['limits']:
                    return []
                if proof['connected']:
                    crossings[i, j] = proof
        # Peel one-contact lines so a lone cross or a detached table cannot pass.
        while crossings:
            active_rows = {i for i, _ in crossings}
            active_columns = {j for _, j in crossings}
            kept_rows = {i for i in active_rows if sum((i, j) in crossings for j in active_columns) >= 2}
            kept_columns = {j for j in active_columns if sum((i, j) in crossings for i in active_rows) >= 2}
            kept = {(i, j): p for (i, j), p in crossings.items() if i in kept_rows and j in kept_columns}
            if len(kept) == len(crossings):
                adjacency = {i: {j for ii, j in kept if i == ii} for i in kept_rows}
                if any(len(adjacency[a] & adjacency[b]) >= 2 for a, b in combinations(sorted(kept_rows), 2)):
                    observed_rows = [witnessed_span(rows[i], [kept[i, j] for j in sorted(adjacency[i])])
                                     for i in sorted(kept_rows)]
                    observed_columns = [witnessed_span(columns[j], [kept[i, j] for i in sorted(kept_rows) if (i, j) in kept])
                                        for j in sorted(kept_columns)]
                    yield corner, observed_rows+observed_columns, list(kept.values())
                break
            crossings = kept


def exact_exclusion(ink, frame, rules, audit):
    exclusion = np.zeros_like(ink, dtype=bool)
    proofs = []
    for index, model in enumerate([side_line(s) for s in frame['sides']]+rules):
        points = np.asarray(model['points_source_px'], float)
        width = model['width_px']
        if not continuous(ink, points, width, audit):
            return None, proofs, 'source_stroke_not_continuous'
        fit, reason = side_masks(ink > 0, points, width, model['uncertainty_px'], audit)
        if reason and reason.endswith('_budget'):
            audit['limits'].append(reason)
        if fit is None and index < 4 and not audit['limits']:
            # A crossing may leave no single exact digital model across the
            # whole side. Use exact, disjoint subintervals of the same traced
            # side and keep unsupported pixels out of the exclusion mask.
            fit = fit_frame_pieces(ink, points, width, model['uncertainty_px'], audit)
        if fit is None:
            if index < 4 or audit['limits']:
                return None, proofs, reason
            proofs.append({'kind': 'ruling', 'source_sections': 0, 'edge_model': None,
                           'safe_pixels': 0, 'reason': reason, 'connectivity_verified': True})
            continue
        sections = fit['sections']
        if index < 4 and sections < POLICY['minimum_exact_frame_fraction']*np.ptp(points, axis=0).max():
            return None, proofs, 'incomplete_exact_frame_support'
        exclusion |= fit['possible']
        proofs.append({'kind': 'frame_side' if index < 4 else 'ruling', 'source_sections': sections,
                       'edge_model': fit['edge_model'], 'safe_pixels': int(np.count_nonzero(fit['safe']))})
    return exclusion, proofs, None


def fit_frame_pieces(ink, points, width, error, audit):
    span = np.ptp(points, axis=0).max()
    count = max(2, math.ceil(span/180))
    limits = points[0]+np.linspace(0, 1, count+1)[:, None]*(points[1]-points[0])
    safe = np.zeros_like(ink, bool)
    possible = np.zeros_like(ink, bool)
    sections = 0
    for start, stop in zip(limits, limits[1:]):
        fit, reason = side_masks(ink > 0, np.stack((start, stop)), width, error, audit)
        if reason and reason.endswith('_budget'):
            audit['limits'].append(reason)
            return None
        if fit is not None:
            safe |= fit['safe']
            possible |= fit['possible']
            sections += fit['sections']
    if sections < POLICY['minimum_exact_frame_fraction']*span:
        return None
    return {'safe': safe, 'possible': possible, 'sections': sections, 'edge_model': 'piecewise_exact_digital_edges'}


def original_closed_view(ink, quad, rules, audit):
    """Find an original white interior outside the attached ruling network."""
    audit['raster_pixels'] += ink.size
    if audit['raster_pixels'] > 64_000_000:
        audit['limits'].append('closed_view_raster_budget')
        return None
    frame = np.zeros_like(ink)
    cv2.fillConvexPoly(frame, np.rint(quad).astype(np.int32), 1)
    grid = np.zeros_like(ink)
    hull = cv2.convexHull(np.concatenate([np.asarray(r['points_source_px'], np.float32) for r in rules]))
    cv2.fillConvexPoly(grid, np.rint(hull).astype(np.int32), 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((ink == 0).astype(np.uint8), connectivity=4)
    if count-1 > 4096:
        audit['limits'].append('closed_view_component_budget')
        return None
    lengths, _, normals = axes(quad)
    span = min(lengths)
    minimum = .12*span
    frame_area = abs(cv2.contourArea(quad.astype(np.float32)))
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if min(w, h) < minimum or w*h > .4*frame_area:
            continue
        region = np.s_[y:y+h, x:x+w]
        mask = labels[region] == i
        if np.any(frame[region][mask] == 0) or np.any(grid[region][mask] > 0):
            continue
        yy, xx = np.nonzero(mask)
        points = np.c_[xx+x, yy+y]
        if min(cv2.minAreaRect(points.astype(np.float32))[1]) < minimum:
            continue
        # Reject voids closed by the external frame rather than a local view.
        if any(np.min(abs((points-start) @ normal)) <= 3 for start, normal in zip(quad, normals)):
            continue
        return {'source_bbox': [int(x), int(y), int(x+w), int(y+h)], 'original_white_pixels': int(area),
                'evidence': 'closed_original_white_interior_away_from_frame_and_grid'}
    return None


def analyze(gray):
    checked_gray(gray)
    baseline = detect(gray)
    previous = attach(gray, baseline)
    supplied = sha(gray)
    empty = {'source_sha256': supplied, 'baseline': baseline, 'baseline_observations': previous,
             'baseline_groups': previous['groups'],
             'groups': [], 'complete': previous['complete'], 'semantic_roles_changed': False}
    if previous['groups'] or not previous['complete']:
        return empty
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    distance = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5)
    paths, topology = graph_paths(skeletonize(ink > 0))
    audit = {'limits': [], 'source_points': 0, 'fragments': 0, 'checks': 0, 'source_support_samples': 0,
             'warp_pixels': 0, 'raster_pixels': 0, 'component_pixels': 0, 'source_samples': 0,
             'contacts': 0, 'contact_pixels': 0, 'decisions': []}
    if topology['search_incomplete']:
        audit['limits'].append('source_graph_budget')
    fragments = list(source_fragments(paths, cv2.distanceTransform(ink, cv2.DIST_L2, 5), audit)) if not audit['limits'] else []
    cache = {}
    frames = list(baseline['frames'])
    if not frames and not audit['limits']:
        for q in candidates(gray, audit):
            frame = refit_frame(ink, distance, q, fragments, audit, cache)
            if frame is not None:
                if not any(np.max(np.linalg.norm(np.asarray(old['quad_overview'])-frame['quad_overview'], axis=1)) < 2 for old in frames):
                    frames.append(frame)
            if audit['limits']:
                break
    found = []
    for frame in frames if not audit['limits'] else []:
        possible = projected_rules(ink, distance, frame, fragments, audit, cache)
        orders = {f'{i}-{j}': len(options) for (i, j), options in possible.items()}
        row = {'directions': orders, 'frame': frame['quad_overview'], 'candidate_groups': []}
        audit['decisions'].append(row)
        for corner, rules, crossings in connected_rules(ink, possible, audit):
            exclusion, fits, reason = exact_exclusion(ink, frame, rules, audit)
            stages = {'corner': corner, 'reason': reason, 'fit': fits}
            row['candidate_groups'].append(stages)
            if exclusion is None:
                continue
            view = connected_view(ink > 0, np.asarray(frame['quad_overview']), rules, exclusion, audit)
            if view is None and not audit['limits']:
                view = original_closed_view(ink, np.asarray(frame['quad_overview']), rules, audit)
            if view is None:
                stages['reason'] = 'no_independent_closed_view'
                continue
            found.append({'frame': frame, 'corner': corner, 'rules': rules, 'crossings': crossings,
                          'exact_source_fits': fits, 'separate_view': view})
        if audit['limits']:
            break
    if sha(gray) != supplied:
        raise ValueError('Stamp witnesses changed the source')
    return {**empty, 'groups': found if not audit['limits'] else [], 'complete': not audit['limits'], 'audit': audit,
            'policy': deepcopy(POLICY), 'evidence_sha256': object_sha(found if not audit['limits'] else [])}


def projection_context(gray, result):
    """Feed source-proven lines into the existing exact crop projection."""
    digest = sha(gray)
    if result['source_sha256'] != digest or result['semantic_roles_changed'] is not False:
        raise ValueError('Stamp context does not belong to this source')
    groups = result['groups']
    previous = result['baseline_observations']
    if not groups and result['complete']:
        if previous['source_sha256'] != digest or previous['detector_sha256'] != object_sha(result['baseline']):
            raise ValueError('Unbound legacy observations')
        return previous
    if groups and result['evidence_sha256'] != object_sha(groups):
        raise ValueError('Stamp evidence changed before projection')
    observations = {'source_sha256': digest, 'size_px': list(gray.shape[::-1]), 'complete': result['complete'],
                    'semantic_GRID_or_PART_assigned': False, 'rules': [], 'groups': []}
    for group_index, group in enumerate(groups) if result['complete'] else []:
        identities = []
        for rule_index, rule in enumerate(group['rules']):
            identity = f'observed-stamp-{group_index}-{rule_index}'
            identities.append(identity)
            observations['rules'].append({**rule, 'id': identity, 'grounded_group_member': True,
                                          'points_source_px': rule['contact_bounded_points_source_px'],
                                          'usage': 'observed_crop_context_only', 'CAD_primitive_created': False})
        observations['groups'].append({'rule_ids': identities, 'semantic_GRID_confirmed': False})
    return observations
