"""Partial source-owned role proposals from independently checked local context.

No context rectangle is painted. Structural and guarded candidates share the
same raster/intervals; neither consumes annotation boxes, names or targets.
"""
from copy import deepcopy
import math
import cv2
import numpy as np
from skimage.morphology import skeletonize
from . import grid_role_revision_v1 as grids
from .boundary_role_overlay_v1 import boundary_witnesses
from .stroke_roles_v1 import checked_gray, foreground
from .supported_role_overlay_v1 import source_ownership

POLICY = {'version': 'source-region-roles/1.0', 'maximum_pixels': 4000000,
          'maximum_intervals': 4096, 'maximum_components': 4096, 'maximum_grids': 64,
          'maximum_circle_roi_pixels': 2000000, 'minimum_circle_veto_count': 3,
          'maximum_enclosure_checks': 250000, 'enclosure_samples_per_grid': 512,
          'minimum_rule_to_glyph_width': 1.4, 'maximum_width_q95_to_q25': 1.5,
          'roles': ['part_contour', 'text', 'title_grid'],
          'editable_roles': ['unknown', 'unreviewed_CV_hatching'],
          'production_admitted': False, 'physical_geometry_approved': False}
ARMS = ('previous', 'structure', 'guarded')


def straight_source(item, radius):
    points = np.asarray(item['points_px'], float)
    if len(points) < 8 or radius <= 0:
        return False
    center = points.mean(0)
    _, _, vectors = np.linalg.svd(points-center, full_matrices=False)
    axis = vectors[0]
    along = (points-center) @ axis
    stroke = max(1., 2*radius-1)
    error = np.quantile(np.abs((points-center) @ np.array([-axis[1], axis[0]])), .95)
    return bool(np.ptp(along) >= 8*stroke and error <= max(.75, .25*stroke))


def enclosing_thick_outline(grid, ink, ownership, items, number, low, audit):
    """A thin mesh inside a thicker observed outline is not a title-grid cue."""
    points = np.concatenate([np.asarray(items[i]['points_px'], float) for i in grid['interval_ids']])
    if len(points) > POLICY['enclosure_samples_per_grid']:
        points = points[np.linspace(0, len(points)-1, POLICY['enclosure_samples_per_grid']).astype(int)]
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ids = [number[i] for i in grid['interval_ids']]
    widths = [2*low[i]-1 for i in ids if low[i] > 0]
    rule_width = float(np.median(widths)) if widths else None
    if rule_width is None:
        return []
    evidence = []
    for contour in contours:
        if abs(cv2.contourArea(contour)) < .8*(grid['local_box'][2]-grid['local_box'][0])*(grid['local_box'][3]-grid['local_box'][1]):
            continue
        audit['enclosure_checks'] += len(points)
        if audit['enclosure_checks'] > POLICY['maximum_enclosure_checks']:
            audit['limits'].append('enclosure_sample_budget')
            return []
        fraction = float(np.mean([cv2.pointPolygonTest(contour, tuple(p), False) >= 0 for p in points]))
        if fraction < .95:
            continue
        q = contour.reshape(-1, 2)
        owners = ownership['owner'][q[:, 1], q[:, 0]]
        outside = [int(i) for i in np.unique(owners) if i > 0 and i not in ids and low[i] > 0]
        boundary_width = float(np.median([2*low[i]-1 for i in outside])) if outside else None
        if boundary_width is not None and boundary_width >= POLICY['minimum_rule_to_glyph_width']*rule_width:
            evidence.append({'boundary_source_owner_numbers': outside, 'outline_width_px': boundary_width,
                             'rule_width_px': rule_width, 'sampled_enclosure_fraction': fraction})
    return evidence


def component_rows(facts):
    options = []
    for swap in (False, True):
        eligible = []
        for f in facts:
            box = f['box'] if not swap else [f['box'][1], f['box'][0], f['box'][3], f['box'][2]]
            width, height = box[2]-box[0], box[3]-box[1]
            # A thick rule must not set the minimum size of nearby thin text.
            if max(5., 1.5*f['stroke']) <= height <= min(120., 24*f['stroke']) and .15 <= width/height <= 2.2:
                eligible.append({**f, 'box': box, 'height': height})
        candidates = {}
        for seed in eligible:
            peers = [g for g in eligible if .6 <= g['height']/seed['height'] <= 1.6
                     and abs(g['box'][3]-seed['box'][3]) <= .5*seed['height']]
            peers.sort(key=lambda g: g['box'][0])
            groups = []
            for g in peers:
                if groups and g['box'][0]-groups[-1][-1]['box'][2] <= 2*max(g['height'], groups[-1][-1]['height']):
                    groups[-1].append(g)
                else:
                    groups.append([g])
            for group in groups:
                if len(group) >= 3:
                    candidates[tuple(sorted(g['id'] for g in group))] = group
        rows, seen = [], set()
        for group in sorted(candidates.values(), key=lambda gs: (np.median([g['box'][3] for g in gs]), tuple(g['id'] for g in gs))):
            baseline = float(np.median([g['box'][3] for g in group]))
            height = float(np.median([g['height'] for g in group]))
            if seen & {g['id'] for g in group} or any(abs(baseline-r['baseline']) < .8*height for r in rows):
                continue
            rows.append({'component_ids': [g['id'] for g in group], 'baseline': baseline, 'height': height})
            seen.update(g['id'] for g in group)
        widths = [g['stroke'] for g in eligible if g['id'] in seen]
        options.append({'rows': rows, 'glyph_count': len(seen), 'glyph_component_ids': sorted(seen),
                        'glyph_stroke_px': float(np.median(widths)) if widths else None, 'swapped_axes': swap})
    best = max(options, key=lambda p: (len(p['rows']), p['glyph_count']))
    best['supported'] = len(best['rows']) >= 2 and best['glyph_count'] >= 6
    best['height_normalization'] = 'each_component_observed_stroke_not_grid_rule_width'
    return best


def glyph_context(grid, ink, labels, stats, distance, items, audit):
    attached = set()
    for identity in grid['interval_ids']:
        points = np.rint(items[identity]['points_px']).astype(int)
        attached.update(labels[points[:, 1], points[:, 0]].tolist())
    u, n = np.asarray(grid['u']), np.asarray(grid['n'])
    left, top, right, bottom = grid['local_box']
    facts = []
    for number in range(1, len(stats)):
        if number in attached:
            continue
        x, y, w, h, area = map(int, stats[number])
        if x <= 0 or y <= 0 or x+w >= ink.shape[1] or y+h >= ink.shape[0] or max(w, h) > 192 or area < 6:
            continue
        center = np.array([x+w/2, y+h/2])
        if not (left < center @ u < right and top < center @ n < bottom):
            continue
        audit['glyph_roi_pixels'] += w*h
        if audit['glyph_roi_pixels'] > 2000000:
            audit['limits'].append('glyph_roi_budget')
            break
        local = labels[y:y+h, x:x+w] == number
        skeleton = skeletonize(local, method='zhang').astype(np.uint8)*255
        degree = grids.POPCOUNT[grids.adjacency_bits(skeleton)]
        ends = int(np.count_nonzero((skeleton > 0) & (degree == 1)))
        branches = int(np.count_nonzero((skeleton > 0) & (degree >= 3)))
        holes = cv2.connectedComponents(np.pad((~local).astype(np.uint8), 1, constant_values=1), connectivity=4)[0]-2
        if ends < 1 or not (branches or holes):
            continue
        yy, xx = np.nonzero(local)
        points = np.column_stack((xx+x, yy+y))
        a, b = points @ u, points @ n
        sy, sx = np.nonzero(skeleton)
        stroke = float(np.quantile(2*distance[y+sy, x+sx]-1, .25))
        facts.append({'id': number, 'box': [float(a.min()), float(b.min()), float(a.max()+1), float(b.max()+1)],
                      'stroke': max(1., stroke), 'endpoints': ends, 'branches': branches, 'holes': int(holes)})
        if len(facts) > 256:
            audit['limits'].append('glyph_count_budget')
            break
    return component_rows(facts)


def circular_context(grid, ink, labels, stats, items, audit):
    """Repeated closed round components veto the automatic table interpretation."""
    attached = set()
    for identity in grid['interval_ids']:
        points = np.rint(items[identity]['points_px']).astype(int)
        attached.update(labels[points[:, 1], points[:, 0]].tolist())
    u, n = np.asarray(grid['u']), np.asarray(grid['n'])
    left, top, right, bottom = grid['local_box']
    rings = []
    for number in range(1, len(stats)):
        if number in attached:
            continue
        x, y, w, h, _ = map(int, stats[number])
        if min(w, h) < 6 or max(w, h) > 192 or not .8 <= w/h <= 1.25:
            continue
        s, t = np.array([x+w/2, y+h/2]) @ u, np.array([x+w/2, y+h/2]) @ n
        if not (left < s < right and top < t < bottom):
            continue
        audit['circle_roi_pixels'] += w*h
        if audit['circle_roi_pixels'] > POLICY['maximum_circle_roi_pixels']:
            audit['limits'].append('circle_roi_budget')
            return []
        mask = (labels[y:y+h, x:x+w] == number).astype(np.uint8)*255
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        outer = [i for i, row in enumerate(hierarchy[0]) if row[3] == -1]
        holes = [i for i, row in enumerate(hierarchy[0]) if row[3] != -1]
        if len(outer) != 1 or len(holes) != 1:
            continue
        contour = contours[outer[0]].reshape(-1, 2).astype(float)
        (cx, cy), radius = cv2.minEnclosingCircle(contour.astype(np.float32))
        area = abs(cv2.contourArea(contours[outer[0]]))
        hole = abs(cv2.contourArea(contours[holes[0]]))
        if radius < 3 or not .15 <= hole/max(area, 1) <= .9:
            continue
        error = float(np.quantile(np.abs(np.linalg.norm(contour-[cx, cy], axis=1)-radius), .95))
        if error <= max(.75, .12*radius):
            rings.append({'component_id': number, 'center_px': [x+cx, y+cy],
                          'radius_px': radius, 'radial_p95_px': error})
    return rings


def analyze(gray, graph, roles, *, authority=None):
    checked_gray(gray)
    if gray.size > POLICY['maximum_pixels'] or len(graph['proposals']) > POLICY['maximum_intervals']:
        raise ValueError('Bounded source raster and intervals required')
    previous = grids.analyze(gray, graph, roles, authority=authority)
    base = previous['guarded']
    permitted = grids.check_authority(gray, roles, authority)
    output = {a: base.copy() for a in ARMS}
    owners = {a: np.zeros(gray.shape, np.uint16) for a in ARMS[1:]}
    audit = {'pair_checks': 0, 'cell_checks': 0, 'glyph_roi_pixels': 0, 'circle_roi_pixels': 0,
             'enclosure_checks': 0, 'limits': []}
    decisions, part_decisions = [], []
    if not np.any(permitted & ((base == 255) | (base == 9))):
        return {**output, 'owners': owners, 'decisions': [], 'part_decisions': [], 'audit': audit,
                'complete': previous['complete'], 'production_admitted': False,
                'physical_geometry_approved': False, 'source_pixels_erased': 0}
    ink = foreground(gray)
    found = grids.grids(ink, graph, audit)
    if len(found) > POLICY['maximum_grids']:
        audit['limits'].append('grid_budget')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count-1 > POLICY['maximum_components']:
        audit['limits'].append('component_budget')
    ownership, work = source_ownership(ink, graph['proposals'])
    audit['source_ownership'] = work
    if work.get('incomplete'):
        audit['limits'].append('source_ownership_budget')
    claims = {a: np.zeros(gray.shape, np.uint16) for a in ARMS[1:]}
    items = {p['id']: p for p in graph['proposals']}
    if len(items) != len(graph['proposals']):
        raise ValueError('Unique source interval IDs required')
    if ownership is not None and not audit['limits']:
        distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        number = {p['id']: i for i, p in enumerate(graph['proposals'], 1)}
        low, high = np.zeros(len(items)+1), np.zeros(len(items)+1)
        for identity, item in items.items():
            points = np.rint(item['points_px']).astype(int)
            measured = distance[points[:, 1], points[:, 0]]
            measured = measured[measured > 0]
            if len(measured) >= 8:
                low[number[identity]], high[number[identity]] = np.quantile(measured, [.25, .95])
        nearest = ownership['owner']
        safe = np.clip(nearest, 0, len(items))
        supported = ((nearest > 0) & (ownership['distance'] <= low[safe]+math.sqrt(.5)) & (ink > 0))
        editable = permitted & ((base == 255) | (base == 9)) & supported
        grid_ids, glyph_ids = set(), set()
        for grid in found:
            context = glyph_context(grid, ink, labels, stats, distance, items, audit)
            rings = circular_context(grid, ink, labels, stats, items, audit)
            grid_ids.update(grid['interval_ids'])
            glyph_ids.update(context['glyph_component_ids'])
            ids = [number[i] for i in grid['interval_ids']]
            stable = [i for i in ids if low[i] > 0 and high[i] <= POLICY['maximum_width_q95_to_q25']*low[i]]
            thick = [i for i in stable if context['supported'] and
                     max(0., 2*low[i]-1) >= POLICY['minimum_rule_to_glyph_width']*context['glyph_stroke_px']]
            straight = [number[i] for i in grid['interval_ids'] if straight_source(items[i], low[number[i]])]
            thick = sorted(set(thick) & set(straight))
            circle_veto = len(rings) >= POLICY['minimum_circle_veto_count']
            enclosures = enclosing_thick_outline(grid, ink, ownership, items, number, low, audit)
            if context['supported']:
                claims['structure'][editable & np.isin(nearest, ids)] |= 1 << 8
                text_pixels = editable & np.isin(labels, context['glyph_component_ids'])
                claims['structure'][text_pixels] |= 1 << 6
                # Text candidates stay separate from the interpretation of rules.
                claims['guarded'][text_pixels] |= 1 << 6
                if not circle_veto and not enclosures:
                    claims['guarded'][editable & np.isin(nearest, thick)] |= 1 << 8
            decisions.append({**grid, 'glyph_context': context, 'circular_components': rings,
                              'table_veto_repeated_circles': circle_veto,
                              'table_veto_thick_enclosure': enclosures,
                              'straight_source_owner_numbers': straight,
                              'stable_source_owner_numbers': stable, 'thick_source_owner_numbers': thick,
                              'role_authority': 'partial_CV_context_proposal_only', 'bbox_filled': False})
            if audit['limits']:
                break
        part_claims, rejected, limited = ({}, [], False)
        if all(k in graph for k in ('boundaries', 'dimension_chains', 'stroke_px')):
            part_claims, rejected, limited = boundary_witnesses(graph, {'dimension_chains': graph['dimension_chains']})
        audit['part_boundary_rejections'] = rejected
        if limited:
            audit['limits'].append('part_boundary_budget')
        for identity, witnesses in part_claims.items():
            i = number[identity]
            candidate = editable & (nearest == i)
            claims['structure'][candidate] |= 1 << 1
            stable = low[i] > 0 and high[i] <= POLICY['maximum_width_q95_to_q25']*low[i]
            independent_role = items[identity].get('role') == 'part_contour'
            allowed = stable and independent_role and identity not in grid_ids
            guarded = candidate & ~np.isin(labels, list(glyph_ids)) if allowed else np.zeros_like(candidate)
            # Contacts with a dimension are an unresolved mixture of ink owners.
            for witness in witnesses:
                chain = graph['dimension_chains'][witness['chain_index']]
                dims = chain['shaft_interval_ids']+[s for group in chain['extension_interval_ids'] for s in group]
                thin = max((low[number[s]] for s in dims), default=0.)
                radius = high[i]+thin+math.sqrt(.5)
                for foot in witness['feet_px']:
                    y, x = np.nonzero(guarded)
                    near = (x-foot[0])**2+(y-foot[1])**2 <= radius**2
                    guarded[y[near], x[near]] = False
            claims['guarded'][guarded] |= 1 << 1
            part_decisions.append({'interval_id': identity, 'witnesses': witnesses,
                                   'independent_existing_interval_role': independent_role,
                                   'stable_width': bool(stable), 'competing_grid_context': identity in grid_ids,
                                   'guarded_candidate_pixels': int(guarded.sum())})
        if not audit['limits']:
            for a in ARMS[1:]:
                bits = claims[a]
                unique = (bits > 0) & ((bits & (bits-1)) == 0)
                for role in (1, 6, 8):
                    selected = unique & (bits == (1 << role))
                    output[a][selected] = role
                    owners[a][selected] = nearest[selected]
                audit[a+'_conflicting_claim_pixels'] = int(np.count_nonzero((bits > 0) & ~unique))
    complete = bool(previous['complete'] and not audit['limits'])
    if not complete:
        for a in ARMS[1:]:
            output[a][:] = base
            owners[a][:] = 0
    return {**output, 'owners': owners, 'decisions': decisions, 'part_decisions': part_decisions,
            'audit': audit, 'complete': complete, 'production_admitted': False,
            'physical_geometry_approved': False, 'source_pixels_erased': 0}
