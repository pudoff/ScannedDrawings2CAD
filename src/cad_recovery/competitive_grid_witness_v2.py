"""Local cell support and competing frame/body hypotheses; conflicts abstain."""
from copy import deepcopy
import math
import cv2
import numpy as np
from . import grid_role_revision_v1 as grids
from .part_admission_v1 import object_sha
from .source_region_roles_v1 import component_rows
from .stroke_roles_v1 import checked_gray, foreground
from .positive_grid_witness_v1 import glyph_fact
from .positive_grid_witness_v3 import observed_rulings, POLICY as PRIOR_POLICY

POLICY = {**PRIOR_POLICY, 'version': 'competitive-grid-witness/2.0',
          'maximum_cell_extent_in_text_heights': 8., 'blank_cell_area_ratio': 2.5,
          'frame_face_area_ratio': 4., 'enclosing_width_ratio': 1.5,
          'conflicts_create_FRAME_or_PART': False}


def analyze(gray, graph, source_owner, ambiguous_owner, proposed_grid, *, frame_owner_numbers=(), other_role_owner_numbers=()):
    checked_gray(gray)
    if any(a.shape != gray.shape for a in (source_owner, ambiguous_owner, proposed_grid)):
        raise ValueError('Aligned source owners and grid proposals required')
    if (source_owner.dtype.kind not in 'iu' or source_owner.min(initial=0) < 0 or
            source_owner.max(initial=0) > len(graph['proposals'])):
        raise ValueError('Foreign source-owner domain')
    before = object_sha(graph)
    snapshots = [grids.sha(a) for a in (gray, source_owner, ambiguous_owner, proposed_grid)]
    audit = {'cell_roi_pixels': 0, 'glyph_roi_pixels': 0, 'source_points': 0, 'limits': []}
    output = np.zeros_like(gray); records = []
    if not np.any(proposed_grid):
        return {'mask': output, 'networks': [], 'complete': True, 'audit': audit,
                'policy': deepcopy(POLICY), 'source_immutable': True}
    ink = foreground(gray)
    ruling, ruling_work = observed_rulings(ink, graph, source_owner, ambiguous_owner)
    audit['source_points'] += ruling_work['source_points']
    audit['observed_ruling_directions'] = ruling_work.get('directions', [])
    audit['limits'].extend(ruling_work['limits'])
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ruling, connectivity=8)
    glyph_count, glyph_labels, glyph_stats, _ = cv2.connectedComponentsWithStats(np.where((ink > 0) & (ruling == 0), 255, 0).astype(np.uint8), connectivity=8)
    networks = [int(v) for v in np.unique(labels[proposed_grid > 0]) if v]
    if max(count-1, glyph_count-1) > POLICY['maximum_components'] or len(networks) > POLICY['maximum_networks']:
        audit['limits'].append('network_or_component_budget')
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    items = graph['proposals']
    radii = np.asarray([0.]+[float(p['width_px'])/2+math.sqrt(.5) for p in items])
    for network in networks:
        if audit['limits']: break
        numbers = [int(v) for v in np.unique(source_owner[labels == network]) if v]
        directions = []
        for number in numbers:
            item = items[number-1]
            if not item['linear_context']: continue
            p = np.asarray(item['points_px'], float)
            length = float(np.linalg.norm(p[-1]-p[0]))
            if length > 0: directions.append((length, (p[-1]-p[0])/length))
        if not directions: continue
        u = max(directions, key=lambda p: p[0])[1].copy()
        if u[0] < 0: u = -u
        n = np.array([-u[1], u[0]])
        pen = float(np.median([items[i-1]['width_px'] for i in numbers]))
        nc, cells, cell_stats, _ = cv2.connectedComponentsWithStats((labels != network).astype(np.uint8), connectivity=4)
        if nc-1 > POLICY['maximum_cells']:
            audit['limits'].append('cell_count_budget'); break
        exterior = set(map(int, np.unique(np.r_[cells[0], cells[-1], cells[:, 0], cells[:, -1]])))
        rectangles = {}
        for identity in range(1, nc):
            if identity in exterior: continue
            x, y, w, h, area = map(int, cell_stats[identity])
            if min(w, h) < POLICY['minimum_cell_extent_in_strokes']*pen: continue
            audit['cell_roi_pixels'] += w*h
            if audit['cell_roi_pixels'] > POLICY['maximum_cell_roi_pixels']:
                audit['limits'].append('cell_roi_budget'); break
            local = (cells[y:y+h, x:x+w] == identity).astype(np.uint8)
            contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 1: continue
            rectangle = cv2.minAreaRect(contours[0]); rw, rh = rectangle[1]
            fill = area/max(1., (rw+1)*(rh+1))
            box = cv2.boxPoints(rectangle)
            axis = box[1]-box[0]; norm = float(np.linalg.norm(axis))
            if norm <= 0: continue
            dot = abs(float(axis@u))/norm
            aligned = dot >= math.cos(math.radians(8)) or dot <= math.sin(math.radians(8))
            if fill >= POLICY['minimum_rectangularity'] and aligned:
                rectangles[identity] = {'cell_id': identity, 'area_px': area, 'rectangularity': float(fill),
                                        'glyphs': [], 'text_rows': [], 'short_extent_px': float(min(rw+1, rh+1))}
        if audit['limits']: break
        facts = {}
        for component in range(1, glyph_count):
            x, y, w, h, _ = map(int, glyph_stats[component])
            if max(w, h) > 192: continue
            occupied = glyph_labels[y:y+h, x:x+w] == component
            membership = np.unique(cells[y:y+h, x:x+w][occupied])
            if len(membership) != 1 or int(membership[0]) not in rectangles: continue
            fact = glyph_fact(component, glyph_labels, glyph_stats, distance, u, n, audit)
            if fact is not None: facts.setdefault(int(membership[0]), []).append(fact)
            if audit['limits']: break
        if audit['limits']: break
        text_cells = set()
        for identity, members in facts.items():
            rows = component_rows(members)
            rectangles[identity].update(glyphs=rows['glyph_component_ids'], text_rows=rows['rows'])
            height = float(np.median([r['height'] for r in rows['rows']])) if rows['rows'] else None
            rectangles[identity]['text_height_px'] = height
            if (height and rows['glyph_count'] >= POLICY['minimum_glyphs_in_cell'] and
                    rectangles[identity]['short_extent_px'] <= POLICY['maximum_cell_extent_in_text_heights']*height):
                text_cells.add(identity)
        # Record actual free faces beside each source interval. A connected
        # network is insufficient to authorize every face in that network.
        side_cache = {}; adjacency = {i: set() for i in rectangles}
        exterior_owner, interior_owner = [], []
        for number in numbers:
            item = items[number-1]
            if not item['linear_context']: continue
            points = np.asarray(item['points_px'], float); q = np.rint(points).astype(int)
            axis = points[-1]-points[0]; norm = float(np.linalg.norm(axis))
            if norm <= 0: continue
            normal = np.array([-axis[1], axis[0]])/norm
            radius = distance[q[:, 1], q[:, 0]]+math.sqrt(.5)+1.
            sides = []
            for sign in (-1, 1):
                probes = np.rint(points+sign*radius[:, None]*normal).astype(int)
                valid = (probes >= 0).all(axis=1) & (probes[:, 0] < gray.shape[1]) & (probes[:, 1] < gray.shape[0])
                faces = np.zeros(len(points), int); where = np.flatnonzero(valid)
                faces[where] = cells[probes[where, 1], probes[where, 0]]
                sides.append(faces)
            side_cache[number] = (q, sides)
            for a, b in np.unique(np.column_stack(sides), axis=0):
                if a in rectangles and b in rectangles and a != b:
                    adjacency[int(a)].add(int(b)); adjacency[int(b)].add(int(a))
            outside = np.isin(sides[0], list(exterior)) | np.isin(sides[1], list(exterior))
            inside = np.isin(sides[0], list(rectangles)) & np.isin(sides[1], list(rectangles)) & (sides[0] != sides[1])
            if outside.mean() >= .5: exterior_owner.append(number)
            if inside.mean() >= .5: interior_owner.append(number)
        approved = set()
        if len(rectangles) >= 3 and len(text_cells) >= 2:
            median_area = float(np.median([rectangles[i]['area_px'] for i in text_cells]))
            max_short = max(rectangles[i]['short_extent_px'] for i in text_cells)
            peers = {i for i, cell in rectangles.items() if median_area/POLICY['blank_cell_area_ratio'] <= cell['area_px'] <= median_area*POLICY['blank_cell_area_ratio'] and cell['short_extent_px'] <= 2*max_short}
            approved = set(text_cells)
            while True:
                grown = approved | {j for i in approved for j in adjacency[i] if j in peers}
                if grown == approved: break
                approved = grown
            if len(approved) < 3: approved.clear()
        median_core = float(np.median([rectangles[i]['area_px'] for i in approved])) if approved else 0.
        frame_present = bool(approved and any(int(cell_stats[i, cv2.CC_STAT_AREA]) >= POLICY['frame_face_area_ratio']*median_core and i not in text_cells for i in range(1, nc) if i not in exterior))
        outer_pen = float(np.median([items[i-1]['width_px'] for i in exterior_owner])) if exterior_owner else None
        inner_pen = float(np.median([items[i-1]['width_px'] for i in interior_owner])) if interior_owner else None
        frame_source_hint = bool(set(numbers) & set(frame_owner_numbers))
        body_conflict = bool(outer_pen and inner_pen and outer_pen >= POLICY['enclosing_width_ratio']*inner_pen and not (frame_present and frame_source_hint))
        qualified = bool(approved)
        record = {'network_component': network, 'rectangular_cells': list(rectangles.values()),
                  'text_cells': sorted(text_cells), 'certified_cells': sorted(approved),
                  'qualified': qualified, 'frame_present': frame_present, 'body_enclosure_conflict': body_conflict,
                  'outer_pen_px': outer_pen, 'inner_pen_px': inner_pen, 'glyph_namespace': 'residual_after_observed_rulings', 'source_owner_numbers': [], 'proposed_grid_pixels': int(((labels == network) & (proposed_grid > 0)).sum())}
        records.append(record)
        if not qualified: continue
        good_cell = np.zeros(nc, bool); good_cell[list(approved)] = True
        seeds = np.zeros_like(gray); conflict_seeds = np.zeros_like(gray); approved_owners = []
        for number, (q, sides) in side_cache.items():
            audit['source_points'] += len(q)
            if audit['source_points'] > POLICY['maximum_source_points']:
                audit['limits'].append('source_point_budget'); break
            outside = np.isin(sides[0], list(exterior)) | np.isin(sides[1], list(exterior))
            conflict = np.full(len(q), body_conflict or number in frame_owner_numbers or number in other_role_owner_numbers, bool)
            if frame_present: conflict |= outside
            conflict_seeds[q[conflict, 1], q[conflict, 0]] = 255
            observed = (good_cell[sides[0]] | good_cell[sides[1]]) & ~conflict
            observed &= labels[q[:, 1], q[:, 0]] == network
            if observed.any():
                seeds[q[observed, 1], q[observed, 0]] = 255; approved_owners.append(number)
        if audit['limits']: break
        if approved_owners:
            proximity = cv2.distanceTransform(255-seeds, cv2.DIST_L2, 5)
            selected = ((proposed_grid > 0) & (labels == network) & np.isin(source_owner, approved_owners) &
                        (ambiguous_owner == 0) & (proximity <= radii[source_owner]))
            if conflict_seeds.any():
                clearance = cv2.distanceTransform(255-conflict_seeds, cv2.DIST_L2, 5)
                selected &= clearance > radii[source_owner]+math.sqrt(.5)
            output[selected] = 255
        record['source_owner_numbers'] = approved_owners
        record['certified_seed_pixels'] = int(np.count_nonzero(seeds))
        record['competing_seed_pixels'] = int(np.count_nonzero(conflict_seeds))
    complete = not audit['limits']
    if not complete: output[:] = 0
    immutable = before == object_sha(graph) and snapshots == [grids.sha(a) for a in (gray, source_owner, ambiguous_owner, proposed_grid)]
    if not immutable: raise ValueError('Positive grid witness mutated its observations')
    return {'mask': output, 'networks': records, 'complete': complete, 'audit': audit,
            'ruling_mask_sha256': grids.sha(ruling), 'glyph_labels_sha256': grids.sha(glyph_labels),
            'policy': deepcopy(POLICY), 'source_immutable': immutable}
