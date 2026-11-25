"""Source ruling separation allows cell evidence even when print touches rules."""
from copy import deepcopy
import math
import cv2
import numpy as np
from . import grid_role_revision_v1 as grids
from .part_admission_v1 import object_sha
from .source_region_roles_v1 import component_rows
from .stroke_roles_v1 import checked_gray, foreground
from .positive_grid_witness_v1 import glyph_fact, POLICY as PRIOR_POLICY

POLICY = {**PRIOR_POLICY, 'version': 'positive-grid-witness/2.0',
          'minimum_cell_extent_in_strokes': 1.5, 'minimum_glyphs_in_cell': 2,
          'variable_cell_areas_allowed': True, 'glyphs_after_observed_rule_separation': True}


def observed_rulings(ink, graph, owner, ambiguous):
    numbers = {p['id']: i for i, p in enumerate(graph['proposals'], 1)}
    eligible = set(); seeds = np.zeros_like(ink); work = {'source_points': 0, 'limits': []}
    for track in graph['tracks']:
        length = np.linalg.norm(np.asarray(track['points'][1])-track['points'][0])
        if length >= max(8*track['width'], .15*min(ink.shape)):
            eligible.update(numbers[i] for i in track['interval_ids'])
    radius = 0.
    for number in sorted(eligible):
        item = graph['proposals'][number-1]
        if not item['linear_context']: continue
        points = np.rint(item['points_px']).astype(np.int32)
        work['source_points'] += len(points)
        if work['source_points'] > POLICY['maximum_source_points']:
            work['limits'].append('ruling_source_point_budget'); return np.zeros_like(ink), work
        cv2.polylines(seeds, [points], False, 255, 1)
        radius = max(radius, item['width_px']/2+math.sqrt(.5))
    if not seeds.any(): return seeds, work
    proximity = cv2.distanceTransform(255-seeds, cv2.DIST_L2, 5)
    # Shared observed junctions close the ruling network; other glyph owners do not.
    selected = (ink > 0) & (np.isin(owner, list(eligible)) | ((ambiguous > 0) & (proximity <= radius)))
    return selected.astype(np.uint8)*255, work


def analyze(gray, graph, source_owner, ambiguous_owner, proposed_grid):
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
        numbers = [int(v) for v in np.unique(source_owner[(labels == network) & (proposed_grid > 0)]) if v]
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
                                        'glyphs': [], 'text_rows': []}
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
            if rows['rows'] and rows['glyph_count'] >= POLICY['minimum_glyphs_in_cell']: text_cells.add(identity)
        approved = set(rectangles) if len(rectangles) >= 3 and len(text_cells) >= 2 else set()
        qualified = bool(approved)
        record = {'network_component': network, 'rectangular_cells': list(rectangles.values()),
                  'text_cells': sorted(text_cells), 'certified_cells': sorted(approved),
                  'qualified': qualified, 'glyph_namespace': 'residual_after_observed_rulings', 'source_owner_numbers': [], 'proposed_grid_pixels': int(((labels == network) & (proposed_grid > 0)).sum())}
        records.append(record)
        if not qualified: continue
        good_cell = np.zeros(nc, bool); good_cell[list(approved)] = True
        seeds = np.zeros_like(gray); approved_owners = []
        for number in numbers:
            item = items[number-1]
            if not item['linear_context']: continue
            p = np.asarray(item['points_px'], float); q = np.rint(p).astype(int)
            audit['source_points'] += len(p)
            if audit['source_points'] > POLICY['maximum_source_points']:
                audit['limits'].append('source_point_budget'); break
            axis = p[-1]-p[0]; norm = float(np.linalg.norm(axis))
            if norm <= 0: continue
            normal = np.array([-axis[1], axis[0]])/norm
            radius = distance[q[:, 1], q[:, 0]]+math.sqrt(.5)+1.
            observed = np.zeros(len(p), bool)
            for sign in (-1, 1):
                probes = np.rint(p+sign*radius[:, None]*normal).astype(int)
                valid = (probes >= 0).all(axis=1) & (probes[:, 0] < gray.shape[1]) & (probes[:, 1] < gray.shape[0])
                where = np.flatnonzero(valid)
                observed[where] |= good_cell[cells[probes[where, 1], probes[where, 0]]]
            observed &= labels[q[:, 1], q[:, 0]] == network
            if observed.any():
                seeds[q[observed, 1], q[observed, 0]] = 255; approved_owners.append(number)
        if audit['limits']: break
        if approved_owners:
            proximity = cv2.distanceTransform(255-seeds, cv2.DIST_L2, 5)
            selected = ((proposed_grid > 0) & (labels == network) & np.isin(source_owner, approved_owners) &
                        (ambiguous_owner == 0) & (proximity <= radii[source_owner]))
            output[selected] = 255
        record['source_owner_numbers'] = approved_owners
        record['certified_seed_pixels'] = int(np.count_nonzero(seeds))
    complete = not audit['limits']
    if not complete: output[:] = 0
    immutable = before == object_sha(graph) and snapshots == [grids.sha(a) for a in (gray, source_owner, ambiguous_owner, proposed_grid)]
    if not immutable: raise ValueError('Positive grid witness mutated its observations')
    return {'mask': output, 'networks': records, 'complete': complete, 'audit': audit,
            'ruling_mask_sha256': grids.sha(ruling), 'glyph_labels_sha256': grids.sha(glyph_labels),
            'policy': deepcopy(POLICY), 'source_immutable': immutable}
