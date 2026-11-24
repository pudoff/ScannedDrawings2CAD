"""Positive, source-local ruling evidence from closed cells containing glyph rows."""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from . import grid_role_revision_v1 as grids
from .part_admission_v1 import object_sha
from .source_region_roles_v1 import component_rows
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'positive-grid-witness/1.0', 'maximum_networks': 64,
          'maximum_components': 4096, 'maximum_cells': 8192,
          'maximum_cell_roi_pixels': 16000000, 'maximum_glyph_roi_pixels': 2000000,
          'maximum_source_points': 2000000, 'minimum_rectangularity': .85,
          'minimum_cell_extent_in_strokes': 3., 'maximum_cell_axis_error_deg': 8.,
          'comparable_cell_area_ratio': 2., 'minimum_comparable_cells': 3,
          'minimum_text_cells': 2, 'minimum_glyphs_in_cell': 3,
          'source_pixels_erased': 0, 'CAD_approved': False, 'production_admitted': False}


def glyph_fact(component, labels, stats, distance, u, n, audit):
    x, y, w, h, area = map(int, stats[component])
    if area < 6 or max(w, h) > 192 or x == 0 or y == 0 or x+w == labels.shape[1] or y+h == labels.shape[0]:
        return None
    audit['glyph_roi_pixels'] += w*h
    if audit['glyph_roi_pixels'] > POLICY['maximum_glyph_roi_pixels']:
        audit['limits'].append('glyph_roi_budget'); return None
    local = labels[y:y+h, x:x+w] == component
    skeleton = skeletonize(local, method='zhang').astype(np.uint8)*255
    degree = grids.POPCOUNT[grids.adjacency_bits(skeleton)]
    ends = int(np.count_nonzero((skeleton > 0) & (degree == 1)))
    branches = int(np.count_nonzero((skeleton > 0) & (degree >= 3)))
    holes = cv2.connectedComponents(np.pad((~local).astype(np.uint8), 1, constant_values=1), connectivity=4)[0]-2
    if not (ends >= 2 or branches or holes >= 2):
        return None
    yy, xx = np.nonzero(local); points = np.column_stack((xx+x, yy+y))
    a, b = points@u, points@n
    sy, sx = np.nonzero(skeleton)
    stroke = max(1., float(np.quantile(2*distance[y+sy, x+sx]-1, .25)))
    center = points.mean(0)
    _, _, vectors = np.linalg.svd(points-center, full_matrices=False)
    direction = vectors[0]; normal = np.array([-direction[1], direction[0]])
    if (np.ptp((points-center)@direction) >= 6*stroke and
            np.quantile(np.abs((points-center)@normal), .95) <= max(.75, .25*stroke)):
        return None
    return {'id': int(component), 'box': [float(a.min()), float(b.min()), float(a.max()+1), float(b.max()+1)],
            'stroke': stroke, 'endpoints': ends, 'branches': branches, 'holes': int(holes)}


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
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    networks = [int(v) for v in np.unique(labels[proposed_grid > 0]) if v]
    if count-1 > POLICY['maximum_components'] or len(networks) > POLICY['maximum_networks']:
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
        for component in range(1, count):
            if component == network: continue
            x, y, w, h, _ = map(int, stats[component])
            if max(w, h) > 192: continue
            occupied = labels[y:y+h, x:x+w] == component
            membership = np.unique(cells[y:y+h, x:x+w][occupied])
            if len(membership) != 1 or int(membership[0]) not in rectangles: continue
            fact = glyph_fact(component, labels, stats, distance, u, n, audit)
            if fact is not None: facts.setdefault(int(membership[0]), []).append(fact)
            if audit['limits']: break
        if audit['limits']: break
        text_cells = set()
        for identity, members in facts.items():
            rows = component_rows(members)
            rectangles[identity].update(glyphs=rows['glyph_component_ids'], text_rows=rows['rows'])
            if rows['rows'] and rows['glyph_count'] >= POLICY['minimum_glyphs_in_cell']: text_cells.add(identity)
        approved = set()
        for identity in text_cells:
            area = rectangles[identity]['area_px']; ratio = POLICY['comparable_cell_area_ratio']
            peers = {i for i, c in rectangles.items() if area/ratio <= c['area_px'] <= area*ratio}
            if len(peers) >= POLICY['minimum_comparable_cells'] and len(peers & text_cells) >= POLICY['minimum_text_cells']:
                approved |= peers
        qualified = bool(approved)
        record = {'network_component': network, 'rectangular_cells': list(rectangles.values()),
                  'text_cells': sorted(text_cells), 'certified_cells': sorted(approved),
                  'qualified': qualified, 'source_owner_numbers': [], 'proposed_grid_pixels': int(((labels == network) & (proposed_grid > 0)).sum())}
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
            'policy': deepcopy(POLICY), 'source_immutable': immutable}
