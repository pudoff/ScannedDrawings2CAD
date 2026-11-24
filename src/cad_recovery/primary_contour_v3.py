"""Source-bound outline/hole proposals without pre-existing PART or dimensions."""
from copy import deepcopy
import math
import cv2
import numpy as np
from .grid_role_revision_v1 import sha, check_authority
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'primary-contour/3.0', 'maximum_pixels': 4000000,
          'maximum_intervals': 4096, 'maximum_boundaries': 512,
          'maximum_source_points': 2000000, 'maximum_polygon_points': 250000,
          'maximum_pair_checks': 250000, 'maximum_boundary_raster_work': 64000000,
          'minimum_body_extent_in_pens': 24.,
          'minimum_hole_extent_in_pens': 6., 'minimum_hole_to_body_extent': .07,
          'minimum_hole_area_fraction': .003, 'maximum_hole_area_fraction': .35,
          'maximum_width_radius_ratio': 2., 'maximum_hole_pen_ratio': 1.67,
          'minimum_outline_vertices': 6, 'maximum_body_oriented_box_fill': .97, 'required_holes': 1,
          'required_previous_part': False, 'required_dimension_chain': False,
          'CAD_semantics_approved': False, 'physical_geometry_approved': False}


def analyze(gray, graph, raw_roles, source_owner, ambiguous_owner, heads, *, authority, source_paths=()):
    checked_gray(gray)
    if gray.size > POLICY['maximum_pixels'] or len(graph['proposals']) > POLICY['maximum_intervals']:
        raise ValueError('Bounded primary contour source required')
    if (raw_roles.dtype != np.uint8 or not np.isin(raw_roles, list(range(10))+[255]).all() or
            source_owner.shape != gray.shape or source_owner.dtype.kind not in 'ui' or
            source_owner.min(initial=0) < 0 or source_owner.max(initial=0) > len(graph['proposals']) or
            ambiguous_owner.shape != gray.shape or raw_roles.shape != gray.shape):
        raise ValueError('Aligned current source owner domain required')
    before = object_sha([graph, heads, source_paths]); snapshots = [sha(a) for a in (gray, raw_roles, source_owner, ambiguous_owner)]
    allowed = check_authority(gray, raw_roles, authority)
    ink = foreground(gray); distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    items = graph['proposals']; numbers = {p['id']: n for n, p in enumerate(items, 1)}
    if len(numbers) != len(items):
        raise ValueError('Unique source interval IDs required')
    audit = {'limits': [], 'source_points': 0, 'polygon_points': 0, 'pair_checks': 0, 'boundary_raster_pixels': 0}
    if graph['search_incomplete'] or graph['boundaries']['search_incomplete']:
        audit['limits'].append('incomplete_source_graph')
    boundaries = list(graph['boundaries']['boundaries'])
    path_ids = [p['id'] for p in source_paths]
    if len(set(path_ids)) != len(path_ids):
        raise ValueError('Unique observed source paths required')
    for path in source_paths:
        if not path['closed']:
            continue
        points = np.asarray(path['points_px'], float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4 or not np.isfinite(points).all():
            raise ValueError('Finite observed closed path required')
        if not np.array_equal(points[0], points[-1]):
            continue
        members = [p['id'] for p in items if p.get('source_observation_id') == path['id']]
        if not members:
            continue
        boundaries.append({'id': 'source-closed/'+path['id'], 'owners': members,
                           'polygon': path['points_px'], 'frame_hypothesis': False})
    if len(boundaries) > POLICY['maximum_boundaries']:
        audit['limits'].append('boundary_budget')
    widths, radii, stable = np.zeros(len(items)+1), np.zeros(len(items)+1), np.zeros(len(items)+1, bool)
    for n, item in enumerate(items, 1):
        points = np.asarray(item['points_px'], float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            raise ValueError('Finite observed source path required')
        q = np.rint(points).astype(int)
        if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any():
            raise ValueError('Source path outside raster')
        audit['source_points'] += len(q)
        if audit['source_points'] > POLICY['maximum_source_points']:
            audit['limits'].append('source_point_budget'); break
        samples = distance[q[:, 1], q[:, 0]]; samples = samples[samples > 0]
        if len(samples) < 3:
            continue
        low, high = np.quantile(samples, [.25, .95]); radii[n] = low+math.sqrt(.5)
        widths[n] = max(1., float(2*low-1))
        stable[n] = high <= POLICY['maximum_width_radius_ratio']*low
    facts, seen, decisions = [], set(), []
    for boundary in boundaries if not audit['limits'] else []:
        ids = boundary['owners']
        if not ids or not set(ids) <= numbers.keys():
            raise ValueError('Boundary refers to foreign source intervals')
        signature = tuple(sorted(set(ids)))
        if signature in seen:
            continue
        seen.add(signature)
        polygon = np.asarray(boundary['polygon'], np.float32)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError('Finite closed source boundary required')
        audit['polygon_points'] += len(polygon)
        if audit['polygon_points'] > POLICY['maximum_polygon_points']:
            audit['limits'].append('polygon_point_budget'); break
        owners = sorted(numbers[i] for i in signature)
        area = abs(float(cv2.contourArea(polygon)))
        pen = float(np.median(widths[owners]))
        lo, hi = polygon.min(0), polygon.max(0); extent = float((hi-lo).min())
        clearance = min(float(lo.min()), float((np.array(gray.shape[::-1])-1-hi).min()))
        simplified = cv2.approxPolyDP(polygon, max(.75, .5*pen), True).reshape(-1, 2)
        rectangle = cv2.minAreaRect(polygon)
        box_area = float(rectangle[1][0]*rectangle[1][1])
        box_fill = area/box_area if box_area > 0 else 1.
        reason = ('frame_hypothesis' if boundary['frame_hypothesis'] else
                  'image_border_contact' if clearance <= 2*pen else
                  'unresolved_width' if not stable[owners].all() else
                  'zero_area' if area <= 0 else None)
        fact = {'id': boundary['id'], 'owners': owners, 'polygon': polygon,
                'area': area, 'extent': extent, 'pen': pen, 'reason': reason,
                'vertices': len(simplified), 'lo': lo, 'hi': hi, 'oriented_box_fill': box_fill}
        facts.append(fact)
        decisions.append({'boundary_id': boundary['id'], 'source_owner_numbers': owners,
                          'area_px2': area, 'extent_px': extent, 'observed_pen_px': pen,
                          'simplified_vertices_for_analysis_only': len(simplified),
                          'oriented_box_fill': box_fill,
                          'rejection': reason, 'holes': [], 'positive_body': False})
    chosen = set()
    for i, body in enumerate(facts) if not audit['limits'] else []:
        if (body['reason'] or body['extent'] < POLICY['minimum_body_extent_in_pens']*body['pen'] or
                body['vertices'] < POLICY['minimum_outline_vertices'] or body['oriented_box_fill'] > POLICY['maximum_body_oriented_box_fill']):
            continue
        interior = np.zeros_like(gray)
        cv2.fillPoly(interior, [np.rint(body['polygon']).astype(np.int32)], 255)
        inside_clearance = cv2.distanceTransform(interior, cv2.DIST_L2, 5)
        audit['boundary_raster_pixels'] += gray.size
        holes = []
        for j, hole in enumerate(facts):
            audit['pair_checks'] += 1
            if audit['pair_checks'] > POLICY['maximum_pair_checks']:
                audit['limits'].append('boundary_pair_budget'); break
            if i == j or hole['reason'] or set(body['owners']) & set(hole['owners']):
                continue
            fraction = hole['area']/body['area']
            ratio = hole['pen']/body['pen']
            if not (POLICY['minimum_hole_area_fraction'] <= fraction <= POLICY['maximum_hole_area_fraction'] and
                    1/POLICY['maximum_hole_pen_ratio'] <= ratio <= POLICY['maximum_hole_pen_ratio'] and
                    hole['extent'] >= max(POLICY['minimum_hole_extent_in_pens']*hole['pen'], POLICY['minimum_hole_to_body_extent']*body['extent'])):
                continue
            if np.any(hole['lo'] <= body['lo']+body['pen']) or np.any(hole['hi'] >= body['hi']-body['pen']):
                continue
            audit['boundary_raster_pixels'] += gray.size
            if audit['boundary_raster_pixels'] > POLICY['maximum_boundary_raster_work']:
                audit['limits'].append('boundary_raster_budget'); break
            # Include the edges, not only vertices: a chord could cross a
            # concavity even when all its vertices lie inside. This is raster
            # evidence, not certification of analytical CAD topology.
            hole_edge = np.zeros_like(gray)
            cv2.polylines(hole_edge, [np.rint(hole['polygon']).astype(np.int32)], True, 255, 1, cv2.LINE_8)
            if np.all(inside_clearance[hole_edge > 0] > body['pen']+math.sqrt(.5)):
                holes.append(j)
        if audit['limits']:
            break
        if len(holes) >= POLICY['required_holes']:
            chosen.update([i]+holes); decisions[i]['positive_body'] = True
            decisions[i]['holes'] = [facts[j]['id'] for j in holes]
    mask = np.zeros_like(gray)
    protect = np.isin(raw_roles, [2, 3, 4, 5, 6, 7, 8, 9])
    for head in heads:
        polygon = np.asarray(head['polygon_px'], float)
        if polygon.shape != (3, 2) or not np.isfinite(polygon).all():
            raise ValueError('Finite observed head required')
        raster = np.zeros_like(gray); cv2.fillConvexPoly(raster, np.rint(polygon).astype(np.int32), 255)
        protect |= raster > 0
    for i in sorted(chosen) if not audit['limits'] else []:
        fact = facts[i]; seeds = np.zeros_like(gray)
        cv2.polylines(seeds, [np.rint(fact['polygon']).astype(np.int32)], True, 255, 1, cv2.LINE_8)
        near = cv2.distanceTransform(255-seeds, cv2.DIST_L2, 5)
        eligible = (allowed & ~protect & (ink > 0) & (ambiguous_owner == 0) &
                    np.isin(source_owner, fact['owners']) & (near <= radii[source_owner]))
        mask[eligible] = 255
    complete = not audit['limits']
    if not complete:
        mask[:] = 0
    immutable = before == object_sha([graph, heads, source_paths]) and snapshots == [sha(a) for a in (gray, raw_roles, source_owner, ambiguous_owner)]
    if not immutable:
        raise ValueError('Primary contour mutated source observations')
    return {'mask': mask, 'decisions': decisions, 'audit': audit, 'complete': complete,
            'source_immutable': immutable, 'policy': deepcopy(POLICY),
            'previous_part_labels_used_for_positive_evidence': False,
            'dimension_chains_used_for_positive_evidence': False,
            'geometry_modified': False, 'source_pixels_erased': 0, 'physical_geometry_approved': False}
