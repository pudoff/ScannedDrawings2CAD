"""Source-pixel contacts ground page-grid observations, never semantic roles."""
from copy import deepcopy
import math

import cv2
import numpy as np

from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .source_corner_rules_v2 import segments, corner_grid
from .stroke_roles_v1 import checked_gray

PIXEL = math.sqrt(.5)
POLICY = {
    'version': 'page-grid-attachment/1.0', 'maximum_pixels': 4194304,
    'maximum_frames': 24, 'maximum_rules': 512, 'maximum_contacts': 2048,
    'maximum_contact_pixels': 2000000, 'maximum_contact_roi_pixels': 65536,
    'axis_tolerance_deg': 8., 'minimum_anchored_per_axis': 2,
    'minimum_crossings_per_rule': 2, 'raw_contact_mask': 'Otsu_without_dilation',
    'semantic_GRID_or_PART_assigned': False, 'source_pixels_erased': 0,
    'physical_geometry_approved': False,
}


def line(points, width, uncertainty):
    p = np.asarray(points, float)
    width, uncertainty = float(width), float(uncertainty)
    if (p.shape != (2, 2) or not np.isfinite(p).all() or not math.isfinite(width+uncertainty)
            or width <= 0 or uncertainty < 0 or np.linalg.norm(p[1]-p[0]) <= 0):
        raise ValueError('Finite nondegenerate observed stroke and measured width required')
    return {'points_source_px': p.tolist(), 'width_px': width, 'uncertainty_px': uncertainty}


def _geometry(stroke):
    p = np.asarray(stroke['points_source_px'], float)
    length = float(np.linalg.norm(p[1]-p[0])); u = (p[1]-p[0])/length
    return p, u, np.array([-u[1], u[0]]), length, stroke['width_px']/2+stroke['uncertainty_px']+PIXEL


def contact(ink, first, second, audit, *, require_first_endpoint):
    """Certify local ink contact, not membership of one global connected component."""
    audit['contacts'] += 1
    if audit['contacts'] > POLICY['maximum_contacts']:
        audit['limits'].append('contact_count_budget'); return {'connected': False, 'reason': 'budget'}
    a, u, n, length_a, radius_a = _geometry(first)
    b, v, normal, length_b, radius_b = _geometry(second)
    if abs(float(u @ v)) > math.sin(math.radians(POLICY['axis_tolerance_deg'])):
        return {'connected': False, 'reason': 'not_transverse'}
    ta, tb = np.linalg.solve(np.column_stack([u, -v]), b[0]-a[0])
    reach = radius_a+radius_b
    if ta < -reach or ta > length_a+reach or tb < -reach or tb > length_b+reach:
        return {'connected': False, 'reason': 'outside_observed_interval'}
    if require_first_endpoint and min(abs(ta), abs(ta-length_a)) > reach:
        return {'connected': False, 'reason': 'not_a_rule_endpoint'}
    intersection = a[0]+ta*u
    clearance = 2*reach
    sign_a = 1 if length_a-ta >= ta else -1
    sign_b = 1 if length_b-tb >= tb else -1
    anchor_a = a[0]+np.clip(ta+sign_a*clearance, 0, length_a)*u
    anchor_b = b[0]+np.clip(tb+sign_b*clearance, 0, length_b)*v
    if (abs((anchor_a-b[0]) @ normal) <= radius_b+PIXEL
            or abs((anchor_b-a[0]) @ n) <= radius_a+PIXEL):
        return {'connected': False, 'reason': 'insufficient_separate_stroke_context'}
    margin = max(radius_a, radius_b)+PIXEL
    vertices = np.stack([intersection, anchor_a, anchor_b])
    low = np.maximum(0, np.floor(vertices.min(0)-margin).astype(int))
    high = np.minimum(ink.shape[::-1], np.ceil(vertices.max(0)+margin).astype(int)+1)
    if np.any(high <= low): return {'connected': False, 'reason': 'outside_raster'}
    pixels = int(np.prod(high-low)); audit['contact_pixels'] += pixels
    if pixels > POLICY['maximum_contact_roi_pixels'] or audit['contact_pixels'] > POLICY['maximum_contact_pixels']:
        audit['limits'].append('contact_pixel_budget'); return {'connected': False, 'reason': 'budget'}
    yy, xx = np.indices(tuple((high-low)[::-1])); points = np.stack([xx+low[0], yy+low[1]], axis=-1)
    pa, pb = points-a[0], points-b[0]
    along_a, along_b = pa @ u, pb @ v
    distance_a, distance_b = np.abs(pa @ n), np.abs(pb @ normal)
    corridor_a = (distance_a <= radius_a) & (along_a >= -radius_a) & (along_a <= length_a+radius_a)
    corridor_b = (distance_b <= radius_b) & (along_b >= -radius_b) & (along_b <= length_b+radius_b)
    actual = ink[low[1]:high[1], low[0]:high[0]] > 0
    supported = actual & (corridor_a | corridor_b)
    _, labels = cv2.connectedComponents(supported.astype(np.uint8), connectivity=8)
    seed_a = (actual & corridor_a & (distance_b > radius_b+PIXEL)
              & (np.abs((points-anchor_a) @ u) <= PIXEL+first['uncertainty_px']))
    seed_b = (actual & corridor_b & (distance_a > radius_a+PIXEL)
              & (np.abs((points-anchor_b) @ v) <= PIXEL+second['uncertainty_px']))
    common = sorted((set(map(int, labels[seed_a])) & set(map(int, labels[seed_b])))-{0})
    result = {'connected': bool(common), 'reason': 'local_source_ink' if common else 'no_local_source_path',
              'intersection_source_px': intersection.tolist(), 'roi_bbox': [*low.tolist(), *high.tolist()],
              'radius_first_px': radius_a, 'radius_second_px': radius_b}
    if common:
        identity = common[0]
        for name, mask in [('first', seed_a), ('second', seed_b)]:
            sy, sx = np.nonzero(mask & (labels == identity))
            result['seed_'+name+'_source_px'] = [int(sx[0]+low[0]), int(sy[0]+low[1])]
        result['contact_component_pixels'] = int(np.count_nonzero(labels == identity))
    return result


def _frame_overlap(rule, sides):
    p, u, _, _, radius = _geometry(rule)
    for side in sides:
        q, v, n, length, other_radius = _geometry(side)
        if abs(float(u @ v)) < math.cos(math.radians(POLICY['axis_tolerance_deg'])): continue
        along = (p-q[0]) @ v
        if (np.max(np.abs((p-q[0]) @ n)) <= radius+other_radius
                and along.min() >= -radius-other_radius and along.max() <= length+radius+other_radius):
            return True
    return False


def analyze(gray, detector):
    checked_gray(gray)
    if gray.size > POLICY['maximum_pixels'] or len(detector['frames']) > POLICY['maximum_frames']:
        raise ValueError('Bounded full-page grid attachment required')
    if detector['source_sha256'] != sha(gray) or detector['size_px'] != list(gray.shape[::-1]):
        raise ValueError('Detector and original source pixels differ')
    if detector.get('GT_used') is True or any(f.get('frame_semantics_confirmed') is not False for f in detector['frames']):
        raise ValueError('Semantic/manual frame targets cannot enter attachment inference')
    before = object_sha(detector); source_hash = sha(gray)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    audit = {'source_segments': 0, 'source_support_samples': 0, 'contacts': 0, 'contact_pixels': 0, 'limits': []}
    if not detector['complete']: audit['limits'].append('incomplete_page_detector')
    candidates = segments(ink, audit) if detector['frames'] and not audit['limits'] else []
    distance = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5)
    rules = []; groups = []; cache = {}
    for frame_index, frame in enumerate(detector['frames']) if not audit['limits'] else []:
        if len(frame['sides']) != 4: raise ValueError('Four observed frame sides required')
        sides = [line(s['points_overview'], s['stroke_width_overview_px'], s['position_uncertainty_overview_px'])
                 for s in frame['sides']]
        quad = np.asarray(frame['quad_overview'], float)
        if quad.shape != (4, 2) or not np.isfinite(quad).all(): raise ValueError('Finite observed frame quad required')
        grid = corner_grid(ink, quad, candidates, distance, audit, cache=cache)
        if audit['limits']: break
        if grid is None: continue
        local = []
        for family in ('rows', 'columns'):
            for item in grid['source_support'][family]:
                if len(rules) >= POLICY['maximum_rules']:
                    audit['limits'].append('rule_count_budget'); break
                rule = {**line(item['source_endpoints'], item['source_width_px'], math.sqrt(2)),
                        'id': 'page-rule-'+str(len(rules)), 'frame_index': frame_index, 'family': family,
                        'source_support': float(item['source_support']), 'attachments': [],
                        'attachment_checks': [], 'frame_overlap': False,
                        'usage': 'source_rule_context_only', 'CAD_primitive_created': False,
                        'continuity_between_contacts_confirmed': False}
                rule['frame_overlap'] = _frame_overlap(rule, sides)
                if not rule['frame_overlap']:
                    for side_index, side in enumerate(sides):
                        proof = contact(ink, rule, side, audit, require_first_endpoint=True)
                        rule['attachment_checks'].append({'side_index': side_index, 'proof': proof})
                        if proof['connected']: rule['attachments'].append({'side_index': side_index, 'proof': proof})
                        if audit['limits']: break
                rules.append(rule); local.append(rule)
                if audit['limits']: break
            if audit['limits']: break
        if audit['limits']: break
        for row_side in range(4):
            for column_side in ((row_side+1) % 4, (row_side-1) % 4):
                rows = [r for r in local if r['family'] == 'rows' and any(a['side_index'] == row_side for a in r['attachments'])]
                columns = [r for r in local if r['family'] == 'columns' and any(a['side_index'] == column_side for a in r['attachments'])]
                if min(len(rows), len(columns)) < POLICY['minimum_anchored_per_axis']: continue
                crossings = []
                for row in rows:
                    for column in columns:
                        proof = contact(ink, row, column, audit, require_first_endpoint=False)
                        if proof['connected']: crossings.append({'row': row['id'], 'column': column['id'], 'proof': proof})
                        if audit['limits']: break
                    if audit['limits']: break
                if audit['limits']: break
                kept_rows, kept_columns = {r['id'] for r in rows}, {r['id'] for r in columns}
                while True:
                    valid = [c for c in crossings if c['row'] in kept_rows and c['column'] in kept_columns]
                    new_rows = {r for r in kept_rows if sum(c['row'] == r for c in valid) >= POLICY['minimum_crossings_per_rule']}
                    new_columns = {r for r in kept_columns if sum(c['column'] == r for c in valid) >= POLICY['minimum_crossings_per_rule']}
                    if new_rows == kept_rows and new_columns == kept_columns: break
                    kept_rows, kept_columns = new_rows, new_columns
                if min(len(kept_rows), len(kept_columns)) >= POLICY['minimum_anchored_per_axis']:
                    groups.append({'frame_index': frame_index, 'row_side': row_side, 'column_side': column_side,
                                   'rule_ids': sorted(kept_rows | kept_columns),
                                   'crossings': [c for c in crossings if c['row'] in kept_rows and c['column'] in kept_columns],
                                   'semantic_GRID_confirmed': False})
            if audit['limits']: break
        if audit['limits']: break
    complete = not audit['limits']
    if not complete: groups = []
    grounded = {identity for group in groups for identity in group['rule_ids']}
    for rule in rules: rule['grounded_group_member'] = rule['id'] in grounded
    if sha(gray) != source_hash or object_sha(detector) != before:
        raise ValueError('Grid attachment mutated source observations')
    return {'schema': 'page-grid-attachment/1.0', 'source_sha256': source_hash, 'detector_sha256': before,
            'size_px': list(gray.shape[::-1]), 'rules': rules, 'groups': groups,
            'grounded_rule_count': len(grounded), 'complete': complete, 'source_immutable': True,
            'audit': audit, 'policy': deepcopy(POLICY), 'semantic_GRID_or_PART_assigned': False,
            'source_pixels_erased': 0, 'physical_geometry_approved': False}
