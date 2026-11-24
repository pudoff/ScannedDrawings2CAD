"""Withdraw new PART proposals only on the same bound, observed grid rule.

Producer/file pins are checked by the caller. No targets or files are read here.
The union of source masks is deliberately never used to authorize a withdrawal.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from .contour_page_competition_v1 import _binary, _whole_footprint_safe
from .frame_grid_guard_v1 import project as transform
from .grid_role_revision_v1 import sha
from .page_grid_attachment_projection_v1 import project
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground


POLICY = {
    'version': 'attached-grid-competition/1.0', 'maximum_pixels': 4194304,
    'maximum_intervals': 4096, 'maximum_rules': 512,
    'maximum_source_points': 2000000, 'maximum_rule_owner_checks': 250000,
    'maximum_local_mask_pixels': 8000000, 'maximum_rule_view_pixels': 64000000,
    'direction_tolerance_deg': 8., 'minimum_source_fraction': .8,
    'threshold_origin': 'unchanged_contour_page_competition_v1',
    'whole_native_footprint_required': True, 'same_rule_mask_required': True,
    'positive_role_created': False, 'geometry_modified': False,
    'physical_geometry_approved': False,
}


def analyze(gray, graph, source_owner, ambiguous_owner, proposed, *, source_gray,
            native_gray, page, observations, source_masks, input_to_native,
            resize_method, baseline_roles, protected):
    """Bind raw page -> ROI -> current graph; preserve baseline/reviewed roles.

    An incomplete search abandons all eligible new proposals, reporting this
    separately from an effective GRID veto. It preserves protected proposals.
    """
    checked_gray(gray)
    items = graph['proposals']
    if gray.size > POLICY['maximum_pixels'] or len(items) > POLICY['maximum_intervals']:
        raise ValueError('Bounded current graph required')
    for array, name in ((proposed, 'proposal'), (ambiguous_owner, 'ambiguity'), (protected, 'protection')):
        _binary(array, gray.shape, name)
    if (baseline_roles.shape != gray.shape or baseline_roles.dtype != np.uint8
            or not np.isin(baseline_roles, list(range(10))+[255]).all()):
        raise ValueError('Aligned accepted baseline roles required')
    if (source_owner.shape != gray.shape or source_owner.dtype.kind not in 'ui'
            or source_owner.max(initial=0) > len(items)):
        raise ValueError('Aligned current source owners required')
    if np.any((proposed > 0) & ((source_owner == 0) | (ambiguous_owner > 0))):
        raise ValueError('Each proposal requires an unambiguous current owner')
    if len({p['id'] for p in items}) != len(items):
        raise ValueError('Unique current interval IDs required')

    # Reuse the exact source/ROI/resize binding, including full original pixels.
    projection = project(source_gray, observations, native_gray, gray, page,
                         input_to_native, resize_method=resize_method)
    if (source_masks['source_sha256'] != sha(source_gray)
            or source_masks['observations_sha256'] != object_sha(observations)
            or source_masks.get('semantic_GRID_or_PART_assigned') is not False
            or source_masks.get('requires_owner_direction_binding') is not True):
        raise ValueError('Source masks must bind to these unconfirmed observations')
    masks = source_masks['rule_masks']
    rules = {r['id']: r for r in observations['rules'] if r['grounded_group_member']}
    if (len(masks) > POLICY['maximum_rules'] or len({m['rule_id'] for m in masks}) != len(masks)
            or any(m['rule_id'] not in rules for m in masks)):
        raise ValueError('Unique grounded per-rule masks required')
    metadata = {k: v for k, v in source_masks.items() if k not in ('safe_mask', 'rule_masks')}
    mask_metadata = [{k: v for k, v in m.items() if k != 'safe'} for m in masks]
    before = object_sha([graph, page, observations, metadata, mask_metadata])
    arrays = (gray, source_gray, native_gray, source_owner, ambiguous_owner,
              proposed, baseline_roles, protected, *[m['safe'] for m in masks])
    hashes = [sha(a) for a in arrays]
    audit = {'source_points': 0, 'rule_owner_checks': 0, 'local_mask_pixels': 0,
             'rule_view_pixels': 0, 'discarded_decisions': 0, 'limits': []}
    if graph.get('search_incomplete') or not projection['complete'] or not source_masks['complete']:
        audit['limits'].append('incomplete_source_grid_or_graph')
    ink = cv2.threshold(source_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    for mask in masks:
        box = mask['bbox_source_px']
        if (len(box) != 4 or any(type(v) is not int for v in box)
                or min(box[:2]) < 0 or box[2] <= box[0] or box[3] <= box[1]
                or box[2] > source_gray.shape[1] or box[3] > source_gray.shape[0]):
            raise ValueError('Valid per-rule source bounds required')
        x0, y0, x1, y1 = box
        _binary(mask['safe'], (y1-y0, x1-x0), 'per-rule safe mask')
        if mask['safe_sha256'] != sha(mask['safe']):
            raise ValueError('Stale per-rule source mask')
        if np.any((mask['safe'] > 0) & (ink[y0:y1, x0:x1] == 0)):
            raise ValueError('Per-rule mask contains unobserved paper')
        audit['local_mask_pixels'] += mask['safe'].size
    if audit['local_mask_pixels'] > POLICY['maximum_local_mask_pixels']:
        audit['limits'].append('local_mask_budget')

    editable = (proposed > 0) & (baseline_roles == 255) & (protected == 0)
    eligible = editable & (foreground(gray) > 0)
    matrix = np.asarray(page['patch_to_source']) @ np.asarray(input_to_native)
    box = page['patch_bbox']; bx, by, ex, ey = box
    directions = {}
    if not audit['limits']:
        for number in sorted(map(int, np.unique(source_owner[eligible]))):
            item = items[number-1]; points = np.asarray(item['points_px'], float)
            if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
                    or (points < 0).any() or (points[:, 0] >= gray.shape[1]).any()
                    or (points[:, 1] >= gray.shape[0]).any()):
                raise ValueError('Finite current source samples inside input required')
            audit['source_points'] += len(points)
            if audit['source_points'] > POLICY['maximum_source_points']:
                audit['limits'].append('source_point_budget'); break
            if len(points) < 2 or not item.get('linear_context', False):
                continue
            points = transform(points, matrix)
            _, singular, vectors = np.linalg.svd(points-points.mean(0), full_matrices=False)
            if len(singular) and singular[0] > 0:
                directions[number] = (points, vectors[0])

    veto = np.zeros_like(gray); decisions = []
    visibility = {r['rule_id']: r['visibility'] for r in projection['rules']}
    for mask in masks if not audit['limits'] else []:
        identity = mask['rule_id']
        if visibility.get(identity) == 'outside_input':
            continue
        audit['rule_view_pixels'] += gray.size
        if audit['rule_view_pixels'] > POLICY['maximum_rule_view_pixels']:
            audit['limits'].append('rule_view_pixel_budget'); break
        # Work with ONE rule at a time; never reduce the union of different rules.
        roi = np.zeros_like(native_gray)
        x0, y0, x1, y1 = mask['bbox_source_px']
        left, top, right, bottom = max(bx, x0), max(by, y0), min(ex, x1), min(ey, y1)
        if right <= left or bottom <= top:
            continue
        roi[top-by:bottom-by, left-bx:right-bx] = mask['safe'][top-y0:bottom-y0, left-x0:right-x0]
        candidate = eligible & _whole_footprint_safe(roi, gray.shape)
        rule = rules[identity]; ends = np.asarray(rule['points_source_px'], float)
        width, error = float(rule['width_px']), float(rule['uncertainty_px'])
        if (ends.shape != (2, 2) or not np.isfinite(ends).all()
                or not math.isfinite(width+error) or width <= 0 or error < 0):
            raise ValueError('Finite measured source rule required')
        length = float(np.linalg.norm(ends[1]-ends[0]))
        if not length:
            raise ValueError('Nondegenerate source rule required')
        u = (ends[1]-ends[0])/length; normal = np.array([-u[1], u[0]])
        radius = width/2+error+.5*np.abs(normal @ matrix[:2, :2]).sum()
        for number in sorted(map(int, np.unique(source_owner[candidate]))):
            if number not in directions:
                continue
            audit['rule_owner_checks'] += 1
            if audit['rule_owner_checks'] > POLICY['maximum_rule_owner_checks']:
                audit['limits'].append('rule_owner_budget'); break
            points, direction = directions[number]
            if abs(direction @ u) < math.cos(math.radians(POLICY['direction_tolerance_deg'])):
                continue
            offsets = points-ends[0]; along = offsets @ u
            support = ((np.abs(offsets @ normal) <= radius)
                       & (along >= -radius) & (along <= length+radius))
            fraction = float(support.mean())
            if fraction < POLICY['minimum_source_fraction']:
                continue
            yy, xx = np.nonzero(candidate & (source_owner == number))
            pixels = transform(np.c_[xx, yy], matrix)-ends[0]
            local = ((np.abs(pixels @ normal) <= radius)
                     & (pixels @ u >= -radius) & (pixels @ u <= length+radius))
            if not local.any():
                continue
            veto[yy[local], xx[local]] = 255
            decisions.append({'rule_id': identity, 'owner_number': number,
                'interval_id': items[number-1]['id'], 'source_fraction': fraction,
                'radius_source_px': float(radius), 'pixels': int(local.sum())})
        if audit['limits']:
            break
    complete = not audit['limits']; output = proposed.copy()
    if complete:
        output[veto > 0] = 0
    else:
        output[editable] = 0; veto[:] = 0
        audit['discarded_decisions'] = len(decisions); decisions = []
    if (hashes != [sha(a) for a in arrays]
            or before != object_sha([graph, page, observations, metadata, mask_metadata])):
        raise ValueError('Attached grid competition mutated source observations')
    return {'mask': output, 'veto_mask': veto, 'complete': complete,
        'source_immutable': True, 'grid_veto_pixels': int(np.count_nonzero(veto)),
        'abstained_pixels': int(np.count_nonzero(proposed)-np.count_nonzero(output)) if not complete else 0,
        'withdrawn_pixels': int(np.count_nonzero(proposed)-np.count_nonzero(output)),
        'decisions': decisions, 'audit': audit, 'policy': deepcopy(POLICY),
        'binding': {'source_sha256': sha(source_gray), 'native_sha256': sha(native_gray),
            'input_sha256': sha(gray), 'graph_sha256': object_sha(graph),
            'source_owner_sha256': sha(source_owner), 'page_sha256': object_sha(page),
            'observations_sha256': object_sha(observations),
            'input_to_native': deepcopy(input_to_native), 'resize_method': resize_method},
        'positive_role_created': False, 'geometry_modified': False,
        'physical_geometry_approved': False}
