"""Withdraw contour proposals only on bound, source-supported page-frame ink."""
from copy import deepcopy
import math

import cv2
import numpy as np
from PIL import Image

from .frame_grid_guard_v1 import affine, project
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {
    'version': 'contour-page-competition/1.0',
    'maximum_pixels': 4194304, 'maximum_intervals': 4096,
    'maximum_source_points': 2000000, 'maximum_sides': 96,
    'maximum_side_owner_checks': 250000,
    'direction_tolerance_deg': 8., 'minimum_source_fraction': .8,
    'mask_variant': 'template.safe', 'whole_native_footprint_required': True,
    'positive_role_created': False, 'geometry_modified': False,
    'physical_geometry_approved': False,
}


def _binary(array, shape, name):
    if (array.shape != shape or array.dtype != np.uint8
            or not np.isin(array, [0, 255]).all()):
        raise ValueError('Aligned binary ' + name + ' required')


def _bound_source(gray, native, matrix, method):
    checked_gray(native)
    if native.size > POLICY['maximum_pixels']:
        raise ValueError('Bounded native ROI required')
    m = affine(matrix)
    sy, sx = native.shape[0] / gray.shape[0], native.shape[1] / gray.shape[1]
    expected_matrix = [[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]]
    if (sx < 1 or sy < 1
            or not np.allclose(m, expected_matrix, rtol=0, atol=1e-12)):
        raise ValueError('Source footprint and exact pixel-centre transform required')
    if method == 'CV_AREA':
        expected = cv2.resize(native, gray.shape[::-1], interpolation=cv2.INTER_AREA)
    elif method == 'PIL_BOX':
        expected = np.array(Image.fromarray(native).resize(gray.shape[::-1], Image.Resampling.BOX))
    else:
        raise ValueError('Explicit AREA/BOX resize provenance required')
    if not np.array_equal(expected, gray):
        raise ValueError('Input pixels differ from authenticated native ROI')
    return m


def _whole_footprint_safe(safe, shape):
    """Require all native cells intersecting a view pixel to be safe.

    Integer division gives exact conservative bounds even for odd-sized crops.
    A fractional native cell at an AREA boundary cannot become trusted via rounding.
    """
    height, width = shape
    x = np.arange(width, dtype=np.int64)
    y = np.arange(height, dtype=np.int64)
    left, top = x*safe.shape[1]//width, y*safe.shape[0]//height
    right = ((x+1)*safe.shape[1]+width-1)//width
    bottom = ((y+1)*safe.shape[0]+height-1)//height
    unsafe = cv2.integral((safe == 0).astype(np.uint8))
    count = (unsafe[bottom[:, None], right] - unsafe[top[:, None], right]
             - unsafe[bottom[:, None], left] + unsafe[top[:, None], left])
    return count == 0


def analyze(gray, graph, source_owner, ambiguous_owner, proposed, *, native_gray,
            page, detection, source_mask, input_to_native, resize_method):
    """Caller authenticates producer/file pins; this pure layer checks pixel bindings.

    A source mask is an automatic observation, never semantic ground truth. This
    function creates no positive evidence and never changes accepted baseline roles.
    """
    checked_gray(gray)
    items = graph['proposals']
    if gray.size > POLICY['maximum_pixels'] or len(items) > POLICY['maximum_intervals']:
        raise ValueError('Bounded page competition required')
    _binary(proposed, gray.shape, 'proposal')
    _binary(ambiguous_owner, gray.shape, 'ambiguity mask')
    if (source_owner.shape != gray.shape or source_owner.dtype.kind not in 'ui'
            or source_owner.max(initial=0) > len(items)):
        raise ValueError('Aligned source owners required')
    if np.any((proposed > 0) & ((source_owner == 0) | (ambiguous_owner > 0))):
        raise ValueError('Contour proposal needs an unambiguous observed owner')
    if len({p['id'] for p in items}) != len(items):
        raise ValueError('Unique source interval identities required')

    m = _bound_source(gray, native_gray, input_to_native, resize_method)
    box = page['patch_bbox']
    if (page.get('source_verified') is not True or len(box) != 4
            or any(type(v) is not int for v in box)
            or list(native_gray.shape[::-1]) != [box[2]-box[0], box[3]-box[1]]
            or box[0] < 0 or box[1] < 0
            or box[2] > page['source_size_px'][0] or box[3] > page['source_size_px'][1]):
        raise ValueError('Authenticated page/ROI bounds required')
    to_page = affine(page['patch_to_source'])
    if (not np.allclose(to_page, [[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]])
            or not np.allclose(to_page @ affine(page['source_to_patch']), np.eye(3))):
        raise ValueError('Page origin or inverse transform differs')
    if (detection['size_px'] != page['size_px']
            or source_mask.get('source_gray_sha256') != sha(native_gray)
            or source_mask.get('page_sha256') != object_sha(page)
            or source_mask.get('detector_sha256') != object_sha(detection)
            or source_mask.get('GT_used') is not False
            or source_mask.get('native_prediction_transfer') is not False
            or any(f.get('frame_semantics_confirmed') is not False for f in detection['frames'])):
        raise ValueError('Rebound or semantic page evidence')
    safe = source_mask['safe']
    _binary(safe, native_gray.shape, 'native safe mask')
    if source_mask['mask_sha256']['safe'] != sha(safe):
        raise ValueError('Stale native safe mask')
    sides = [side for frame in detection['frames'] for side in frame['sides']]
    if len(sides) > POLICY['maximum_sides']:
        raise ValueError('Bounded page-frame sides required')
    if safe.any() and not sides:
        raise ValueError('Native safe mask has no observed page-frame side')
    before = object_sha([graph, page, detection, {k: v for k, v in source_mask.items() if k != 'safe'}])
    arrays = (gray, native_gray, source_owner, ambiguous_owner, proposed, safe)
    hashes = [sha(a) for a in arrays]
    audit = {'source_points': 0, 'side_owner_checks': 0, 'limits': []}
    if (graph.get('search_incomplete') or not detection['complete'] or not source_mask['complete']
            or source_mask.get('template_search_complete', True) is not True):
        audit['limits'].append('incomplete_page_or_source_search')

    # All source pixels, not the rounded average or nearest neighbour, must be safe.
    fully_safe = _whole_footprint_safe(safe, gray.shape)
    candidate = (proposed > 0) & fully_safe & (foreground(gray) > 0)
    withdrawn = np.zeros_like(candidate)
    matrix = to_page @ m
    overview = affine(page['overview_to_source'])
    decisions = []
    for number in sorted(map(int, np.unique(source_owner[candidate]))) if not audit['limits'] else []:
        item = items[number-1]
        p = np.asarray(item['points_px'], float)
        if (p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all()
                or (p < 0).any() or (p[:, 0] >= gray.shape[1]).any()
                or (p[:, 1] >= gray.shape[0]).any()):
            raise ValueError('Finite observed source points inside ROI required')
        audit['source_points'] += len(p)
        if audit['source_points'] > POLICY['maximum_source_points']:
            audit['limits'].append('source_point_budget')
            break
        if len(p) < 2 or not item.get('linear_context', False):
            continue
        points = project(p, matrix)
        _, singular, vectors = np.linalg.svd(points-points.mean(0), full_matrices=False)
        if not len(singular) or singular[0] <= 0:
            continue
        yy, xx = np.nonzero(candidate & (source_owner == number))
        pixels = project(np.c_[xx, yy], matrix)
        matched = []
        for side_index, side in enumerate(sides):
            audit['side_owner_checks'] += 1
            if audit['side_owner_checks'] > POLICY['maximum_side_owner_checks']:
                audit['limits'].append('side_owner_budget')
                break
            observed = np.asarray(side['points_overview'], float)
            if observed.shape != (2, 2) or not np.isfinite(observed).all():
                raise ValueError('Finite observed page side required')
            ends = project(observed, overview)
            length = float(np.linalg.norm(ends[1]-ends[0]))
            original_length = float(np.linalg.norm(observed[1]-observed[0]))
            if not length or not original_length:
                raise ValueError('Nondegenerate observed page side required')
            u = (ends[1]-ends[0])/length
            if abs(vectors[0] @ u) < math.cos(math.radians(POLICY['direction_tolerance_deg'])):
                continue
            normal = np.array([-u[1], u[0]])
            original_u = (observed[1]-observed[0])/original_length
            original_n = np.array([-original_u[1], original_u[0]])
            width, error = float(side['stroke_width_overview_px']), float(side['position_uncertainty_overview_px'])
            if not math.isfinite(width+error) or width <= 0 or error < 0:
                raise ValueError('Finite measured side width and uncertainty required')
            scale = abs(normal @ overview[:2, :2] @ original_n)
            radius = (width/2+error)*scale + .5*np.abs(normal @ matrix[:2, :2]).sum()
            along = (points-ends[0]) @ u
            support = ((np.abs((points-ends[0]) @ normal) <= radius)
                       & (along >= -radius) & (along <= length+radius))
            if support.mean() < POLICY['minimum_source_fraction']:
                continue
            position = pixels-ends[0]
            on_side = ((np.abs(position @ normal) <= radius)
                       & (position @ u >= -radius) & (position @ u <= length+radius))
            withdrawn[yy[on_side], xx[on_side]] = True
            matched.append({'side': side_index, 'source_fraction': float(support.mean()),
                            'radius_page_px': float(radius), 'pixels': int(on_side.sum())})
        if matched:
            decisions.append({'owner_number': number, 'interval_id': item['id'], 'sides': matched})
        if audit['limits']:
            break
    complete = not audit['limits']
    mask = proposed.copy()
    if complete:
        mask[withdrawn] = 0
    else:
        mask[:] = 0
        withdrawn[:] = False
    if (hashes != [sha(a) for a in arrays]
            or before != object_sha([graph, page, detection, {k: v for k, v in source_mask.items() if k != 'safe'}])):
        raise ValueError('Page competition mutated source observations')
    return {'mask': mask, 'veto_mask': withdrawn.astype(np.uint8)*255,
            'fully_safe_mask': fully_safe.astype(np.uint8)*255,
            'complete': complete, 'source_immutable': True, 'context_available': True,
            'page_veto_pixels': int(withdrawn.sum()), 'withdrawn_pixels': int(np.count_nonzero(proposed)-np.count_nonzero(mask)),
            'decisions': decisions, 'audit': audit, 'policy': deepcopy(POLICY),
            'positive_role_created': False, 'geometry_modified': False, 'physical_geometry_approved': False}
