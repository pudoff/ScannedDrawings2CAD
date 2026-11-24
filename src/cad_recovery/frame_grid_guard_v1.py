"""Withdraw new grid roles on source-supported parent-frame hypotheses only."""
from copy import deepcopy
import math
import numpy as np
import cv2
from .grid_role_revision_v1 import sha, check_authority
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'frame-grid-guard/1.0', 'maximum_intervals': 4096,
          'maximum_source_points': 2000000, 'maximum_frame_sides': 96,
          'maximum_side_owner_checks': 250000, 'minimum_source_fraction': .8,
          'direction_tolerance_deg': 8., 'local_edge_margin_in_strokes': 3.,
          'new_frame_roles': False, 'production_admitted': False, 'physical_geometry_approved': False}


def affine(value):
    m = np.asarray(value, float)
    if m.shape != (3, 3) or not np.isfinite(m).all() or not np.allclose(m[2], [0, 0, 1], atol=1e-12) or abs(np.linalg.det(m[:2, :2])) < 1e-12:
        raise ValueError('Finite invertible affine pixel-centre transform required')
    return m


def project(points, matrix):
    p = np.asarray(points, float)
    return np.c_[p, np.ones(len(p))] @ matrix[:2].T


def bind_context(gray, graph, previous, base, owners, page, detector, decoded_page_sha, input_to_native, producer):
    if not page['source_verified'] or detector['source_sha256'] != decoded_page_sha or detector['size_px'] != page['size_px']:
        raise ValueError('Frame detection must match the authenticated page pixels')
    if any(f.get('frame_semantics_confirmed') is not False for f in detector['frames']):
        raise ValueError('Manual/semantic frame targets cannot enter the automatic context')
    m = affine(input_to_native); to_page = affine(page['patch_to_source']); to_patch = affine(page['source_to_patch'])
    if not np.allclose(to_page @ to_patch, np.eye(3), atol=1e-9): raise ValueError('ROI/page transforms disagree')
    box = page['patch_bbox']; native_size = [box[2]-box[0], box[3]-box[1]]
    if not np.allclose(to_page, [[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]]): raise ValueError('ROI origin rebound')
    corners = project([[-.5, -.5], [gray.shape[1]-.5, gray.shape[0]-.5]], m)
    if not np.allclose(corners, [[-.5, -.5], [native_size[0]-.5, native_size[1]-.5]], atol=1e-8):
        raise ValueError('Pixel centres and ROI edges use inconsistent transforms')
    overview_to_page = affine(page['overview_to_source'])
    sides = []
    for frame in detector['frames']:
        for side in frame['sides']:
            p = np.asarray(side['points_overview'], float); u = (p[1]-p[0])/np.linalg.norm(p[1]-p[0]); n = np.array([-u[1], u[0]])
            q = project(p, overview_to_page); v = (q[1]-q[0])/np.linalg.norm(q[1]-q[0]); normal = np.array([-v[1], v[0]])
            scale = abs(normal @ overview_to_page[:2, :2] @ n)
            footprint = .5*np.abs(normal @ (to_page @ m)[:2, :2]).sum()
            radius = (side['stroke_width_overview_px']/2+side['position_uncertainty_overview_px'])*scale+footprint
            sides.append({'points_page': q.tolist(), 'radius_page_px': float(radius),
                          'source_sample_count': side['source_sample_count']})
    if len(sides) > POLICY['maximum_frame_sides']: raise ValueError('Frame side budget exceeded')
    return {'producer': deepcopy(producer), 'kind': 'automatic_page_pixel_context',
            'input_sha256': sha(gray), 'graph_sha256': object_sha(graph), 'previous_sha256': sha(previous),
            'base_sha256': sha(base), 'owners_sha256': sha(owners), 'page_pixel_sha256': decoded_page_sha,
            'page_file': deepcopy(page['source']), 'source_size_px': list(page['source_size_px']),
            'input_to_page': (to_page @ m).tolist(), 'sides': sides,
            'detector_complete': detector['complete'], 'GT_used': False, 'new_frame_roles': False}


def analyze(gray, graph, raw_roles, previous, base, owners, *, authority=None, context=None):
    checked_gray(gray)
    if len(graph['proposals']) > POLICY['maximum_intervals']: raise ValueError('Bounded source intervals required')
    for a in (previous, base, raw_roles):
        if a.shape != gray.shape or a.dtype != np.uint8: raise ValueError('Aligned uint8 role maps required')
    if owners.shape != gray.shape or owners.dtype.kind not in 'iu' or owners.min() < 0 or owners.max() > len(graph['proposals']):
        raise ValueError('Source owner map outside its interval domain')
    permitted = check_authority(gray, raw_roles, authority)
    candidate = permitted & (base == 8) & (previous != 8)
    if np.any(candidate & (owners == 0)): raise ValueError('New grid roles need observed source owners')
    output = {'local_edge': base.copy(), 'page_context': base.copy()}
    audit = {'source_points': 0, 'side_owner_checks': 0, 'limits': [], 'context_available': context is not None}
    decisions = []
    if context is not None:
        hashes = {'input_sha256': sha(gray), 'graph_sha256': object_sha(graph), 'previous_sha256': sha(previous),
                  'base_sha256': sha(base), 'owners_sha256': sha(owners)}
        if any(context.get(k) != v for k, v in hashes.items()) or context.get('kind') != 'automatic_page_pixel_context' or context.get('GT_used') is not False:
            raise ValueError('Stale or nonautomatic frame context')
        matrix = affine(context['input_to_page'])
        if not context['detector_complete']: audit['limits'].append('page_detector_incomplete')
        if len(context['sides']) > POLICY['maximum_frame_sides']: audit['limits'].append('frame_side_budget')
    if candidate.any() and not audit['limits']:
        ink = foreground(gray); distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        for number in sorted(int(i) for i in np.unique(owners[candidate])):
            item = graph['proposals'][number-1]; p = np.asarray(item['points_px'], float)
            if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or (p[:, 0] > gray.shape[1]-1).any() or (p[:, 1] > gray.shape[0]-1).any():
                raise ValueError('Finite observed owner points required')
            audit['source_points'] += len(p)
            if audit['source_points'] > POLICY['maximum_source_points']:
                audit['limits'].append('source_point_budget'); break
            q = np.rint(p).astype(int); radii = distance[q[:, 1], q[:, 0]]; radii = radii[radii > 0]
            if not len(radii): continue
            pen = max(1., float(2*np.quantile(radii, .25)-1))
            margin = POLICY['local_edge_margin_in_strokes']*pen+math.sqrt(.5)
            near = np.minimum.reduce([p[:, 0], p[:, 1], gray.shape[1]-1-p[:, 0], gray.shape[0]-1-p[:, 1]]) <= margin
            selected = candidate & (owners == number)
            if near.mean() >= POLICY['minimum_source_fraction']: output['local_edge'][selected] = previous[selected]
            admitted_sides = []
            if context is not None and len(p) >= 2:
                points = project(p, matrix); center = points.mean(0)
                _, _, vectors = np.linalg.svd(points-center, full_matrices=False); direction = vectors[0]
                yy, xx = np.nonzero(selected); pixels = project(np.c_[xx, yy], matrix)
                withdraw = np.zeros(len(pixels), bool)
                for side_index, side in enumerate(context['sides']):
                    audit['side_owner_checks'] += 1
                    if audit['side_owner_checks'] > POLICY['maximum_side_owner_checks']:
                        audit['limits'].append('side_owner_budget'); break
                    a, b = np.asarray(side['points_page'], float); u = (b-a)/np.linalg.norm(b-a); n = np.array([-u[1], u[0]])
                    if abs(direction @ u) < math.cos(math.radians(POLICY['direction_tolerance_deg'])): continue
                    radius = float(side['radius_page_px'])
                    if not math.isfinite(radius) or radius <= 0: raise ValueError('Finite observed frame radius required')
                    along = (points-a) @ u; normal = np.abs((points-a) @ n)
                    support = (normal <= radius) & (along >= -radius) & (along <= np.linalg.norm(b-a)+radius)
                    if support.mean() < POLICY['minimum_source_fraction']: continue
                    d = (pixels-a) @ u
                    on_side = (np.abs((pixels-a) @ n) <= radius) & (d >= -radius) & (d <= np.linalg.norm(b-a)+radius)
                    withdraw |= on_side
                    admitted_sides.append({'side_index': side_index, 'source_fraction': float(support.mean()),
                                           'radius_page_px': radius, 'withdrawn_pixels': int(on_side.sum())})
                output['page_context'][yy[withdraw], xx[withdraw]] = previous[yy[withdraw], xx[withdraw]]
            decisions.append({'owner_number': number, 'interval_id': item['id'], 'observed_pen_px': pen,
                              'local_edge_source_fraction': float(near.mean()), 'parent_side_matches': admitted_sides})
            if audit['limits']: break
    complete = not audit['limits']
    if not complete:
        for a in output.values(): a[:] = base
    return {**output, 'decisions': decisions, 'audit': audit, 'complete': complete,
            'new_frame_roles': 0, 'source_pixels_erased': 0, 'production_admitted': False, 'physical_geometry_approved': False}
