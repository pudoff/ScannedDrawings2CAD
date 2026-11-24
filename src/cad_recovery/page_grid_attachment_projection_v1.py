"""Project bound full-source rule observations; clipped spans are not CAD targets."""
from copy import deepcopy
import numpy as np
import cv2
from PIL import Image

from .frame_grid_guard_v1 import affine
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray


def clip(points, width, height):
    p = np.asarray(points, float); delta = p[1]-p[0]
    lo, hi = 0., 1.
    for axis, span in ((0, width), (1, height)):
        minimum, maximum = -.5, span-.5
        if abs(delta[axis]) <= 1e-12:
            if p[0, axis] < minimum or p[0, axis] > maximum: return None
        else:
            a, b = sorted([(minimum-p[0, axis])/delta[axis], (maximum-p[0, axis])/delta[axis]])
            lo, hi = max(lo, a), min(hi, b)
            if hi <= lo: return None
    return lo, hi


def project(source_gray, observations, native_gray, input_gray, page, input_to_native, *, resize_method):
    for gray in (source_gray, native_gray, input_gray): checked_gray(gray)
    if any(gray.size > 4194304 for gray in (source_gray, native_gray, input_gray)):
        raise ValueError('Bounded source and ROI projection required')
    source_size = list(source_gray.shape[::-1])
    if (observations['source_sha256'] != sha(source_gray) or observations['size_px'] != source_size
            or observations.get('semantic_GRID_or_PART_assigned') is not False):
        raise ValueError('Source-bound unconfirmed observations required')
    if (page.get('source_verified') is not True or page['size_px'] != source_size
            or page['source_size_px'] != source_size
            or not np.allclose(affine(page['overview_to_source']), np.eye(3), rtol=0, atol=1e-12)):
        raise ValueError('This projection requires original full-source pixels, not a downsampled overview')
    box = page['patch_bbox']
    if (len(box) != 4 or any(type(i) is not int for i in box) or min(box[:2]) < 0
            or box[2] <= box[0] or box[3] <= box[1] or box[2] > source_size[0] or box[3] > source_size[1]):
        raise ValueError('Valid original ROI bounds required')
    native_to_source = affine(page['patch_to_source'])
    expected = [[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]]
    if (not np.allclose(native_to_source, expected, rtol=0, atol=1e-12)
            or not np.allclose(native_to_source @ affine(page['source_to_patch']), np.eye(3), rtol=0, atol=1e-12)):
        raise ValueError('ROI origin or inverse is rebound')
    if not np.array_equal(native_gray, source_gray[box[1]:box[3], box[0]:box[2]]):
        raise ValueError('ROI pixels do not match original source')
    matrix = affine(input_to_native)
    sx, sy = native_gray.shape[1]/input_gray.shape[1], native_gray.shape[0]/input_gray.shape[0]
    if (min(sx, sy) < 1 or not np.allclose(matrix, [[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]], rtol=0, atol=1e-12)):
        raise ValueError('Exact pixel-centre resize transform required')
    if resize_method == 'CV_AREA':
        expected_pixels = cv2.resize(native_gray, input_gray.shape[::-1], interpolation=cv2.INTER_AREA)
    elif resize_method == 'PIL_BOX':
        expected_pixels = np.array(Image.fromarray(native_gray).resize(input_gray.shape[::-1], Image.Resampling.BOX))
    else:
        raise ValueError('Explicit AREA/BOX provenance required')
    if not np.array_equal(input_gray, expected_pixels): raise ValueError('Resized source pixels differ')
    before = object_sha([observations, page]); hashes = [sha(a) for a in (source_gray, native_gray, input_gray)]
    groups = {identity for group in observations['groups'] for identity in group['rule_ids']}
    identifiers = [rule['id'] for rule in observations['rules']]
    if len(set(identifiers)) != len(identifiers) or not groups <= set(identifiers):
        raise ValueError('Unique bound rule IDs required')
    rows = []
    for rule in observations['rules']:
        if rule['grounded_group_member'] != (rule['id'] in groups): raise ValueError('Grounded rule membership differs')
        if not rule['grounded_group_member'] or not observations['complete']: continue
        source = np.asarray(rule['points_source_px'], float)
        if source.shape != (2, 2) or not np.isfinite(source).all() or np.linalg.norm(source[1]-source[0]) <= 0:
            raise ValueError('Finite observed rule endpoints required')
        native = source-[box[0], box[1]]
        interval = clip(native, native_gray.shape[1], native_gray.shape[0])
        row = {'rule_id': rule['id'], 'source_points_px': source.tolist(), 'visibility': 'outside_input',
               'visible_source_parameter_interval': [], 'points_native_px': [], 'points_input_px': [],
               'usage': 'role_context_only', 'CAD_primitive_created': False}
        if interval is not None:
            lo, hi = interval; points = native[0]+np.array([lo, hi])[:, None]*(native[1]-native[0])
            inverse = np.linalg.inv(matrix)
            pixels = np.c_[points, np.ones(2)] @ inverse[:2].T
            row.update(visibility='full' if lo == 0 and hi == 1 else 'partial',
                       visible_source_parameter_interval=[lo, hi], points_native_px=points.tolist(), points_input_px=pixels.tolist())
        rows.append(row)
    immutable = before == object_sha([observations, page]) and hashes == [sha(a) for a in (source_gray, native_gray, input_gray)]
    if not immutable: raise ValueError('Rule projection mutated its source')
    return {'source_sha256': hashes[0], 'native_sha256': hashes[1], 'input_sha256': hashes[2],
            'observations_sha256': object_sha(observations), 'input_to_native': matrix.tolist(),
            'native_to_source': native_to_source.tolist(), 'rules': rows, 'complete': observations['complete'],
            'source_immutable': True, 'semantic_GRID_or_PART_assigned': False,
            'coordinate_system': 'original page / native ROI / input pixel centres',
            'resize_method': resize_method, 'physical_geometry_approved': False}
