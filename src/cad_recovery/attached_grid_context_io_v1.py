"""Read bounded original-page pixels for grid competition, never a scaled overview."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .contour_page_competition_v1 import _whole_footprint_safe
from .dataset_layout_v1 import checked
from .frame_grid_guard_v1 import affine
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .raster_regions import read_prepared_region

MAX_PIXELS = 4194304


def _gray(root, reference):
    with Image.open(checked(root, reference)) as picture:
        if picture.width*picture.height > MAX_PIXELS:
            raise ValueError('Bounded original pixels required')
        return np.array(picture.convert('L'))


def load(root, record, native_ref):
    """A missing/oversized full page is unavailable; corrupt bindings are errors.

    Prepared original samples are authenticated via preparation and tile hashes,
    not by rehashing the original large source file. Only coordinates enter the
    normalized page; the overview image is never relabelled as original pixels.
    """
    root = Path(root).resolve()
    packet = json.loads(checked(root, record).read_text(encoding='utf-8'))
    if not packet:
        return {'available': False, 'reason': 'missing_page_context', 'page_record': deepcopy(record)}
    page = packet['page']; size = page['source_size_px']; box = page['patch_bbox']
    if (page.get('source_verified') is not True or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
            or len(box) != 4 or any(type(v) is not int for v in box)
            or not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])):
        raise ValueError('Authenticated original page bounds required')
    expected = [[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]]
    forward, inverse = affine(page['patch_to_source']), affine(page['source_to_patch'])
    if (not np.allclose(forward, expected, rtol=0, atol=1e-12)
            or not np.allclose(forward @ inverse, np.eye(3), rtol=0, atol=1e-12)):
        raise ValueError('Rebound original page crop')
    if size[0]*size[1] > MAX_PIXELS:
        return {'available': False, 'reason': 'original_page_pixel_budget',
            'source_size_px': deepcopy(size), 'maximum_pixels': MAX_PIXELS,
            'page_record': deepcopy(record), 'overview_not_substituted': True}
    native = _gray(root, native_ref)
    if page['kind'] == 'authenticated_page_overview':
        preparation = checked(root, page['preparation'])
        rgb, proof = read_prepared_region(preparation.parent, [0, 0, *size], max_pixels=MAX_PIXELS)
        if proof['source_size_px'] != size:
            raise ValueError('Preparation original dimensions differ')
        source = np.array(Image.fromarray(rgb).convert('L'))
        integrity = {'scope': 'checked_full_preparation_tiles', 'preparation': deepcopy(page['preparation']),
                     'proof': proof, 'source_original_rehashed': False}
    elif page['kind'] in ('synthetic_full_page_pixels', 'synthetic_original_page'):
        source = _gray(root, page['source'])
        if (page['size_px'] != size or list(source.shape[::-1]) != size
                or not np.allclose(affine(page['overview_to_source']), np.eye(3), rtol=0, atol=1e-12)):
            raise ValueError('Original source is not the authenticated overview')
        integrity = {'scope': 'checked_whole_source_file', 'source': deepcopy(page['source']),
                     'source_original_rehashed': True}
    else:
        raise ValueError('Unsupported original-page provenance')
    if not np.array_equal(native, source[box[1]:box[3], box[0]:box[2]]):
        raise ValueError('Native ROI differs from authenticated original pixels')
    normalized = {k: deepcopy(page[k]) for k in ('source_size_px', 'patch_bbox',
                                              'patch_to_source', 'source_to_patch')}
    normalized.update(source_verified=True, size_px=deepcopy(size), overview_to_source=np.eye(3).tolist())
    return {'available': True, 'reason': None, 'source_gray': source, 'native_gray': native,
        'page': normalized, 'provenance': {'page_record': deepcopy(record), 'native': deepcopy(native_ref),
            'original_page_sha256': object_sha(page), 'decoded_original_sha256': sha(source),
            'normalized_page_sha256': object_sha(normalized), **integrity}}


def protect_native_baseline(baseline, input_shape):
    """Protect an input pixel if any intersecting native cell is already accepted."""
    if (baseline.dtype != np.uint8 or baseline.ndim != 2
            or not np.isin(baseline, list(range(10))+[255]).all()):
        raise ValueError('Native baseline roles required')
    if (len(input_shape) != 2 or min(input_shape) < 1 or baseline.size > MAX_PIXELS
            or input_shape[0] > baseline.shape[0] or input_shape[1] > baseline.shape[1]):
        raise ValueError('Bounded native/downsampled input shape required')
    unknown = (baseline == 255).astype(np.uint8)*255
    return (~_whole_footprint_safe(unknown, input_shape)).astype(np.uint8)*255
