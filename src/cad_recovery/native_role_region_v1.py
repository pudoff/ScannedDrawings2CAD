"""Load a crop and its surrounding source pixels from checked original tiles."""
from pathlib import Path

import numpy as np
from PIL import Image

from .attached_grid_context_io_v1 import MAX_PIXELS
from .dataset_layout_v1 import checked
from .raster_regions import read_prepared_region
from .stroke_roles_v1 import checked_gray


def load(root, page, native):
    checked_gray(native)
    if native.size > MAX_PIXELS:
        return None
    if page['kind'] != 'authenticated_page_overview' or page['source_verified'] is not True:
        raise ValueError('Verified prepared original required')
    x0, y0, x1, y1 = page['patch_bbox']
    width, height = page['source_size_px']
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Native crop bounds outside the original')
    if native.shape != (y1-y0, x1-x0):
        raise ValueError('Original crop dimensions differ')
    transform = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]])
    if not (np.array_equal(transform, page['patch_to_source'])
            and np.array_equal(np.linalg.inv(transform), page['source_to_patch'])):
        raise ValueError('Original crop transform differs')

    # Use at most one crop-width of additional context in each dimension.
    # Padding is symmetric before clipping to the actual raster bounds.
    margin_x, margin_y = native.shape[1]//2, native.shape[0]//2
    while True:
        left, top = max(0, x0-margin_x), max(0, y0-margin_y)
        right, bottom = min(width, x1+margin_x), min(height, y1+margin_y)
        if (right-left)*(bottom-top) <= MAX_PIXELS:
            break
        margin_x //= 2
        margin_y //= 2
    ref = page['preparation']
    preparation = checked(Path(root), ref)
    rgb, proof = read_prepared_region(preparation.parent, [left, top, right, bottom], max_pixels=MAX_PIXELS)
    if (proof['preparation_report_sha256'] != ref['sha256'] or proof['source_size_px'] != [width, height]
            or proof['source_sha256'] != page['original_source']['sha256']):
        raise ValueError('Prepared region belongs to a different original')
    gray = np.array(Image.fromarray(rgb).convert('L'))
    local_box = [x0-left, y0-top, x1-left, y1-top]
    l, t, r, b = local_box
    if not np.array_equal(native, gray[t:b, l:r]):
        raise ValueError('Native crop pixels differ from the authenticated region')
    return {'gray': gray, 'crop_in_region_px': local_box, 'provenance': proof}
