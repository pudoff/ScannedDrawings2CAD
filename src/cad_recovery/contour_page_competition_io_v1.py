"""Authenticate frozen page observations before the pure contour-page veto."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .raster_regions import read_prepared_region


def _gray(root, ref):
    with Image.open(checked(root, ref)) as picture:
        if picture.width * picture.height > 4194304:
            raise ValueError('Bounded page context image required')
        return np.array(picture.convert('L'))


def load_context(root, record, native_ref):
    """Read automatic observations, not prediction maps or semantic labels.

    Rechecks the native crop against source pixels/tiles. Does not re-execute the
    historical detector/template producer and does not claim that it does.
    """
    root = Path(root).resolve()
    packet = json.loads(checked(root, record).read_text(encoding='utf-8'))
    if not packet:
        return None
    checked(root, packet['source_template_model'])
    native = _gray(root, native_ref)
    if not np.array_equal(native, _gray(root, packet['native_gray'])):
        raise ValueError('Page packet uses another native ROI')
    page, detection = packet['page'], packet['detection']
    overview = _gray(root, page['source'])
    if (packet['decoded_sha256'] != sha(overview)
            or detection['source_sha256'] != sha(overview)
            or detection['size_px'] != list(overview.shape[::-1])
            or page.get('source_verified') is not True):
        raise ValueError('Detector is not bound to authenticated overview pixels')
    box = page['patch_bbox']
    if (len(box) != 4 or any(type(v) is not int for v in box)
            or min(box[:2]) < 0 or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError('Valid native ROI bounds required')
    if page['kind'] == 'authenticated_page_overview':
        preparation = checked(root, page['preparation'])
        rgb, proof = read_prepared_region(preparation.parent, box, max_pixels=4194304)
        expected = np.array(Image.fromarray(rgb).convert('L'))
        if proof['source_size_px'] != page['source_size_px']:
            raise ValueError('Source preparation dimensions differ')
        source_scope = 'checked_preparation_and_intersecting_tiles'
    else:
        if list(overview.shape[::-1]) != page['source_size_px']:
            raise ValueError('Unprepared source and overview dimensions differ')
        expected = overview[box[1]:box[3], box[0]:box[2]]
        source_scope = 'checked_whole_source_file'
    if not np.array_equal(native, expected):
        raise ValueError('Native ROI pixels do not match their original page crop')
    source = packet['masks']['template']
    # Keep metadata JSON-only; the selected array is checked independently.
    safe = _gray(root, source['mask_paths']['safe'])
    if (source['source_gray_sha256'] != sha(native)
            or source['page_sha256'] != object_sha(page)
            or source['detector_sha256'] != object_sha(detection)
            or source['mask_sha256']['safe'] != sha(safe)
            or source.get('GT_used') is not False
            or source.get('native_prediction_transfer') is not False):
        raise ValueError('Page source/template bindings differ')
    return {'native_gray': native, 'page': page, 'detection': detection,
            'source_mask': {**source, 'safe': safe},
            'provenance': {'page_record': record, 'native': native_ref,
                           'source_scope': source_scope, 'historical_template_recomputed': False,
                           'native_predictions_used': False, 'GT_used': False}}
