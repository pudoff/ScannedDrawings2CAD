"""Observe bounded table rules on original tiles next to a verified sheet frame."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import native_role_region_v1
from .attached_grid_context_io_v1 import MAX_PIXELS, load as load_original
from .dataset_layout_v1 import checked, read
from .frame_view_guard_v1 import detect
from .grid_role_revision_v1 import sha
from .native_frame_v1 import source_masks
from .native_table_roles_v1 import MAX_PEN, analyze as table_roles
from .page_grid_source_masks_v1 import POLICY as MASK_POLICY, build as rule_masks
from .part_admission_v1 import object_sha
from .source_stamp_context_v1 import analyze as stamp_observations, projection_context
from .stroke_roles_v1 import checked_gray


def translation(x, y):
    return [[1, 0, x], [0, 1, y], [0, 0, 1]]


def axis_open(mask, run, axis):
    shape = (1, run) if axis == 0 else (run, 1)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones(shape, np.uint8)) > 0


def observe_window(gray, frame_safe, *, edges):
    """Keep only isolated, observed rules from cells attached to the frame."""
    checked_gray(gray)
    if gray.size > MAX_PIXELS or frame_safe.shape != gray.shape or frame_safe.dtype != bool:
        raise ValueError('Bounded aligned original frame evidence required')
    if len(edges) != 4 or any(type(edge) is not bool for edge in edges):
        raise ValueError('Original window edge flags required')
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if np.any(frame_safe & (ink == 0)):
        raise ValueError('Frame evidence contains original paper')
    digest = sha(gray)
    observations = {'source_sha256': digest, 'size_px': list(gray.shape[::-1]), 'complete': True,
                    'semantic_GRID_or_PART_assigned': False, 'rules': [], 'groups': []}
    table = table_roles(gray, original_edge=any(edges), thin_rules=True)
    observations['complete'] = table['complete']
    audit = {'networks': len(table['networks']), 'attached_networks': 0, 'perpendicular_contacts': [],
             'local_mask_pixels': 0, 'limits': []}
    fitted = []
    if table['complete'] and any(edges) and table['networks'] and frame_safe.any():
        run = table['minimum_run_px'] | 1
        axes = [axis_open(ink, run, axis) for axis in range(2)]
        safe = frame_safe.astype(np.uint8)
        crossings = (axis_open(safe, run, 0) & axes[1]) | (axis_open(safe, run, 1) & axes[0])
        stable = (table['stable_grid'] | table['thin_grid']) & ~frame_safe
        _, labels = cv2.connectedComponents(table['table_context'].astype(np.uint8), connectivity=8)
        for network in sorted(set(map(int, np.unique(labels[crossings]))) - {0}):
            cell_network = labels == network
            count, _, _, centers = cv2.connectedComponentsWithStats((cell_network & crossings).astype(np.uint8))
            if count < 3 or max(np.ptp(centers[1:], axis=0)) < run:
                continue
            audit['attached_networks'] += 1
            audit['perpendicular_contacts'].append(count - 1)
            identities = []
            for axis in range(2):
                admissible = stable & cell_network & axes[axis] & ~axes[1-axis]
                _, components, bounds, _ = cv2.connectedComponentsWithStats(admissible.astype(np.uint8), connectivity=8)
                for component, (left, top, width, height, _) in enumerate(bounds[1:], 1):
                    length, pen = (width, height) if axis == 0 else (height, width)
                    if length < max(run, 5*pen) or pen > MAX_PEN:
                        continue
                    if len(fitted) >= MASK_POLICY['maximum_rules']:
                        audit['limits'].append('original_window_rule_budget')
                        break
                    if audit['local_mask_pixels'] + int(width)*int(height) > MASK_POLICY['maximum_local_mask_pixels']:
                        audit['limits'].append('original_window_mask_budget')
                        break
                    right, bottom = int(left + width), int(top + height)
                    left, top = int(left), int(top)
                    rule_mask = (components[top:bottom, left:right] == component).astype(np.uint8)*255
                    # Avoid end-to-end projection over an internal span with
                    # missing original evidence, including clipped intersections.
                    if not np.all((rule_mask > 0).any(axis=0 if axis == 0 else 1)):
                        continue
                    middle = (top + bottom - 1)/2 if axis == 0 else (left + right - 1)/2
                    points = ([[left, middle], [right-1, middle]] if axis == 0
                              else [[middle, top], [middle, bottom-1]])
                    identity = f'original-window-rule-{len(fitted)}'
                    identities.append(identity)
                    observations['rules'].append({'id': identity, 'points_source_px': points,
                        'width_px': int(pen), 'uncertainty_px': .5, 'grounded_group_member': True,
                        'usage': 'observed_crop_context_only', 'CAD_primitive_created': False})
                    fitted.append({'rule_id': identity, 'bbox_source_px': [left, top, right, bottom],
                                   'safe': rule_mask, 'safe_sha256': sha(rule_mask)})
                    audit['local_mask_pixels'] += rule_mask.size
                if audit['limits']:
                    break
            if identities:
                observations['groups'].append({'rule_ids': identities, 'semantic_GRID_confirmed': False})
            if audit['limits']:
                break
    if audit['limits']:
        observations.update(complete=False, rules=[], groups=[])
        fitted = []
    union = np.zeros_like(gray)
    for rule in fitted:
        x0, y0, x1, y1 = rule['bbox_source_px']
        union[y0:y1, x0:x1] |= rule['safe']
    if np.any((union > 0) & ((ink == 0) | frame_safe)):
        raise ValueError('Rule mask escaped authorized original ink')
    masks = {'source_sha256': digest, 'observations_sha256': object_sha(observations),
             'rule_masks': fitted, 'safe_mask': union, 'safe_mask_sha256': sha(union),
             'complete': observations['complete'], 'semantic_GRID_or_PART_assigned': False,
             'requires_owner_direction_binding': True, 'audit': audit}
    return observations, masks


def load(root, record, native_ref):
    """Prefer whole original pixels. Fall back to a bounded authenticated window."""
    root = Path(root)
    baseline = load_original(root, record, native_ref)
    if baseline['available']:
        source = baseline['source_gray']
        cv2.setRNGSeed(0)
        observations = projection_context(source, stamp_observations(source))
        return {**baseline, 'scope': 'whole_original', 'whole_page_processed': True, 'observations': observations,
                'masks': rule_masks(source, observations)}
    if baseline['reason'] != 'original_page_pixel_budget':
        return baseline
    page = read(checked(root, record))['page']
    if page['kind'] != 'authenticated_page_overview':
        return baseline
    with Image.open(checked(root, native_ref)) as source:
        expected = (page['patch_bbox'][2]-page['patch_bbox'][0], page['patch_bbox'][3]-page['patch_bbox'][1])
        if source.size != expected or source.width*source.height > MAX_PIXELS:
            raise ValueError('Bounded original crop dimensions required')
        native = np.array(source.convert('L'))
    window = native_role_region_v1.load(root, page, native)
    gray = window['gray']
    left, top, right, bottom = window['provenance']['source_box_px']
    width, height = page['source_size_px']
    preparation = read(checked(root, page['preparation']))
    preview = preparation['overview']
    if (page['source']['sha256'] != preview['sha256'] or page['size_px'] != preview['size_px']
            or page['overview_to_source'] != preview['overview_to_source']):
        raise ValueError('Frame overview is unrelated to the original tile preparation')
    edges = (left == 0, top == 0, right == width, bottom == height)
    if not any(edges):
        return {**baseline, 'reason': 'window_does_not_reach_original_edge', 'window': window['provenance']}
    with Image.open(checked(root, page['source'])) as source:
        if list(source.size) != page['size_px'] or source.width*source.height > MAX_PIXELS:
            raise ValueError('Bounded registered overview required')
        overview = np.array(source.convert('L'))
    cv2.setRNGSeed(0)
    detection = detect(overview)
    if not detection['complete']:
        return {**baseline, 'reason': 'incomplete_overview_frame_detection', 'window': window['provenance']}
    window_page = {**page, 'patch_bbox': [left, top, right, bottom],
                   'patch_to_source': translation(left, top), 'source_to_patch': translation(-left, -top)}
    frame = source_masks(gray, window_page, detection)
    if not frame['complete']:
        return {**baseline, 'reason': 'incomplete_original_frame_fit', 'window': window['provenance']}
    observations, masks = observe_window(gray, frame['safe'] > 0, edges=edges)
    local_box = window['crop_in_region_px']
    normalized = {'source_verified': True, 'source_size_px': list(gray.shape[::-1]),
                  'size_px': list(gray.shape[::-1]), 'patch_bbox': local_box,
                  'patch_to_source': translation(*local_box[:2]),
                  'source_to_patch': translation(-local_box[0], -local_box[1]),
                  'overview_to_source': translation(0, 0)}
    return {'available': True, 'reason': None, 'scope': 'authenticated_original_window',
            'whole_page_processed': False, 'source_gray': gray, 'native_gray': native,
            'page': normalized, 'observations': observations, 'masks': masks,
            'provenance': {'original_page_sha256': object_sha(page), 'window': window['provenance'],
                           'window_to_original': translation(left, top), 'frame_mask_sha256': sha(frame['safe']),
                           'decoded_window_sha256': sha(gray), 'outward_sides': list(edges)}}
