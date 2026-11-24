"""Recheck unlinked filled-head protection in verified original PNG windows.

Failure to qualify a native window preserves protection. Only a registered,
insufficiently filled unlinked triangle loses its provisional head mask;
independent hard protection and all other accepted heads are retained.
"""
import cv2
import numpy as np

from .source_arrow_observation_v1 import checked_matrix, transform

POLICY = dict(version='native-head-protection/1.0', maximum_heads=64,
    maximum_window_side=1024, minimum_native_fill=.7, minimum_context_support=.8,
    context_margin_fraction=.25, native_pixel_margin=2,
    original_pixels_required=True, linked_heads_released=False)


def mask(shape, polygon):
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.rint(polygon).astype(np.int32)], 1)
    return result > 0


def qualify(gray, original_protected, hard_protected, heads, linked_ids, *, source_size, source_to_working, read_windows):
    if (gray.ndim != 2 or gray.dtype != np.uint8 or gray.size > 4194304
            or any(a.shape != gray.shape or a.dtype != bool for a in (original_protected, hard_protected))):
        raise ValueError('Aligned bounded grayscale and explicit protection layers required')
    matrix = checked_matrix(source_to_working); inverse = np.linalg.inv(matrix)
    if len(heads) > POLICY['maximum_heads'] or len({row['id'] for row in heads}) != len(heads):
        raise ValueError('Unique bounded head observations required')
    _, work_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    rows = []; layers = {}; boxes = []
    for head in heads:
        polygon = np.asarray(head.get('polygon_px', []), float)
        if (polygon.shape != (3, 2) or not np.isfinite(polygon).all() or (polygon < 0).any()
                or (polygon[:, 0] >= gray.shape[1]).any() or (polygon[:, 1] >= gray.shape[0]).any()):
            continue
        layer = mask(gray.shape, polygon); area = int(layer.sum())
        if not 3 <= area <= 4096 or np.mean(work_ink[layer] > 0) < .7:
            continue  # Exactly the previous loader's provisional mask domain.
        layers[head['id']] = layer; native_polygon = transform(polygon, inverse)
        margin = np.ceil(np.ptp(native_polygon, axis=0)*POLICY['context_margin_fraction']).astype(int)+POLICY['native_pixel_margin']
        low = np.floor(native_polygon.min(axis=0)).astype(int)-margin
        high = np.ceil(native_polygon.max(axis=0)).astype(int)+margin+1
        row = dict(id=head['id'], polygon_px=polygon.tolist(), native_polygon_px=native_polygon.tolist(),
            status='keep_provisional', reason='native_window_unqualified', linked=head['id'] in linked_ids)
        if row['linked']:
            row['reason'] = 'linked_head_never_released'
        elif (low >= 0).all() and (high <= source_size).all() and (high-low <= POLICY['maximum_window_side']).all():
            row['box'] = [*low.tolist(), *high.tolist()]; row['window_index'] = len(boxes); boxes.append(row['box'])
        rows.append(row)
    union = hard_protected.copy()
    for layer in layers.values():
        union |= layer
    if not np.array_equal(union, original_protected | hard_protected):
        raise ValueError('Undeclared protection: cannot infer an independent hard mask')
    windows = read_windows(boxes) if boxes else []
    if len(windows) != len(boxes):
        raise ValueError('Native head windows are incomplete')
    for row in rows:
        if 'window_index' not in row:
            continue
        native = windows[row['window_index']]; low = np.array(row['box'][:2]); high = np.array(row['box'][2:])
        if native.dtype != np.uint8 or native.shape != tuple((high-low)[::-1]):
            raise ValueError('Native window dimensions differ')
        if native.min() == native.max():
            row['reason'] = 'native_window_without_contrast'; continue
        _, ink = cv2.threshold(native, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        triangle = mask(native.shape, np.asarray(row['native_polygon_px'])-low)
        if triangle.sum() < 9:
            row['reason'] = 'native_triangle_too_small'; continue
        # Independent registration check in the working frame, within one
        # working raster cell. No content/nominals from reference DXF are used.
        p = transform(np.argwhere(ink > 0)[:, ::-1]+low, matrix)
        q = np.rint(p).astype(int); valid = (q >= 0).all(1) & (q[:, 0] < gray.shape[1]) & (q[:, 1] < gray.shape[0])
        projected = np.zeros_like(gray, np.uint8); projected[q[valid, 1], q[valid, 0]] = 1
        work_box = transform(np.array([[low[0], low[1]], [high[0]-1, low[1]],
            [high[0]-1, high[1]-1], [low[0], high[1]-1]]), matrix)
        lo = np.maximum(0, np.floor(work_box.min(0)).astype(int)); hi = np.minimum(gray.shape[::-1], np.ceil(work_box.max(0)).astype(int)+1)
        selection = np.zeros_like(gray, bool); selection[lo[1]:hi[1], lo[0]:hi[0]] = True
        observed = (work_ink > 0) & selection
        projected = (projected > 0) & selection
        a = np.mean(cv2.dilate(projected.astype(np.uint8), np.ones((3, 3), np.uint8))[observed] > 0) if observed.any() else 0.
        b = np.mean(cv2.dilate(observed.astype(np.uint8), np.ones((3, 3), np.uint8))[projected] > 0) if projected.any() else 0.
        row.update(native_fill=float(np.mean(ink[triangle] > 0)), context_support=[float(a), float(b)])
        if min(a, b) < POLICY['minimum_context_support']:
            row['reason'] = 'native_context_registration_insufficient'; continue
        if row['native_fill'] >= POLICY['minimum_native_fill']:
            row.update(status='keep_native', reason='native_filled_head_supported')
        else:
            row.update(status='release_unlinked_head_mask', reason='native_filled_triangle_not_supported')
    result = hard_protected.copy()
    for row in rows:
        if row['status'] != 'release_unlinked_head_mask':
            result |= layers[row['id']]
    if np.any(hard_protected & ~result):
        raise ValueError('Native head recheck changed hard protection')
    return result, dict(policy=POLICY, heads=rows,
        released_pixels=int(np.count_nonzero(original_protected & ~result)), hard_protection_preserved=True,
        pixel_ink_erased=False, roles_changed=False, physical_geometry_approved=False)
