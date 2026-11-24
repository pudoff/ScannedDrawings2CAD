"""Lossless native ink classes and set-valued coarse interval claims.

Observed carrier ink is geometric evidence, not a semantic DIM approval. Shared
cells retain every current claimant; disconnected ink is never labelled TEXT.
"""
from collections import Counter, defaultdict
from copy import deepcopy

import cv2
import numpy as np

from .native_dimension_transfer_v1 import _sha, _supported_matrix
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix

POLICY = {'maximum_roi_pixels': 4_000_000, 'maximum_working_pixels': 4_000_000,
    'maximum_intervals': 4096, 'maximum_source_points': 2_000_000,
    'maximum_cell_pixels': 8_000_000, 'maximum_events': 2_000_000,
    'maximum_rle_rows': 1_000_000, 'maximum_components': 16384}
CLASSES = {'1': 'observed_carrier_ink', '2': 'protected',
           '3': 'foreign_component', '4': 'unresolved'}


class _Limit(Exception):
    pass


def _charge(work, key, count, limit):
    work[key] = work.get(key, 0)+int(count)
    if work[key] > POLICY[limit]:
        raise _Limit(limit)


def _bounds(points, matrix, origin):
    centers = points@matrix[:2, :2].T+matrix[:2, 2]-origin
    half = np.abs(matrix[:2, :2]).sum(axis=1)/2
    low = np.ceil(centers-half-.5+1e-9).astype(int)
    high = np.floor(centers+half+.5-1e-9).astype(int)+1
    return np.column_stack((low, high))


def _runs(row):
    cuts = np.r_[0, np.flatnonzero(row[1:] != row[:-1])+1, len(row)]
    return [(int(a), int(b), int(row[a])) for a, b in zip(cuts, cuts[1:]) if row[a]]


def observe(gray, working_gray, geometry, roles, chain_row, matrix, *,
            protection, targets, ink, source_protection):
    """Partition a freshly observed source ROI; leave roles/intervals untouched."""
    matrix = checked_matrix(matrix)
    arrays = (gray, working_gray, roles, protection, ink, source_protection)
    before = [_sha(a) for a in arrays]
    metadata = object_sha([geometry, chain_row, matrix.tolist()])
    def target_digest():
        return object_sha([{'role': t['role'], 'ids': sorted(t['ids']),
            'a': np.asarray(t['a']).tolist(), 'b': np.asarray(t['b']).tolist(),
            'body': _sha(t['body']), 'safe': _sha(t['safe'])} for t in targets])
    target_hash = target_digest()
    for a in (gray, working_gray, roles, protection):
        if a.ndim != 2 or a.dtype != np.uint8:
            raise ValueError('Two dimensional uint8 source/working maps required')
    if working_gray.shape != roles.shape or protection.shape != roles.shape:
        raise ValueError('Aligned working maps required')
    if (ink.shape != gray.shape or source_protection.shape != gray.shape
            or ink.dtype != bool or source_protection.dtype != bool):
        raise ValueError('Aligned boolean native masks required')
    box = np.asarray(chain_row['source_box_px'], float)
    if (box.shape != (4,) or not np.isfinite(box).all() or not np.array_equal(box, np.rint(box))
            or tuple((box[2:]-box[:2])[::-1]) != gray.shape):
        raise ValueError('Native ROI differs from its source box')
    if chain_row.get('source_window_sha256', _sha(gray)) != _sha(gray):
        raise ValueError('Native ROI differs from current chain')
    report = {'schema_version': 'native-dimension-source-partition/1.0',
        'chain_id': chain_row['id'], 'source_box_px': box.astype(int).tolist(),
        'size_px': list(gray.shape[::-1]), 'source_window_sha256': _sha(gray),
        'working_gray_sha256': _sha(working_gray), 'roles_sha256': _sha(roles),
        'protection_sha256': _sha(protection), 'source_protection_sha256': _sha(source_protection),
        'geometry_sha256': object_sha(geometry), 'working_to_source': matrix.tolist(),
        'policy': deepcopy(POLICY), 'classes': deepcopy(CLASSES),
        'classes_rle': [], 'claims_rle': [], 'intervals': [], 'components': [],
        'limits': [], 'work': {}, 'complete': False, 'semantic_DIM_approved': False,
        'roles_changed': 0, 'geometry_changed': False, 'source_pixels_erased': 0,
        'single_interval_owner_assumed': False, 'claim_domain': 'all_cells_of_intervals_touching_finite_carrier_body',
        'foreign_component_is_TEXT': False, 'physical_geometry_approved': False}
    work = report['work']
    try:
        if gray.size > POLICY['maximum_roi_pixels'] or roles.size > POLICY['maximum_working_pixels']:
            raise _Limit('raster_budget')
        if not _supported_matrix(matrix):
            raise _Limit('unsupported_source_cell_transform')
        current_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
        if not np.array_equal(current_ink, ink):
            raise ValueError('Native ink differs from current source threshold')
        carriers = [t for t in targets if t['role'] == 2]
        if not carriers:
            raise _Limit('measured_carrier_missing')
        body, safe = np.zeros_like(ink, bool), np.zeros_like(ink, bool)
        for target in carriers:
            for key in ('body', 'safe'):
                if target[key].shape != gray.shape or target[key].dtype != bool:
                    raise ValueError('Aligned boolean observed carrier masks required')
            if np.any(target['safe'] & ~target['body']):
                raise ValueError('Safe carrier exceeds measured body')
            body |= target['body']; safe |= target['safe']
        report['carrier_masks_sha256'] = {'body': _sha(body), 'safe': _sha(safe)}
        blocked = np.asarray(source_protection, bool).copy()
        py, px = np.nonzero((protection > 0) | (roles == 6))
        _charge(work, 'source_points', len(px), 'maximum_source_points')
        for left, top, right, bottom in _bounds(np.column_stack((px, py)), matrix, box[:2]):
            left, top = max(0, int(left)), max(0, int(top))
            right, bottom = min(gray.shape[1], int(right)), min(gray.shape[0], int(bottom))
            if right > left and bottom > top:
                _charge(work, 'cell_pixels', (right-left)*(bottom-top), 'maximum_cell_pixels')
                blocked[top:bottom, left:right] = True
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        if count-1 > POLICY['maximum_components']:
            raise _Limit('maximum_components')
        carrier_ids = set(map(int, np.unique(labels[ink & safe])))-{0}
        if not carrier_ids:
            raise _Limit('observed_carrier_ink_missing')
        typed = {r['id']: r for r in geometry['candidates']}
        intervals = geometry.get('ownership_intervals', geometry['candidates'])
        identities = [r['id'] for r in intervals]
        if len(set(identities)) != len(identities) or len(typed) != len(geometry['candidates']) or not typed.keys() <= set(identities):
            raise ValueError('Unique and complete typed/untyped interval inventory required')
        if len(intervals) > POLICY['maximum_intervals']:
            raise _Limit('maximum_intervals')
        events = defaultdict(list); touched = set(); records = []
        for interval in intervals:
            identity = interval['id']
            points = np.asarray(interval.get('support_points', interval.get('points_px', [])), float)
            if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
                    or not np.array_equal(points, np.rint(points)) or (points < 0).any()
                    or (points[:, 0] >= roles.shape[1]).any() or (points[:, 1] >= roles.shape[0]).any()):
                raise ValueError('Finite integer source samples within working raster required')
            if identity in typed and not np.array_equal(points, np.asarray(typed[identity]['support_points'])):
                raise ValueError('Typed interval differs from original ownership walk')
            _charge(work, 'source_points', len(points), 'maximum_source_points')
            bounds = _bounds(points, matrix, box[:2])
            clipped = bounds.copy()
            clipped[:, :2] = np.maximum(clipped[:, :2], 0)
            clipped[:, 2:] = np.minimum(clipped[:, 2:], gray.shape[::-1])
            inside = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
            indices = np.flatnonzero(inside)
            if not any(body[t:b, l:r].any() for l, t, r, b in clipped[indices]):
                continue
            row = {'id': identity, 'kind': typed.get(identity, {}).get('kind', 'untyped'),
                'source_members': deepcopy(typed.get(identity, {}).get('source_members', interval.get('source_members', []))),
                'source_cells': []}
            for index in indices:
                left, top, right, bottom = map(int, clipped[index])
                _charge(work, 'cell_pixels', (right-left)*(bottom-top), 'maximum_cell_pixels')
                actual = labels[top:bottom, left:right]
                ids, counts = np.unique(actual[actual > 0], return_counts=True)
                touched.update(map(int, ids))
                row['source_cells'].append({'point': points[index].astype(int).tolist(),
                    'bounds': [left, top, right, bottom], 'clipped': not np.array_equal(bounds[index], clipped[index]),
                    'component_ink': {str(int(i)): int(n) for i, n in zip(ids, counts)}})
                _charge(work, 'events', 2*(bottom-top), 'maximum_events')
                for y in range(top, bottom):
                    events[y].extend([(left, identity, 1), (right, identity, -1)])
            records.append(row)
        foreign_ids = set()
        for number in sorted(touched | carrier_ids):
            x, y, width, height, area = map(int, stats[number])
            clipped = x == 0 or y == 0 or x+width == gray.shape[1] or y+height == gray.shape[0]
            foreign = number not in carrier_ids and not clipped
            if foreign:
                foreign_ids.add(number)
            report['components'].append({'id': number, 'box_px': [x, y, x+width, y+height],
                'ink_pixels': area, 'touches_ROI_boundary': clipped,
                'touches_measured_carrier': number in carrier_ids, 'foreign_separation_observed': foreign})
        plane = np.where(ink, 4, 0).astype(np.uint8)
        plane[np.isin(labels, list(foreign_ids))] = 3
        plane[ink & safe] = 1
        plane[ink & blocked] = 2
        class_rows = []
        for y, line in enumerate(plane):
            for left, right, kind in _runs(line):
                _charge(work, 'rle_rows', 1, 'maximum_rle_rows')
                class_rows.append([y, left, right, kind])
        claims = []
        for y, row_events in sorted(events.items()):
            active = Counter(); positions = defaultdict(list)
            for x, identity, delta in row_events:
                positions[x].append((identity, delta))
            last = None
            for x, changes in sorted(positions.items()):
                if last is not None and x > last and active:
                    owners = sorted(active)
                    for a, b, _ in _runs(ink[y, last:x].astype(np.uint8)):
                        _charge(work, 'rle_rows', 1, 'maximum_rle_rows')
                        claims.append([y, last+a, last+b, owners])
                for identity, delta in changes:
                    active[identity] += delta
                    if active[identity] == 0:
                        del active[identity]
                last = x
        report.update(classes_rle=class_rows, claims_rle=claims, intervals=records,
            class_map_sha256=_sha(plane), ink_sha256=_sha(ink),
            effective_source_protection_sha256=_sha(blocked),
            class_pixels={CLASSES[str(k)]: int(np.count_nonzero(plane == k)) for k in range(1, 5)},
            source_ink_pixels=int(ink.sum()), shared_claim_pixels=sum(b-a for _, a, b, ids in claims if len(ids) > 1),
            source_partition_exact=True, complete=True)
    except _Limit as exc:
        report.update(classes_rle=[], claims_rle=[], intervals=[], components=[], limits=[str(exc)],
                      source_partition_exact=False, complete=False)
    if (metadata != object_sha([geometry, chain_row, matrix.tolist()])
            or before != [_sha(a) for a in arrays] or target_hash != target_digest()):
        raise ValueError('Native partition mutated its inputs')
    return {'complete': report['complete'], 'report': report}
