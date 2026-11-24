"""Bind a matched original stamp rule to an observed working-grid owner."""

from math import cos, radians

import numpy as np

from .attached_grid_competition_v1 import POLICY


MINIMUM_POINTS = 3
MINIMUM_DOT = cos(radians(POLICY['direction_tolerance_deg']))


class RuleProjection:
    def __init__(self, original_to_working, shape):
        self.matrix = original_to_working
        self.combined = np.zeros(shape, np.uint8)
        self.rules = []
        self.frame_rules = []
        self.frame = np.zeros(shape, np.uint8)

    def _pixels(self, points):
        height, width = self.combined.shape
        mapped = np.rint(points @ self.matrix[:2, :2].T + self.matrix[:2, 2]).astype(int)
        x, y = mapped.T
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        return np.unique(mapped[valid], axis=0)

    def add_frame(self, origin, fit):
        if not fit['complete']:
            return
        y, x = np.nonzero(fit['safe'])
        points = np.column_stack((x, y))
        for side in fit['decisions']:
            if not side['fit']:
                continue
            endpoints = np.asarray(side['points_native'])
            delta = endpoints[1] - endpoints[0]
            length = np.linalg.norm(delta)
            axis = delta / length
            normal = np.array([-axis[1], axis[0]])
            offsets = points - endpoints[0]
            distance = abs(offsets @ normal)
            along = offsets @ axis
            nearby = ((along >= 0) & (along <= length)
                      & (distance <= side['width_native_px'] / 2 + side['error_native_px']))
            pixels = self._pixels(points[nearby] + origin)
            if not len(pixels):
                continue
            mask = np.zeros_like(self.frame, bool)
            x, y = pixels.T
            mask[y, x] = True
            direction = self.matrix[:2, :2] @ axis
            self.frame_rules.append((direction / np.linalg.norm(direction), mask))
            self.frame[mask] = 255

    def add(self, origin, matched, observations, masks):
        by_id = {row['id']: row for row in observations['rules']}
        for rule in masks['rule_masks']:
            identity = rule['rule_id']
            left, top, right, bottom = rule['bbox_source_px']
            yy, xx = np.nonzero((matched[top:bottom, left:right] > 0) & (rule['safe'] > 0))
            if not len(xx):
                continue
            points = np.column_stack((xx + left + origin[0], yy + top + origin[1]))
            pixels = self._pixels(points)
            if not len(pixels):
                continue
            ends = np.asarray(by_id[identity]['points_source_px'])
            direction = self.matrix[:2, :2] @ (ends[1] - ends[0])
            direction /= np.linalg.norm(direction)
            bit = 1 if abs(direction[0]) > abs(direction[1]) else 2
            x, y = pixels.T
            self.combined[y, x] |= bit
            self.rules.append((f'{origin[0]},{origin[1]}/{identity}', direction, pixels))


def stamp_pixels(rules, eligible, owners, source_owner):
    """Match one rule at a time, preserving unassigned and competing ink."""
    assigned = np.zeros_like(eligible)
    decisions = []
    minimum_fraction = POLICY['minimum_source_fraction']
    for identity, direction, pixels in rules:
        x, y = pixels.T
        owner_numbers = source_owner[y, x]
        probe = eligible[y, x] & (owner_numbers > 0)
        if not probe.any():
            continue
        support = np.zeros_like(eligible)
        support[y, x] = True
        along = pixels @ direction
        start, end = along.min(), along.max()
        for number in np.unique(owner_numbers[probe]):
            owner = owners[number - 1]
            points = np.asarray(owner['points_px'])
            if len(points) < MINIMUM_POINTS or not owner.get('linear_context', False):
                continue
            _, singular, axes = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
            if singular[0] == 0 or abs(axes[0] @ direction) < MINIMUM_DOT:
                continue
            longitudinal = points @ direction
            interval = points[(longitudinal >= start - 1e-8) & (longitudinal <= end + 1e-8)]
            if len(interval) < MINIMUM_POINTS:
                continue
            u, v = np.rint(interval).astype(int).T
            fraction = np.count_nonzero(support[v, u]) / len(interval)
            if fraction < minimum_fraction:
                continue
            used = probe & (owner_numbers == number) & ~assigned[y, x]
            if used.any():
                assigned[y[used], x[used]] = True
                decisions.append({'rule': identity, 'owner': owner['id'], 'pixels': int(used.sum()),
                                  'centreline_fraction': fraction})
    return assigned, decisions


def guard_export_layers(assembly, frame_rules):
    """Withhold a GRID layer where a joined LINE follows a verified frame side."""
    withheld = []
    for obj in assembly['objects']:
        if obj['role'] != 8 or obj['kind'] != 'line':
            continue
        x, y = np.asarray(obj['support_points'], dtype=int).T
        ends = np.asarray(obj['points_px'], dtype=float)
        axis = ends[1] - ends[0]
        axis /= np.linalg.norm(axis)
        confirmed = max((int(mask[y, x].sum()) for direction, mask in frame_rules
                         if abs(direction @ axis) >= MINIMUM_DOT), default=0)
        if confirmed >= MINIMUM_POINTS:
            withheld.append({'object_id': obj['id'], 'frame_support': confirmed, 'support': len(x)})
            obj['role'] = 255
    assembly['stamp_export_withheld'] = withheld
    return assembly
