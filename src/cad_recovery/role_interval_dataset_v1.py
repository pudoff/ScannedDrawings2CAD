"""Source-only stroke observations, with separate partial-mask supervision.

The learning task remains experimental partial pixel-role supervision. Targets
are attached to fixed source intervals only after observations/features exist.
Unreviewed, ambiguous, clipped and mixed-role intervals remain unlabelled.
"""
from collections import Counter
import hashlib
import math
import unicodedata

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .curve_graph_v1 import graph_paths

FEATURES = ('length_in_strokes', 'chord_fraction', 'bbox_aspect', 'cos_2angle', 'sin_2angle',
    'width_relative_median', 'width_q90_to_q25', 'mean_turn', 'max_turn',
    'parent_length_in_strokes', 'parent_chord_fraction', 'parent_closed',
    'start_degree', 'end_degree', 'sheet_x', 'sheet_y', 'nearest_sheet_edge',
    'ink_density_3_widths', 'ink_density_6_widths', 'ink_density_12_widths', 'local_context_clipped')
POLICY = dict(version='role-interval-observations/1.0', maximum_pixels=1024**2,
    maximum_skeleton_pixels=120000, maximum_paths=4096, interval_edges=32, minimum_edges=8,
    targets='all unique source pixels: same reviewed role, state=1, exactly one matching role bit',
    clipped_source_paths='not training targets', ignore=255, color_features=False,
    physical_geometry_approved=False, dense_ground_truth=False)


def canonical(path):
    value = unicodedata.normalize('NFC', path.replace('\\', '/')).casefold()
    return value[8:] if value.startswith('dataset/') else value


class SourceBoundary:
    """Metadata-only unions. Closed families are forbidden; development stays known."""
    def __init__(self, group_sets, inventory, registries, fresh_sources, development_sources):
        self.parent = {}
        for groups in group_sets:
            for group in groups:
                self.union(group['members'])
        for registry in registries:
            for group in registry.get('group_commitments', []):
                self.union(group.get('member_paths', group.get('members', [])))
        by_sha = {}
        for row in inventory:
            key = row.get('sha256')
            if key:
                by_sha.setdefault(key, []).append(row['path'])
        for paths in by_sha.values():
            self.union(paths)
        self.family_splits = {}
        for row in inventory:
            if row.get('declared_split'):
                self.family_splits.setdefault(self.find(row['path']), set()).add(row['declared_split'])
        closed_paths = set(); self.closed_hashes = set()
        for registry in registries:
            for row in registry['assignments']:
                if row['split'] == 'test':
                    closed_paths.add(canonical(row['source_path']))
                    self.closed_hashes.add(row['source_sha256'])
            for group in registry.get('group_commitments', []):
                if group.get('split') == 'test' or 'test' in group.get('declared_splits', []):
                    closed_paths.update(map(canonical, group.get('member_paths', group.get('members', []))))
        for row in fresh_sources:
            closed_paths.add(canonical(row['input']))
            self.closed_hashes.add(row['input_sha256'])
        for key in self.closed_hashes:
            closed_paths.update(map(canonical, by_sha.get(key, [])))
        self.closed_roots = {self.find(path) for path in closed_paths}
        self.development_paths = {canonical(row['path']) for row in development_sources}
        self.development_hashes = {row['sha256'] for row in development_sources}
        self.development_roots = {self.find(path) for path in self.development_paths}

    def find(self, value):
        value = canonical(value)
        self.parent.setdefault(value, value)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            old = self.parent[value]; self.parent[value] = root; value = old
        return root

    def union(self, members):
        roots = sorted({self.find(path) for path in members})
        for root in roots[1:]:
            self.parent[root] = roots[0]

    def screen(self, source):
        path = canonical(source['path']); root = self.find(path)
        reasons = []
        if root in self.closed_roots or source['sha256'] in self.closed_hashes:
            reasons.append('closed_or_reserved_source_family')
        if len(self.family_splits.get(root, set())) > 1:
            reasons.append('cross_directory_split_family')
        if path in self.development_paths or source['sha256'] in self.development_hashes:
            reasons.append('current_development_source_no_new_training_targets')
        return dict(blocked=bool(reasons), reasons=reasons,
            canonical_family='role-family-'+hashlib.sha256(root.encode()).hexdigest()[:20],
            development_related_family=root in self.development_roots,
            independent_evaluation=False)


def partition(rows):
    """Choose an internal selection group from train metadata, before pixel reads."""
    groups = sorted({row['boundary']['canonical_family'] for row in rows if row['split'] == 'train'},
                    key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    if len(groups) < 3:
        return {}, dict(ready=False, reason='fewer_than_three_admissible_train_groups', groups=groups)
    held = groups[-1]
    usage = {row['id']: ('diagnostic_validation' if row['split'] == 'validation' else
             'internal_validation' if row['boundary']['canonical_family'] == held else 'fit') for row in rows}
    return usage, dict(ready=True, held_train_family=held,
        policy='largest SHA256(canonical family ID) held before decoding; all crops/aliases stay together',
        global_directory_split_unchanged=True, original_validation_model_selection_allowed=False)


def _features(points, parent, gray, distance, global_width, transform, source_size, integral):
    delta = np.diff(points, axis=0); lengths = np.linalg.norm(delta, axis=1)
    length = float(lengths.sum()); chord = points[-1]-points[0]
    angle = math.atan2(chord[1], chord[0]); widths = 2*distance[points[:, 1], points[:, 0]]-1
    widths = np.maximum(1., widths); width = float(np.median(widths))
    units = delta/np.maximum(lengths[:, None], 1e-12)
    turns = np.arccos(np.clip(np.sum(units[:-1]*units[1:], axis=1), -1, 1))
    parent_points = np.asarray(parent['points']); parent_length = float(np.linalg.norm(np.diff(parent_points, axis=0), axis=1).sum())
    center = points.mean(0); source_center = transform[:2, :2]@center+transform[:2, 2]
    position = source_center/np.maximum(np.asarray(source_size)-1, 1)
    densities = []; clipped = False
    for factor in (3, 6, 12):
        radius = min(128, max(2, math.ceil(factor*width)))
        l, t = np.floor(center-radius).astype(int); r, b = np.ceil(center+radius+1).astype(int)
        clipped |= l < 0 or t < 0 or r > gray.shape[1] or b > gray.shape[0]
        l, t = max(0, l), max(0, t); r, b = min(gray.shape[1], r), min(gray.shape[0], b)
        densities.append(float((integral[b, r]-integral[t, r]-integral[b, l]+integral[t, l])/max(1, (r-l)*(b-t))))
    span = np.ptp(points, axis=0)+1
    return [math.log1p(length/width), float(np.linalg.norm(chord)/max(length, 1e-12)),
        float(np.log(span[0]/span[1])), math.cos(2*angle), math.sin(2*angle), width/global_width,
        float(np.quantile(widths, .9)/np.quantile(widths, .25)),
        float(turns.mean()) if len(turns) else 0., float(turns.max()) if len(turns) else 0.,
        math.log1p(parent_length/width), float(np.linalg.norm(parent_points[-1]-parent_points[0])/max(parent_length, 1e-12)),
        float(parent['closed']), float(parent['start_degree']), float(parent['end_degree']),
        float(position[0]), float(position[1]), float(np.min(np.r_[position, 1-position])),
        *densities, float(clipped)]


def observe(gray, *, patch_to_source, source_size):
    """No label argument, filenames, DXF layer or OCR target is available here."""
    if gray.ndim != 2 or gray.dtype != np.uint8 or gray.size > POLICY['maximum_pixels']:
        raise ValueError('Bounded grayscale patch required')
    transform = np.asarray(patch_to_source, float)
    if (transform.shape != (3, 3) or not np.isfinite(transform).all()
            or not np.array_equal(transform[:2, :2], np.eye(2)) or not np.array_equal(transform[2], [0, 0, 1])):
        raise ValueError('Exact source crop translation required')
    if not np.array_equal(transform[:2, 2], np.floor(transform[:2, 2])):
        raise ValueError('Integer source crop translation required')
    if len(source_size) != 2 or any(type(n) is not int or n <= 0 for n in source_size):
        raise ValueError('Source dimensions required')
    lo = transform[:2, 2]; hi = lo+np.asarray(gray.shape[::-1])
    if (lo < 0).any() or (hi > source_size).any():
        raise ValueError('Patch lies outside source')
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    skeleton = skeletonize(ink > 0, method='zhang')
    paths, audit = graph_paths(skeleton)
    if audit['search_incomplete']:
        return [], dict(complete=False, graph=audit, roles_inferred=False)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    widths = np.maximum(1., 2*distance[skeleton]-1)
    global_width = float(np.median(widths)) if len(widths) else 1.
    integral = cv2.integral((ink > 0).astype(np.uint8), sdepth=cv2.CV_64F)
    rows = []
    for index, path in enumerate(paths):
        points = np.asarray(path['points'], np.int32)
        clipped = bool(((points == [0, 0]) | (points == np.asarray(gray.shape[::-1])-1)).any())
        for start in range(0, len(points)-1, POLICY['interval_edges']):
            stop = min(len(points)-1, start+POLICY['interval_edges'])
            if stop-start < POLICY['minimum_edges']:
                continue
            part = points[start:stop+1]
            features = _features(part, path, gray, distance, global_width, transform, source_size, integral)
            if not np.isfinite(features).all():
                raise ValueError('Non-finite source features')
            rows.append(dict(id=f'path-{index:05d}/{start}-{stop}', points_px=part.tolist(),
                source_path_clipped=clipped, features=features, source_interval=[start, stop]))
    return rows, dict(complete=True, graph=audit, observations=len(rows), roles_inferred=False)


def attach_targets(observations, labels, state, role_bits):
    if labels.ndim != 2 or labels.dtype != np.uint8 or state.shape != labels.shape or role_bits.shape != labels.shape:
        raise ValueError('Aligned partial masks required')
    if not np.isin(labels, [*range(10), 255]).all() or np.any(state > 3):
        raise ValueError('Invalid partial mask code')
    output = []; counts = Counter()
    for row in observations:
        points = np.unique(np.asarray(row['points_px'], int), axis=0); x, y = points.T
        if (points < 0).any() or (x >= labels.shape[1]).any() or (y >= labels.shape[0]).any():
            raise ValueError('Observed source samples outside masks')
        values = labels[y, x]; unique = np.unique(values)
        role = int(unique[0]) if len(unique) == 1 and 1 <= unique[0] <= 9 else 255
        if row['source_path_clipped']:
            reason = 'clipped_source_path'; role = 255
        elif role == 255:
            reason = 'unreviewed_background_or_mixed_role'
        elif not np.all((state[y, x] == 1) & (role_bits[y, x] == (1 << (role-1)))):
            reason = 'ambiguous_or_unverified_role_bits'; role = 255
        else:
            reason = 'all_source_samples_same_reviewed_role'
        counts[reason] += 1
        output.append(dict(observation_id=row['id'], target=role, reason=reason,
            source_points=len(points), dense_ground_truth=False))
    return output, dict(counts)
