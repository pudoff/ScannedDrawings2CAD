"""Keep artificial UNKNOWN split boundaries out of semantic residual ink.

Two adjacent pieces of one frozen model can both abstain from a role. Their
shared source endpoint is not a conflict between physical owners. Resolve only
that exact indexed endpoint; other shared seeds keep the original exclusion.
"""
from collections import defaultdict

import cv2
import numpy as np

from .cad_object_export_v2 import _ownership as original_ownership
from .grid_role_revision_v1 import sha


def boundary_seeds(result, geometry):
    """Return independently indexed, two-sided UNKNOWN boundary certificates."""
    lookup = {c['id']: c for c in geometry}
    if len(lookup) != len(geometry):
        raise ValueError('Unique source candidates required')
    source_at, claims = defaultdict(list), defaultdict(list)
    for c in geometry:
        points = c.get('support_points', c.get('points_px'))
        values = np.asarray(points, float)
        if not np.isfinite(values).all() or not np.array_equal(values, np.rint(values)):
            raise ValueError('Integer raster source samples required')
        low, high = c['source_index_interval']
        if high-low+1 != len(points):
            raise ValueError('Source index interval differs from observed walk')
        for offset, point in enumerate(points):
            source_at[tuple(point)].append((c['id'], low+offset))
    for index, obj in enumerate(result['objects'], 1):
        for member in obj['source_members']:
            c = lookup[member['candidate_id']]
            first, last = c['source_index_interval']
            low, high = member['source_index_interval']
            if not first <= low <= high <= last:
                raise ValueError('Object membership exceeds source walk')
            points = c.get('support_points', c.get('points_px'))
            for source_index in range(low, high+1):
                point = tuple(points[source_index-first])
                claims[point].append((index, member, source_index))
    partitions = {row['parent_object_id']: row for row in
                  result.get('geometry_preserving_role_export', {}).get('changed_role_or_partition', [])}
    accepted = []
    for point, owners in claims.items():
        if len(owners) != 2 or owners[0][0] == owners[1][0]:
            continue
        left, right = sorted(owners, key=lambda row: row[1]['source_index_interval'])
        li, lm, ls = left
        ri, rm, rs = right
        lo, ro = result['objects'][li-1], result['objects'][ri-1]
        lp, rp = lo.get('semantic_role_guard', {}), ro.get('semantic_role_guard', {})
        parent = lp.get('parent_object_id')
        if (lo['role'] != 255 or ro['role'] != 255 or not parent
                or rp.get('parent_object_id') != parent
                or lp.get('decision') != 'confirmed_role_boundary'
                or rp.get('decision') != 'confirmed_role_boundary'
                or lp.get('assigned_role') != 255 or rp.get('assigned_role') != 255):
            continue
        proof = partitions.get(parent)
        if (not proof or lo['id'] not in proof['output_object_ids']
                or ro['id'] not in proof['output_object_ids']):
            continue
        keys = ('candidate_id', 'source_observation_id', 'path_id')
        if any(not lm.get(key) or lm[key] != rm.get(key) for key in keys):
            continue
        a, b = lm['source_index_interval']
        c, d = rm['source_index_interval']
        if not (a < b == c < d and ls == rs == b):
            continue
        # A coincident point from an unconverted candidate or a repeated visit
        # elsewhere on this path remains a real competing seed.
        if source_at[point] != [(lm['candidate_id'], b)]:
            continue
        accepted.append({'point_px': list(map(int, point)), 'owner_index': min(li, ri),
            'object_ids': [lo['id'], ro['id']], 'parent_object_id': parent,
            'source_identity': {key: lm[key] for key in keys},
            'source_intervals': [[a, b], [c, d]], 'shared_source_index': b})
    return sorted(accepted, key=lambda row: (row['point_px'][1], row['point_px'][0]))


def ownership(gray, result, geometry, protected):
    """Preserve the old measured-radius ownership except proven draft joints."""
    ink, removed, owner = original_ownership(gray, result, geometry, protected)
    boundaries = boundary_seeds(result, geometry)
    corrections = np.zeros_like(ink)
    if boundaries:
        seeds = np.zeros_like(gray, bool)
        for c in geometry:
            p = np.asarray(c.get('support_points', c.get('points_px')), int)
            seeds[p[:, 1], p[:, 0]] = True
        distance, labels = cv2.distanceTransformWithLabels(
            np.where(seeds, 0, 255).astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_PIXEL)
        labels_to_owner = np.zeros(int(labels.max())+1, np.int32)
        for row in boundaries:
            x, y = row['point_px']
            labels_to_owner[labels[y, x]] = row['owner_index']
        radius = cv2.distanceTransform(ink.astype(np.uint8)*255, cv2.DIST_L2, 5)
        yy, xx = np.nonzero(seeds)
        radii = np.zeros(len(labels_to_owner), np.float32)
        radii[labels[yy, xx]] = radius[yy, xx]
        proposed = labels_to_owner[labels]
        corrections = ink & ~removed & ~protected & (proposed > 0) & (distance <= radii[labels]+.75)
        owner[corrections] = proposed[corrections]
        removed[corrections] = True
    result['residual_split_ownership'] = {
        'schema_version': 'residual-split-ownership/1.0',
        'policy': 'only_adjacent_same_source_same_parent_UNKNOWN_endpoints',
        'boundaries': boundaries, 'resolved_ink_pixels': int(corrections.sum()),
        'resolved_mask_sha256': sha(corrections.astype(np.uint8)),
        'ownership_sha256': sha(owner), 'removed_sha256': sha(removed.astype(np.uint8)),
        'new_known_roles': 0, 'geometry_changed': False,
        'other_shared_seeds_preserved': True}
    return ink, removed, owner
