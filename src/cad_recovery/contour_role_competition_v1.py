"""Negative source-context veto for a new contour hypothesis; creates no role."""
from copy import deepcopy
import math
import cv2
import numpy as np
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'contour-role-competition/1.0', 'maximum_pixels': 4000000,
          'maximum_intervals': 4096, 'maximum_source_points': 2000000,
          'negative_roles': ['sheet_frame', 'title_grid', 'dimension_line', 'extension_line', 'leader_line', 'axis_construction'],
          'negative_hint_is_positive_PART_evidence': False,
          'clearance': 'observed_q95_radii_both_owners_plus_sqrt_half_pixel',
          'geometry_modified': False, 'physical_geometry_approved': False}


def guard(gray, graph, source_owner, ambiguous_owner, proposed, context):
    checked_gray(gray); items = graph['proposals']
    if gray.size > POLICY['maximum_pixels'] or len(items) > POLICY['maximum_intervals']:
        raise ValueError('Bounded contour competition required')
    if (source_owner.shape != gray.shape or source_owner.dtype.kind not in 'ui' or source_owner.min(initial=0) < 0 or
            source_owner.max(initial=0) > len(items) or ambiguous_owner.shape != gray.shape or
            proposed.shape != gray.shape or proposed.dtype != np.uint8 or not np.isin(proposed, [0, 255]).all()):
        raise ValueError('Aligned current owners and binary contour proposal required')
    if np.any((proposed > 0) & (source_owner == 0)): raise ValueError('Every proposed pixel requires a current source owner')
    before = object_sha([graph, context]); snapshots = [sha(a) for a in (gray, source_owner, ambiguous_owner, proposed)]
    numbers = {p['id']: n for n, p in enumerate(items, 1)}
    if len(numbers) != len(items): raise ValueError('Unique source interval IDs required')
    negatives = set(context['frame_conflict_owner_numbers']) | set(context['other_conflict_owner_numbers'])
    tracks = {t['id']: t for t in graph['tracks']}
    for line in context['context']['lines']:
        if not set(line['alternatives']) & set(POLICY['negative_roles']): continue
        if line['id'] not in tracks: raise ValueError('Foreign negative context track')
        for identity in tracks[line['id']]['interval_ids']:
            if identity not in numbers: raise ValueError('Foreign negative context interval')
            negatives.add(numbers[identity])
    if any(type(n) is not int or n < 1 or n > len(items) for n in negatives): raise ValueError('Foreign negative source owner')
    ink = foreground(gray); distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    radii = np.zeros(len(items)+1, np.float32); seeds = np.zeros_like(gray); seed_radius = np.zeros(gray.shape, np.float32)
    audit = {'source_points': 0, 'limits': []}
    if not context['complete'] or graph.get('search_incomplete'): audit['limits'].append('incomplete_negative_context')
    for n, item in enumerate(items, 1) if not audit['limits'] else []:
        points = np.asarray(item['points_px'], float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all(): raise ValueError('Finite source interval required')
        q = np.rint(points).astype(int)
        if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any(): raise ValueError('Source interval outside raster')
        audit['source_points'] += len(q)
        if audit['source_points'] > POLICY['maximum_source_points']:
            audit['limits'].append('source_point_budget'); break
        samples = distance[q[:, 1], q[:, 0]]; samples = samples[samples > 0]
        if len(samples): radii[n] = float(np.quantile(samples, .95))
        if n in negatives:
            seeds[q[:, 1], q[:, 0]] = 255
            np.maximum.at(seed_radius, (q[:, 1], q[:, 0]), radii[n])
    eligible = (proposed > 0) & (ambiguous_owner == 0) & (ink > 0)
    direct = eligible & np.isin(source_owner, list(negatives)); near_conflict = np.zeros_like(eligible)
    if seeds.any() and not audit['limits']:
        distances, labels = cv2.distanceTransformWithLabels(255-seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        nearest_radius = np.zeros(int(labels.max())+1, np.float32)
        where = seeds > 0; np.maximum.at(nearest_radius, labels[where], seed_radius[where])
        near_conflict = eligible & (distances <= radii[source_owner]+nearest_radius[labels]+math.sqrt(.5))
    output = np.where(eligible & ~direct & ~near_conflict, 255, 0).astype(np.uint8)
    complete = not audit['limits']
    if not complete: output[:] = 0
    immutable = before == object_sha([graph, context]) and snapshots == [sha(a) for a in (gray, source_owner, ambiguous_owner, proposed)]
    if not immutable: raise ValueError('Contour competition mutated source observations')
    return {'mask': output, 'complete': complete, 'source_immutable': immutable,
            'negative_source_owner_numbers': sorted(negatives), 'direct_veto_pixels': int(direct.sum()),
            'contact_veto_pixels': int((near_conflict & ~direct).sum()),
            'withdrawn_pixels': int(np.count_nonzero(proposed)-np.count_nonzero(output)),
            'audit': audit, 'policy': deepcopy(POLICY), 'positive_role_created': False,
            'geometry_modified': False, 'physical_geometry_approved': False}
