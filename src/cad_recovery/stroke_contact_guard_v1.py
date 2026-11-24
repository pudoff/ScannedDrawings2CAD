"""Protect curved source and thicker-stroke contacts from a raw width veto."""
from copy import deepcopy
import math
import cv2
import numpy as np

from .contour_page_competition_v1 import _binary
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .primary_stroke_guard_v1 import POLICY as RAW_POLICY
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'stroke-contact-guard/1.0', 'maximum_pixels': 4194304,
    'maximum_intervals': 4096, 'maximum_source_points': 2000000,
    'maximum_raster_pixels': 64000000, 'maximum_seed_samples': 16000000,
    'curved_source_protected': True,
    'clearance': 'base_radius_plus_nearest_thicker_q95_radius_plus_sqrt_half',
    'positive_role_created': False, 'geometry_modified': False,
    'semantic_AXIS_assigned': False, 'physical_geometry_approved': False}


def analyze(gray, graph, source_owner, ambiguous_owner, proposed, *, raw, binding, baseline_roles, protected):
    checked_gray(gray); items = graph['proposals']
    if gray.size > POLICY['maximum_pixels'] or len(items) > POLICY['maximum_intervals']:
        raise ValueError('Bounded contact guard required')
    for value, name in ((proposed, 'proposal'), (raw['veto_mask'], 'raw veto'),
                        (ambiguous_owner, 'ambiguity'), (protected, 'protection')):
        _binary(value, gray.shape, name)
    if (source_owner.shape != gray.shape or source_owner.dtype.kind not in 'ui'
            or source_owner.max(initial=0) > len(items)
            or np.any((proposed > 0) & ((source_owner == 0) | (ambiguous_owner > 0)))
            or np.any((raw['veto_mask'] > 0) & (proposed == 0))):
        raise ValueError('Current unambiguous owners and negative subset required')
    if baseline_roles.shape != gray.shape or baseline_roles.dtype != np.uint8:
        raise ValueError('Aligned baseline roles required')
    if (binding['source_sha256'] != sha(gray) or binding['graph_sha256'] != object_sha(graph)
            or binding['source_owner_sha256'] != sha(source_owner) or binding['proposal_sha256'] != sha(proposed)
            or binding['raw_veto_sha256'] != sha(raw['veto_mask']) or raw['policy'] != RAW_POLICY
            or raw.get('source_immutable') is not True):
        raise ValueError('Pixel-bound frozen raw negative evidence required')
    if set(binding['producer']) != {'path', 'sha256'} or len(binding['producer']['sha256']) != 64:
        raise ValueError('Pinned raw producer required')
    before = object_sha([graph, binding, {k: v for k, v in raw.items() if k != 'veto_mask'}])
    arrays = (gray, source_owner, ambiguous_owner, proposed, raw['veto_mask'], baseline_roles, protected)
    hashes = [sha(a) for a in arrays]
    audit = {'source_points': 0, 'seed_samples': 0, 'raster_pixels': 0, 'limits': []}
    if not raw['complete'] or graph.get('search_incomplete'): audit['limits'].append('incomplete_raw_or_graph')
    veto = raw['veto_mask'].copy(); veto[(baseline_roles != 255) | (protected > 0)] = 0
    allowed = np.array([False]+[p.get('linear_context', False) and p.get('parent_kind') not in ('circle', 'arc') for p in items])
    curved_preserved = int(np.count_nonzero((veto > 0) & ~allowed[source_owner]))
    veto[~allowed[source_owner]] = 0
    distance = cv2.distanceTransform(foreground(gray), cv2.DIST_L2, 5)
    points_by_owner = {}; radii = np.zeros(len(items)+1)
    for number, item in enumerate(items, 1) if veto.any() and not audit['limits'] else []:
        p = np.asarray(item['points_px'], float)
        if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all():
            raise ValueError('Finite current source samples required')
        q = np.rint(p).astype(int)
        if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any():
            raise ValueError('Current source samples outside input')
        audit['source_points'] += len(q)
        if audit['source_points'] > POLICY['maximum_source_points']:
            audit['limits'].append('source_point_budget'); break
        samples = distance[q[:, 1], q[:, 0]]; samples = samples[samples > 0]
        if len(samples): radii[number] = float(np.quantile(samples, .95))
        points_by_owner[number] = q
    junction_preserved = 0; decisions = []; cache = {}
    for decision in raw['decisions'] if not audit['limits'] else []:
        number = decision['owner_number']
        if not 1 <= number <= len(items) or decision['interval_id'] != items[number-1]['id']:
            raise ValueError('Foreign raw decision owner')
        selected = (veto > 0) & (source_owner == number)
        if not selected.any(): continue
        radius = (float(decision['observed_pen_px'])+1)/2
        if not math.isfinite(radius) or radius < 1: raise ValueError('Measured candidate radius required')
        if radius not in cache:
            audit['raster_pixels'] += gray.size
            if audit['raster_pixels'] > POLICY['maximum_raster_pixels']:
                audit['limits'].append('distance_raster_budget'); break
            seeds = np.zeros_like(gray); seed_radius = np.zeros(gray.shape, np.float32)
            for other, q in points_by_owner.items():
                audit['seed_samples'] += len(q)
                if audit['seed_samples'] > POLICY['maximum_seed_samples']:
                    audit['limits'].append('thicker_seed_sample_budget'); break
                values = distance[q[:, 1], q[:, 0]]; thick = values > radius
                local = q[thick]
                seeds[local[:, 1], local[:, 0]] = 255
                np.maximum.at(seed_radius, (local[:, 1], local[:, 0]), np.maximum(radii[other], values[thick]))
            if audit['limits']: break
            protected_contact = np.zeros(gray.shape, bool)
            if seeds.any():
                distances, labels = cv2.distanceTransformWithLabels(255-seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
                nearest = np.zeros(int(labels.max())+1, np.float32)
                where = seeds > 0; np.maximum.at(nearest, labels[where], seed_radius[where])
                protected_contact = distances <= radius+nearest[labels]+math.sqrt(.5)
            cache[radius] = protected_contact
        preserve = selected & cache[radius]
        junction_preserved += int(preserve.sum()); veto[preserve] = 0
        count = int(np.count_nonzero((veto > 0) & (source_owner == number)))
        if count: decisions.append({**decision, 'withdrawn_pixels': count})
    complete = not audit['limits']; output = proposed.copy()
    if complete: output[veto > 0] = 0
    else:
        output[(baseline_roles == 255) & (protected == 0)] = 0; veto[:] = 0; decisions = []
    if before != object_sha([graph, binding, {k: v for k, v in raw.items() if k != 'veto_mask'}]) or hashes != [sha(a) for a in arrays]:
        raise ValueError('Contact guard mutated its source')
    return {'mask': output, 'veto_mask': veto, 'width_veto_pixels': int(np.count_nonzero(veto)),
        'abstained_pixels': int(np.count_nonzero(proposed)-np.count_nonzero(output)) if not complete else 0,
        'raw_width_veto_pixels': int(np.count_nonzero(raw['veto_mask'])),
        'curved_source_preserved_pixels': curved_preserved, 'junction_preserved_pixels': junction_preserved,
        'decisions': decisions, 'audit': audit, 'complete': complete, 'source_immutable': True,
        'policy': deepcopy(POLICY), 'positive_role_created': False, 'geometry_modified': False,
        'semantic_AXIS_assigned': False, 'physical_geometry_approved': False}
