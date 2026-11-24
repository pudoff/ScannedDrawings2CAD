"""Gate new part-role proposals without treating dimension hypotheses as truth."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import cv2
import numpy as np
from .grid_role_revision_v1 import check_authority, sha
from .stroke_roles_v1 import checked_gray, foreground
from .boundary_role_overlay_v1 import boundary_witnesses

POLICY = {'version': 'part-admission/1.0', 'maximum_intervals': 4096, 'maximum_tracks': 1024,
          'maximum_chains': 4096, 'maximum_source_points': 2000000, 'maximum_track_checks': 250000,
          'parallel_angle_deg': 8., 'parallel_neighbourhood_in_strokes': 8.,
          'maximum_width_ratio': 1.5, 'minimum_overlapping_fraction': .6,
          'minimum_distinct_parallel_rows': 3, 'known_hatching_protected': True,
          'production_admitted': False, 'physical_geometry_approved': False}
CHAIN_FIELDS = ('boundary_id', 'shaft_interval_ids', 'extension_interval_ids', 'feet_px', 'head_ids')


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def provenance(gray, graph, raw_roles, previous, candidate, owners, canonical, producer):
    if set(producer) != {'path', 'sha256'} or len(producer['sha256']) != 64:
        raise ValueError('Explicit checked CV producer pin required')
    return {'producer_kind': 'frozen_CV_positive_roles', 'producer': deepcopy(producer),
            'source_sha256': sha(gray), 'graph_sha256': object_sha(graph), 'raw_roles_sha256': sha(raw_roles),
            'previous_sha256': sha(previous), 'candidate_sha256': sha(candidate),
            'owners_sha256': sha(owners), 'canonical_sha256': object_sha(canonical)}


def source_lines(ink, graph, audit):
    items = {p['id']: p for p in graph['proposals']}
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    output = []
    for track in graph['tracks']:
        if not track['interval_ids'] or not set(track['interval_ids']) <= items.keys():
            raise ValueError('Missing track source intervals')
        arrays = [np.asarray(items[i]['points_px'], float) for i in track['interval_ids']]
        if any(p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all() for p in arrays):
            raise ValueError('Finite observed source points required')
        audit['source_points'] += sum(len(p) for p in arrays)
        if audit['source_points'] > POLICY['maximum_source_points']:
            audit['limits'].append('source_point_budget'); return []
        points = np.unique(np.concatenate(arrays), axis=0)
        if (points < 0).any() or (points[:, 0] > ink.shape[1]-1).any() or (points[:, 1] > ink.shape[0]-1).any():
            raise ValueError('Source points outside current raster')
        if len(points) < 8: continue
        q = np.rint(points).astype(int)
        if np.mean(ink[q[:, 1], q[:, 0]] > 0) < .95: continue
        radii = distance[q[:, 1], q[:, 0]]
        pen = max(1., float(2*np.quantile(radii[radii > 0], .25)-1))
        center = points.mean(0)
        _, _, vectors = np.linalg.svd(points-center, full_matrices=False)
        axis = vectors[0]
        normal = np.array([-axis[1], axis[0]])
        if np.quantile(np.abs((points-center) @ normal), .95) > max(.75, .25*pen): continue
        along = points @ axis
        if np.ptp(along) < 6*pen: continue
        output.append({'id': track['id'], 'interval_ids': list(track['interval_ids']),
                       'points': points, 'axis': axis, 'pen': pen})
    return output


def parallel_context(identity, lines, audit):
    evidence = []
    for seed in [t for t in lines if identity in t['interval_ids']]:
        axis, pen = seed['axis'], seed['pen']
        normal = np.array([-axis[1], axis[0]])
        a = seed['points'] @ axis
        lo, hi = float(a.min()), float(a.max())
        offset = float(np.median(seed['points'] @ normal))
        rows = []
        for other in lines:
            audit['track_checks'] += 1
            if audit['track_checks'] > POLICY['maximum_track_checks']:
                audit['limits'].append('track_check_budget'); return []
            if abs(axis @ other['axis']) < math.cos(math.radians(POLICY['parallel_angle_deg'])): continue
            ratio = other['pen']/pen
            if not 1/POLICY['maximum_width_ratio'] <= ratio <= POLICY['maximum_width_ratio']: continue
            position = float(np.median(other['points'] @ normal))
            if abs(position-offset) > POLICY['parallel_neighbourhood_in_strokes']*max(pen, other['pen']): continue
            b = other['points'] @ axis
            overlap = max(0., min(hi, float(b.max()))-max(lo, float(b.min())))
            if overlap < POLICY['minimum_overlapping_fraction']*min(hi-lo, float(np.ptp(b))): continue
            rows.append({'track_id': other['id'], 'interval_ids': other['interval_ids'],
                         'offset': position, 'pen': other['pen'], 'overlap_px': overlap})
        grouped = []
        for row in sorted(rows, key=lambda r: (r['offset'], r['track_id'])):
            if grouped and row['offset']-grouped[-1]['offset'] <= 1.5*max(row['pen'], grouped[-1]['pen']):
                grouped[-1]['members'].append(row)
            else: grouped.append({'offset': row['offset'], 'pen': row['pen'], 'members': [row]})
        if len(grouped) >= POLICY['minimum_distinct_parallel_rows']:
            evidence.append({'seed_track_id': seed['id'], 'observed_pen_px': pen,
                             'distinct_offsets': len(grouped), 'groups': grouped})
    return evidence


def canonical_witnesses(graph, canonical, audit):
    if not all(k in graph for k in ('boundaries', 'dimension_chains', 'stroke_px')):
        return {}, {}, {'reason': 'no_boundary_context', 'canonical_is_semantic_authority': False}
    eligible = []
    source_signatures = {object_sha({k: c[k] for k in CHAIN_FIELDS}) for c in graph['dimension_chains']
                         if all(k in c for k in CHAIN_FIELDS)}
    for c in canonical:
        if (all(k in c for k in CHAIN_FIELDS) and len(c['head_ids']) == 2 and len(set(c['head_ids'])) == 2
                and object_sha({k: c[k] for k in CHAIN_FIELDS}) in source_signatures):
            eligible.append(c)
    claims, rejected, incomplete = boundary_witnesses(graph, {'dimension_chains': eligible})
    if incomplete: audit['limits'].append('boundary_witness_budget')
    boundaries = {b['id']: b for b in graph['boundaries']['boundaries']}
    shared = Counter(i for b in boundaries.values() if b['orientation'] == 'bounded_face' for i in set(b['owners']))
    admissible = {}
    for identity, witnesses in claims.items():
        supported = []
        for witness in witnesses:
            boundary = boundaries[witness['boundary_id']]
            # Adjacent internal faces between hatch rows do not establish a
            # physical outer contour, even when a dimension meets that face.
            # A thin internal hatch edge may be shared while this particular
            # outer owner is not. Test the proposed owner, not every neighbour.
            if boundary['orientation'] == 'bounded_face' and shared[identity] > 1: continue
            supported.append(witness)
        if supported: admissible[identity] = supported
    return admissible, boundaries, {'canonical_records': len(canonical), 'source_matched_records': len(eligible),
                                  'boundary_rejections': rejected, 'raw_claims': len(claims),
                                  'unshared_boundary_claims': len(admissible),
                                  'canonical_is_semantic_authority': False}


def analyze(gray, graph, raw_roles, previous, candidate, owners, canonical, *, authority=None, evidence=None):
    checked_gray(gray)
    if len(graph['proposals']) > POLICY['maximum_intervals'] or len(graph['tracks']) > POLICY['maximum_tracks'] or len(canonical) > POLICY['maximum_chains']:
        raise ValueError('Bounded source context required')
    for array in (raw_roles, previous, candidate):
        if array.shape != gray.shape or array.dtype != np.uint8 or not np.isin(array, list(range(10))+[255]).all():
            raise ValueError('Aligned uint8 role map required')
    if owners.shape != gray.shape or owners.dtype.kind not in 'iu' or owners.min() < 0 or owners.max() > len(graph['proposals']):
        raise ValueError('Aligned owners in the observed source domain required')
    permitted = check_authority(gray, raw_roles, authority)
    if evidence is None:
        permitted[:] = False
    else:
        expected = provenance(gray, graph, raw_roles, previous, candidate, owners, canonical, evidence['producer'])
        if any(evidence.get(k) != value for k, value in expected.items() if k != 'producer_kind'):
            raise ValueError('Stale source/candidate/owners/canonical evidence')
        if evidence.get('producer_kind') != expected['producer_kind']: permitted[:] = False
    changed = candidate != previous
    if np.any(changed & (~np.isin(previous, [9, 255]) | ~np.isin(candidate, [1, 6, 8]))):
        raise ValueError('Unexpected producer role changes')
    if np.any(changed & (owners == 0)): raise ValueError('New producer pixels require an observed source owner')
    seed = previous.copy()
    seed[permitted] = candidate[permitted]
    new_part = permitted & (candidate == 1) & (previous != 1)
    no_part = seed.copy(); no_part[new_part] = previous[new_part]
    guarded = no_part.copy()
    audit = {'source_points': 0, 'track_checks': 0, 'limits': []}
    decisions = []
    if new_part.any():
        ink = foreground(gray)
        lines = source_lines(ink, graph, audit)
        claims, _, context = canonical_witnesses(graph, canonical, audit)
        audit['canonical_context'] = context
        for number in sorted(int(i) for i in np.unique(owners[new_part])):
            identity = graph['proposals'][number-1]['id']
            repeat = parallel_context(identity, lines, audit) if identity in claims else []
            allowed = identity in claims and not repeat and not audit['limits']
            selected = new_part & (owners == number) & (previous != 9)
            if allowed: guarded[selected] = candidate[selected]
            decisions.append({'source_owner_number': number, 'interval_id': identity,
                              'matched_unshared_boundary': identity in claims,
                              'canonical_witnesses': claims.get(identity, []), 'parallel_repeat_veto': repeat,
                              'candidate_pixels': int((new_part & (owners == number)).sum()),
                              'known_hatching_protected_pixels': int((new_part & (owners == number) & (previous == 9)).sum()),
                              'admitted_pixels': int(selected.sum()) if allowed else 0})
    complete = not audit['limits']
    if not complete: guarded[:] = no_part
    return {'no_part': no_part, 'canonical': guarded, 'decisions': decisions, 'audit': audit, 'complete': complete,
            'new_part_candidate_pixels': int(new_part.sum()), 'source_pixels_erased': 0,
            'production_admitted': False, 'physical_geometry_approved': False}
