"""Source-derived grid context; raw geometry and semantic authority stay separate."""
from copy import deepcopy
import math
import cv2
import numpy as np
from . import grid_role_revision_v1 as previous
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'grid-context/1.0', 'maximum_pixels': 4000000,
          'maximum_source_points': 2000000, 'maximum_tracks': 1024,
          'maximum_intervals': 4096, 'maximum_source_fit_p95_in_radii': 1.,
          'maximum_merge_pairs': 250000, 'maximum_bridge_samples': 2000000,
          'angle_deg': 8., 'production_admitted': False,
          'physical_geometry_approved': False}


def source_track_context(ink, graph):
    """Fit only the exact source intervals already owned by each frozen track."""
    if len(graph['tracks']) > POLICY['maximum_tracks'] or len(graph['proposals']) > POLICY['maximum_intervals']:
        raise ValueError('Bounded source graph required')
    items = {p['id']: p for p in graph['proposals']}
    if len(items) != len(graph['proposals']):
        raise ValueError('Unique source interval IDs required')
    tracks, evidence, used = [], [], 0
    for track in graph['tracks']:
        original = np.asarray(track['points'], float)
        width = float(track['width'])
        if (original.shape != (2, 2) or not np.isfinite(original).all()
                or not math.isfinite(width) or width <= 0 or np.linalg.norm(original[1]-original[0]) == 0):
            raise ValueError('Finite non-degenerate source track and positive width required')
        ids = track['interval_ids']
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= items.keys():
            raise ValueError('Track must map uniquely to existing source intervals')
        arrays = [np.asarray(items[i]['points_px'], float) for i in ids]
        if any(p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all() for p in arrays):
            raise ValueError('Finite source samples required')
        used += sum(len(p) for p in arrays)
        if used > POLICY['maximum_source_points']:
            return None, {'complete': False, 'reason': 'source_sample_budget', 'source_samples': used}
        points = np.unique(np.concatenate(arrays), axis=0)
        if (points < 0).any() or (points[:, 0] >= ink.shape[1]).any() or (points[:, 1] >= ink.shape[0]).any():
            raise ValueError('Source samples must stay in the current raster')
        q = np.rint(points).astype(int)
        if not np.all(ink[q[:, 1], q[:, 0]] > 0):
            evidence.append({'track_id': track['id'], 'status': 'unobserved_source_sample'})
            tracks.append(deepcopy(track))
            continue
        if len(points) < 8:
            evidence.append({'track_id': track['id'], 'status': 'short_source_support'})
            tracks.append(deepcopy(track))
            continue
        vx, vy, x, y = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        u = np.array([vx, vy], float)
        u /= np.linalg.norm(u)
        normal = np.array([-u[1], u[0]])
        center = np.array([x, y], float)
        errors = np.abs((points-center) @ normal)
        tolerance = width/2 + math.sqrt(.5)
        error = float(np.quantile(errors, .95))
        if error > tolerance:
            evidence.append({'track_id': track['id'], 'status': 'source_not_linear', 'p95_px': error})
            tracks.append(deepcopy(track))
            continue
        offsets = (points-center) @ u
        a, b = center+offsets.min()*u, center+offsets.max()*u
        tracks.append({**deepcopy(track), 'points': [a.tolist(), b.tolist()], 'u': u.tolist(),
                       'length': float(offsets.max()-offsets.min())})
        evidence.append({'track_id': track['id'], 'status': 'source_fitted',
                         'source_interval_ids': list(ids), 'source_samples': len(points),
                         'p95_px': error, 'maximum_p95_px': tolerance,
                         'original_points': deepcopy(track['points']), 'context_points': [a.tolist(), b.tolist()]})
    return {'proposals': graph['proposals'], 'tracks': tracks}, {
        'complete': True, 'source_samples': used, 'tracks': evidence,
        'raw_graph_changed': False, 'new_interval_ids': [], 'external_nodes_used': False}


def merge_source_tracks(ink, context, audit):
    """Recover fragmented context lines only across continuously observed ink."""
    items = {p['id']: np.asarray(p['points_px'], float) for p in context['proposals']}
    tracks = context['tracks']
    groups = [{i} for i in range(len(tracks))]
    active = {i: deepcopy(t) for i, t in enumerate(tracks)}
    pairs, samples, accepted = 0, 0, []
    # Raster quantisation permits at most the half-pixel diagonal; gaps in the
    # drawing are not bridged by increasing a geometric distance tolerance.
    blank_distance = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5)

    def fitted(members):
        ids = sorted({s for i in members for s in tracks[i]['interval_ids']})
        points = np.unique(np.concatenate([items[s] for s in ids]), axis=0)
        vx, vy, x, y = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
        unit = np.array([vx, vy], float)
        unit /= np.linalg.norm(unit)
        center = np.array([x, y], float)
        width = min(tracks[i]['width'] for i in members)
        errors = np.abs((points-center) @ np.array([-unit[1], unit[0]]))
        if np.quantile(errors, .95) > width/2:
            return None
        along = (points-center) @ unit
        a, b = center+along.min()*unit, center+along.max()*unit
        return {'id': 'context:'+'+'.join(tracks[i]['id'] for i in sorted(members)),
                'points': [a.tolist(), b.tolist()], 'u': unit.tolist(),
                'length': float(along.max()-along.min()), 'width': float(width),
                'interval_ids': ids, 'source_track_ids': [tracks[i]['id'] for i in sorted(members)]}

    for i in range(len(tracks)):
        if i not in active:
            continue
        for j in range(i+1, len(tracks)):
            if j not in active:
                continue
            pairs += 1
            if pairs > POLICY['maximum_merge_pairs']:
                return None, {**audit, 'complete': False, 'reason': 'merge_pair_budget'}
            a, b = active[i], active[j]
            pa, pb = np.asarray(a['points']), np.asarray(b['points'])
            ua = (pa[1]-pa[0])/np.linalg.norm(pa[1]-pa[0])
            ub = (pb[1]-pb[0])/np.linalg.norm(pb[1]-pb[0])
            if abs(ua @ ub) < math.cos(math.radians(POLICY['angle_deg'])):
                continue
            normal = np.array([-ua[1], ua[0]])
            if abs((pb.mean(0)-pa.mean(0)) @ normal) > min(a['width'], b['width'])/2:
                continue
            candidate = fitted(groups[i] | groups[j])
            if candidate is None:
                continue
            endpoints = np.asarray(candidate['points'])
            count = int(math.ceil(candidate['length']*2))+1
            samples += count
            if samples > POLICY['maximum_bridge_samples']:
                return None, {**audit, 'complete': False, 'reason': 'bridge_sample_budget'}
            points = np.linspace(endpoints[0], endpoints[1], count)
            q = np.rint(points).astype(int)
            if (q < 0).any() or (q[:, 0] >= ink.shape[1]).any() or (q[:, 1] >= ink.shape[0]).any():
                continue
            observed = blank_distance[q[:, 1], q[:, 0]] <= math.sqrt(.5)
            change = np.diff(np.r_[False, ~observed, False].astype(int))
            missing = np.flatnonzero(change == -1)-np.flatnonzero(change == 1)
            if observed.mean() < .95 or max(missing, default=0)*.5 > 1.:
                continue
            groups[i] |= groups[j]
            del active[j]
            active[i] = candidate
            accepted.append({'source_track_ids': candidate['source_track_ids'],
                             'interval_ids': candidate['interval_ids'],
                             'source_coverage': float(observed.mean()),
                             'maximum_missing_run_px': float(max(missing, default=0)*.5)})
    return {'proposals': context['proposals'], 'tracks': list(active.values())}, {
        **audit, 'merge_pairs': pairs, 'bridge_samples': samples, 'merges': accepted,
        'output_tracks': len(active), 'complete': True}


def analyze(gray, graph, roles, *, authority=None):
    """Compare fixed previous, source-fit and source-merged diagnostic maps.

    Each call consumes one raster only. All alternatives retain the previous
    guarded withdrawals. A context budget failure rolls back all new changes.
    """
    checked_gray(gray)
    if gray.size > POLICY['maximum_pixels']:
        raise ValueError('Bounded context raster required')
    old = previous.analyze(gray, graph, roles, authority=authority)
    ink = foreground(gray)
    fit, work = source_track_context(ink, graph)
    contexts = [('source_fit', fit)]
    merged, merged_work = merge_source_tracks(ink, fit, work) if fit is not None else (None, work)
    contexts.append(('source_merged', merged))
    output = {'baseline': roles.copy(), 'previous': old['guarded'].copy(),
              'source_fit': old['guarded'].copy(), 'source_merged': old['guarded'].copy()}
    owners = {a: old['revision_owner'].copy() for a in ('previous', 'source_fit', 'source_merged')}
    details = {'previous': {'decisions': old['decisions'], 'audit': old['audit']}}
    structures = {}
    complete = bool(old['complete'] and work['complete'] and merged_work['complete'])
    for name, context in [('previous', graph)]+contexts:
        if context is None:
            structures[name] = []
            continue
        audit = {'pair_checks': 0, 'cell_checks': 0, 'limits': []}
        structures[name] = previous.grids(ink, context, audit)
        complete &= not audit['limits']
        if name == 'previous':
            continue
        candidate = previous.analyze(gray, context, roles, authority=authority)
        complete &= candidate['complete']
        changed = candidate['guarded'] != roles
        output[name][changed] = candidate['guarded'][changed]
        owners[name][changed] = candidate['revision_owner'][changed]
        details[name] = {'decisions': candidate['decisions'], 'audit': candidate['audit'],
                         'structural_audit': audit}
    if not complete:
        for a in ('source_fit', 'source_merged'):
            output[a][:] = old['guarded']
            owners[a][:] = old['revision_owner']
    return {**output, 'owners': owners, 'context_evidence': {'source_fit': work, 'source_merged': merged_work},
            'structures': structures, 'details': details, 'complete': bool(complete),
            'production_admitted': False, 'physical_geometry_approved': False,
            'source_pixels_erased': 0, 'new_positive_roles': 0, 'native_or_GT_inputs': False}
