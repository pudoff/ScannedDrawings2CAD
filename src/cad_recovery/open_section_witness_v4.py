"""Source-point probes and native pen corroboration for local open sections."""
from copy import deepcopy
import math
import cv2
import numpy as np

from . import open_section_source_walk_v1 as previous
from .open_source_continuation_v1 import observe as source_continuation
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {**previous.POLICY, 'version': 'open-section-witness/4.0',
          'native_profile_sites': 9, 'minimum_native_profiles': 5,
          'maximum_native_profile_samples': 2000000,
          'analysis_probe_is_CAD_geometry': False,
          'native_margin': 'input half diagonal + native pixel half diagonal'}


def observed_graph(graph):
    """Recompute only analysis probes. Exact source points/IDs/owners stay fixed."""
    output = deepcopy(graph); lookup = {p['id']: p for p in output['proposals']}
    work = 0
    if len(lookup) != len(output['proposals']): raise ValueError('Unique source intervals required')
    if len(output.get('tracks', [])) > POLICY['maximum_lines']:
        return output, {'complete': False, 'reason': 'line_budget'}
    for track in output.get('tracks', []):
        if not track['interval_ids'] or not set(track['interval_ids']) <= lookup.keys():
            raise ValueError('Track refers to foreign source intervals')
        if any(lookup[i].get('parent_kind') in ('arc', 'circle') for i in track['interval_ids']): continue
        points = np.concatenate([np.asarray(lookup[i]['points_px'], float) for i in track['interval_ids']])
        work += len(points)
        if work > POLICY['maximum_source_points']: return output, {'complete': False, 'reason': 'source_point_budget'}
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('Finite observed source points required')
        if len(points) < 3: continue
        fit = cv2.fitLine(points.astype(np.float32), cv2.DIST_L1, 0, .01, .01).reshape(-1)
        axis = fit[:2].astype(float); center = fit[2:].astype(float)
        t = (points-center)@axis
        # Observed extremes, including the raster's true junction location.
        ends = points[[int(np.argmin(t)), int(np.argmax(t))]]
        if np.array_equal(ends[0], ends[1]): continue
        track['points'] = ends.tolist()
        for identity in track['interval_ids']: lookup[identity]['linear_context'] = True
    return output, {'complete': True, 'source_points': work,
                    'source_points_sha256': object_sha([p['points_px'] for p in output['proposals']]),
                    'original_graph_sha256': object_sha(graph), 'probe_graph_sha256': object_sha(output)}


def native_pen(track, owners, gray, matrix, *, upper, audit):
    ends = np.asarray(track['points'], float)
    mapped = np.c_[ends, np.ones(2)]@matrix.T; mapped = mapped[:, :2]
    delta = mapped[1]-mapped[0]; length = float(np.linalg.norm(delta))
    if length <= 0: return None
    normal = np.array([-delta[1], delta[0]])/length
    # Radius search is tied to observed source pen and resize footprint.
    points = np.concatenate([np.asarray(p['points_px'], float) for p in owners])
    q = np.rint((np.c_[points, np.ones(len(points))]@matrix.T)[:, :2]).astype(int)
    inside = (q >= 0).all(1) & (q[:, 0] < gray.shape[1]) & (q[:, 1] < gray.shape[0])
    q = q[inside]
    if not len(q): return None
    # Native DT provides only the bounded search radius; the measured width is
    # the perpendicular ink section, avoiding directional DT pen aliasing.
    radius = max(1., float(np.quantile(audit['distance'][q[:, 1], q[:, 0]], .95)))
    extent = 2*radius+np.linalg.norm(matrix[:2, :2], ord=2)*math.sqrt(.5)
    positions = np.arange(-extent, extent+.25, .25)
    widths = []
    for fraction in np.linspace(.25, .75, POLICY['native_profile_sites']):
        center = mapped[0]+fraction*delta
        sampled = np.rint(center+positions[:, None]*normal).astype(int)
        audit['samples'] += len(sampled)
        if audit['samples'] > POLICY['maximum_native_profile_samples']: return None
        if ((sampled < 0).any() or (sampled[:, 0] >= gray.shape[1]).any() or (sampled[:, 1] >= gray.shape[0]).any()): continue
        values = audit['ink'][sampled[:, 1], sampled[:, 0]] > 0
        origin = int(np.argmin(abs(positions)))
        if not values[origin]: continue
        lo = origin; hi = origin
        while lo > 0 and values[lo-1]: lo -= 1
        while hi+1 < len(values) and values[hi+1]: hi += 1
        if lo == 0 or hi == len(values)-1: continue
        widths.append((hi-lo+1)*.25)
    if len(widths) < POLICY['minimum_native_profiles']: return None
    return {'pen_px': float(np.quantile(widths, .95 if upper else .25)),
            'sections_px': widths, 'endpoints_px': mapped.tolist(), 'normal': normal.tolist()}


def analyze(gray, graph, raw_roles, source_owner, ambiguous_owner, heads, *,
            authority, original_gray=None, source_binding=None, page_negative=None):
    checked_gray(gray); original_graph_sha = object_sha(graph)
    if len(graph['proposals']) > POLICY['maximum_intervals']: raise ValueError('Bounded source graph required')
    probe, observer = source_continuation(gray, graph)
    if not observer['complete']: probe['search_incomplete'] = True
    old = previous.analyze(gray, probe, raw_roles, source_owner, ambiguous_owner, heads,
        authority=authority, original_gray=original_gray, source_binding=source_binding, page_negative=page_negative)
    raw_mask = old['mask']; mask = np.zeros_like(raw_mask)
    native = np.zeros_like(original_gray) if original_gray is not None else None
    checks = []; samples = 0
    if old['complete'] and original_gray is not None:
        matrix = np.asarray(source_binding['matrix'], float)
        tracks = {t['id']: t for t in probe.get('tracks', [])}
        tracks.update({p['id']: {'id': p['id'], 'interval_ids': [p['id']],
                               'points': [p['points_px'][0], p['points_px'][-1]]}
                       for p in probe['proposals'] if p['id'] not in tracks})
        lookup = {p['id']: p for p in probe['proposals']}
        ink = foreground(original_gray); distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        work = {'ink': ink, 'distance': distance, 'samples': 0}; cache = {}
        for witness in old['witnesses']:
            identities = witness['boundary_track_ids']+witness['hatch_track_ids']; facts = []
            for index, identity in enumerate(identities):
                t = tracks[identity]
                key = (identity, index >= 2)
                if key not in cache:
                    cache[key] = native_pen(t, [lookup[i] for i in t['interval_ids']], original_gray,
                                            matrix, upper=index >= 2, audit=work)
                facts.append(cache[key])
                if work['samples'] > POLICY['maximum_native_profile_samples']: break
            if work['samples'] > POLICY['maximum_native_profile_samples']: break
            accepted = all(f is not None for f in facts)
            if accepted:
                ratio = min(f['pen_px'] for f in facts[:2])/max(f['pen_px'] for f in facts[2:])
                accepted = ratio >= POLICY['minimum_pen_ratio']
            else: ratio = None
            checks.append({'boundary_track_ids': witness['boundary_track_ids'], 'accepted': accepted,
                           'native_pen_ratio': ratio, 'profiles': facts})
            if not accepted: continue
            local = np.zeros_like(mask)
            for identity, span in zip(witness['boundary_track_ids'], witness['local_spans_px']):
                track = tracks[identity]; ids = {i for i in track['interval_ids']}
                numbers = [n for n, p in enumerate(probe['proposals'], 1) if p['id'] in ids]
                yy, xx = np.nonzero((raw_mask > 0) & np.isin(source_owner, numbers))
                endpoints = np.asarray(track['points'], float); u = endpoints[1]-endpoints[0]; u /= np.linalg.norm(u)
                along = (np.c_[xx, yy]-endpoints[0])@u
                keep = (along > span[0]) & (along < span[1]); local[yy[keep], xx[keep]] = 255
            mask |= local
            transferred = cv2.resize(local, original_gray.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
            margin = math.hypot(matrix[0, 0]/2, matrix[1, 1]/2)+math.sqrt(.5)
            yy, xx = np.nonzero(transferred & (distance > margin)); xy = np.c_[xx, yy]
            keep = np.ones(len(xy), bool)
            for fact in facts[2:]:
                keep &= abs((xy-np.asarray(fact['endpoints_px'][0]))@np.asarray(fact['normal'])) > fact['pen_px']/2+math.sqrt(.5)
            native[yy[keep], xx[keep]] = 255
        samples = work['samples']
    complete = old['complete'] and samples <= POLICY['maximum_native_profile_samples']
    if not complete:
        mask[:] = 0
        if native is not None: native[:] = 0
    audit = deepcopy(old['audit'])
    if samples > POLICY['maximum_native_profile_samples']: audit['limits'].append('native_profile_budget')
    if original_graph_sha != object_sha(graph): raise ValueError('Source observer changed the original graph')
    return {**{k: v for k, v in old.items() if k not in ('mask', 'policy')},
            'mask': mask, 'native_mask': native, 'raw_witness_mask_sha256': sha(raw_mask),
            'complete': complete, 'audit': audit, 'policy': deepcopy(POLICY), 'source_observer': observer,
            'native_checks': checks, 'native_profile_samples': samples,
            'original_sha256': sha(original_gray) if original_gray is not None else None,
            'original_binding_sha256': object_sha(source_binding),
            'graph_sha256': original_graph_sha, 'source_graph_modified': False,
            'native_admission_is_physical_CAD_approval': False}
