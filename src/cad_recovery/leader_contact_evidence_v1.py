"""Witness a tapered source head and connected shaft without guessing its tip."""
import math

import cv2
import numpy as np

from .grid_role_revision_v1 import sha
from .local_contour_profiles_v2 import measure
from .part_admission_v1 import object_sha
from .shared_stroke_ownership_v1 import POLICY as TRACK_POLICY, track_records
from .source_stamp_context_v1 import continuous
from .stroke_roles_v1 import checked_gray, foreground


def tapered_profile(ink, origin, toward_shaft, pen, audit):
    """Find two independently observed, diverging edges behind a shaft seed."""
    normal = np.array([-toward_shaft[1], toward_shaft[0]])
    positions = np.arange(-40*pen, 0., max(1., pen))
    offsets = np.arange(-18*pen, 18*pen+.25, .25)
    audit['taper_samples'] += len(positions)*len(offsets)
    if audit['taper_samples'] > TRACK_POLICY['maximum_profile_samples']:
        audit['limits'].append('taper_sample_budget')
        return None
    q = np.rint(origin+positions[:, None, None]*toward_shaft+offsets[None, :, None]*normal).astype(int)
    valid = ((q >= 0) & (q < ink.shape[::-1])).all(axis=-1)
    hit = np.zeros(valid.shape, bool)
    hit[valid] = ink[q[..., 1][valid], q[..., 0][valid]] > 0
    rows = []
    for sample, row in zip(positions, hit):
        boundaries = np.diff(np.r_[False, row, False].astype(int))
        spans = [(lo, hi) for lo, hi in zip(np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1))
                 if lo and hi < len(row) and offsets[lo] <= .75*pen and offsets[hi-1] >= -.75*pen]
        if len(spans) != 1:
            rows.append([sample, np.nan, np.nan])
            continue
        lo, hi = spans[0]
        rows.append([sample, offsets[lo], offsets[hi-1]])
    rows = np.asarray(rows)
    best = None
    for span in (16, 14, 12, 10):
        for start in range(len(rows)-span+1):
            samples = rows[start:start+span]
            t, lo, hi = samples.T
            if not np.isfinite(samples).all() or (np.max(abs(lo+hi)) > 2.5*pen):
                continue
            width = hi-lo
            if width[-3:].mean() < max(4*pen, 1.65*width[:3].mean()) or width[0] > 6*pen:
                continue
            edges = np.column_stack((lo, hi))
            design = np.c_[t, np.ones(len(t))]
            parameters = np.linalg.lstsq(design, edges, rcond=None)[0]
            slopes = parameters[0]*np.array([-1., 1.])
            if not np.all((slopes >= .13) & (slopes <= 1.4)) or max(slopes) > 2.5*min(slopes):
                continue
            error = float(np.quantile(abs(design @ parameters-edges), .95))
            if error > max(1.1, .55*pen):
                continue
            if not continuous(ink, np.stack((origin+t[-1]*toward_shaft, origin+5*pen*toward_shaft)), pen, audit):
                continue
            record = {'distance_interval_along_source_px': [float(t[0]), float(t[-1])],
                      'source_cross_sections': samples.tolist(), 'bilateral_edge_slopes': slopes.tolist(),
                      'edge_error_p95_px': error, 'head_core_connected_to_shaft': True,
                      'tip_and_full_triangle_unconfirmed': True}
            if best is None or (span, -error) > best[0]:
                best = ((span, -error), record)
        if best is not None or audit['limits']:
            break
    return best[1] if best and not audit['limits'] else None


def observe(gray, graph):
    checked_gray(gray)
    audit = {'limits': [], 'taper_samples': 0, 'raster_pixels': 0, 'track_queries': 0}
    result = {'source_sha256': sha(gray), 'graph_sha256': object_sha(graph), 'witnesses': [], 'audit': audit}
    if graph.get('search_incomplete') or len(graph['tracks']) > TRACK_POLICY['maximum_tracks']:
        audit['limits'].append('source_graph_incomplete')
        return {**result, 'complete': False}
    ink = foreground(gray)
    work = {'ink': ink, 'distance': cv2.distanceTransform(ink, cv2.DIST_L2, 5), 'samples': 0}
    lines = list(track_records(gray, graph, work, audit, profile_measure=measure))
    for record in lines if not audit['limits'] else []:
        pen = record['pen_px']
        if record['length'] < 20*pen or record['upper_pen_px'] > 2*pen:
            continue
        for a, b in (record['ends'], record['ends'][::-1]):
            u = (b-a)/record['length']
            profile = tapered_profile(ink, a, u, pen, audit)
            if profile is None:
                continue
            continuations = []
            for tail in lines:
                audit['track_queries'] += 1
                if audit['track_queries'] > TRACK_POLICY['maximum_pairs']:
                    audit['limits'].append('continuation_pair_budget')
                    break
                if tail['id'] == record['id'] or tail['upper_pen_px'] > 2*pen or abs(tail['unit'] @ u) < math.cos(math.radians(8)):
                    continue
                delta = tail['ends']-b
                along = delta @ u
                normal = abs(delta @ np.array([-u[1], u[0]]))
                nearest = int(along.argmin())
                if not (0 < along[nearest] <= 55*pen and normal.max() <= pen+1.):
                    continue
                if continuous(ink, np.stack((b, tail['ends'][nearest])), pen, audit):
                    continuations.append({'track_id': tail['id'], 'source_members': tail['members'],
                                          'points_px': tail['ends'].tolist(), 'gap_is_observed_ink': True})
            result['witnesses'].append({'shaft_track_id': record['id'], 'source_members': record['members'],
                'shaft_points_px': [a.tolist(), b.tolist()], 'pen_px': pen, 'direction_toward_taper': (-u).tolist(),
                'taper': profile, 'observed_continuations': continuations, 'semantic_role_assigned': False})
        if audit['limits']:
            break
    if audit['limits']:
        result['witnesses'] = []
    return {**result, 'complete': not audit['limits']}
