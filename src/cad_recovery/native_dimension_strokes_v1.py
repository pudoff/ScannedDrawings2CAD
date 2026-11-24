"""Rebind straight seeds to measured strokes in a bounded original-pixel window.

This observes geometry only. A prior working-view role is neither copied nor
upscaled into source evidence, and unresolved seeds stay in the event ledger.
"""
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np

from .source_line_extension_v1 import recover


POLICY = {'version': 'native-dimension-strokes/1.0', 'maximum_roi_pixels': 4_000_000,
          'maximum_lines': 128, 'maximum_source_samples': 4_000_000,
          'minimum_sections': 24, 'maximum_cross_section': 256,
          'pixel_radius': math.sqrt(.5), 'roles_inferred': False}


class _Budget(Exception):
    pass


def _consume(audit, count):
    audit['source_support_samples'] += int(count)
    if audit['source_support_samples'] > POLICY['maximum_source_samples']:
        audit['limits'].append('source_sample_budget')
        raise _Budget()


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _parallel_alternatives(ink, points, width, uncertainty, audit):
    """Do not let a nearest-line fit hide two supported parallel choices."""
    delta = points[1]-points[0]
    major = int(np.argmax(abs(delta)))
    minor = 1-major
    normal = abs(delta[major])/np.linalg.norm(delta)
    radius = (uncertainty+width/2+POLICY['pixel_radius'])/normal
    if 2*math.ceil(radius)+3 > POLICY['maximum_cross_section']:
        audit['limits'].append('cross_section_budget')
        raise _Budget()
    data = ink.T if major == 0 else ink
    start, stop = sorted(points[:, major])
    xs = np.arange(max(0, math.ceil(start)), min(data.shape[0], math.floor(stop)+1))
    slope = delta[minor]/delta[major]
    centers = points[0, minor]+(xs-points[0, major])*slope
    ambiguous = []
    for x, center in zip(xs, centers):
        left = max(0, math.floor(center-radius)-1)
        right = min(data.shape[1], math.ceil(center+radius)+2)
        _consume(audit, max(0, right-left))
        intervals = [(a+left, b+left-1) for a, b in _runs(data[x, left:right] > 0)]
        eligible = [(a, b) for a, b in intervals
                    if a > left and b < right-1 and abs((a+b)/2-center) <= radius]
        ambiguous.append(len(eligible) > 1)
    lengths = [b-a for a, b in _runs(np.asarray(ambiguous, bool))]
    return {'sections': len(xs), 'ambiguous_sections': int(sum(ambiguous)),
            'maximum_ambiguous_run_sections': int(max(lengths, default=0)),
            'search_radius_minor_px': float(radius)}


def _observed_points(ink, fitted, audit):
    """Sample measured centres, never filling gaps with synthetic support."""
    a, b = fitted
    count = max(2, int(math.ceil(max(abs(b-a))))+1)
    probes = np.rint(np.linspace(a, b, count)).astype(int)
    _consume(audit, len(probes))
    inside = ((probes >= 0).all(1) & (probes[:, 0] < ink.shape[1]) & (probes[:, 1] < ink.shape[0]))
    probes = probes[inside]
    probes = probes[ink[probes[:, 1], probes[:, 0]] > 0]
    if not len(probes):
        return probes
    # Keep path order; np.unique(axis=0) would reorder a descending line.
    keep = np.r_[True, (np.diff(probes, axis=0) != 0).any(1)]
    return probes[keep]


def observe(gray, candidates, *, maximum_lines=128):
    """Return source-fitted LINE observations, plus every unresolved seed.

    `complete` means this bounded list was examined without a resource stop;
    it does not certify complete geometry, ownership, or semantic roles.
    """
    if (not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8
            or not gray.size):
        raise ValueError('A nonempty uint8 original grayscale window is required')
    if type(maximum_lines) is not int or not 0 <= maximum_lines <= POLICY['maximum_lines']:
        raise ValueError('Invalid bounded line count')
    candidates = list(candidates)
    identities = [c.get('id') for c in candidates]
    if any(not isinstance(v, str) or not v for v in identities) or len(set(identities)) != len(identities):
        raise ValueError('Unique named source candidates required')
    audit = {'source_support_samples': 0, 'limits': [], 'input_candidates': len(candidates),
             'input_pixels_sha256': hashlib.sha256(gray.tobytes()).hexdigest(),
             'source_pixels_changed': 0, 'roles_inferred': False,
             'full_geometry_or_ownership_coverage_claimed': False}
    events, output = [], []
    if gray.size > POLICY['maximum_roi_pixels']:
        audit['limits'].append('roi_pixel_budget')
    if sum(c.get('kind') == 'line' for c in candidates) > maximum_lines:
        audit['limits'].append('line_count_budget')
    if not audit['limits']:
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        distance = cv2.distanceTransform(255-ink, cv2.DIST_L2, 5)
        try:
            for candidate in candidates:
                event = {'candidate_id': candidate['id'], 'status': 'unresolved', 'reasons': []}
                events.append(event)
                if candidate.get('kind') != 'line':
                    event['reasons'].append('not_a_straight_seed')
                    continue
                points = np.asarray(candidate.get('points_px'), float)
                if (points.shape != (2, 2) or not np.isfinite(points).all()
                        or np.linalg.norm(points[1]-points[0]) <= 0):
                    raise ValueError('Finite nonzero straight seed required')
                width = candidate.get('width_px', 1.)
                uncertainty = candidate.get('uncertainty_px', 1.)
                if (isinstance(width, bool) or isinstance(uncertainty, bool)
                        or not np.isfinite(width) or not np.isfinite(uncertainty)
                        or width <= 0 or uncertainty < 0):
                    raise ValueError('Finite positive width and nonnegative uncertainty required')
                alternative = _parallel_alternatives(ink, points, width, uncertainty, audit)
                event['parallel_search'] = alternative
                if alternative['maximum_ambiguous_run_sections'] >= POLICY['minimum_sections']:
                    event['reasons'].append('ambiguous_parallel_source_strokes')
                    continue
                # The transformed location error determines the search band;
                # the returned thickness is still measured from source ink.
                search_width = max(width, uncertainty)
                fitted = recover(ink, distance, points, search_width, audit)
                if audit['limits'] or audit['source_support_samples'] > POLICY['maximum_source_samples']:
                    if not audit['limits']:
                        audit['limits'].append('source_sample_budget')
                    raise _Budget()
                if fitted is None:
                    event['reasons'].append('source_sections_do_not_confirm_straight_seed')
                    continue
                ends = fitted['points']
                unit = (ends[1]-ends[0])/np.linalg.norm(ends[1]-ends[0])
                normal = np.array([-unit[1], unit[0]])
                displacement = float(max(abs((points-ends[0]) @ normal)))
                bound = float(uncertainty+fitted['width']/2+POLICY['pixel_radius'])
                event.update(seed_to_source_max_px=displacement, allowed_seed_to_source_px=bound)
                if displacement > bound:
                    event['reasons'].append('refit_displaced_beyond_seed_uncertainty')
                    continue
                support = _observed_points(ink, ends, audit)
                if len(support) < POLICY['minimum_sections']:
                    event['reasons'].append('too_few_actual_source_ink_samples')
                    continue
                edge_margin = fitted['width']/2+POLICY['pixel_radius']
                clipped = [bool((point < edge_margin).any()
                                or (point >= np.asarray(gray.shape[::-1])-edge_margin-1).any()) for point in ends]
                proof = {'seed_points_px': points.tolist(), 'source_sections': fitted['source_sections'],
                         'residual_minor_px': fitted['residual_minor_px'],
                         'seed_to_source_max_px': displacement, 'parallel_search': alternative,
                         'source_pixels_changed': 0, 'support_samples_are_actual_ink': True,
                         'endpoint_clipped': clipped, 'endpoints_are_material_boundaries': False,
                         'claim_scope': 'geometry_evidence_only_not_source_interval_ownership',
                         'extended_support_ownership_confirmed': False,
                         'measurement_coordinate_space': 'source_window_pixels',
                         'input_pixels_sha256': audit['input_pixels_sha256'], 'roles_inferred': False}
                output.append({'id': candidate['id'], 'kind': 'line', 'points_px': ends.tolist(),
                               'support_points': support.tolist(), 'width_px': float(fitted['width']),
                               'uncertainty_px': float(fitted['residual_minor_px']+POLICY['pixel_radius']),
                               'source_members': deepcopy(candidate.get('source_members', [])),
                               'source_profile': proof})
                event.update(status='observed', actual_ink_samples=len(support), endpoint_clipped=clipped)
        except _Budget:
            if events:
                events[-1]['reasons'].append('source_observation_budget_exhausted')
    complete = not audit['limits']
    if not complete:
        audit['withheld_observations'] = len(output)
        output = []
    audit['examined_candidates'] = len(events)
    return {'schema_version': POLICY['version'], 'lines': output, 'events': events,
            'audit': audit, 'complete': complete, 'policy': deepcopy(POLICY)}
