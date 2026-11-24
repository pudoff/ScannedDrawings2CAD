"""Source-profile head proposals when morphology joins a head to a contour.

The two visible taper flanks determine an apex hypothesis. A subsequent return
to a thin observed shaft bounds the base. An occluded base is explicitly an
interval, never an exact recovered triangle or accepted CAD geometry.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from . import arrow_head_completion_v1 as completion
from . import arrow_head_validation_v1 as validation
from .arrow_detection import _profile, _runs
from .arrow_path_recovery_v1 import native_head_check, combine
from .leader_contact_evidence_v1 import tapered_profile
from .scaled_arrow_detection_v1 import _stroke


POLICY = {'version': 'source-arrow-taper/1.0', 'maximum_pixels': 1_048_576,
          'maximum_lines': 128, 'maximum_samples': 2_000_000,
          'maximum_profile_pixels': 2_000_000, 'maximum_tip_extrapolation_strokes': 16.,
          'maximum_head_length_px': 192., 'minimum_head_length_px': 6.,
          'minimum_shaft_run_px': 8.,
          'base_interval_semantics': 'conservative gap from accepted taper core to observed thin return; not a calibrated or minimal interval',
          'base_occluded_semantics': 'possibly occluded or otherwise unresolved transition; not proof of an exact occluder footprint',
          'production_admitted': False,
          'physical_geometry_approved': False, 'manual_references_used': False}


def possible_taper_core(ink, origin, toward_shaft, pen, account):
    """Cheap necessary condition for the unchanged bilateral profile search.

    Its accepted final three widths average at least 4*pen, so one row must
    have width >=4*pen. All accepted rows have |left+right|<=2.5*pen. Therefore
    that row contains both +/-0.75*pen and at least one of +/-2*pen. We sample
    inward-rounded positions on the exact quarter-pixel grid used by the full
    search. This can admit crossings, but cannot reject a qualifying taper row.
    """
    positions = np.arange(-40*pen, 0., max(1., pen))
    offsets = np.arange(-18*pen, 18*pen+.25, .25)
    indices = [np.searchsorted(offsets, -.75*pen, side='left'),
               np.searchsorted(offsets, .75*pen, side='right')-1,
               np.searchsorted(offsets, -2*pen, side='left'),
               np.searchsorted(offsets, 2*pen, side='right')-1]
    chosen = offsets[np.asarray(indices)]
    normal = np.array([-toward_shaft[1], toward_shaft[0]])
    account(len(positions)*len(chosen))
    q = np.rint(origin+positions[:, None, None]*toward_shaft+
                chosen[None, :, None]*normal).astype(int)
    valid = ((q >= 0) & (q < ink.shape[::-1])).all(axis=-1)
    hit = np.zeros(valid.shape, bool)
    hit[valid] = ink[q[..., 1][valid], q[..., 0][valid]] > 0
    return bool(np.any(hit[:, 0] & hit[:, 1] & (hit[:, 2] | hit[:, 3])))


def proposals(gray, *, lines, stroke_px=None, text_polygons=()):
    """Derive bounded candidates from every input line endpoint, without GT."""
    if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2
            or not 0 < gray.size <= POLICY['maximum_pixels'] or len(lines) > POLICY['maximum_lines']):
        raise ValueError('Bounded native window and local lines required')
    for polygon in text_polygons:
        p = np.asarray(polygon, float)
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError('Finite text exclusion polygon required')
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    audit = {'limits': [], 'taper_samples': 0, 'raster_pixels': 0}
    profile_pixels = samples = 0
    events, accepted = [], []

    class Limit(Exception):
        pass

    def account(count):
        nonlocal samples
        samples += count
        if samples > POLICY['maximum_samples']:
            raise Limit()

    try:
        for line in lines:
            if line.get('kind') != 'line':
                continue
            points = np.asarray(line['points_px'], float)
            if (points.shape != (2, 2) or not np.isfinite(points).all()
                    or np.abs(points).max() > 1e7):
                raise ValueError('Finite source line required')
            delta = points[1]-points[0]
            length = float(np.linalg.norm(delta))
            if length < 12:
                continue
            unit = delta/length
            pen = _stroke(ink, {'points': points, 'length': length, 'u': unit, 'original': line}, account)
            if pen is None or not 1 <= pen <= 24 or length < 20*pen:
                events.append({'geometry_id': line['id'], 'reason': 'no_supported_thin_shaft', 'pen_px': pen})
                continue
            for endpoint in (0, 1):
                origin = points[endpoint]
                toward_shaft = unit if endpoint == 0 else -unit
                normal = np.array([-toward_shaft[1], toward_shaft[0]])
                event = {'geometry_id': line['id'], 'endpoint': endpoint, 'pen_px': pen}
                if not possible_taper_core(ink, origin, toward_shaft, pen, account):
                    events.append({**event, 'reason': 'no_source_pixels_for_minimum_bilateral_width'})
                    continue
                before = audit['taper_samples']
                taper = tapered_profile(ink, origin, toward_shaft, pen, audit)
                account(audit['taper_samples']-before)
                if audit['limits']:
                    raise Limit()
                if taper is None:
                    events.append({**event, 'reason': 'no_bilateral_visible_taper'})
                    continue
                sections = np.asarray(taper['source_cross_sections'], float)
                params = np.linalg.lstsq(np.c_[sections[:, 0], np.ones(len(sections))], sections[:, 1:], rcond=None)[0]
                spread = float(params[0, 1]-params[0, 0])
                apex_t = float((params[1, 0]-params[1, 1])/spread)
                if not sections[0, 0]-POLICY['maximum_tip_extrapolation_strokes']*pen <= apex_t < sections[0, 0]:
                    events.append({**event, 'reason': 'apex_outside_bounded_taper_extrapolation'})
                    continue
                apex_offset = float(params[0, 0]*apex_t+params[1, 0])
                tip = origin+apex_t*toward_shaft+apex_offset*normal
                # Observe the first sustained thin run beyond the visible core.
                # No manually selected base or expected head position is used.
                start = float(sections[-1, 0])
                span = -start+max(12., 12*pen)
                cost = 81*(math.ceil(span)+97)
                if profile_pixels+cost > POLICY['maximum_profile_pixels']:
                    raise Limit()
                profile_pixels += cost
                along, widths, _ = _profile(ink, origin+start*toward_shaft, toward_shaft, span)
                positions = along+start
                narrow = ((positions >= start) & (positions <= max(12., 12*pen))
                          & (widths > 0) & (widths <= max(3., 2*pen)))
                runs = [(a, b) for a, b in _runs(narrow)
                        if positions[b-1]-positions[a] >= POLICY['minimum_shaft_run_px']]
                if not runs:
                    events.append({**event, 'reason': 'no_sustained_thin_return_after_taper'})
                    continue
                first, _ = runs[0]
                base_t = float(positions[first])
                edge_offsets = params[0]*base_t+params[1]
                base_points = origin+base_t*toward_shaft+edge_offsets[:, None]*normal
                base = base_points.mean(0)
                direction = tip-base
                head_length = float(np.linalg.norm(direction))
                width = float(edge_offsets[1]-edge_offsets[0])
                if (not POLICY['minimum_head_length_px'] <= head_length <= POLICY['maximum_head_length_px']
                        or width <= 0 or not .65 <= head_length/width <= 8):
                    events.append({**event, 'reason': 'taper_triangle_outside_profile_geometry_scope'})
                    continue
                polygon = np.vstack([tip, base_points])
                if (polygon < 0).any() or (np.ceil(polygon.max(0))+1 > gray.shape[::-1]).any():
                    events.append({**event, 'reason': 'triangle_sampling_clipped'})
                    continue
                # Existing frozen profile geometry uses the .65..8 aspect
                # range; this is not the morphology component-shape filter.
                candidate = {'id': f'source-taper-{len(events)+1:05d}', 'tip_px': tip.tolist(),
                    'polygon_px': polygon.tolist(), 'direction': (direction/head_length).tolist(),
                    'classification': 'filled_arrow_candidate', 'confidence': .55,
                    'uncertainty_px': max(2., taper['edge_error_p95_px']/spread+math.sqrt(.5)),
                    'evidence': {'method': POLICY['version'], 'path_kind': 'bilateral_source_taper',
                        'head_length_px': head_length, 'arrow_length_px': head_length,
                        'base_width_px': width, 'shaft_width_px': pen,
                        'geometry_ids': [line['id']], 'source_members': deepcopy(line.get('source_members', [])),
                        'source_taper': taper, 'apex_t_along_shaft_px': apex_t,
                        'base_visibility_interval_along_shaft_px': [start, base_t],
                        'base_occluded': base_t-start > 2*pen,
                        'base_position_precisely_confirmed': False,
                        'triangle_geometry_approved': False, 'confidence_calibrated': False}}
                if not completion.tip_has_native_ink(candidate, ink, pen):
                    events.append({**event, 'reason': 'extrapolated_apex_without_source_ink'})
                    continue
                valid, head = native_head_check(candidate, ink, pen, text_polygons)
                if not valid:
                    events.append({**event, 'reason': head['reason']})
                    continue
                cost = validation.profile_cost(candidate)+81*(math.ceil(max(12, min(40, head_length*.6)))+97)
                if profile_pixels+cost > POLICY['maximum_profile_pixels']:
                    raise Limit()
                profile_pixels += cost
                valid, shaft = completion.shaft_evidence(candidate, ink, pen)
                if not valid or shaft.get('base_to_shaft_ratio', 0) < validation.POLICY['minimum_base_to_shaft_ratio']:
                    events.append({**event, 'reason': shaft['reason'] if not valid else 'insufficient_head_shaft_contrast'})
                    continue
                valid, shape = validation.shape_check(candidate, ink)
                if not valid:
                    events.append({**event, 'reason': shape['reason']})
                    continue
                candidate['evidence'].update(native_head_check=head, native_shape_check=shape,
                    native_shaft=shaft, shaft_width_px=shaft['width_px'], native_tip_ink=True)
                accepted.append(candidate)
                events.append({**event, 'reason': 'visible_apex_taper_and_thin_return',
                    'tip_px': tip.tolist(), 'base_occluded': candidate['evidence']['base_occluded'],
                    'base_visibility_interval_along_shaft_px': [start, base_t]})
    except Limit:
        audit['limits'].append('source_taper_sample_or_profile_budget')
    return {'schema_version': POLICY['version'], 'policy': deepcopy(POLICY),
            'candidates': combine(accepted), 'events': events,
            'analysis': {'samples': samples, 'profile_pixels': profile_pixels,
                         'taper_audit': audit, 'limits': sorted(set(audit['limits']))},
            'search_incomplete': bool(audit['limits']), 'production_admitted': False,
            'physical_geometry_approved': False}
