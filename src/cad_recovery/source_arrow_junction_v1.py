"""Localize morphology-seeded heads at unique, source-observed junctions.

Enclosing triangles may extrapolate their apex beyond a real extension line.
The frozen head-first producer rejects these before junction localization. This
bounded auxiliary search uses its same proposals and checks, with the existing
two-dimensional native-junction check before the final apex-ink gate. It neither
raises confidence nor identifies the transverse junction line as the shaft.
"""

from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from . import arrow_head_completion_v1 as completion
from . import arrow_head_validation_v1 as validation
from .arrow_path_recovery_v1 import native_head_check, native_junction, combine, POLICY as PATH_POLICY


POLICY = {
    'version': 'source-arrow-junction/1.0',
    'maximum_pixels': 1_048_576,
    'maximum_morphology_pixels': 5_242_880,
    'maximum_components_per_scale': 256,
    'maximum_proposals': 512,
    'maximum_lines': 128,
    'maximum_junction_line_checks': 32_768,
    'maximum_profile_pixels': 2_000_000,
    'opening_sides': [3, 5, 7, 9, 11],
    'confidence_calibrated': False,
    'manual_references_used': False,
    'production_admitted': False,
    'physical_geometry_approved': False,
}


def proposals(gray, *, lines, stroke_px, text_polygons=()):
    """Return localized auxiliary candidates and every attempted seed decision.

    Inputs and output coordinates refer to this exact source-pixel window. The
    caller must bind the window to its source and resolve its own shaft IDs.
    Search limits remain explicit even if an earlier seed succeeded.
    """
    if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8
            or gray.ndim != 2 or not 0 < gray.size <= POLICY['maximum_pixels']):
        raise ValueError('Bounded native uint8 window required')
    if not np.isfinite(stroke_px) or not 1 <= stroke_px <= 24:
        raise ValueError('Finite source stroke width in [1, 24] required')
    if len(lines) > POLICY['maximum_lines']:
        raise ValueError('Too many local junction lines')
    for line in lines:
        if line.get('kind') != 'line':
            continue
        points = np.asarray(line['points_px'], float)
        if (points.shape != (2, 2) or not np.isfinite(points).all()
                or np.abs(points).max() > 1e7
                or np.linalg.norm(points[1]-points[0]) <= 0):
            raise ValueError('Finite, nondegenerate local line required')
    for polygon in text_polygons:
        points = np.asarray(polygon, float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or np.abs(points).max() > 1e7):
            raise ValueError('Finite text polygon required')

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    accepted, events, limits = [], [], set()
    reasons = Counter()
    morphology = examined = profile_pixels = junction_checks = 0

    def reject(event, reason):
        event.update(status='rejected', reason=reason)
        reasons[reason] += 1
        events.append(event)

    for side in POLICY['opening_sides']:
        if morphology + gray.size > POLICY['maximum_morphology_pixels']:
            limits.add('morphology_pixel_budget')
            break
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((side, side), np.uint8))
        morphology += gray.size
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        eligible = []
        for contour in sorted(contours, key=lambda c: -cv2.contourArea(c)):
            _, _, width, height = cv2.boundingRect(contour)
            if (cv2.contourArea(contour) >= completion.POLICY['minimum_component_area']
                    and max(width, height) <= completion.POLICY['maximum_component_extent_px']):
                eligible.append(contour)
        if len(eligible) > POLICY['maximum_components_per_scale']:
            limits.add('component_budget')
        for contour in eligible[:POLICY['maximum_components_per_scale']]:
            if examined >= POLICY['maximum_proposals']:
                limits.add('proposal_budget')
                break
            examined += 1
            identity = f'source-junction-{examined:05d}'
            event = {'id': identity, 'opening_side_px': side,
                     'component_bbox_px': list(cv2.boundingRect(contour))}
            candidate = completion.triangle_proposal(contour, identity)
            if candidate is None:
                reject(event, 'not_elongated_triangle')
                continue
            event['original_tip_px'] = deepcopy(candidate['tip_px'])
            # A native thin shaft is observed before using its width to describe
            # the seed. This observation does not pick an owning CAD primitive.
            length = candidate['evidence']['head_length_px']
            shaft_cost = 81 * (math.ceil(max(12, min(40, length*.6))) + 97)
            if profile_pixels + shaft_cost > POLICY['maximum_profile_pixels']:
                limits.add('profile_pixel_budget')
                events.append({**event, 'status': 'unknown', 'reason': 'profile_pixel_budget'})
                break
            profile_pixels += shaft_cost
            valid, shaft = completion.shaft_evidence(candidate, mask, stroke_px)
            if not valid:
                reject(event, shaft['reason'])
                continue
            candidate['evidence'].update(native_shaft=shaft,
                shaft_width_px=shaft['width_px'], arrow_length_px=length)
            if junction_checks + len(lines) > POLICY['maximum_junction_line_checks']:
                limits.add('junction_line_budget')
                events.append({**event, 'status': 'unknown', 'reason': 'junction_line_budget'})
                break
            # The frozen native check uses ceil(max)+1 for a local triangle
            # buffer. Its complete possible junction envelope must stay inside
            # the available pixel centers, not merely below width/height.
            tip = np.asarray(candidate['tip_px'], float)
            unit = np.asarray(candidate['direction'], float)
            forward = max(3., candidate['uncertainty_px'] *
                          PATH_POLICY['maximum_native_forward_shift_in_uncertainties'])
            backward = length * PATH_POLICY['maximum_native_backward_shift_in_head_lengths']
            envelope = np.vstack([candidate['polygon_px'][1:],
                                  tip + forward*unit, tip - backward*unit])
            size = np.array(gray.shape[::-1])
            if (envelope.min(axis=0) < 0).any() or (np.ceil(envelope.max(axis=0))+1 > size).any():
                reject(event, 'junction_search_envelope_clipped')
                continue
            junction_checks += len(lines)
            localized = native_junction(deepcopy(candidate), mask, stroke_px, lines, text_polygons)
            if localized['evidence'].get('tip_localization') != 'native_unique_transverse_line_junction':
                reject(event, 'no_unique_source_supported_junction')
                continue
            polygon = np.asarray(localized['polygon_px'], float)
            delta = polygon[0] - polygon[1:].mean(axis=0)
            length = float(np.linalg.norm(delta))
            localized['direction'] = (delta/length).tolist()
            localized['evidence'].update(head_length_px=length, arrow_length_px=length)
            event['localized_tip_px'] = deepcopy(localized['tip_px'])
            if not completion.tip_has_native_ink(localized, mask, stroke_px):
                reject(event, 'localized_apex_without_native_ink')
                continue
            if (polygon.min(axis=0) < 0).any() or (np.ceil(polygon.max(axis=0))+1 > size).any():
                reject(event, 'localized_triangle_sampling_clipped')
                continue
            valid, head = native_head_check(localized, mask, stroke_px, text_polygons)
            if not valid:
                reject(event, head['reason'])
                continue
            shaft_cost = 81 * (math.ceil(max(12, min(40, length*.6))) + 97)
            cost = shaft_cost + validation.profile_cost(localized)
            if profile_pixels + cost > POLICY['maximum_profile_pixels']:
                limits.add('profile_pixel_budget')
                events.append({**event, 'status': 'unknown', 'reason': 'profile_pixel_budget'})
                break
            profile_pixels += cost
            valid, shaft = completion.shaft_evidence(localized, mask, stroke_px)
            if not valid:
                reject(event, shaft['reason'])
                continue
            if shaft['base_to_shaft_ratio'] < validation.POLICY['minimum_base_to_shaft_ratio']:
                reject(event, 'insufficient_head_to_native_shaft_contrast')
                continue
            valid, shape = validation.shape_check(localized, mask)
            if not valid:
                reject(event, shape['reason'])
                continue
            localized['evidence'].update(method=POLICY['version'],
                path_kind='source_head_seed_unique_transverse_junction',
                native_head_check=head, native_shaft=shaft, native_shape_check=shape,
                shaft_width_px=shaft['width_px'], native_tip_ink=True,
                opening_side_px=side, original_tip_px=candidate['tip_px'],
                shaft_owner_unresolved=True, confidence_inherited_without_recalibration=True)
            # native_junction_lines identifies transverse evidence, not a shaft.
            event.update(status='localized', reason='unique_native_junction_and_rechecked_head',
                         junction_line_ids=localized['evidence']['native_junction_lines'])
            events.append(event)
            accepted.append(localized)
        if limits & {'proposal_budget', 'profile_pixel_budget', 'junction_line_budget'}:
            break
    return {'schema_version': POLICY['version'], 'policy': deepcopy(POLICY),
            'candidates': combine(accepted), 'events': events,
            'analysis': {'morphology_pixels': int(morphology), 'components_examined': examined,
                         'profile_pixels': profile_pixels, 'junction_line_checks': junction_checks,
                         'rejections': dict(reasons), 'limits': sorted(limits)},
            'search_incomplete': bool(limits), 'production_admitted': False,
            'physical_geometry_approved': False}
