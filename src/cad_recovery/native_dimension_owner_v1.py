"""Recheck a short observed shaft without granting a longer ownership interval.

A large context may contain digits parallel to a shaft. The actual source
shaft already measured behind the head bounds a second profile observation;
it does not license shrinking the transverse search band or choosing an owner.
"""
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np

from . import source_arrow_observation_v1 as frozen
from .arrow_head_completion_v1 import shaft_evidence
from .native_dimension_strokes_v1 import observe
from .part_admission_v1 import object_sha


POLICY = {'version': 'native-dimension-owner/1.0', 'maximum_lines': 128,
    'maximum_roi_pixels': 4_000_000, 'minimum_profile_sections': 24,
    'minimum_owner_dot': .98, 'confidence_recalibrated': False,
    'source_interval_ownership_extended': False, 'semantic_roles_inferred': False}


def _line(points):
    ends = np.asarray(points, float)
    if ends.shape != (2, 2) or not np.isfinite(ends).all():
        raise ValueError('Finite source line endpoints required')
    extent = float(np.linalg.norm(ends[1]-ends[0]))
    if extent <= 0:
        raise ValueError('Nonzero source line required')
    return ends, (ends[1]-ends[0])/extent, extent


def _seeds(lines, probes, direction, measured, gray_shape):
    selected, originals = [], {}
    for line in lines:
        ends, unit, extent = _line(line['points_px'])
        if abs(float(unit@direction)) < POLICY['minimum_owner_dot']:
            continue
        uncertainty = float(line.get('uncertainty_px', 1.))
        width = float(line.get('width_px', 1.))
        if not math.isfinite(uncertainty) or uncertainty < 0 or not math.isfinite(width) or width <= 0:
            raise ValueError('Finite positive source width and nonnegative uncertainty required')
        along = (probes-ends[0])@unit
        across = abs((probes-ends[0])@np.array([-unit[1], unit[0]]))
        tolerance = measured+uncertainty
        if not np.all((along >= -tolerance) & (along <= extent+tolerance) & (across <= tolerance)):
            continue
        lo, hi = max(0., float(along.min())), min(extent, float(along.max()))
        current = deepcopy(line)
        current['points_px'] = (ends[0]+np.array([lo, hi])[:, None]*unit).tolist()
        clipped = np.asarray(current['points_px'])
        # A plausible alternative with too little actual original ownership
        # cannot silently disappear while another owner is admitted.
        major = int(np.argmax(abs(unit)))
        if (hi <= lo or math.floor(clipped[:,major].max())-math.ceil(clipped[:,major].min())+1
                < POLICY['minimum_profile_sections']):
            return None, None
        minor = 1-major
        radius = (uncertainty+width/2+math.sqrt(.5))/abs(unit[major])
        if (clipped[:,major].min() < 0 or clipped[:,major].max() >= gray_shape[1-major]
                or clipped[:,minor].min()-radius-1 < 0
                or clipped[:,minor].max()+radius+1 >= gray_shape[1-minor]):
            return None, None
        selected.append(current)
        originals[line['id']] = deepcopy(line)
    return selected, originals


def verify(candidate, gray, lines, text_polygons, stroke):
    """Same API as the frozen verifier; all pixels and owners are rechecked."""
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
        raise ValueError('Nonempty uint8 original source ROI required')
    lines = list(lines)
    ids = [line.get('id') for line in lines]
    if any(not isinstance(identity, str) or not identity for identity in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique source line identities required')
    if gray.size > POLICY['maximum_roi_pixels'] or len(lines) > POLICY['maximum_lines']:
        return None, 'no_observed_carrier_owner'
    checked, reason = frozen.verify(candidate, gray, lines, text_polygons, stroke)
    if checked is not None or reason != 'no_observed_carrier_owner':
        return checked, reason
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    polygon = np.asarray(candidate['polygon_px'], float)
    base = polygon[1:].mean(0)
    delta = np.asarray(candidate['tip_px'])-base
    direction = delta/np.linalg.norm(delta)
    local = deepcopy(candidate)
    local['evidence'].update(head_length_px=float(np.linalg.norm(delta)),
        arrow_length_px=float(np.linalg.norm(delta)), base_width_px=float(np.linalg.norm(polygon[1]-polygon[2])))
    valid, shaft = shaft_evidence(local, mask, stroke)
    if not valid:
        return None, reason
    measured = float(shaft['width_px'])
    near, far = 2*measured, min(float(shaft['observed_length_px']), 8*measured)
    if far <= near:
        return None, reason
    probes = base-np.linspace(near, far, 9)[:, None]*direction
    seeds, originals = _seeds(lines, probes, direction, measured, gray.shape)
    if not seeds:
        return None, reason
    observed = observe(gray, seeds)
    if not observed['complete'] or set(line['id'] for line in observed['lines']) != set(originals):
        return None, reason
    fitted = {line['id']: line for line in observed['lines']}
    revised = deepcopy(lines)
    ownership = []
    for line in revised:
        if line['id'] not in fitted:
            continue
        measured_line = fitted[line['id']]
        ends, unit, _ = _line(measured_line['points_px'])
        original = np.asarray(line['points_px'])
        along = (original-ends[0])@unit
        line['points_px'] = (ends[0]+along[:, None]*unit).tolist()
        line['uncertainty_px'] = measured_line['uncertainty_px']
        ownership.append({'candidate_id': line['id'],
            'original_points_px': original.tolist(), 'owner_points_px': deepcopy(line['points_px']),
            'observed_geometry_points_px': deepcopy(measured_line['points_px']),
            'original_uncertainty_px': originals[line['id']].get('uncertainty_px', 1.),
            'measured_uncertainty_px': measured_line['uncertainty_px'],
            'measured_width_px': measured_line['width_px'],
            'source_members': deepcopy(line.get('source_members', [])),
            'source_profile': deepcopy(measured_line['source_profile'])})
    verified, final_reason = frozen.verify(candidate, gray, revised, text_polygons, stroke)
    if verified is None:
        return None, final_reason
    owners = verified['evidence']['geometry_ids']
    if len(owners) != 1:
        return None, 'ambiguous_carrier_owners'
    if owners != candidate['evidence'].get('geometry_ids', []):
        return None, 'no_observed_carrier_owner'
    verified['evidence']['local_shaft_owner_recheck'] = {
        'schema_version': POLICY['version'], 'policy': deepcopy(POLICY),
        'source_pixels_sha256': hashlib.sha256(gray.tobytes()).hexdigest(),
        'candidate_sha256': object_sha(candidate), 'input_lines_sha256': object_sha(lines),
        'shaft_measurement': deepcopy(shaft), 'probe_points_px': probes.tolist(),
        'probe_interval_behind_base_px': [near, far],
        'candidate_profiles': deepcopy(seeds), 'observed_owner_candidates': ownership,
        'observation_events': deepcopy(observed['events']),
        'observation_audit': deepcopy(observed['audit']),
        'all_plausible_owner_candidates_observed': True,
        'original_longitudinal_ownership_preserved': True,
        'source_interval_ownership_extended': False,
        'confidence_recalibrated': False, 'semantic_roles_inferred': False,
        'physical_geometry_approved': False, 'source_pixels_changed': 0,
        'coordinate_space': candidate.get('source_ref', {}).get('coordinate_space', 'source_window_pixels')}
    return verified, final_reason
