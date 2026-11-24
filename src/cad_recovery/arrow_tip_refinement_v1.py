"""Experimental visible-tip localization for existing, unverified arrow observations.

The old detector and its acceptance gates remain intact. Its measured taper
defines a bounded search for a unique transverse raster junction; an unverified
width-only ablation is available for comparison. No arrows or CAD dimensions
are created by this adapter.
"""
from copy import deepcopy

import cv2
import numpy as np

from .family_catalog_v2 import require
from .arrow_detection import _profile, _runs

POLICY = {
    'version': 'taper-guided-raster-junction-tip/1.0',
    'original_taper_endpoint_width_px': 1.0,
    'maximum_shift_fraction_of_head': .5,
    'raster_support_radius_px': 2,
    'junction_width_relative_to_head': 1.5,
    'junction_width_relative_to_shaft': 4,
    'modes': ['width_only_ablation', 'verified_junction'],
    'default_mode': 'verified_junction',
    'new_arrow_discovery': False,
    'confidence_calibrated': False,
    'physical_geometry_approved': False,
    'production_admitted': False,
}


def refine_candidate(candidate, foreground, text_polygons=(), *, mode='verified_junction'):
    """Return (observation or None, decision), keeping caller data unchanged."""
    require(candidate.get('classification') == 'filled_arrow_candidate', 'Expected a filled-arrow observation')
    require(foreground.ndim == 2 and foreground.dtype == np.uint8, 'Expected a uint8 foreground raster')
    require(mode in POLICY['modes'], 'Unknown tip refinement mode')
    tip = np.asarray(candidate['tip_px'], dtype=float)
    unit = np.asarray(candidate['direction'], dtype=float)
    polygon = np.asarray(candidate['polygon_px'], dtype=float)
    require(tip.shape == unit.shape == (2,) and polygon.shape == (3, 2)
            and np.isfinite(tip).all() and np.isfinite(unit).all() and np.isfinite(polygon).all()
            and np.isclose(np.linalg.norm(unit), 1, atol=1e-6), 'Invalid arrow geometry')
    polygons = []
    for points in text_polygons:
        array = np.asarray(points, dtype=np.float32)
        require(array.ndim == 2 and array.shape[1] == 2 and len(array) >= 3
                and np.isfinite(array).all(), 'Invalid text polygon')
        polygons.append(array)
    evidence = candidate['evidence']
    shaft, base, length = (float(evidence[k]) for k in ('shaft_width_px', 'base_width_px', 'arrow_length_px'))
    require(np.isfinite([shaft, base, length]).all() and shaft >= 1 and base > 1 and length > 0,
            'Invalid measured taper')
    base_center = polygon[1:].mean(axis=0)
    require(np.allclose(polygon[0], tip, atol=1e-5)
            and np.allclose(tip-base_center, unit*length, atol=1e-4)
            and np.isclose(np.linalg.norm(polygon[1]-polygon[2]), base, atol=1e-4),
            'Taper evidence does not match the arrow polygon')
    fraction = (shaft - 1) / (base - 1)
    decision = {'id': candidate['id'], 'original_tip_px': tip.tolist(),
                'shaft_width_px': shaft, 'original_head_length_px': length,
                'shift_fraction': float(fraction), 'shift_px': float(fraction*length)}
    result = deepcopy(candidate)
    if not 0 <= fraction < POLICY['maximum_shift_fraction_of_head']:
        return result, {**decision, 'status': 'unchanged', 'reason': 'insufficient_taper_contrast'}
    decision['mode'] = mode
    refined = tip - unit * (fraction*length)
    if mode == 'verified_junction':
        back = fraction*length + 2*shaft
        front = 2*shaft
        first = tip-unit*back
        along, widths, _ = _profile(foreground, first, unit, back+front)
        threshold = max(base*POLICY['junction_width_relative_to_head'],
                        shaft*POLICY['junction_width_relative_to_shaft'])
        eligible = (along >= 0) & (along <= back+front) & (widths >= threshold)
        junctions = []
        for begin, end in _runs(eligible):
            if end-begin <= 2*shaft+3:
                position = float(np.mean(along[begin:end]))
                shift = back-position
                if -2*shaft <= shift < POLICY['maximum_shift_fraction_of_head']*length:
                    junctions.append(position)
        decision['junction_count'] = len(junctions)
        if len(junctions) != 1:
            return result, {**decision, 'status': 'unchanged', 'reason': 'no_unique_transverse_junction'}
        refined = first+unit*junctions[0]
        decision['actual_signed_shift_px'] = float((tip-refined)@unit)
    height, width = foreground.shape
    if not (0 <= refined[0] < width and 0 <= refined[1] < height):
        return result, {**decision, 'status': 'unchanged', 'reason': 'refined_tip_outside_raster'}
    x, y = np.rint(refined).astype(int)
    radius = POLICY['raster_support_radius_px']
    support = foreground[max(0, y-radius):min(height, y+radius+1), max(0, x-radius):min(width, x+radius+1)]
    if not support.any():
        return result, {**decision, 'status': 'unchanged', 'reason': 'no_raster_support'}
    polygon[0] = refined
    centroid = polygon.mean(axis=0)
    if any(cv2.pointPolygonTest(p, tuple(refined), False) >= 0
           or cv2.pointPolygonTest(p, tuple(centroid), False) >= 0 for p in polygons):
        return None, {**decision, 'status': 'excluded', 'reason': 'refined_tip_or_centroid_inside_text'}
    result['tip_px'] = refined.tolist()
    result['polygon_px'] = polygon.tolist()
    result['uncertainty_px'] = max(float(candidate['uncertainty_px']), shaft/2)
    result['evidence'].update({
        'tip_localization': POLICY['version'], 'original_tip_px': tip.tolist(),
        'original_arrow_length_px': length, 'arrow_length_px': length*(1-fraction),
        'tip_shift_px': float((tip-refined)@unit), 'taper_endpoint_width_px': shaft,
        'tip_refinement_mode': mode,
        'confidence_inherited_without_recalibration': True,
    })
    result['evidence']['arrow_length_px'] = float(np.linalg.norm(refined-base_center))
    return result, {**decision, 'status': 'refined', 'reason': 'measured_shaft_taper_intersection' if mode == 'width_only_ablation' else 'unique_raster_supported_transverse_junction',
                    'refined_tip_px': refined.tolist()}


def refine_report(report, foreground, text_polygons=(), *, mode='verified_junction'):
    require(report['processed_size_px'] == [foreground.shape[1], foreground.shape[0]], 'Changed arrow raster size')
    result = deepcopy(report)
    candidates, decisions = [], []
    for original in report['candidates']:
        candidate, decision = refine_candidate(original, foreground, text_polygons, mode=mode)
        decisions.append(decision)
        if candidate is not None:
            candidates.append(candidate)
    result['candidates'] = candidates
    result['tip_refinement'] = {'policy': POLICY, 'decisions': decisions,
                                'new_candidates': 0, 'production_admitted': False}
    # Historical preview/report hashes belong to the original file, never this new observation.
    result.pop('artifact_sha256', None)
    result.pop('artifacts', None)
    result.pop('preview_scale_xy', None)
    result.pop('timings', None)
    return result
