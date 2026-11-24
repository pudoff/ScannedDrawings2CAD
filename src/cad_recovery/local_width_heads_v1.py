"""Bounded head-first proposals measured in the untouched raster's local frame."""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from .arrow_path_recovery_v1 import native_head_check
from .native_head_bounds_v1 import safe_validator
from .family_catalog_v2 import require

POLICY = {'version': 'local-width-heads/1.0', 'opening_radii_px': [1, 2, 3, 4, 5, 7],
    'maximum_pixels': 25000000, 'maximum_morphology_pixels': 40000000,
    'maximum_components_per_scale': 256, 'maximum_proposals': 512,
    'maximum_profile_samples': 8000000, 'maximum_component_extent_px': 192,
    'minimum_head_length_px': 5, 'maximum_head_length_px': 192,
    'head_aspect_range': [1.2, 8.], 'maximum_side_ratio': 1.5,
    'profile_step_px': .5, 'minimum_shaft_length_px': 6., 'minimum_base_to_shaft_ratio': 2.5,
    'maximum_taper_relative_error': .30, 'maximum_tip_shift_fraction': .20,
    'minimum_shaft_continuity': .95, 'maximum_candidates': 512,
    'confidence_calibrated': False, 'manual_references_used': False,
    'LINE_fitting_required': False, 'DT_head_proposals_used': False,
    'source_erasure_allowed': False, 'production_admitted': False, 'physical_geometry_approved': False}


def triangle(contour):
    _, raw = cv2.minEnclosingTriangle(contour)
    if raw is None or not np.isfinite(raw).all():
        return None
    p = raw.reshape(3, 2).astype(float)
    edges = [np.linalg.norm(p[(i+1)%3]-p[(i+2)%3]) for i in range(3)]
    apex = int(np.argmin(edges))
    p = p[[apex, (apex+1)%3, (apex+2)%3]]
    base = p[1:].mean(0); length = float(np.linalg.norm(p[0]-base)); width = float(np.linalg.norm(p[1]-p[2]))
    sides = np.linalg.norm(p[1:]-p[0], axis=1)
    if (not 5 <= length <= 192 or width <= 0 or sides.min() <= 0
        or not 1.2 <= length/width <= 8 or sides.max()/sides.min() > 1.5):
        return None
    return p


def profiles(mask, polygon):
    """Subpixel stations; widths of the ink run nearest the measured local axis."""
    tip, base = polygon[0], polygon[1:].mean(0)
    length = float(np.linalg.norm(tip-base)); unit = (tip-base)/length
    normal = np.array([-unit[1], unit[0]])
    width = float(np.linalg.norm(polygon[1]-polygon[2]))
    along = np.arange(-min(40., max(14., length)), length+1., .5)
    across = np.arange(-min(96., width+3), min(96., width+3)+.25, .5)
    points = base+along[:, None, None]*unit+across[None, :, None]*normal
    pixel = np.rint(points).astype(int)
    inside = (pixel >= 0).all(2) & (pixel[:, :, 0] < mask.shape[1]) & (pixel[:, :, 1] < mask.shape[0])
    # Clipped samples are unavailable evidence, never white paper.
    values = np.zeros(inside.shape, bool)
    values[inside] = mask[pixel[:, :, 1][inside], pixel[:, :, 0][inside]] > 0
    widths, centers = np.zeros(len(along)), np.full(len(along), np.inf)
    for i, row in enumerate(values):
        changes = np.diff(np.r_[False, row, False].astype(int))
        runs = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
        if not runs:
            continue
        a, b = min(runs, key=lambda r: abs((across[r[0]]+across[r[1]-1])/2))
        center = float((across[a]+across[b-1])/2)
        if abs(center) > max(1.5, width*.12) or not inside[i, a:b].all():
            continue
        widths[i], centers[i] = across[b-1]-across[a]+.5, center
    return along, widths, centers, int(values.size)


def validate(polygon, mask, text_polygons, budget):
    length = float(np.linalg.norm(polygon[0]-polygon[1:].mean(0)))
    base_width = float(np.linalg.norm(polygon[1]-polygon[2]))
    # Bound allocation before building a profile.
    estimated = math.ceil((min(40., max(14., length))+length+1)/.5)*math.ceil((2*min(96., base_width+3)+1)/.5)
    if budget['samples']+estimated > POLICY['maximum_profile_samples']:
        return None, 'profile_sample_budget'
    along, widths, centers, cost = profiles(mask, polygon)
    budget['samples'] += cost
    narrow = (along < -1) & (widths > 0) & (widths <= base_width/2.5)
    # Require a connected shaft starting at the base, rather than a distant parallel line.
    runs = np.diff(np.r_[False, narrow, False].astype(int))
    eligible = []
    for a, b in zip(np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)):
        local_width = float(np.median(widths[a:b]))
        if along[b-1] < -max(3., 2*local_width):
            continue
        if along[b-1]-along[a] < max(6., 2*local_width):
            continue
        bridge = (along >= along[b-1]) & (along <= 0)
        if np.mean(widths[bridge] > 0) < .95:
            continue
        eligible.append((a, b, local_width))
    if not eligible:
        return None, 'no_connected_local_shaft'
    a, b, local_width = max(eligible, key=lambda r: r[1]-r[0])
    chosen = (along >= .15*length) & (along <= .75*length)
    if chosen.sum() < 4 or np.mean(widths[chosen] > 0) < .95:
        return None, 'head_profile_missing_ink'
    slope, intercept = np.polyfit(along[chosen], widths[chosen], 1)
    expected = slope*along[chosen]+intercept
    error = float(np.median(np.abs(widths[chosen]-expected)/np.maximum(local_width, expected)))
    if slope >= 0 or error > .30 or intercept/max(local_width, .5) < 2.5:
        return None, 'no_locally_normalized_triangle_taper'
    root = float(-intercept/slope)
    if not .8*length <= root <= 1.2*length:
        return None, 'extrapolated_tip_outside_local_taper'
    tip, base = polygon[0], polygon[1:].mean(0)
    unit = (tip-base)/length
    normal = np.array([-unit[1], unit[0]])
    # Correct only the observed taper apex; never move it to a supplied target or LINE.
    p = np.array([base+root*unit, base+normal*intercept/2, base-normal*intercept/2])
    c = {'tip_px': p[0].tolist(), 'direction': unit.tolist(), 'polygon_px': p.tolist(),
        'classification': 'filled_arrow_candidate', 'uncertainty_px': max(.75, local_width/2),
        'confidence': .55, 'evidence': {'method': POLICY['version'], 'path_kind': 'local_source_profile',
            'local_shaft_width_px': local_width, 'local_normalization_scale': 3/local_width,
            'shaft_observed_length_px': float(along[b-1]-along[a]),
            'base_to_shaft_ratio': float(intercept/local_width), 'taper_relative_error': error,
            'head_length_px': root, 'base_width_px': float(intercept),
            'tip_shift_px': float(abs(root-length)), 'confidence_calibrated': False}}
    events = []
    valid, check = safe_validator(native_head_check, events)(c, mask, local_width, text_polygons)
    if not valid:
        return None, check['reason']
    # Native apex must have ink within one observed shaft width, at least one pixel.
    x, y = np.rint(p[0]).astype(int); radius = max(1, math.ceil(local_width))
    lo = np.maximum([0, 0], [x-radius, y-radius]); hi = np.minimum(mask.shape[::-1], [x+radius+1, y+radius+1])
    if (hi <= lo).any() or not mask[lo[1]:hi[1], lo[0]:hi[0]].any():
        return None, 'apex_without_native_ink'
    c['evidence']['native_head_check'] = check
    c['evidence']['source_checked'] = True
    return c, None


def append_preserving(baseline, proposals):
    """Preserve original observations and IDs byte-for-byte; append novel evidence."""
    output = deepcopy(list(baseline)); added = []
    for candidate in proposals:
        duplicate = False
        for old in output:
            if np.dot(candidate['direction'], old['direction']) < math.cos(math.radians(20)):
                continue
            p, q = np.asarray(candidate['polygon_px'], np.float32), np.asarray(old['polygon_px'], np.float32)
            intersection, _ = cv2.intersectConvexConvex(p, q)
            union = abs(cv2.contourArea(p))+abs(cv2.contourArea(q))-intersection
            radius = max(2., candidate['uncertainty_px']+old.get('uncertainty_px', 1.))
            if np.linalg.norm(np.asarray(candidate['tip_px'])-old['tip_px']) <= radius or union > 0 and intersection/union >= .25:
                duplicate = True; break
        if duplicate:
            continue
        c = deepcopy(candidate); identity = 'local-width-head-'+str(len(added)+1).zfill(5)
        while identity in {h['id'] for h in output}:
            identity += '-new'
        c['id'] = identity; added.append(c); output.append(c)
    return output, added


def search(gray, *, baseline=(), text_polygons=()):
    require(isinstance(gray, np.ndarray) and gray.ndim == 2 and gray.dtype == np.uint8
            and 0 < gray.size <= POLICY['maximum_pixels'], 'Expected bounded uint8 grayscale')
    require(len(baseline) <= POLICY['maximum_candidates'], 'Too many baseline heads')
    require(len(text_polygons) <= 256, 'Too many text polygons')
    for p in text_polygons:
        q = np.asarray(p, float)
        require(q.ndim == 2 and q.shape[1] == 2 and 3 <= len(q) <= 4096
                and np.isfinite(q).all() and np.abs(q).max() <= 1e7, 'Invalid text polygon')
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    candidates, reasons, limits = [], Counter(), set()
    pixels = examined = 0; budget = {'samples': 0}
    for radius in POLICY['opening_radii_px']:
        if pixels+gray.size > POLICY['maximum_morphology_pixels']:
            limits.add('morphology_pixel_budget'); break
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel); pixels += gray.size
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        eligible = [c for c in sorted(contours, key=lambda c: -cv2.contourArea(c))
                    if cv2.contourArea(c) >= 4 and max(cv2.boundingRect(c)[2:]) <= 192]
        if len(eligible) > POLICY['maximum_components_per_scale']:
            limits.add('component_budget')
        for co in eligible[:POLICY['maximum_components_per_scale']]:
            if examined >= POLICY['maximum_proposals']:
                limits.add('proposal_budget'); break
            examined += 1; p = triangle(co)
            if p is None:
                reasons['not_local_triangle'] += 1; continue
            c, reason = validate(p, mask, text_polygons, budget)
            if reason:
                reasons[reason] += 1
                if reason == 'profile_sample_budget': limits.add(reason); break
            else:
                c['evidence']['opening_radius_px'] = radius; candidates.append(c)
        if 'proposal_budget' in limits or 'profile_sample_budget' in limits:
            break
    combined, added = append_preserving(baseline, candidates)
    if len(combined) > POLICY['maximum_candidates']:
        limits.add('candidate_budget')
        combined = combined[:POLICY['maximum_candidates']]
        added = combined[len(baseline):]
    return {'schema': POLICY['version'], 'policy': deepcopy(POLICY), 'candidates': combined, 'added': added,
        'analysis': {'morphology_pixels': pixels, 'components_examined': examined, 'profile_samples': budget['samples'],
            'raw_accepted': len(candidates), 'added': len(added), 'rejections': dict(reasons), 'limits': sorted(limits)},
        'search_incomplete': bool(limits), 'source_modified': False, 'baseline_modified': False,
        'production_admitted': False, 'physical_geometry_approved': False, 'CAD_promotions': 0}
