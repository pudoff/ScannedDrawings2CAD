"""Validate the final, tangent-trimmed arc against the native resolution.

The v2 finite discovery queue is unchanged. Final acceptance cannot inherit
length measured on an earlier radial run. A bend below twice its existing
positional uncertainty remains unresolved rather than an asserted circular arc.
"""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from . import open_arc_search_v2 as previous
from .family_catalog_v2 import require

POLICY = deepcopy(previous.POLICY)
POLICY.update(version='resolved-final-open-arcs/3.0',
    minimum_final_sagitta_in_uncertainties=2., final_length_checked=True,
    maximum_final_centerline_samples=5000000)


def final_geometry(curve):
    radius = float(curve['radius_px']); angle = math.radians(curve['sweep_angle_deg'])
    # For major arcs, distance from the middle of the observed arc to its
    # endpoint chord is still r*(1-cos(sweep/2)), rather than the minor sagitta.
    length = radius*angle
    sagitta = radius*(1-math.cos(angle/2))
    uncertainty = float(curve['uncertainty_px'])
    require(all(math.isfinite(v) for v in (radius, angle, uncertainty)) and radius > 0 and
        0 < angle <= 2*math.pi and uncertainty > 0, 'Invalid final arc geometry')
    minimum_sagitta = POLICY['minimum_final_sagitta_in_uncertainties']*uncertainty
    reason = ('final_arc_too_short' if length < POLICY['minimum_arc_length_px'] else
              'final_bend_unresolved' if sagitta < minimum_sagitta else 'resolved_final_arc')
    return {'length_px': length, 'minimum_length_px': POLICY['minimum_arc_length_px'],
        'chord_px': 2*radius*math.sin(angle/2), 'sagitta_px': sagitta,
        'uncertainty_px': uncertainty, 'minimum_sagitta_px': minimum_sagitta,
        'accepted': reason == 'resolved_final_arc', 'reason': reason}


def search(gray, *, stroke_px, lines=(), text_polygons=()):
    result = previous.search(gray, stroke_px=stroke_px, lines=lines, text_polygons=text_polygons)
    stats = result['analysis']; queue = stats['queue']
    retained = []; rejected = Counter(); sampled = 0; limited = False
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    for curve in result['candidates']:
        old_id = curve['id']; witness = final_geometry(curve)
        row = next(r for r in queue if r.get('candidate_id') == old_id)
        row['pre_final_validation_id'] = old_id
        row['final_geometry'] = witness
        row['candidate_id'] = None
        if not witness['accepted']:
            row['refinement_reason'] = witness['reason']; rejected[witness['reason']] += 1
            continue
        band = min(previous.base.POLICY['maximum_radial_band_px'],
            max(8, math.ceil(stroke_px*2), math.ceil(curve['radius_px']*.03)))
        cost = previous.base.POLICY['angular_samples']*(2*band+1)
        if sampled+cost > POLICY['maximum_final_centerline_samples']:
            limited = True
            check = {'accepted': False, 'reason': 'final_centerline_budget'}
        else:
            sampled += cost
            radial = previous.base.radial_profile(mask, np.asarray(curve['center_px']), curve['radius_px'], stroke_px)
            relative = (np.degrees(radial['angles'])-curve['start_angle_deg']) % 360
            selected = (relative <= curve['sweep_angle_deg']) & radial['supported']
            residual = np.linalg.norm(radial['observed'][selected]-curve['center_px'], axis=1)-curve['radius_px']
            rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else None
            # Reapply the existing seed fit precision to the *final* native
            # centreline, excluding locally thick intersections. Merely lying
            # somewhere within a wide ink stroke does not verify its centre.
            accepted = len(residual) >= 12 and rmse <= POLICY['maximum_seed_rmse_px']
            check = {'sample_count': int(selected.sum()), 'rmse_px': rmse,
                'maximum_rmse_px': POLICY['maximum_seed_rmse_px'], 'accepted': bool(accepted),
                'reason': 'observed_final_centerline' if accepted else 'final_centerline_residual'}
        row['native_centerline'] = check
        if not check['accepted']:
            row['refinement_reason'] = check['reason']; rejected[check['reason']] += 1
            continue
        curve['id'] = f'open-arc-{len(retained)+1:05d}'
        row['candidate_id'] = curve['id']
        curve['evidence'].update(method=POLICY['version'], pre_final_validation_id=old_id,
            final_geometry=witness, native_centerline=check)
        retained.append(curve)
    # Recompute from actual refined rows; no rejected observation is silently
    # relabelled as an unexamined seed or a successful candidate.
    stats['refinement_rejections'] = dict(Counter(r['refinement_reason'] for r in queue
        if r['state'] == 'refined' and r['refinement_reason'] != 'accepted'))
    stats['final_validation'] = {'examined': len(result['candidates']), 'retained': len(retained),
        'rejections': dict(rejected), 'queue_order_changed': False,
        'duplicate_seeds_revisited': False, 'centerline_sample_pixels': sampled}
    if limited: stats['limits'].append('final_centerline_budget')
    result.update(policy=deepcopy(POLICY), candidates=retained, search_incomplete=bool(stats['limits']))
    return result
