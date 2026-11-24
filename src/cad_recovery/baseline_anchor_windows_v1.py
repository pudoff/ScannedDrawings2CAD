"""Windowed support geometry for head-comparison baseline, never CAD reconstruction."""
from copy import deepcopy
import numpy as np

from .anchored_curve_fit_v1 import extract as original_extract

POLICY = {'version': 'baseline-anchor-windows/1.0', 'single_raster_limit': 4194304,
    'core_side': 1024, 'halo_px': 192, 'maximum_total_window_pixels': 40000000,
    'maximum_total_fit_calls': 8192, 'maximum_candidates': 2500,
    'scope': 'baseline LINE/ARC support only, no stitched CAD topology certificate'}


def extract(ink, skeleton, *, conditioned, input_to_native):
    if ink.size <= POLICY['single_raster_limit']:
        return original_extract(ink, skeleton, conditioned=conditioned, input_to_native=input_to_native)
    if ink.dtype != np.uint8 or skeleton.dtype != np.uint8 or ink.shape != skeleton.shape or ink.ndim != 2:
        raise ValueError('Aligned uint8 ink/skeleton required')
    height, width = ink.shape; side, halo = POLICY['core_side'], POLICY['halo_px']
    candidates, windows, pixels, fit_calls = [], [], 0, 0
    for y in range(0, height, side):
        for x in range(0, width, side):
            left, top, right, bottom = max(0, x-halo), max(0, y-halo), min(width, x+side+halo), min(height, y+side+halo)
            pixels += (right-left)*(bottom-top)
            if pixels > POLICY['maximum_total_window_pixels']: raise ValueError('Baseline window aggregate pixel budget')
            packet = original_extract(ink[top:bottom, left:right], skeleton[top:bottom, left:right],
                conditioned=conditioned, input_to_native=input_to_native)
            fit_calls += packet['analysis']['fit_calls']
            if packet['analysis']['search_incomplete'] or fit_calls > POLICY['maximum_total_fit_calls']:
                raise ValueError('Baseline window aggregate fitting budget')
            kept = 0
            for original in packet['candidates']:
                c = deepcopy(original)
                if 'points_px' in c: anchor = np.asarray(c['points_px'], float).mean(0)+[left, top]
                elif 'center_px' in c: anchor = np.asarray(c['center_px'], float)+[left, top]
                else: continue
                if not (x <= anchor[0] < min(width, x+side) and y <= anchor[1] < min(height, y+side)): continue
                for key in ('points_px', 'support_points'):
                    if key in c: c[key] = (np.asarray(c[key])+[left, top]).tolist()
                if 'center_px' in c: c['center_px'] = (np.asarray(c['center_px'])+[left, top]).tolist()
                c['id'] = 'window-support-'+str(len(candidates)+1).zfill(5)
                c['baseline_support_window'] = [left, top, right, bottom]
                c['baseline_owner_core'] = [x, y, min(width, x+side), min(height, y+side)]
                c['width_evidence_frame'] = 'window_local_measurements; scalar widths unchanged'
                candidates.append(c); kept += 1
                if len(candidates) > POLICY['maximum_candidates']: raise ValueError('Baseline window candidate budget')
            windows.append({'source_box': [left, top, right, bottom], 'fit_calls': packet['analysis']['fit_calls'], 'kept': kept})
    return {'candidates': candidates, 'source': [], 'analysis': {'fit_calls': fit_calls, 'search_incomplete': False,
        'window_policy': POLICY, 'windows': windows, 'total_window_pixels': pixels, 'source_geometry_is_not_stitched_CAD': True}}
