"""Source checks shared by new arc searches; historical producers stay frozen."""
import math
import numpy as np

from . import open_arc_search_v3 as previous


def validate(mask, curve, stroke, policy, budget, radial):
    geometry = previous.final_geometry(curve)
    if not geometry['accepted']:
        return False, geometry['reason'], {'final_geometry': geometry}
    band = min(policy['maximum_radial_band_px'], max(8, math.ceil(stroke*2), math.ceil(curve['radius_px']*.03)))
    cost = policy['angular_samples']*(2*band+1)
    if budget['used']+cost > policy['maximum_final_centerline_samples']:
        budget['limited'] = True
        return False, 'final_centerline_budget', {'final_geometry': geometry}
    budget['used'] += cost
    profile = radial(mask, np.asarray(curve['center_px']), curve['radius_px'], stroke)
    relative = (np.degrees(profile['angles'])-curve['start_angle_deg']) % 360
    selected = (relative <= curve['sweep_angle_deg']) & profile['supported']
    residual = np.linalg.norm(profile['observed'][selected]-curve['center_px'], axis=1)-curve['radius_px']
    rmse = float(np.sqrt(np.mean(residual**2))) if len(residual) else None
    accepted = len(residual) >= 12 and rmse <= policy['maximum_seed_rmse_px']
    check = {'sample_count': int(selected.sum()), 'rmse_px': rmse,
        'maximum_rmse_px': policy['maximum_seed_rmse_px'], 'accepted': bool(accepted),
        'reason': 'observed_final_centerline' if accepted else 'final_centerline_residual'}
    return bool(accepted), check['reason'], {'final_geometry': geometry, 'native_centerline': check}
