"""Recover a shaft-axis proposal before requiring its full observed continuity."""
import math

import numpy as np

from .economical_head_profiles_v1 import Profiles as ProgressiveProfiles
from .economical_head_profiles_v1 import prepare as prepare_original, finish as finish_original
from .sparse_local_heads_v1 import runs


class Profiles(ProgressiveProfiles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovered_axes = {}
        self.axis_proposals = 0


def signature(prepared):
    base, unit, length, width, _ = prepared
    return tuple(np.r_[base, unit, length, width])


def prepare(base, unit, length, width, profiles):
    prepared, reason = prepare_original(base, unit, length, width, profiles)
    if reason != 'no_connected_source_shaft': return prepared, reason
    entry = profiles.cache.get(tuple(np.r_[base, unit, length, width]))
    if entry is None: return None, reason
    along, widths, centers = entry['along'], entry['widths'], entry['centers']
    # A short run proposes an axis only. It never qualifies the head or reduces
    # the final >=max(6 px, 2*observed_width), 95% shaft continuity requirement.
    options = []
    for a, b in runs((along < 0) & entry['observed'] & np.isfinite(centers)
                     & (widths > 0) & (widths <= width*.8)):
        local = float(np.median(widths[a:b])); extent = float(along[b-1]-along[a]+.5)
        if extent < 3 or along[b-1] < -max(3., 2*local): continue
        options.append((extent, a, b, local))
    if not options: return None, reason
    extent, a, b, local = max(options)
    slope, intercept = np.polyfit(along[a:b], centers[a:b], 1)
    if abs(slope) > math.tan(math.radians(20)) or abs(intercept) > max(2., local):
        return None, 'short_source_run_axis_unstable'
    normal = np.array([-unit[1], unit[0]])
    adjusted = unit+slope*normal; adjusted /= np.linalg.norm(adjusted)
    shifted = base+intercept*normal
    profiles.axis_proposals += 1
    prepared, reason = prepare_original(shifted, adjusted, length, width, profiles)
    if prepared is not None:
        profiles.recovered_axes[signature(prepared)] = {
            'method': 'observed_short_run_axis_proposal_then_full_shaft_validation',
            'preliminary_run_extent_px': extent, 'seed_direction': unit.tolist(),
            'proposed_direction': adjusted.tolist(), 'final_shaft_extent_px': float(prepared[4][3]),
            'final_shaft_width_px': float(prepared[4][2]), 'full_shaft_checks_unchanged': True}
    return prepared, reason


def finish(prepared, profiles, text_polygons):
    head, reason = finish_original(prepared, profiles, text_polygons)
    if head is not None:
        head['evidence']['method'] = 'local-head-context/1.0'
        if signature(prepared) in profiles.recovered_axes:
            head['evidence']['axis_recovery'] = profiles.recovered_axes[signature(prepared)]
    return head, reason
