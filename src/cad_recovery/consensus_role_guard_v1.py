"""Require source witnesses before promoting consensus PART proposals."""

import numpy as np

from .full_role_recovery_v1 import observed_part_core, reconcile


def reconcile_observed(gray, baseline, protected, *, part, stamp, independent=(),
                       graph, source_grid, source_frame):
    if any(mask.shape != gray.shape or mask.dtype != bool for mask in (source_grid, source_frame)):
        raise ValueError('Aligned original-role evidence required')
    competitors = source_grid | source_frame
    measured_part = part & ~competitors
    checked = []
    if independent:
        agreement = np.logical_and.reduce([row == independent[0] for row in independent])
        proposal = agreement & (independent[0] == 1) & ~competitors
        anchored = observed_part_core(gray, graph, proposal, accepted=(baseline == 1) | measured_part)
        allowed = measured_part | anchored
        for row in independent:
            value = row.copy()
            value[(value == 1) & ~allowed] = 255
            checked.append(value)
    return reconcile(gray, baseline, protected, part=measured_part,
                     stamp=stamp | source_grid, independent=tuple(checked))


def exclude_new_competitors(roles, baseline, source_grid, source_frame):
    """Keep prior roles and withhold newly inferred PART on measured other ink."""
    output = roles.copy()
    conflict = ((baseline == 255) & (roles == 1) & (source_grid | source_frame))
    output[conflict] = 255
    return output, int(conflict.sum())
