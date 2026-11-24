"""Withdraw newly assigned roles on unobserved native paper, preserving baseline."""
import numpy as np
from .stroke_roles_v1 import checked_gray, foreground
from .grid_role_revision_v1 import sha


def refine(native_gray, baseline, candidate):
    checked_gray(native_gray)
    for a in (baseline, candidate):
        if a.shape != native_gray.shape or a.dtype != np.uint8 or not np.isin(a, list(range(10))+[255]).all():
            raise ValueError('Native aligned uint8 roles required')
    supported = foreground(native_gray) > 0
    change = candidate != baseline
    withdraw = change & (candidate != 255) & ~supported
    result = candidate.copy(); result[withdraw] = baseline[withdraw]
    return {'mapping': result, 'withdrawn_pixels': int(withdraw.sum()),
            'native_source_sha256': sha(native_gray), 'support_sha256': sha(supported.astype(np.uint8)),
            'existing_baseline_assignments_preserved': True, 'native_predictions_transferred': False,
            'source_pixels_erased': 0, 'physical_geometry_approved': False}
