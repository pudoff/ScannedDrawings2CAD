"""Compose observed open stroke evidence with the stage 1.2 contour baseline."""
import numpy as np

from .contour_contact_recovery_v2 import analyze as contact_baseline
from .grid_role_revision_v1 import sha
from .material_contours_v1 import guarded
from .open_contour_joint_v5 import metadata
from .open_stroke_core_v1 import analyze as core_witness
from .part_admission_v1 import object_sha


def extend(packet, baseline, *, authority, source_pin, page_context, protected):
    arrays = tuple(packet[k] for k in ('gray', 'roles', 'source_owner', 'ambiguous_owner_mask')) + (protected,)
    observations = [packet[k] for k in ('graph', 'geometry', 'heads')]
    before_arrays = [sha(array) for array in arrays]
    before_observations = object_sha(observations)
    proposed, audit = core_witness(packet, authority=authority, protected=protected)
    proposed = np.where(baseline['mask'] == 0, proposed, 0).astype(np.uint8)
    guard = None
    if baseline['complete'] and not audit['limits'] and proposed.any():
        guard = guarded(packet, proposed, authority=authority, source_pin=source_pin,
                        page_context=page_context, protected=protected)
    added = guard['mask'] if guard is not None else np.zeros_like(proposed)
    mask = baseline['mask'] | added
    if np.any((added > 0) & ((proposed == 0) | (packet['roles'] != 255) | protected)):
        raise ValueError('Open contour recovery added unbound pixels')
    if before_arrays != [sha(array) for array in arrays] or before_observations != object_sha(observations):
        raise ValueError('Open contour recovery changed source observations')
    roles = packet['roles'].copy()
    roles[mask > 0] = 1
    return {'mask': mask, 'raw_mask': baseline['raw_mask'] | proposed, 'full_roles': roles,
            'complete': bool(baseline['complete'] and not audit['limits'] and (guard is None or guard['complete'])),
            'baseline_mask': baseline['mask'], 'added_mask': added, 'material_raw_mask': proposed,
            'guarded_positive_pixels': int(np.count_nonzero(mask)), 'new_material_pixels': int(np.count_nonzero(added)),
            'source_immutable': True, 'core_audit': audit, 'guard': metadata(guard) if guard else None,
            'policy': {'version': 'open-contour-recovery/1.0', 'source_geometry_modified': False,
                       'interpolated_gap_edges': 0, 'negative_guards_relaxed': False}}


def analyze(packet, *, authority, source_pin, page_context, protected=None):
    if protected is None:
        protected = np.zeros_like(packet['gray'], dtype=bool)
    options = dict(authority=authority, source_pin=source_pin, page_context=page_context, protected=protected)
    baseline = contact_baseline(packet, **options)
    return extend(packet, baseline, **options)
