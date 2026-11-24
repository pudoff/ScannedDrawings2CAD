"""Version 2 material witness, with the frozen negative guards and baseline."""
from copy import deepcopy
import numpy as np
from .material_contour_witness_v2 import analyze as witness
from .material_contours_v1 import guarded
from .local_shared_contacts_v2 import analyze as baseline
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .open_contour_joint_v5 import metadata

POLICY = {'version': 'material-contours/2.0', 'old_branch_preserved': True,
          'negative_guards_relaxed': False, 'geometry_modified': False,
          'source_owner_remapped': False, 'Web_changed': False,
          'production_admitted': False, 'physical_geometry_approved': False}


def analyze(packet, *, authority, source_pin, page_context, protected=None):
    gray = packet['gray']
    protection = np.zeros(gray.shape, bool) if protected is None else protected
    arrays = (gray, packet['roles'], packet['source_owner'], packet['ambiguous_owner_mask'], protection)
    snapshots = [sha(a) for a in arrays]
    before = object_sha([packet['graph'], packet['geometry'], packet['heads']])
    kw = dict(authority=authority, source_pin=source_pin, page_context=page_context, protected=protection)
    old = baseline(packet, **kw)
    positive = witness(packet, authority=authority, protected=protection)
    guard = None; added = np.zeros_like(gray)
    if positive['complete'] and positive['mask'].any():
        guard = guarded(packet, positive['mask'], **kw)
        if guard['complete']:
            added = guard['mask']
    complete = bool(old['complete'] and positive['complete'] and (guard is None or guard['complete']))
    mask = old['mask'].copy()
    if complete:
        mask |= added
    if np.any((mask > 0) & ((packet['roles'] != 255) | protection)):
        raise ValueError('Material composition changed baseline or protected pixels')
    if np.any((old['mask'] > 0) & (mask == 0)):
        raise ValueError('Material composition lost the frozen branch')
    if snapshots != [sha(a) for a in arrays] or before != object_sha([packet['graph'], packet['geometry'], packet['heads']]):
        raise ValueError('Material composition changed source observations')
    full = packet['roles'].copy(); full[mask > 0] = 1
    return {'mask': mask, 'raw_mask': old['raw_mask'] | positive['mask'], 'full_roles': full,
            'added_mask': np.where((mask > 0) & (old['mask'] == 0), 255, 0).astype(np.uint8),
            'material_raw_mask': positive['mask'], 'baseline_mask': old['mask'],
            'complete': complete, 'baseline': metadata(old), 'positive': metadata(positive),
            'guard': metadata(guard) if guard else None, 'source_immutable': True,
            'raw_positive_pixels': int(np.count_nonzero(old['raw_mask'] | positive['mask'])),
            'guarded_positive_pixels': int(np.count_nonzero(mask)),
            'new_material_pixels': int(np.count_nonzero((mask > 0) & (old['mask'] == 0))),
            'legacy_packet_complete': packet.get('complete'),
            'legacy_graph_search_incomplete': packet['graph'].get('search_incomplete'),
            'policy': deepcopy(POLICY)}
