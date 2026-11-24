"""Pixel-only inference inputs; target masks are deliberately not read here."""
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import checked
from run_topology_robust_v1 import read


def prepare(root, view):
    from cad_recovery.interval_roles_v1 import analyze as intervals
    from cad_recovery.local_role_guard_v1 import analyze as local_roles
    root = Path(root)
    gray = np.array(Image.open(checked(root, view['gray'])).convert('L'))
    if view.get('cached_contact'):
        cache = view['cached_contact']; row = read(checked(root, cache['row']))
        source, geometry = read(checked(root, cache['source'])), read(checked(root, cache['geometry']))
        old_baseline = np.array(Image.open(checked(root, cache['baseline'])))
        inherited = np.array(Image.open(checked(root, cache['contact'])))
        heads, extra = row['head_candidates']['baseline'], row['head_candidates']['DT_novel']
        raw = intervals(gray, geometry, source, heads, input_to_native=view['input_to_native'], diagnostic_heads=extra)
        previous = {k: v for k, v in raw.items() if k not in ('pixel_roles', 'text_mask')}
        local = local_roles(gray, previous, old_baseline, raw['text_mask'], heads, diagnostic_heads=extra)
        if local['hybrid']['dimension_chains'] != row['details']['dimension_chains']:
            raise ValueError('Reconstructed local evidence differs from sealed comparison')
        prep = {'geometry_extractions': 0, 'head_DT_skeleton_calls': 0, 'legacy_role_calls': 0,
                'interval_calls': 1, 'local_guard_calls': 1, 'contact_overlay_calls': 0}
    else:
        from contact_role_inputs_v1 import prepare as original_prepare
        from cad_recovery.contact_role_overlay_v2 import analyze as contact
        gray, source, geometry, old_baseline, text, previous, local, heads, extra, prep = original_prepare(root, view)
        inherited = contact(gray, previous, old_baseline, local, heads=heads, diagnostic_heads=extra)['pixel_roles']
        prep = {**prep, 'contact_overlay_calls': 1}
    return gray, source, geometry, inherited, previous, local, heads, extra, prep
