"""Reuse frozen pixel-only producers; target masks never enter this module."""
from pathlib import Path
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked
from cad_recovery.boundary_role_overlay_v2 import analyze as previous_boundary
from boundary_role_inputs_v1 import prepare as previous_prepare


def prepare(root,view):
    gray,source,geometry,contact,previous,local,heads,extra,calls=previous_prepare(root,view)
    if view.get('prior_chain_profile'):
        inherited=np.array(Image.open(checked(Path(root),view['prior_chain_profile']['roles'])))
        calls={**calls,'previous_boundary_calls':0}
    else:
        inherited=previous_boundary(gray,previous,contact,local,heads=heads,diagnostic_heads=extra)['pixel_roles']
        calls={**calls,'previous_boundary_calls':1}
    return gray,source,geometry,inherited,previous,local,heads,extra,calls
