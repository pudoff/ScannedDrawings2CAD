"""V5 homogeneous primitives plus exact observed closed-cycle recognition."""
import numpy as np

from .cad_object_assembly_v5 import assemble as previous
from .closed_primitive_cycle_v1 import apply


def assemble(gray, geometry, *, roles=None, protected=None):
    roles = np.full_like(gray, 255) if roles is None else roles
    protected = np.zeros_like(gray, bool) if protected is None else protected
    base = previous(gray, geometry, roles=roles, protected=protected)
    return apply(base, gray, roles=roles, protected=protected)
