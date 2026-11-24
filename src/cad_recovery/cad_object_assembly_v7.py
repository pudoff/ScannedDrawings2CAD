"""Retain V6 closed cycles and add independently corroborated weak ARC sectors."""
from unittest.mock import patch
import numpy as np

from . import cad_object_assembly_v5 as stage
from .cad_primitive_fit_v6 import Fitter
from .closed_primitive_cycle_v1 import apply


def assemble(gray, geometry, *, roles=None, protected=None):
    roles = np.full_like(gray, 255) if roles is None else roles
    protected = np.zeros_like(gray, bool) if protected is None else protected
    # This version is an isolated experimental worker, never a concurrent Web
    # handler. Reuse the pinned assembler without copying its ownership logic.
    with patch.object(stage, 'Fitter', Fitter):
        base = stage.assemble(gray, geometry, roles=roles, protected=protected)
    return apply(base, gray, roles=roles, protected=protected)
