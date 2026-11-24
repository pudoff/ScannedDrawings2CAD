"""Bounded screened weak-curvature discovery, retaining prior source ownership."""
from unittest.mock import patch

from . import cad_object_assembly_v7 as previous
from .cad_primitive_fit_v7 import Fitter


def assemble(*args, **kwargs):
    with patch.object(previous, 'Fitter', Fitter):
        return previous.assemble(*args, **kwargs)
