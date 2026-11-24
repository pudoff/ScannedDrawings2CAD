"""Collective ARC discovery without requiring a valid isolated short-sector fit."""
from unittest.mock import patch

from . import cad_object_assembly_v7 as previous
from .cad_primitive_fit_v8 import Fitter


def assemble(*args, **kwargs):
    with patch.object(previous, 'Fitter', Fitter):
        return previous.assemble(*args, **kwargs)
