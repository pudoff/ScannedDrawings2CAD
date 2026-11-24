"""Exact indexed equivalent of V12 for an isolated experimental worker."""
from unittest.mock import patch
from . import cad_object_assembly_v12 as previous
from .arc_junction_merge_v3 import merge


def assemble(*args, **kwargs):
    with patch.object(previous, 'merge', merge):
        return previous.assemble(*args, **kwargs)
