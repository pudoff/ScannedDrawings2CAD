"""Collective ARC discovery, source-connected junctions and bounded corners."""
from unittest.mock import patch
import numpy as np
from . import cad_object_assembly_v5 as stage
from .cad_object_assembly_v10 import assemble as previous
from .arc_junction_merge_v1 import merge


def assemble(gray, geometry, *, roles=None, protected=None):
    roles = np.full_like(gray, 255) if roles is None else roles
    protected = np.zeros_like(gray, bool) if protected is None else protected
    audit = dict(version='source-connected-ARC-junction/1.0', routes_checked=0,
        foreground_routes_found=0, search_complete=True, white_gap_bridging=False)
    with patch.object(stage, '_merge_curves', lambda items, fitter, ink: merge(items, fitter, ink, protected | (roles == 6), audit)):
        result = previous(gray, geometry, roles=roles, protected=protected)
    result['source_connected_arc_junctions'] = audit
    result['search_complete'] &= audit['search_complete']
    return result
