"""Complete uncertainty-neighbour ARC search with optional native head context."""
from unittest.mock import patch
import numpy as np
from . import cad_object_assembly_v5 as stage
from .cad_object_assembly_v10 import assemble as previous
from .arc_junction_merge_v2 import merge
from .native_head_arc_context_v1 import load_context


def assemble(gray, geometry, *, roles=None, protected=None, native_head_context=None):
    roles = np.full_like(gray, 255) if roles is None else roles
    protected = np.zeros_like(gray, bool) if protected is None else protected
    context = load_context(native_head_context, gray)
    audit = dict(version='source-connected-ARC-junction/2.0', routes_checked=0,
        foreground_routes_found=0, search_complete=True, white_gap_bridging=False,
        native_head_context_supplied=context is not None)
    with patch.object(stage, '_merge_curves', lambda items, fitter, ink: merge(items, fitter, ink, protected | (roles == 6), audit, context)):
        result = previous(gray, geometry, roles=roles, protected=protected)
    result['source_connected_arc_junctions'] = audit
    result['search_complete'] &= audit['search_complete']
    return result
