"""Collective ARC discovery with bounded common-endpoint refinement."""
from unittest.mock import patch

from . import cad_object_assembly_v5 as stage
from .cad_object_assembly_v9 import assemble as previous
from .arc_endpoint_alignment_v1 import refine, POLICY


def assemble(*args, **kwargs):
    audit = dict(policy=POLICY, tests=0, fallbacks_accepted=0, search_complete=True)
    with patch.object(stage, '_align_endpoints', lambda items, support: refine(items, support, audit)):
        result = previous(*args, **kwargs)
    result['bounded_arc_endpoint_search'] = audit
    result['search_complete'] &= audit['search_complete']
    return result
