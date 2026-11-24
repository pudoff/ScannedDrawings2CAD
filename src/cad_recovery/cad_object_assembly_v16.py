"""Source-certified LINE crossing continuation, preserving V15 native ARC evidence."""
from unittest.mock import patch
from . import cad_object_assembly_v13 as stage
from .cad_object_assembly_v15 import assemble as previous
from .arc_junction_merge_v4 import merge


def assemble(*args,**kwargs):
    with patch.object(stage,'merge',merge):
        result=previous(*args,**kwargs)
    result['source_connected_arc_junctions']['version']='source-connected-LINE-ARC-junction/4.0'
    return result
