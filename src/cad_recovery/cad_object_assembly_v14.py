"""Native evidence at short LINE/ARC boundaries, atop the indexed assembler."""
from pathlib import Path
from unittest.mock import patch
import numpy as np

from . import cad_object_assembly_v5 as stage
from .cad_object_assembly_v13 import assemble as previous
from .native_corner_input_v1 import NativeInput
from .short_arc_boundary_v1 import refine


def assemble(gray,geometry,*,roles=None,protected=None,native_head_context=None,native_corner_context=None):
    roles=np.full_like(gray,255) if roles is None else roles
    protected=np.zeros_like(gray,bool) if protected is None else protected
    source=NativeInput(Path(__file__).resolve().parents[2],native_corner_context,gray) if native_corner_context else None
    audit=dict(version='native-short-arc-boundary/1.0',candidates=0,accepted=0,decisions=[],search_complete=True)
    old=stage.refine_corners
    def boundaries(items,fitter):
        return refine(old(items,fitter),fitter,gray,roles,protected,source,audit)
    with patch.object(stage,'refine_corners',boundaries):
        result=previous(gray,geometry,roles=roles,protected=protected,native_head_context=native_head_context)
    for decision in audit['decisions']:
        if not decision['native']['accepted']: continue
        edges={tuple(sorted(map(tuple,pair))) for pair in zip(decision['source_points'],decision['source_points'][1:])}
        for obj in result['objects']:
            if obj['kind']!='arc': continue
            own={tuple(sorted(map(tuple,pair))) for pair in zip(obj['support_points'],obj['support_points'][1:])}
            if edges <= own:
                obj.setdefault('native_short_arc_evidence',[]).append(decision)
    result['short_arc_boundaries']=audit; result['search_complete'] &= audit['search_complete']
    return result
