"""Safe experimental entry point for one isolated CV child; proposals need review."""
from . import stroke_roles_v1 as core
from .arrow_head_completion_v1 import search
from .native_head_bounds_v1 import guarded_native_heads


def analyze(gray,geometry,widths,*,stroke_px):
    """Use only in a single worker child, never concurrent API request threads.

    The scoped legacy validator hook is restored on return or exception. Source
    pixels/geometry are not mutated; callers retain source SHA and frame binding.
    """
    core.checked_gray(gray)
    with guarded_native_heads() as events:
        base=search(gray,lines=[g for g in geometry if g['kind']=='line'],
            arcs=[g for g in geometry if g['kind']=='arc'],stroke_px=stroke_px)
        extra=core.dt_heads(gray,stroke_px)
        combined=core.append_heads(base['candidates'],extra['candidates'])
    text=core.text_components(gray,stroke_px)
    kept,excluded=core.guard_heads(combined,text['component_mask'])
    roles=core.role_proposals(gray,geometry,widths,kept,text['component_mask'])
    return {'schema':'stroke-role-proposals/2.0','base_heads':base,'DT_proposals':extra,
        'heads':kept,'heads_excluded_by_text':excluded,'text':text,'roles':roles,
        'guard_events':events,'source_modified':False,'production_admitted':False,
        'physical_geometry_approved':False,'requires_review':True}
