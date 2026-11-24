"""Compose source-route hypotheses with immutable inherited pixel roles."""
from copy import deepcopy

import numpy as np

from .local_role_guard_v3 import analyze as local_analyze
from .contact_role_overlay_v2 import analyze as contact_analyze
from .boundary_role_overlay_v2 import analyze as boundary_analyze


def analyze(gray,previous,inherited_roles,text_mask,heads,*,diagnostic_heads=()):
    local=local_analyze(gray,previous,inherited_roles,text_mask,heads,diagnostic_heads=diagnostic_heads)
    contact=contact_analyze(gray,previous,inherited_roles,local,heads=heads,diagnostic_heads=diagnostic_heads)
    boundary=boundary_analyze(gray,previous,contact['pixel_roles'],local,heads=heads,diagnostic_heads=diagnostic_heads)
    old=previous['proposals'];owners=np.zeros_like(boundary['pixel_interval_owner'])
    change_contact=contact['pixel_roles']!=inherited_roles
    owners[change_contact]=contact['pixel_interval_owner'][change_contact]
    change_boundary=boundary['pixel_roles']!=contact['pixel_roles']
    owners[change_boundary]=boundary['pixel_interval_owner'][change_boundary]
    return {'pixel_roles':boundary['pixel_roles'],'pixel_interval_owner':owners,
            'dimension_chains':deepcopy(local['hybrid']['dimension_chains']),
            'source_route_audit':local['hybrid'].get('source_route_audit',{}),
            'recovered_chains':local['hybrid'].get('recovered_chains',0),
            'head_ids_used':deepcopy(local['hybrid']['head_ids_used']),
            'head_events':deepcopy(local['hybrid']['head_events']),
            'contact_audit':deepcopy(contact['contact_guard_audit']),
            'boundary_competitor_audit':deepcopy(boundary['competitor_guard']),
            'added_pixels':int(np.count_nonzero(boundary['pixel_roles']!=inherited_roles)),
            'source_intervals':[{'id':i['id'],'path_id':i['path_id'],'source_index_interval':i['source_index_interval']} for i in old],
            'search_incomplete':local['hybrid']['search_incomplete'] or contact['search_incomplete'] or boundary['search_incomplete'],
            'production_admitted':False,'physical_geometry_approved':False,'semantic_ownership_confirmed':False,'source_pixels_erased':0}


def diagnostic_heads(gray,baseline):
    """Additive old-prefix → local-width → local-axis → eligibility ablation."""
    from .local_width_heads_v1 import search as width_search
    from .local_axis_heads_v2 import guarded_search
    from .head_eligibility_v2 import apply
    width=width_search(gray,baseline=baseline)
    axis=guarded_search(gray,baseline=width['candidates'])
    sidecar=apply(gray,axis['candidates'],inherited_morphology_pixels=axis['analysis']['morphology_pixels'])
    original_ids={h['id'] for h in baseline};retained=set(sidecar['diagnostic_candidate_ids'])|original_ids
    candidates=[h for h in axis['candidates'] if h['id'] in retained]
    return {'candidates':candidates,'raw_candidates':axis['candidates'],
            'analysis':{'width':width['analysis'],'axis':axis['analysis'],'eligibility':sidecar['analysis']},
            'prefix_unchanged':candidates[:len(baseline)]==list(baseline),
            'search_incomplete':width['search_incomplete'] or axis['search_incomplete'] or sidecar['search_incomplete']}


def head_only_profile(gray,previous,inherited_roles,text_mask,heads,*,diagnostic_heads=()):
    """Same overlay/geometry with the legacy chain verifier for the head ablation."""
    from .local_role_guard_v1 import analyze as legacy_local
    local=legacy_local(gray,previous,inherited_roles,text_mask,heads,diagnostic_heads=diagnostic_heads)
    contact=contact_analyze(gray,previous,inherited_roles,local,heads=heads,diagnostic_heads=diagnostic_heads)
    boundary=boundary_analyze(gray,previous,contact['pixel_roles'],local,heads=heads,diagnostic_heads=diagnostic_heads)
    return {'pixel_roles':boundary['pixel_roles'],'dimension_chains':deepcopy(local['hybrid']['dimension_chains']),
            'search_incomplete':local['hybrid']['search_incomplete'] or contact['search_incomplete'] or boundary['search_incomplete']}
