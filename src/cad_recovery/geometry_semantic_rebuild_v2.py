"""Fresh dependency replay with additive role-only source context.

The v1 rebuild body is retained for a controlled intervention: only the initial
role context differs. Geometry selection and all subsequent algorithms stay fixed.
"""
import cv2
import numpy as np
from .geometry_bundle_v1 import _namespace
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import foreground
from .source_interval_context_v1 import propose


def rebuild(gray, source, geometry, *, input_to_native, prefix, context_mode):
    """Fresh heads, intervals, local ownership and chains on selected primitives."""
    from .interval_roles_v1 import analyze as intervals
    from .local_role_guard_v1 import analyze as local_guard
    from .stroke_roles_v2 import analyze as safe_heads
    from .stroke_roles_v1 import stroke_floor, text_components, guard_heads, role_proposals, append_heads
    from .contact_role_overlay_v2 import analyze as contact
    from .boundary_role_overlay_v2 import analyze as boundary
    from .source_chain_profile_v2 import analyze as routes, diagnostic_heads
    from .head_junction_ownership_v1 import apply
    from .open_reference_chains_v2 import extend
    from .open_reference_context_v2 import filter_context
    from .joint_cv_profile_v1 import analyze as joint
    from .supported_role_overlay_v1 import source_ownership
    from .head_refinement_v1 import refine as refine_heads
    from .head_source_context_v1 import analyze as head_context
    from .curved_source_heads_v1 import search as source_heads
    widths = []
    for c in geometry:
        w = c.get('local_width', {}); local = w.get('local_observations', [])
        widths.append({'id': c['id'], 'width_px': w.get('width_px'),
            'coverage': w.get('coverage', float(np.mean([p['coverage'] for p in local])) if local else 0.),
            'heterogeneous': w.get('heterogeneous', any(p['heterogeneous'] for p in local))})
    stroke = stroke_floor(widths) or 3.
    h = safe_heads(gray, geometry, widths, stroke_px=stroke)
    heads = h['base_heads']['candidates']
    extra = [c for c in append_heads(heads, h['DT_proposals']['candidates']) if c['id'].startswith('DT-added-')]
    text = text_components(gray, stroke); kept, _ = guard_heads(heads, text['component_mask'])
    raw = intervals(gray, geometry, source, heads, input_to_native=input_to_native, diagnostic_heads=extra)
    if context_mode == 'selected_only':
        context = None
        baseline = role_proposals(gray, geometry, widths, kept, text['component_mask'], context=True)['pixel_roles']
    else:
        context = propose(gray, {k: v for k, v in raw.items() if k not in ('pixel_roles', 'text_mask')}, kept, text['component_mask'], mode=context_mode)
        baseline = role_proposals(gray, geometry, widths, kept, text['component_mask'], context=True)['pixel_roles']
        unknown = baseline == 255
        baseline[unknown] = context['pixel_roles'][unknown]
    graph = {k: v for k, v in raw.items() if k not in ('pixel_roles', 'text_mask')}
    local = local_guard(gray, graph, baseline, raw['text_mask'], heads, diagnostic_heads=extra)
    contact_map = contact(gray, graph, baseline, local, heads=heads, diagnostic_heads=extra)
    boundary_map = boundary(gray, graph, contact_map['pixel_roles'], local, heads=heads, diagnostic_heads=extra)
    inherited = boundary_map['pixel_roles']
    routed = routes(gray, graph, inherited, local['local']['text_mask'], heads, diagnostic_heads=extra)
    advanced = diagnostic_heads(gray, heads)
    proposals = routes(gray, graph, routed['pixel_roles'], local['local']['text_mask'], advanced['candidates'], diagnostic_heads=extra)
    base = apply(gray, graph, routed['pixel_roles'], proposals, advanced['candidates'])
    localized = refine_heads(gray, advanced['candidates'], graph['tracks'])
    head_ctx = head_context(gray, localized['localized'])
    detector = source_heads(gray, baseline=head_ctx['combined'], prior_observations=localized['localized'])
    refined = {'previous': head_ctx['combined'], **{a: detector[a] for a in ('shape_only', 'path_only', 'guarded')}}
    allowed = set(base['head_ids_used']); original_ids = {h['id'] for h in localized['localized']}
    effective = [h for h in refined['guarded'] if h['id'] in allowed or h['id'] not in original_ids]
    links = filter_context(gray, graph, base, extend(gray, graph, base, effective))
    fullbase = {**base, 'dimension_chains': base['dimension_chains']+links['open_reference_chains']}
    packet = joint(gray, graph, fullbase, refined['previous'], refined['guarded'], prior_binding_ids=allowed & {h['id'] for h in refined['previous']})
    owner, work = source_ownership(foreground(gray), graph['proposals'])
    nearest = owner['owner'] if owner is not None else np.zeros(gray.shape, np.int32)
    source_owner = np.maximum(nearest, 0).astype(np.uint16)
    ambiguous = (nearest < 0).astype(np.uint8)*255
    # The historical nearest-seed helper does not rasterize one-vertex paths.
    # Keep isolated observed vertices explicitly addressable, without a LINE.
    for number, interval in enumerate(graph['proposals'], 1):
        if len(interval['points_px']) == 1:
            x, y = np.rint(interval['points_px'][0]).astype(int)
            if source_owner[y, x] not in (0, number): ambiguous[y, x] = 255; source_owner[y, x] = 0
            elif not ambiguous[y, x]: source_owner[y, x] = number
    inventory = {h['id']: h for h in heads+extra+advanced['raw_candidates']+packet['width_contract']['normalized_heads']}
    bound = _namespace({'graph': graph, 'heads': packet['width_contract']['normalized_heads'],
                        'head_observations': list(inventory.values()), 'chains': packet['joint']['dimension_chains']}, prefix)
    state = {'semantic_context': context is None or context['complete'], 'intervals': not graph['search_incomplete'], 'routes': not routed['search_incomplete'],
             'local': not local['hybrid']['search_incomplete'], 'contact': not contact_map['search_incomplete'],
             'boundary': not boundary_map['search_incomplete'],
             'advanced_heads': not advanced['search_incomplete'], 'localized_heads': not localized['search_incomplete'],
             'head_context': bool(head_ctx['complete']), 'head_detector': bool(detector['complete']),
             'open_links': not links['search_incomplete'], 'joint_roles': not packet['joint']['search_incomplete'],
             'ownership': not work['incomplete']}
    return {**bound, 'semantic_context': None if context is None else {k: v for k, v in context.items() if k != 'pixel_roles'}, 'baseline_roles': baseline, 'roles': packet['joint']['pixel_roles'],
            'source_owner': source_owner, 'ambiguous_owner_mask': ambiguous,
            'role_owner': packet['joint']['pixel_interval_owner'].astype(np.uint16),
            'complete': all(state.values()), 'dependency_state': state, 'source_ownership_work': work,
            'dependent_graph_rebuilt': True, 'cached_dependent_inputs_used': False}

