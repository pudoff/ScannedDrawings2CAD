"""Choose source-bound geometry before rebuilding any dependent CV observations."""
from collections import defaultdict
from copy import deepcopy

import cv2
import numpy as np

from .current_cv_qualification_v1 import geometry_qa, inspect_bundle
from .curve_graph_metrics_v1 import native
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'geometry-bundle/1.0', 'maximum_pixels': 4194304,
          'maximum_paths': 4096, 'maximum_source_points': 2000000,
          'geometry_choice': 'observed_QA_or_exact_source_before_dependencies',
          'fallback_scope': 'whole observed view; no partial unverified promotion',
          'source_pixels_erased': 0, 'production_admitted': False, 'physical_geometry_approved': False}


def matrix(value):
    m = np.asarray(value, dtype=float)
    if m.shape != (3, 3) or not np.isfinite(m).all(): raise ValueError('Finite 3x3 input-to-native matrix required')
    native({'kind': 'line', 'points_px': [[0, 0], [1, 1]]}, m)
    return m


def validate_geometry(gray, source, candidates):
    """Source intervals must cover every observed edge exactly once."""
    checked_gray(gray)
    if not 0 < gray.size <= POLICY['maximum_pixels']: raise ValueError('Geometry input pixel budget exceeded')
    if len(source) > POLICY['maximum_paths']: raise ValueError('Source path budget exceeded')
    paths = {s['path_id']: s for s in source}
    if len(paths) != len(source) or len({s['id'] for s in source}) != len(source): raise ValueError('Duplicate source identity')
    ink = foreground(gray); total = 0
    for s in source:
        p = np.asarray(s['points_px'], float); total += len(p)
        if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all() or s['kind'] != 'polyline':
            raise ValueError('Invalid observed source path')
        if not np.array_equal(p, np.rint(p)): raise ValueError('Source vertices must be observed pixel centres')
        steps = np.abs(np.diff(p, axis=0))
        if len(p) > 1 and (np.any(steps.max(axis=1) > 1) or np.any(steps.max(axis=1) == 0)):
            raise ValueError('Source vertices must follow observed adjacent pixels')
        if s.get('closed') and not np.array_equal(p[0], p[-1]): raise ValueError('Unobserved source closure')
        q = p.astype(int)
        if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any() or not np.all(ink[q[:, 1], q[:, 0]]):
            raise ValueError('Source path outside observed ink')
    if total > POLICY['maximum_source_points']: raise ValueError('Source point budget exceeded')
    if len({g['id'] for g in candidates}) != len(candidates): raise ValueError('Duplicate primitive identity')
    pieces = defaultdict(list)
    for g in candidates:
        s = paths.get(g['path_id']); lo, hi = g['source_index_interval']
        if s is None or g['source_observation_id'] != s['id']: raise ValueError('Primitive/source identity differs')
        if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi < len(s['points_px']): raise ValueError('Invalid primitive source interval')
        if g['kind'] not in ('line', 'polyline', 'arc', 'circle'): raise ValueError('Unsupported primitive kind')
        if g.get('support_points') != s['points_px'][lo:hi+1]: raise ValueError('Primitive source support differs')
        if g['kind'] in ('line', 'polyline'):
            p = np.asarray(g['points_px'], float)
            if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all() or (g['kind'] == 'line' and len(p) != 2):
                raise ValueError('Invalid primitive points')
        elif not np.isfinite([*g['center_px'], g['radius_px']]).all() or g['radius_px'] <= 0:
            raise ValueError('Invalid primitive circle')
        if g['kind'] == 'arc' and (not np.isfinite([g['start_angle_deg'], g['sweep_angle_deg']]).all() or not 0 < abs(g['sweep_angle_deg']) < 360):
            raise ValueError('Invalid primitive arc')
        pieces[g['path_id']].append((lo, hi))
    if set(pieces) != set(paths): raise ValueError('Missing source paths')
    for identity, spans in pieces.items():
        spans.sort(); last = len(paths[identity]['points_px'])-1
        if spans[0][0] != 0 or spans[-1][1] != last or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError('Primitive intervals do not partition source')
        if last == 0 and len(spans) != 1 or last > 0 and any(a == b for a, b in spans): raise ValueError('Empty or duplicated source interval')


def select(gray, source, proposed, *, input_to_native, models, extraction_complete=True):
    m = matrix(input_to_native); validate_geometry(gray, source, proposed)
    before = object_sha([source, proposed]); qa = geometry_qa(source, proposed, m)
    accepted = bool(qa['decision']['accepted'] and qa['decision']['complete'] and extraction_complete)
    revision = object_sha({'pixels': sha(gray), 'input_size_px': list(gray.shape[::-1]), 'matrix': m.tolist(), 'source': source, 'proposed': proposed,
                           'models': models, 'policy': POLICY, 'modeled_accepted': accepted})
    prefix = 'gb1-'+revision
    bound_source = []
    for s in source:
        bound_source.append({**deepcopy(s), 'original_id': s['id'], 'original_path_id': s['path_id'],
                             'id': prefix+'/source/'+s['id'], 'path_id': prefix+'/path/'+s['path_id']})
    lookup = {s['original_path_id']: s for s in bound_source}; selected = []
    ink = foreground(gray)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    from .curve_graph_v1 import width_evidence
    for i, g in enumerate(proposed if accepted else source):
        s = lookup[g['path_id']]
        c = deepcopy(g)
        lo, hi = g['source_index_interval'] if accepted else (0, len(g['points_px'])-1)
        c.update(id=prefix+'/primitive/'+str(i), path_id=s['path_id'], source_observation_id=s['id'],
                 source_index_interval=[lo, hi], support_points=deepcopy(s['points_px'][lo:hi+1]),
                 classification='unclassified', semantic_role='unknown', semantic_ownership_confirmed=False,
                 physical_geometry_approved=False)
        if not accepted: c['review_status'] = 'exact_source_fallback_observation'
        c['local_width'] = width_evidence(c, ink, distance)
        selected.append(c)
    validate_geometry(gray, bound_source, selected)
    if before != object_sha([source, proposed]): raise ValueError('Geometry selection mutated its input')
    return {'revision': revision, 'prefix': prefix, 'source': bound_source, 'geometry': selected,
            'selection': {'mode': 'modeled_QA_pass' if accepted else 'exact_source_fallback',
                          'candidate_QA': qa, 'candidate_accepted': accepted,
                          'extraction_complete': bool(extraction_complete), 'source_unchanged': True,
                          'fallback_is_CAD_approval': False, 'geometry_export_approved': False}}


def _namespace(values, prefix):
    identities = set()
    def collect(v):
        if isinstance(v, dict):
            if isinstance(v.get('id'), str): identities.add(v['id'])
            for x in v.values(): collect(x)
        elif isinstance(v, list):
            for x in v: collect(x)
    collect(values)
    mapping = {i: prefix+'/dependent/'+i for i in identities if not i.startswith(prefix+'/')}
    def rewrite(v):
        if isinstance(v, str): return mapping.get(v, v)
        if isinstance(v, dict): return {mapping.get(k, k): rewrite(x) for k, x in v.items()}
        if isinstance(v, list): return [rewrite(x) for x in v]
        return v
    return rewrite(values)


def rebuild(gray, source, geometry, *, input_to_native, prefix):
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
    baseline = role_proposals(gray, geometry, widths, kept, text['component_mask'], context=True)['pixel_roles']
    raw = intervals(gray, geometry, source, heads, input_to_native=input_to_native, diagnostic_heads=extra)
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
    context = head_context(gray, localized['localized'])
    detector = source_heads(gray, baseline=context['combined'], prior_observations=localized['localized'])
    refined = {'previous': context['combined'], **{a: detector[a] for a in ('shape_only', 'path_only', 'guarded')}}
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
    state = {'intervals': not graph['search_incomplete'], 'routes': not routed['search_incomplete'],
             'local': not local['hybrid']['search_incomplete'], 'contact': not contact_map['search_incomplete'],
             'boundary': not boundary_map['search_incomplete'],
             'advanced_heads': not advanced['search_incomplete'], 'localized_heads': not localized['search_incomplete'],
             'head_context': bool(context['complete']), 'head_detector': bool(detector['complete']),
             'open_links': not links['search_incomplete'], 'joint_roles': not packet['joint']['search_incomplete'],
             'ownership': not work['incomplete']}
    return {**bound, 'roles': packet['joint']['pixel_roles'],
            'source_owner': source_owner, 'ambiguous_owner_mask': ambiguous,
            'role_owner': packet['joint']['pixel_interval_owner'].astype(np.uint16),
            'complete': all(state.values()), 'dependency_state': state, 'source_ownership_work': work,
            'dependent_graph_rebuilt': True, 'cached_dependent_inputs_used': False}


def inspect(gray, packet, input_to_native):
    if packet['roles'].shape != gray.shape or packet['roles'].dtype != np.uint8 or not np.isin(packet['roles'], [*range(10), 255]).all():
        raise ValueError('Invalid aligned role map')
    result = inspect_bundle(gray, packet['graph'], packet['heads'], packet['chains'], packet['roles'], np.eye(3),
                            geometry=packet['geometry'], source=packet['source'])
    # Role maps are in the input pixel frame here. The package's native matrix
    # is a separate authenticated frame; never resample the ownership map.
    proposals = packet['graph']['proposals']; ids = {p['id'] for p in proposals}
    tracks = packet['graph']['tracks']; issues = []
    for t in tracks:
        if not set(t['interval_ids']) <= ids: issues.append('dangling_track_interval')
    for b in packet['graph']['boundaries']['boundaries']:
        if not set(b['owners']) <= ids: issues.append('dangling_boundary_owner')
    boundaries = {b['id'] for b in packet['graph']['boundaries']['boundaries']}
    head_inventory = {h['id'] for h in packet['head_observations']}
    if not set(packet['graph']['head_ids_used']) <= head_inventory: issues.append('dangling_graph_head_observation')
    for c in packet['chains']:
        if c.get('boundary_id') is not None and c['boundary_id'] not in boundaries: issues.append('dangling_chain_boundary')
        evidence = c.get('local_reference_evidence', {})
        body_route = evidence.get('body_route', {})
        if not set(body_route.get('interval_ids', [])) <= ids: issues.append('dangling_open_body_route')
        for ref in evidence.get('references', []):
            for field in ('reference_ids', 'connectors', 'extension_interval_ids', 'extension_route_ids'):
                if not set(ref.get(field, [])) <= ids: issues.append('dangling_open_reference')
            if ref.get('extension_track') and ref['extension_track'] not in {t['id'] for t in tracks}: issues.append('dangling_open_reference_track')
    for b in packet['graph']['hatch_bundles']:
        if not set(b['interval_ids']) <= ids or not set(b['track_ids']) <= {t['id'] for t in tracks}: issues.append('dangling_hatch_owner')
    for field in ('source_owner', 'role_owner'):
        owner = packet[field]
        if owner.shape != gray.shape or owner.dtype != np.uint16 or np.any(owner > len(proposals)):
            issues.append('invalid_'+field)
    if packet['ambiguous_owner_mask'].shape != gray.shape or np.any((packet['ambiguous_owner_mask'] > 0) & (packet['source_owner'] != 0)):
        issues.append('ambiguous_owner_was_assigned')
    if np.any((packet['role_owner'] > 0) & ~np.isin(packet['roles'], range(1, 10))): issues.append('unassigned_role_has_owner')
    prefix = packet['prefix']+'/'
    for value in [*proposals, *tracks, *packet['heads'], *packet['head_observations']]:
        if not value['id'].startswith(prefix): issues.append('unbound_dependent_identity')
    result['ownership_and_identity_issues'] = sorted(set(issues))
    result['valid'] = bool(result['geometry_binding']['valid'] and not result['source_graph_identity_issues'] and
                           not result['chain_identity_issues'] and not issues)
    result['input_to_native'] = matrix(input_to_native).tolist()
    return result


def build(gray, *, input_to_native, models):
    """One extraction and one dependency rebuild; no snapshot names or GT input."""
    from skimage.morphology import skeletonize
    from .joint_shape_v1 import extract
    checked_gray(gray); m = matrix(input_to_native); before = sha(gray)
    if gray.size > POLICY['maximum_pixels']: raise ValueError('Geometry input pixel budget exceeded')
    ink = foreground(gray); skeleton = skeletonize(ink > 0, method='zhang').astype(np.uint8)*255
    raw_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    extracted = extract(ink, skeleton, input_to_native=m, move_nodes=False)
    chosen = select(gray, extracted['source'], extracted['candidates'], input_to_native=m, models=models,
                    extraction_complete=not extracted['analysis']['search_incomplete'])
    dependents = rebuild(gray, chosen['source'], chosen['geometry'], input_to_native=m, prefix=chosen['prefix'])
    packet = {**chosen, **dependents, 'raw_source': extracted['source'], 'proposed_geometry': extracted['candidates'],
              'input_sha256': before, 'input_size_px': list(gray.shape[::-1]), 'input_to_native': m.tolist(), 'native_to_input': np.linalg.inv(m).tolist(),
              'models': deepcopy(models), 'policy': deepcopy(POLICY), 'source_pixels_erased': 0,
              'raw_ink': raw_ink, 'preprocessing_excluded_ink': np.where((raw_ink > 0) & (ink == 0), 255, 0).astype(np.uint8),
              'production_changed': False, 'physical_geometry_approved': False}
    packet['inspection'] = inspect(gray, packet, m)
    if sha(gray) != before: raise ValueError('Geometry bundle changed input pixels')
    if not packet['inspection']['valid']: raise ValueError('Rebuilt geometry bundle is inconsistent: '+str(packet['inspection']))
    return packet
