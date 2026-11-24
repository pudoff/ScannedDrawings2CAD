"""Monotone composition of the frozen contour branch and new material witnesses."""
from copy import deepcopy
import numpy as np

from .material_contour_witness_v1 import analyze as witness
from .local_shared_contacts_v2 import analyze as baseline
from .negative_source_graph_v1 import build
from .local_negative_context_v1 import check_certificate
from .open_source_continuation_v1 import observe
from .competitive_grid_admission_v2 import analyze as role_context
from .contour_role_competition_v1 import guard as role_guard
from .contour_page_competition_v1 import analyze as frame_guard
from .attached_grid_competition_v1 import analyze as grid_guard
from .part_contact_v1 import bind_source
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha
from .open_contour_joint_v5 import metadata, negative_subset
from .stroke_roles_v1 import foreground
import cv2

POLICY = {'version': 'material-contours/1.0', 'old_branch_preserved': True,
          'negative_guards_relaxed': False, 'geometry_modified': False,
          'source_owner_remapped': False, 'Web_changed': False,
          'production_admitted': False, 'physical_geometry_approved': False}


def guarded(packet, mask, *, authority, source_pin, page_context, protected):
    gray = packet['gray']; graph = packet['graph']; certificate = None
    empty = np.zeros_like(mask)
    if page_context is None or not page_context['complete']:
        return {'mask': empty, 'complete': False, 'reason': 'page_context_unavailable_or_incomplete', 'stages': {}}
    if page_context['source_sha256'] != sha(gray) or page_context.get('GT_used') is not False:
        raise ValueError('Material page belongs to another source or GT')
    if graph.get('search_incomplete', True) or 'stroke_px' not in graph:
        graph, certificate = build(packet); check_certificate(packet, graph, certificate)
    graph, continuation = observe(gray, graph)
    context = role_context(gray, graph, packet['roles'], packet['heads'], packet['source_owner'],
                           packet['ambiguous_owner_mask'], authority=authority)
    negative = {k: v for k, v in context.items() if k not in ('previous', 'old_candidate', 'old_owners', 'candidates', 'owners')}
    roles = role_guard(gray, graph, packet['source_owner'], packet['ambiguous_owner_mask'], mask, negative)
    stages = {'role': metadata(roles)}
    candidate = negative_subset(mask, roles)
    binding = bind_source(gray, gray, np.eye(3).tolist(), source_pin, resize_method='CV_AREA')
    frame = frame_guard(gray, graph, packet['source_owner'], packet['ambiguous_owner_mask'], candidate,
        native_gray=gray, page=page_context['page'], detection=page_context['detection'],
        source_mask=page_context['frame_mask'], input_to_native=binding['matrix'], resize_method=binding['resize_method'])
    stages['frame'] = metadata(frame); candidate = negative_subset(candidate, frame)
    # A small gap can open an internal frame cell, defeating a closed-hole
    # observation. Do not extend the new positive branch along a component
    # already containing source-qualified page-frame ink. This is an explicit
    # conservative refusal; it does not erase an earlier accepted PART.
    ink = foreground(gray)
    observed = np.where((ink > 0) & (packet['source_owner'] > 0) & (packet['ambiguous_owner_mask'] == 0), 255, 0).astype(np.uint8)
    probe = frame_guard(gray, graph, packet['source_owner'], packet['ambiguous_owner_mask'], observed,
        native_gray=gray, page=page_context['page'], detection=page_context['detection'],
        source_mask=page_context['frame_mask'], input_to_native=binding['matrix'], resize_method=binding['resize_method'])
    _, labels = cv2.connectedComponents(ink, connectivity=8)
    frame_labels = sorted(set(map(int, labels[probe['veto_mask'] > 0]))-{0})
    removed = (candidate > 0) & np.isin(labels, frame_labels)
    candidate = candidate.copy(); candidate[removed] = 0
    stages['frame_component'] = {'complete': probe['complete'], 'frame_component_ids': frame_labels,
        'withdrawn_pixels': int(removed.sum()), 'source_frame_probe': metadata(probe),
        'source_connected_ink_only': True, 'shared_material_admission_deferred': True}
    grid = grid_guard(gray, graph, packet['source_owner'], packet['ambiguous_owner_mask'], candidate,
        source_gray=gray, native_gray=gray, page=page_context['page'], observations=page_context['observations'],
        source_masks=page_context['grid_masks'], input_to_native=binding['matrix'], resize_method=binding['resize_method'],
        baseline_roles=packet['roles'], protected=protected.astype(np.uint8)*255)
    stages['grid'] = metadata(grid); candidate = negative_subset(candidate, grid)
    complete = bool(continuation['complete'] and context['complete'] and all(s['complete'] for s in stages.values())
                    and (certificate is None or certificate['complete']))
    return {'mask': candidate if complete else empty, 'complete': complete,
            'reason': None if complete else 'incomplete_material_negative_context', 'stages': stages,
            'negative_certificate': certificate, 'source_continuation': continuation,
            'role_context_sha256': object_sha(negative)}


def analyze(packet, *, authority, source_pin, page_context, protected=None):
    gray = packet['gray']
    protection = np.zeros(gray.shape, bool) if protected is None else protected
    arrays = (gray, packet['roles'], packet['source_owner'], packet['ambiguous_owner_mask'], protection)
    snapshots = [sha(a) for a in arrays]
    before = object_sha([packet['graph'], packet['geometry'], packet['heads']])
    kw = dict(authority=authority, source_pin=source_pin, page_context=page_context, protected=protection)
    old = baseline(packet, **kw)
    positive = witness(packet, authority=authority, protected=protection)
    guard = None; added = np.zeros_like(gray)
    if positive['complete'] and positive['mask'].any():
        guard = guarded(packet, positive['mask'], **kw)
        if guard['complete']:
            added = guard['mask']
    complete = bool(old['complete'] and positive['complete'] and (guard is None or guard['complete']))
    mask = old['mask'].copy()
    if complete:
        mask |= added
    if np.any((mask > 0) & ((packet['roles'] != 255) | protection)):
        raise ValueError('Material composition changed baseline or protected pixels')
    if np.any((old['mask'] > 0) & (mask == 0)):
        raise ValueError('Material composition lost the frozen branch')
    if snapshots != [sha(a) for a in arrays] or before != object_sha([packet['graph'], packet['geometry'], packet['heads']]):
        raise ValueError('Material composition changed source observations')
    full = packet['roles'].copy(); full[mask > 0] = 1
    return {'mask': mask, 'raw_mask': old['raw_mask'] | positive['mask'], 'full_roles': full,
            'added_mask': np.where((mask > 0) & (old['mask'] == 0), 255, 0).astype(np.uint8),
            'material_raw_mask': positive['mask'], 'baseline_mask': old['mask'],
            'complete': complete, 'baseline': metadata(old), 'positive': metadata(positive),
            'guard': metadata(guard) if guard else None, 'source_immutable': True,
            'raw_positive_pixels': int(np.count_nonzero(old['raw_mask'] | positive['mask'])),
            'guarded_positive_pixels': int(np.count_nonzero(mask)),
            'new_material_pixels': int(np.count_nonzero((mask > 0) & (old['mask'] == 0))),
            'legacy_packet_complete': packet.get('complete'),
            'legacy_graph_search_incomplete': packet['graph'].get('search_incomplete'),
            'policy': deepcopy(POLICY)}
