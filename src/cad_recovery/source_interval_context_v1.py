"""Role-only straight context from observed intervals; never CAD primitives."""
from copy import deepcopy

import cv2
import numpy as np

from .curve_graph_v1 import width_evidence
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground, role_proposals

POLICY = {'version': 'source-interval-context/1.0', 'maximum_lines': 512,
          'modes': ['interval', 'track'], 'geometry_export_approved': False,
          'semantic_ownership_confirmed': False, 'source_pixels_erased': 0}


def propose(gray, graph, heads, text_mask, *, mode):
    """Reuse existing straightness and width tests without granting analytic type."""
    checked_gray(gray)
    if mode not in POLICY['modes']:
        raise ValueError('Unknown source-context mode')
    if text_mask.shape != gray.shape:
        raise ValueError('Aligned text mask required')
    before = object_sha([graph, heads])
    gray_before, text_before = sha(gray), sha(text_mask)
    lookup = {p['id']: p for p in graph['proposals']}
    if len(lookup) != len(graph['proposals']):
        raise ValueError('Duplicate observed interval identity')
    if mode == 'interval':
        entries = [(p['id'], [p['points_px'][0], p['points_px'][-1]], [p['id']])
                   for p in graph['proposals'] if p['linear_context']]
    else:
        entries = [(t['id'], t['points'], t['interval_ids']) for t in graph['tracks']]
    incomplete = bool(graph['search_incomplete'] or len(entries) > POLICY['maximum_lines'])
    # Abstain on the entire context instead of silently selecting a prefix.
    if incomplete:
        return {'pixel_roles': np.full(gray.shape, 255, np.uint8), 'lines': [],
                'context': [], 'complete': False, 'policy': deepcopy(POLICY),
                'reason': 'source_search_or_line_budget', 'source_immutable': True}
    ink = foreground(gray)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    contexts, widths, bindings = [], [], []
    for identity, endpoints, owners in entries:
        if not owners or not set(owners) <= set(lookup):
            raise ValueError('Context has foreign source intervals')
        if any(not lookup[i]['linear_context'] for i in owners):
            raise ValueError('Nonlinear source context cannot become a straight probe')
        points = np.asarray(endpoints, float)
        if points.shape != (2, 2) or not np.isfinite(points).all() or np.linalg.norm(points[1]-points[0]) <= 0:
            raise ValueError('Finite nondegenerate source endpoints required')
        c = {'id': identity, 'kind': 'line', 'points_px': points.tolist()}
        w = width_evidence(c, ink, distance)
        contexts.append(c)
        widths.append({'id': identity, 'width_px': w['width_px'],
                       'coverage': w['coverage'], 'heterogeneous': w['heterogeneous']})
        bindings.append({'id': identity, 'interval_ids': list(owners),
                         'parent_primitive_ids': sorted({lookup[i]['parent_primitive_id'] for i in owners}),
                         'endpoints_px': points.tolist(), 'width_evidence': w,
                         'usage': 'role_context_only', 'CAD_primitive_created': False})
    result = role_proposals(gray, contexts, widths, heads, text_mask, context=True)
    # Role paint is supported only on the same foreground used to extract paths.
    result['pixel_roles'][ink == 0] = 255
    immutable = (before == object_sha([graph, heads]) and gray_before == sha(gray)
                 and text_before == sha(text_mask))
    if not immutable:
        raise ValueError('Role context mutated its observations')
    return {**result, 'context': bindings, 'complete': True, 'mode': mode,
            'policy': deepcopy(POLICY), 'source_immutable': immutable}
