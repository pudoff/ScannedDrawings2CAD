"""Offer source context to existing part/frame gates, never raw role authority."""
from copy import deepcopy

import numpy as np

from .grid_role_revision_v1 import check_authority, sha
from .part_admission_v1 import object_sha
from .source_interval_context_v1 import propose as context_proposals
from .source_region_roles_v1 import analyze as previous_producer
from .stroke_roles_v1 import text_components

POLICY = {'version': 'source-context-admission/1.0', 'new_roles': [1, 8],
          'previous_producer_preserved': True, 'context_overrides_known_roles': False,
          'context_is_role_authority': False, 'CAD_primitive_created': False,
          'source_pixels_erased': 0, 'physical_geometry_approved': False}


def analyze(gray, graph, raw_roles, heads, source_owner, ambiguous_owner, *, authority, mode='track'):
    before = object_sha([graph, heads])
    arrays = [gray, raw_roles, source_owner, ambiguous_owner, authority['mask']]
    hashes = [sha(a) for a in arrays]
    if (source_owner.shape != gray.shape or source_owner.dtype.kind not in 'iu' or
            source_owner.min(initial=0) < 0 or source_owner.max(initial=0) > len(graph['proposals']) or
            ambiguous_owner.shape != gray.shape):
        raise ValueError('Source owner is outside the current interval domain')
    old = previous_producer(gray, graph, raw_roles, authority=authority)
    text = text_components(gray, graph['stroke_px'])
    context = context_proposals(gray, graph, heads, text['component_mask'], mode=mode)
    candidate, owners = old['guarded'].copy(), old['owners']['guarded'].copy()
    allowed = check_authority(gray, raw_roles, authority)
    eligible = (allowed & (candidate == 255) & np.isin(context['pixel_roles'], POLICY['new_roles']) &
                (source_owner > 0) & (ambiguous_owner == 0))
    if not context['complete']:
        eligible[:] = False
    candidate[eligible] = context['pixel_roles'][eligible]
    owners[eligible] = source_owner[eligible]
    immutable = before == object_sha([graph, heads]) and hashes == [sha(a) for a in arrays]
    if not immutable:
        raise ValueError('Positive source-context producer mutated observations')
    return {'previous': old['previous'], 'candidate': candidate, 'owners': owners,
            'old_candidate': old['guarded'], 'old_owners': old['owners']['guarded'],
            'context': {k: v for k, v in context.items() if k != 'pixel_roles'},
            'context_added_pixels': {str(k): int((eligible & (candidate == k)).sum()) for k in POLICY['new_roles']},
            'complete': old['complete'] and context['complete'],
            'old_complete': old['complete'], 'context_complete': context['complete'],
            'source_immutable': immutable, 'policy': deepcopy(POLICY)}
