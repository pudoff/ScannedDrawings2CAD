"""Require positive cell/glyph evidence for each newly offered grid stroke."""
import numpy as np
from .source_context_admission_v1 import analyze as propose
from .positive_grid_witness_v2 import analyze as witness
from .grid_role_revision_v1 import sha, check_authority


def analyze(gray, graph, raw_roles, heads, source_owner, ambiguous_owner, *, authority):
    base = propose(gray, graph, raw_roles, heads, source_owner, ambiguous_owner, authority=authority)
    new_grid = (base['candidate'] == 8) & (base['old_candidate'] != 8)
    ruling_ids = set()
    interval_numbers = {p['id']: i for i, p in enumerate(graph['proposals'], 1)}
    for track in graph['tracks']:
        length = np.linalg.norm(np.asarray(track['points'][1])-track['points'][0])
        if length >= max(8*track['width'], .15*min(gray.shape)):
            ruling_ids.update(interval_numbers[i] for i in track['interval_ids'])
    potential = (check_authority(gray, raw_roles, authority) & (base['old_candidate'] == 255) &
                 np.isin(source_owner, list(ruling_ids)) & (ambiguous_owner == 0))
    proof = witness(gray, graph, source_owner, ambiguous_owner, potential.astype(np.uint8)*255)
    candidate, owners = base['candidate'].copy(), base['owners'].copy()
    withdraw = new_grid & (proof['mask'] == 0)
    candidate[withdraw] = base['old_candidate'][withdraw]
    owners[withdraw] = base['old_owners'][withdraw]
    positive = potential & (proof['mask'] > 0)
    candidate[positive] = 8
    owners[positive] = source_owner[positive]
    return {**base, 'candidate': candidate, 'owners': owners,
            'context_added_pixels': {'1': int(((candidate == 1) & (base['old_candidate'] != 1)).sum()),
                                    '8': int(((candidate == 8) & (base['old_candidate'] != 8)).sum())},
            'positive_grid': {k: v for k, v in proof.items() if k != 'mask'},
            'positive_grid_mask_sha256': sha(proof['mask']),
            'grid_withdrawn_pixels': int(withdraw.sum()),
            'independent_ruling_context_pixels': int(positive.sum()),
            'complete': base['complete'] and proof['complete'],
            'source_immutable': base['source_immutable'] and proof['source_immutable']}
