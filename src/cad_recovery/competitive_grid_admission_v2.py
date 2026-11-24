"""Two source-bound proposal arms, sharing cell and competition evidence."""
import numpy as np
from .source_context_admission_v1 import analyze as propose
from .competitive_grid_witness_v2 import analyze as witness
from .grid_role_revision_v1 import sha, check_authority

ARMS = ('agreement', 'local')


def analyze(gray, graph, raw_roles, heads, source_owner, ambiguous_owner, *, authority):
    base = propose(gray, graph, raw_roles, heads, source_owner, ambiguous_owner, authority=authority)
    numbers = {p['id']: i for i, p in enumerate(graph['proposals'], 1)}
    tracks = {t['id']: t for t in graph['tracks']}
    ruling, frame, other = set(), set(), set()
    for track in graph['tracks']:
        length = np.linalg.norm(np.asarray(track['points'][1])-track['points'][0])
        if length >= max(8*track['width'], .15*min(gray.shape)):
            ruling.update(numbers[i] for i in track['interval_ids'])
    for line in base['context']['lines']:
        owners = {numbers[i] for i in tracks[line['id']]['interval_ids']}
        hints = set(line['alternatives'])
        if 'sheet_frame' in hints: frame |= owners
        if hints & {'dimension_line', 'extension_line', 'leader_line', 'axis_construction'}: other |= owners
    potential = (check_authority(gray, raw_roles, authority) & (base['old_candidate'] == 255) &
                 np.isin(source_owner, list(ruling)) & (ambiguous_owner == 0))
    proof = witness(gray, graph, source_owner, ambiguous_owner, potential.astype(np.uint8)*255,
                    frame_owner_numbers=frame, other_role_owner_numbers=other)
    old_grid = (base['candidate'] == 8) & (base['old_candidate'] != 8)
    candidates, owner_maps, added = {}, {}, {}
    for arm in ARMS:
        candidate, owners = base['candidate'].copy(), base['owners'].copy()
        withdraw = old_grid & (proof['mask'] == 0)
        candidate[withdraw] = base['old_candidate'][withdraw]; owners[withdraw] = base['old_owners'][withdraw]
        if arm == 'local':
            positive = potential & (proof['mask'] > 0)
            candidate[positive] = 8; owners[positive] = source_owner[positive]
        candidates[arm], owner_maps[arm] = candidate, owners
        added[arm] = {str(k): int(((candidate == k) & (base['old_candidate'] != k)).sum()) for k in (1, 8)}
    return {'previous': base['previous'], 'old_candidate': base['old_candidate'], 'old_owners': base['old_owners'],
            'candidates': candidates, 'owners': owner_maps, 'context_added': added,
            'context': base['context'], 'competition': {k: v for k, v in proof.items() if k != 'mask'},
            'competition_mask_sha256': sha(proof['mask']),
            'frame_conflict_owner_numbers': sorted(frame), 'other_conflict_owner_numbers': sorted(other),
            'complete': base['complete'] and proof['complete'],
            'source_immutable': base['source_immutable'] and proof['source_immutable'],
            'geometry_modified': False, 'source_pixels_erased': 0, 'physical_geometry_approved': False}
