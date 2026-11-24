"""Local cell witnesses may contradict a new contour, without assigning GRID."""
from copy import deepcopy

import numpy as np

from .competitive_grid_witness_v2 import analyze as grid_witness
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {
    'version': 'contour-grid-competition/1.0', 'maximum_pixels': 4000000,
    'maximum_intervals': 4096, 'maximum_source_points': 2000000,
    'cell_producer': 'competitive_grid_witness_v2_unchanged',
    'search': 'all_unambiguous_source_ink_including_enclosure',
    'revision': 'withdraw_new_PART_only_at_local_cell_witness_pixels',
    'positive_role_created': False, 'geometry_modified': False,
    'physical_geometry_approved': False,
}


def analyze(gray, graph, source_owner, ambiguous_owner, proposed, *,
            frame_owner_numbers=(), other_role_owner_numbers=()):
    checked_gray(gray)
    items = graph['proposals']
    if gray.size > POLICY['maximum_pixels'] or len(items) > POLICY['maximum_intervals']:
        raise ValueError('Bounded cell competition required')
    if (source_owner.shape != gray.shape or source_owner.dtype.kind not in 'ui'
            or source_owner.max(initial=0) > len(items)):
        raise ValueError('Aligned current source owners required')
    for array in (proposed, ambiguous_owner):
        if array.shape != gray.shape or array.dtype != np.uint8 or not np.isin(array, [0, 255]).all():
            raise ValueError('Aligned binary proposal and ambiguity required')
    if np.any((proposed > 0) & ((source_owner == 0) | (ambiguous_owner > 0))):
        raise ValueError('Every contour proposal needs an unambiguous source owner')
    if len({p['id'] for p in items}) != len(items):
        raise ValueError('Unique source interval IDs required')
    frame, other = tuple(frame_owner_numbers), tuple(other_role_owner_numbers)
    if any(type(i) is not int or i < 1 or i > len(items) for i in frame+other):
        raise ValueError('Foreign conflicting source owner')
    before = object_sha(graph)
    arrays = (gray, source_owner, ambiguous_owner, proposed)
    hashes = [sha(a) for a in arrays]
    audit = {'source_points': 0, 'limits': []}
    if graph.get('search_incomplete'):
        audit['limits'].append('incomplete_source_search')
    if np.any(proposed) and not audit['limits']:
        for item in items:
            points = np.asarray(item['points_px'], float)
            if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
                    or (points < 0).any() or (points[:, 0] >= gray.shape[1]).any()
                    or (points[:, 1] >= gray.shape[0]).any()):
                raise ValueError('Finite source interval inside the raster required')
            audit['source_points'] += len(points)
            if audit['source_points'] > POLICY['maximum_source_points']:
                audit['limits'].append('source_point_budget')
                break
    evidence = None
    veto = np.zeros_like(proposed)
    if np.any(proposed) and not audit['limits']:
        search = ((source_owner > 0) & (ambiguous_owner == 0) & (foreground(gray) > 0)).astype(np.uint8)*255
        evidence = grid_witness(gray, graph, source_owner, ambiguous_owner, search,
                                frame_owner_numbers=frame, other_role_owner_numbers=other)
        if not evidence['complete']:
            audit['limits'].append('incomplete_cell_witness')
        else:
            veto[(proposed > 0) & (evidence['mask'] > 0)] = 255
    complete = not audit['limits']
    output = proposed.copy()
    if complete:
        output[veto > 0] = 0
    else:
        output[:] = 0
        veto[:] = 0
    if before != object_sha(graph) or hashes != [sha(a) for a in arrays]:
        raise ValueError('Cell competition mutated source observations')
    return {'mask': output, 'veto_mask': veto, 'complete': complete, 'source_immutable': True,
            'grid_veto_pixels': int(np.count_nonzero(veto)),
            'withdrawn_pixels': int(np.count_nonzero(proposed)-np.count_nonzero(output)),
            'grid_evidence': None if evidence is None else {k: v for k, v in evidence.items() if k != 'mask'},
            'audit': audit, 'policy': deepcopy(POLICY), 'positive_role_created': False,
            'geometry_modified': False, 'physical_geometry_approved': False}
