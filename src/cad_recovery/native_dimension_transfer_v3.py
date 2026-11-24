"""Refine specific coarse competitor vetoes after fresh source ownership checks."""
from collections import Counter
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from . import native_dimension_transfer_v2 as previous
from .native_dimension_competitors_v1 import admit
from .native_dimension_transfer_v1 import _sha
from .part_admission_v1 import object_sha


def bind(gray, working_gray, geometry, roles, chain_row, context_row, matrix, *, protection=None):
    observed = {}
    source_masks = previous._source_target_masks

    def targets(*args, **kwargs):
        value = source_masks(*args, **kwargs)
        if value[0] is not None:
            observed['masks'] = value[0]
        return value

    inputs = object_sha([geometry, chain_row, context_row, np.asarray(matrix).tolist()])
    hashes = [_sha(array) for array in (gray, working_gray, roles)]
    protected = np.zeros_like(roles) if protection is None else protection
    protection_hash = _sha(protected)
    with patch.object(previous, '_source_target_masks', targets):
        result = previous.bind(gray, working_gray, geometry, roles, chain_row,
                               context_row, matrix, protection=protected)
    report = result['report']
    report['schema_version'] = 'native-dimension-transfer/3.0'
    report['policy'] = {**deepcopy(report['policy']), 'version': 'native-dimension-transfer/3.0',
                        'competitor_refinement': 'fresh_component_and_adjacent_interval_ownership',
                        'shared_seeds_released': False}
    if not report['complete'] or report['holds'] or 'masks' not in observed:
        report['competitor_refinement'] = {'complete': True, 'status': 'baseline_not_admissible',
                                          'additional_points': 0}
        return result
    source_targets, ink, blocked = observed['masks']
    extra = admit(gray, working_gray, geometry, roles, chain_row, context_row, matrix,
                  protection=protected, baseline=result, targets=source_targets,
                  ink=ink, source_protection=blocked)
    report['competitor_refinement'] = extra['report']
    if not extra['complete']:
        report['limits'].append('source_competitor_observation_incomplete')
        report['complete'] = False
        report['rolled_back'] = True
        result['candidate_roles'] = roles.copy()
        result['proposed_roles'][:] = 255
        result['binding_owner'][:] = 0
        for row in report['rows']:
            row['admitted_working_points'] = []
            row['previous_role_counts'] = {}
    else:
        intervals = geometry.get('ownership_intervals', geometry['candidates'])
        owners = {row['id']: index for index, row in enumerate(intervals, 1)}
        rows = {row['interval_id']: row for row in report['rows']}
        for identity, points in extra['points_by_interval'].items():
            row = rows[identity]
            permitted = {tuple(item['point']) for item in row['withheld_points']
                         if item['reason'] == 'competing_working_interval_footprint'}
            if row['role'] != 2 or len(points) != len({tuple(q) for q in points}):
                raise ValueError('Invalid source competitor release target')
            for x, y in points:
                if ((x, y) not in permitted or protected[y, x]
                        or result['proposed_roles'][y, x] != 255):
                    raise ValueError('Source competitor release bypassed an existing guard')
                result['proposed_roles'][y, x] = 2
                result['binding_owner'][y, x] = owners[identity]
                if roles[y, x] == 255:
                    result['candidate_roles'][y, x] = 2
            released = {tuple(q) for q in points}
            row['withheld_points'] = [item for item in row['withheld_points']
                                      if tuple(item['point']) not in released]
            row['admitted_working_points'] = sorted([*row['admitted_working_points'], *points])
            row['source_competitor_released_points'] = deepcopy(points)
            counts = Counter(int(roles[y, x]) for x, y in row['admitted_working_points'])
            row['previous_role_counts'] = {str(k): v for k, v in counts.items()}
    proposed = result['proposed_roles']
    report.update(candidate_role_changes=int(np.count_nonzero(result['candidate_roles'] != roles)),
                  proposed_role_pixels=int(np.count_nonzero(proposed != 255)),
                  summary={str(int(k)): int(v) for k, v in
                           zip(*np.unique(proposed[proposed != 255], return_counts=True))},
                  candidate_roles_sha256=_sha(result['candidate_roles']),
                  geometric_roles_sha256=_sha(proposed),
                  binding_owner_sha256=_sha(result['binding_owner']))
    if (inputs != object_sha([geometry, chain_row, context_row, np.asarray(matrix).tolist()])
            or hashes != [_sha(array) for array in (gray, working_gray, roles)]
            or protection_hash != _sha(protected)):
        raise ValueError('Competitor refinement mutated source inputs')
    return result
