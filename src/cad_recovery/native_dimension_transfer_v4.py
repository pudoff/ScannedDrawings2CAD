"""Attach lossless source subcells to the unchanged conservative role binding."""
from unittest.mock import patch

import numpy as np

from . import native_dimension_transfer_v2 as measured
from . import native_dimension_transfer_v3 as previous
from .native_dimension_source_partition_v1 import observe


def bind(gray, working_gray, geometry, roles, chain_row, context_row, matrix, *, protection=None):
    captured = {}
    original = measured._source_target_masks

    def masks(*args, **kwargs):
        result = original(*args, **kwargs)
        if result[0] is not None:
            captured['masks'] = result[0]
        return result

    with patch.object(measured, '_source_target_masks', masks):
        result = previous.bind(gray, working_gray, geometry, roles, chain_row, context_row,
                               matrix, protection=protection)
    if 'masks' in captured and result['report']['complete'] and not result['report']['holds']:
        targets, ink, blocked = captured['masks']
        partition = observe(gray, working_gray, geometry, roles, chain_row, matrix,
            protection=np.zeros_like(roles) if protection is None else protection,
            targets=targets, ink=ink, source_protection=blocked)
        result['report']['source_partition'] = partition['report']
    else:
        result['report']['source_partition'] = {'complete': False, 'status': 'source_binding_unavailable',
                                               'roles_changed': 0}
    return result
