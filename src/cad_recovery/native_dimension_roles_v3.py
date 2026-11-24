"""Publish source-separated dimension bodies with unchanged semantic authority."""
from copy import deepcopy
from unittest.mock import patch

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v1 as previous
from .native_dimension_transfer_v3 import bind

_propose = previous.propose
_joint_propose = joint.propose


def propose(*args, **kwargs):
    with patch.object(previous, 'bind', bind):
        result = _propose(*args, **kwargs)
    report = result['report']
    report['schema_version'] = 'native-dimension-roles/3.0'
    report['policy'] = {**deepcopy(report['policy']),
        'version': 'native-dimension-roles/3.0',
        'carrier_body_basis': 'all_current_measured_source_axes',
        'competitor_refinement': 'fresh_component_and_adjacent_interval_ownership',
        'head_uncertainty_or_export_thresholds_relaxed': False,
        'shared_seeds_released': False}
    return result


def joint_propose(*args, **kwargs):
    with patch.object(joint, 'native_propose', propose):
        result = _joint_propose(*args, **kwargs)
    result['report']['schema_version'] = 'joint-dimension-roles/8.0'
    return result
