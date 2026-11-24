"""Publish dimension bodies measured on source shafts, preserving admission gates."""
from copy import deepcopy
from unittest.mock import patch

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v1 as previous
from .native_dimension_transfer_v2 import bind

_propose = previous.propose
_joint_propose = joint.propose


def propose(*args, **kwargs):
    """Re-observe the chain and use measured carrier axes for body ownership."""
    with patch.object(previous, 'bind', bind):
        result = _propose(*args, **kwargs)
    report = result['report']
    report['schema_version'] = 'native-dimension-roles/2.0'
    report['policy'] = {
        **deepcopy(report['policy']),
        'version': 'native-dimension-roles/2.0',
        'carrier_body_basis': 'all_current_measured_source_axes',
        'head_uncertainty_or_export_thresholds_relaxed': False,
    }
    return result


def joint_propose(*args, **kwargs):
    """Retain working decisions and bind the new native report in the joint result."""
    with patch.object(joint, 'native_propose', propose):
        result = _joint_propose(*args, **kwargs)
    result['report']['schema_version'] = 'joint-dimension-roles/7.0'
    return result
