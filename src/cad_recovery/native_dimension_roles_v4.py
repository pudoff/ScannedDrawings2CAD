"""Preserve semantic V3 release; publish source-resolution evidence separately."""
from unittest.mock import patch

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v3 as previous
from .native_dimension_transfer_v4 import bind


def propose(*args, **kwargs):
    with patch.object(previous, 'bind', bind):
        result = previous.propose(*args, **kwargs)
    result['report']['source_partition_observation_added'] = True
    return result


def joint_propose(*args, **kwargs):
    with patch.object(joint, 'native_propose', propose):
        result = joint.propose(*args, **kwargs)
    result['report']['schema_version'] = 'joint-dimension-roles/8.0'
    return result
