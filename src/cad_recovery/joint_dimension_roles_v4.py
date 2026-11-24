"""Use locally rechecked owners without weakening dimension role rules."""
from unittest.mock import patch

from . import joint_dimension_roles_v3 as previous
from .native_head_admission_v2 import build


_propose = previous.propose


def propose(source_image, **kwargs):
    with patch.object(previous, 'build', build):
        result = _propose(source_image, **kwargs)
    result['report']['schema_version'] = 'joint-dimension-roles/4.0'
    return result
