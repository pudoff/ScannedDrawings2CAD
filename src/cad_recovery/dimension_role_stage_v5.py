"""Fresh source heads, local shaft ownership, and unchanged joint-role rules."""
from unittest.mock import patch

from . import dimension_role_stage_v4 as previous
from .joint_dimension_roles_v4 import propose


def analyze(*args, **kwargs):
    with patch.object(previous, 'propose', propose):
        return previous.analyze(*args, **kwargs)
