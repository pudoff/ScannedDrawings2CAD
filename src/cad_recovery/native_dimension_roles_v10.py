"""Add external thin-wall dimensions while retaining straight and ARC branches."""
from unittest.mock import patch

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v9 as previous
from .native_external_linear_v1 import observe_context


def propose(*args,**kwargs):
    with patch.object(previous,'observe_context',observe_context):
        return previous.propose(*args,**kwargs)


def joint_propose(*args,**kwargs):
    with patch.object(joint,'native_propose',propose):result=joint.propose(*args,**kwargs)
    result['report']['schema_version']='joint-dimension-roles/13.0'
    return result
