"""Add admitted native body roles while preserving the older working decisions."""
from unittest.mock import patch

from . import joint_dimension_roles_v5 as previous
from .native_dimension_roles_v1 import propose as native_propose
from .part_admission_v1 import object_sha

_propose = previous.propose


def propose(source_image, *, authority=None, authority_root=None, protected=None, **kwargs):
    native = native_propose(kwargs['original_source'], source_image,
        **{key: kwargs[key] for key in ('geometry', 'ocr', 'roles', 'source_heads', 'working_to_source', 'arrows')},
        authority=authority, authority_root=authority_root, protected=protected)
    with patch.object(previous, 'inspect', lambda *args, **unused: native['chain_result']):
        result = _propose(source_image, **kwargs)
    output = result['roles'].copy()
    changed = native['roles'] != kwargs['roles']
    conflict = changed & (output != kwargs['roles']) & (output != native['roles'])
    if conflict.any():
        raise ValueError('Native and working dimension decisions conflict')
    output[changed] = native['roles'][changed]
    result.update(roles=output, native_roles=native)
    result['report'].update(schema_version='joint-dimension-roles/6.0', native_role_release={
        'artifact': 'native-dimension-roles.json', 'object_sha256': object_sha(native['report']),
        'summary': native['report']['summary'], 'complete': native['report']['complete'],
        'roles_changed': native['report']['roles_changed'], 'prior_PART_revised': native['report']['prior_PART_revised'],
        'roles_added': native['report']['roles_added'], 'physical_geometry_approved': False,
        'legacy_working_decisions_retained_separately': True, 'CAD_DIMENSION_created': False})
    return result
