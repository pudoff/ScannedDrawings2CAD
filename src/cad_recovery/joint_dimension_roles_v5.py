"""Attach coupled source geometry without silently releasing semantic roles."""
from unittest.mock import patch

from . import joint_dimension_roles_v4 as previous
from .native_dimension_chain_v1 import inspect
from .part_admission_v1 import object_sha


_propose = previous.propose


def propose(source_image, **kwargs):
    chain = inspect(kwargs['original_source'],source_image,geometry=kwargs['geometry'],
        ocr=kwargs['ocr'],roles=kwargs['roles'],source_heads=kwargs['source_heads'],
        working_to_source=kwargs['working_to_source'],arrows=kwargs['arrows'])
    # The bridge was just rechecked by inspect; do not reread saved evidence or
    # independently rebuild a different set of heads for the semantic rules.
    with patch.object(previous,'build',lambda *args, **unused:chain['admission']):
        result = _propose(source_image, **kwargs)
    result['native_chain'] = chain['report']
    result['report'].update(schema_version='joint-dimension-roles/5.0',native_chain={
        'object_sha256':object_sha(chain['report']), 'artifact':'native-dimension-chain.json',
        'summary':chain['report']['summary'], 'complete':chain['report']['complete'],
        'local_observation_complete':chain['report']['local_observation_complete'],
        'roles_changed':0, 'semantic_dimension_approved':False,
        'physical_geometry_approved':False})
    return result
