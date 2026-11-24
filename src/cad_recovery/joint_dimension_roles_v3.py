"""Feed admitted source tip revisions through the unchanged semantic vetoes."""
from unittest.mock import patch

from . import joint_dimension_roles_v2 as previous
from .native_head_admission_v1 import build


def propose(source_image, **kwargs):
    bridge = build(kwargs['original_source'], source_image, geometry=kwargs['geometry'],
        ocr=kwargs['ocr'], roles=kwargs['roles'], source_heads=kwargs['source_heads'],
        working_to_source=kwargs['working_to_source'])
    with patch.object(previous, 'build', lambda *args, **unused: bridge):
        result = previous.propose(source_image, **kwargs)
    result['report'].update(schema_version='joint-dimension-roles/3.0',
        source_tip_admission=bridge['report']['tip_admission'])
    result['native_context'] = bridge['context']
    return result
