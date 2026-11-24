"""Fresh original tip admission followed by the existing joint role checks."""
from pathlib import Path
from unittest.mock import patch

from . import dimension_role_stage_v2 as previous
from .dimension_role_stage_v3 import _summary
from .joint_dimension_roles_v3 import propose


def analyze(folder, packet, gray, roles, *, original_source, working_to_source,
            frame_rules=(), ocr=None):
    observed = {}

    def with_admission(image, **kwargs):
        result = propose(image, **kwargs)
        observed['context'] = result['native_context']
        observed['admission'] = result['native_bridge']['report']['tip_admission']
        result['report']['native_context'] = _summary(observed['context'])
        return result

    with patch.object(previous, 'propose', with_admission):
        result = previous.analyze(folder, packet, gray, roles, original_source=original_source,
            working_to_source=working_to_source, frame_rules=frame_rules, ocr=ocr)
    for name, value in (('native-dimension-context.json', observed['context']),
                        ('native-head-admission.json', observed['admission'])):
        previous.previous.write(Path(folder) / name, value)
    return result
