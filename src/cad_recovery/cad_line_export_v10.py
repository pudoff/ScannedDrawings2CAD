"""Change semantic layers on frozen input-role geometry, without a second fit."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import cad_line_export_v9 as previous
from .geometry_preserving_role_export_v1 import apply as preserve
from .grid_role_revision_v1 import sha
from .stamp_role_projection_v1 import guard_export_layers

_export = previous.export_objects


def export_objects(packet, target, *, input_roles, protected=None, ocr=None, render=True):
    final = packet['roles']
    if input_roles.shape != final.shape or input_roles.dtype != np.uint8:
        raise ValueError('Aligned original automatic role map required for frozen geometry')
    if not np.array_equal(input_roles == 6, final == 6):
        raise ValueError('A changed TEXT exclusion requires a separate geometry reconstruction')
    before = sha(input_roles)
    base = previous.previous
    original_assemble, pattern, review = base.assemble, base.apply_pattern, base.apply_review
    guard = previous.apply
    observed = {}

    def assemble(gray, geometry, *, roles, protected):
        return original_assemble(gray, geometry, roles=input_roles, protected=protected)

    def original_pattern(assembly, gray, roles, *args, **kwargs):
        return pattern(assembly, gray, input_roles, *args, **kwargs)

    def original_review(assembly, gray, roles, *args, **kwargs):
        return review(assembly, gray, input_roles, *args, **kwargs)

    def fixed_geometry(assembly, gray, geometry, roles):
        baseline = guard_export_layers(guard(assembly, gray, geometry, input_roles), packet.get('frame_rules', []))
        result = preserve(baseline, gray, geometry, roles)
        observed['proof'] = deepcopy(result['geometry_preserving_role_export'])
        return result

    with (patch.object(base, 'assemble', assemble), patch.object(base, 'apply_pattern', original_pattern),
          patch.object(base, 'apply_review', original_review), patch.object(previous, 'apply', fixed_geometry)):
        result = _export(packet, target, protected=protected, ocr=ocr, render=render)
    if sha(input_roles) != before:
        raise ValueError('Frozen role-map input changed during export')
    result.update(schema_version='cad-lines-semantic-export/10.0',
        frozen_input_roles_sha256=before, geometry_preserving_role_export=observed['proof'])
    Path(target).with_name(Path(target).stem+'-objects.json').write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    return result
