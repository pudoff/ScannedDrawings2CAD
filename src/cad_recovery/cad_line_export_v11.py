"""Frozen geometry export preserving UNKNOWN ownership at artificial splits."""
import json
from pathlib import Path
from unittest.mock import patch

from . import cad_object_export_v2 as objects
from .cad_line_export_v10 import export_objects as frozen_export
from .residual_split_ownership_v1 import ownership


def export_objects(packet, target, *, input_roles, protected=None, ocr=None, render=True):
    with patch.object(objects, '_ownership', ownership):
        report = frozen_export(packet, target, input_roles=input_roles,
                               protected=protected, ocr=ocr, render=render)
    report['schema_version'] = 'cad-lines-semantic-export/11.0'
    Path(target).with_name(Path(target).stem+'-objects.json').write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    return report
