"""Native CAD export with source-interval semantic checks and readback proof."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import ezdxf

from . import cad_line_export_v8 as previous
from .cad_line_export_v3 import _record
from .cad_object_export_v2 import entity_geometry, read_provenance
from .part_admission_v1 import object_sha
from .primitive_role_guard_v1 import apply
from .stamp_role_projection_v1 import guard_export_layers


CODE = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('primitive_role_guard_v1.py', 'cad_line_export_v9.py')}


def export_objects(packet, target, *, protected=None, ocr=None, render=True):
    original_export = previous.export_geometry
    guarded = None

    def export_geometry(*args, assembler, **kwargs):
        def assemble(gray, geometry, *, roles, protected):
            nonlocal guarded
            result = assembler(gray, geometry, roles=roles, protected=protected)
            guarded = guard_export_layers(apply(result, gray, geometry, roles), packet.get('frame_rules', []))
            by_id = {o['id']: o for o in guarded['objects']}
            for veto in guarded['stamp_export_withheld']:
                proof = by_id[veto['object_id']]['semantic_role_guard']
                proof.update(decision='original_frame_veto', assigned_role=255, frame_veto=veto)
            guarded['primitive_role_guard'].update(
                decisions=dict(Counter(o['semantic_role_guard']['decision'] for o in guarded['objects'])),
                changed_role_objects=sum(o['role'] != o['semantic_role_guard']['previous_role']
                                         for o in guarded['objects']))
            # The axis exporter also reads this exact object list after writing.
            result.clear()
            result.update(guarded)
            return result
        return original_export(*args, assembler=assemble, **kwargs)

    target = Path(target)
    with patch.object(previous, 'export_geometry', export_geometry):
        report = previous.export_objects(packet, target, protected=protected, ocr=ocr, render=render)
    doc = ezdxf.readfile(target)
    expected = {}
    for obj, row in zip(guarded['objects'], report['objects'], strict=True):
        entity = doc.entitydb[row['handle']]
        proof = read_provenance(entity)
        proof['semantic_role_guard'] = obj['semantic_role_guard']
        proof['semantic_role_guard']['export_layer'] = entity.dxf.layer
        _record(entity, proof)
        row.update(proof)
        expected[entity.dxf.handle] = object_sha([entity_geometry(entity), proof])
    doc.saveas(target)
    reopened = ezdxf.readfile(target)
    audit = reopened.audit()
    if audit.errors or audit.fixes:
        raise ValueError('Guarded DXF failed audit')
    for handle, checksum in expected.items():
        entity = reopened.entitydb[handle]
        if object_sha([entity_geometry(entity), read_provenance(entity)]) != checksum:
            raise ValueError('Semantic role proof changed on DXF readback')
    report.update(schema_version='cad-lines-semantic-export/9.0',
                  dxf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  semantic_role_readback_exact=True,
                  semantic_role_code=CODE)
    target.with_name(target.stem+'-objects.json').write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    return report
