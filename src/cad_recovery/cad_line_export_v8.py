"""Geometry draft export with the stage 1.4 bounded axis search."""
import hashlib
import json
from pathlib import Path

import ezdxf

from .cad_axis_pattern_v5 import apply as apply_pattern
from .cad_axis_preview_v1 import render as render_dxf
from .cad_axis_review_v1 import apply_review
from .cad_line_export_v3 import _record
from .cad_line_export_v7 import assemble
from .cad_object_export_v2 import entity_geometry, export_objects as export_geometry, read_provenance
from .part_admission_v1 import object_sha


def export_objects(packet, target, *, protected=None, ocr=None, render=True):
    assembled = None

    def assembler(gray, geometry, *, roles, protected):
        nonlocal assembled
        base = assemble(gray, geometry, roles=roles, protected=protected)
        if packet.get('confirmed_axis_groups'):
            assembled = apply_review(base, gray, roles, geometry, packet['confirmed_axis_groups'])
        else:
            assembled = apply_pattern(base, gray, roles, geometry,
                packet.get('axis_observed_heads', packet.get('heads', [])),
                protected=protected, text_protection=packet.get('axis_text_protection'))
        return assembled

    report = export_geometry(packet, target, protected=protected, ocr=ocr, render=False, assembler=assembler)
    target = Path(target)
    doc = ezdxf.readfile(target)
    expected = {}
    for obj, row in zip(assembled['objects'], report['objects'], strict=True):
        entity = doc.entitydb[row['handle']]
        data = read_provenance(entity)
        data['line_continuation_evidence'] = obj.get('line_continuation_evidence')
        axis_evidence = next((key for key in ('axis_pattern_evidence', 'axis_review') if key in obj), None)
        if axis_evidence:
            pattern = obj['linetype_pattern_px']
            token = object_sha(pattern)[:8]
            automatic = axis_evidence == 'axis_pattern_evidence'
            linetype = ('CENTER_PATTERN_' if automatic else 'CENTER_REVIEW_') + token
            layer = ('AXES_PATTERN_DRAFT_' if automatic else 'AXES_REVIEWED_') + token
            if linetype not in doc.linetypes:
                doc.linetypes.new(linetype, dxfattribs={'description': 'Source-measured dash-gap-dot-gap',
                    'pattern': [sum(map(abs, pattern)), *pattern]})
            if layer not in doc.layers:
                doc.layers.new(layer, dxfattribs={'color': 7, 'lineweight': 13, 'linetype': linetype})
            entity.dxf.layer = layer
            entity.dxf.linetype = 'BYLAYER'
            entity.dxf.ltscale = 1
            data.update(linetype_pattern_px=pattern, status=obj['status'])
            data[axis_evidence] = obj[axis_evidence]
            if automatic:
                data['semantic_ownership_confirmed'] = False
        _record(entity, data)
        row.update(data, layer=entity.dxf.layer)
        expected[entity.dxf.handle] = object_sha([entity_geometry(entity), data])
    doc.header['$LTSCALE'] = 1.
    doc.saveas(target)
    reopened = ezdxf.readfile(target)
    audit = reopened.audit()
    if audit.errors or audit.fixes or reopened.units != 0:
        raise ValueError('Axis draft DXF failed readback')
    for handle, checksum in expected.items():
        entity = reopened.entitydb[handle]
        if object_sha([entity_geometry(entity), read_provenance(entity)]) != checksum:
            raise ValueError('Axis draft geometry or provenance changed on readback')
    report.update(schema_version='cad-lines-semantic-export/8.0',
                  dxf_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    if render:
        preview = target.with_name(target.stem+'-preview.png')
        render_dxf(target, preview)
        report['artifacts']['preview'] = preview.name
    target.with_name(target.stem+'-objects.json').write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    return report
