"""Current primitive assembly plus the existing automatic axis export contract.

Patches are limited to an isolated experiment worker; production exports stay frozen.
"""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import ezdxf

from . import cad_line_export_v8 as axis_export
from . import cad_line_export_v9 as semantic_export
from . import cad_object_export_v2 as native_export
from .cad_axis_contract_v1 import apply, audit_entities
from .cad_line_export_v3 import _record
from .cad_object_assembly_v16 import assemble
from .arc_identity_role_guard_v1 import apply as role_guard


def export_objects(packet, target, *, assembly=None, protected=None, render=False):
    if packet.get('confirmed_axis_groups'):
        raise ValueError('Automatic CENTER qualification cannot use manual axis groups')
    old_provenance = native_export._provenance
    def provenance(entity, obj, source_sha):
        proof = old_provenance(entity, obj, source_sha)
        if obj.get('native_short_arc_evidence'):
            proof['native_short_arc_evidence'] = obj['native_short_arc_evidence']
            _record(entity, proof)
        return proof
    def source_assembly(*args, **kwargs):
        return assemble(*args, **kwargs) if assembly is None else assembly
    with (patch.object(axis_export, 'assemble', source_assembly),
          patch.object(axis_export, 'apply_pattern', apply),
          patch.object(semantic_export, 'apply', role_guard),
          patch.object(native_export, '_provenance', provenance)):
        report = semantic_export.export_objects(packet, target, protected=protected, render=render)
    target = Path(target)
    doc = ezdxf.readfile(target)
    expected = [r['object_id'] for r in report['objects'] if r.get('axis_pattern_evidence')]
    report['whole_CENTER_readback'] = audit_entities(doc, packet['gray'], expected_ids=expected)
    report['axis_component_scope'] = 'automatic_source_pattern_draft_not_confirmed_semantic_owner'
    report['dxf_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_name(target.stem+'-objects.json').write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    return report
