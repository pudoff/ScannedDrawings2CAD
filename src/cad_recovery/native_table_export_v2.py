"""Versioned native-table export with explicit safe text reconciliation."""
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter
import hashlib

import ezdxf

from .native_table_export_v1 import export as previous_export
from .table_text_reconcile_v1 import reconcile, LEGACY
from .table_cad_export_v1 import _snapshot
from .dataset_layout_v1 import encoded


def export(page, graph, baseline, target, native_to_working, working_size, *, choices=None):
    target = Path(target)
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError(target)
    if choices is not None:
        raise ValueError('For explicit selection, first export a stable candidate and reconcile its pinned files')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='table-text-private-', dir=target.parent) as scratch:
        intermediate = Path(scratch)/'table.dxf'
        previous = previous_export(page, graph, baseline, intermediate, native_to_working, working_size)
        result = reconcile(baseline, intermediate, target, graph=graph, native_to_DXF=previous['native_to_DXF'])
        doc, old = ezdxf.readfile(target), ezdxf.readfile(baseline)
        emitted = []
        for item in previous['emitted']:
            entity = doc.entitydb[item['handle']]
            emitted.append({**item, 'layer': entity.dxf.layer,
                'tags_sha256': hashlib.sha256(encoded(_snapshot(entity))).hexdigest()})
        report = {**previous, 'schema_version': 'native-table-export/2.0',
            'dxf_sha256': result['dxf_sha256'], 'text_reconciliation': result,
            'emitted': emitted,
            'legacy_text_retained_for_review': [e.dxf.handle for e in old.modelspace().query('TEXT MTEXT')
                if e.dxf.layer != LEGACY and doc.entitydb[e.dxf.handle].dxf.layer == LEGACY],
            'counts': dict(Counter(e.dxftype() for e in doc.modelspace())),
            'layer_counts': dict(Counter(e.dxf.layer for e in doc.modelspace())),
            'role_mapping': {**previous['role_mapping'], **result['role_mapping']}}
        target.with_suffix('.json').write_bytes(encoded(report))
        return report
