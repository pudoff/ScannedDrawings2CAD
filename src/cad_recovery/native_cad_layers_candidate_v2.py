"""Source-binding safety over the unchanged layer/annotation exporter.

Use only in an isolated single-job process, like V1. Invalid primitives are
withheld; the existing ownership exporter retains their ink as residual.
"""
from pathlib import Path
from unittest.mock import patch

from . import cad_axis_candidate_export_v1 as geometry_export
from .cad_source_binding_guard_v1 import apply
from .dataset_layout_v1 import encoded
from .native_cad_layers_candidate_v1 import export_candidate as previous_export


def export_candidate(packet, annotations, target, *, protected=None, render=True):
    previous_guard = geometry_export.role_guard
    evidence = []

    def guarded(assembly, gray, geometry, roles):
        semantic = previous_guard(assembly, gray, geometry, roles)
        result = apply(semantic, geometry)
        evidence.append(result.get('source_binding_export_guard', {
            'version': '1.0', 'withheld': [], 'source_intervals_repaired': False,
            'withheld_ink': 'unconverted residual'}))
        return result

    with patch.object(geometry_export, 'role_guard', guarded):
        report = previous_export(packet, annotations, target, protected=protected, render=render)
    if len(evidence) != 1:
        raise ValueError('Exactly one source-binding export pass required')
    report['source_binding_guard'] = evidence[0]
    target = Path(target)
    target.with_name('source-binding-guard.json').write_bytes(encoded(evidence[0]))
    target.with_suffix('.json').write_bytes(encoded(report))
    return report
