"""Explicit process-local compatible intake over the unchanged V7 core.

Run in a dedicated conversion process, like the existing candidate adapters.
The base processing binding describes the CV/OCR/export core; the additional
execution record pins this input adapter separately. No queued Web binding is
silently changed.
"""
from pathlib import Path
from unittest.mock import patch

from . import profile_pipeline_v6 as adapted
from .profile_pipeline_v7 import run_profile as baseline
from .raster_input_compat_v1 import prepare, VERSION
from .dataset_layout_v1 import digest
from .pipeline import write_json


def run_profile(source, output, options, *, models=None, expected_binding=None):
    with patch.object(adapted, 'prepare', prepare):
        report = baseline(source, output, options, models=models,
                          expected_binding=expected_binding)
    output = Path(output)
    adaptation = report['common_input']
    normalized = bool(adaptation.get('compatibility_normalization'))
    root = Path(__file__).resolve().parents[2]
    files = ('src/cad_recovery/raster_input_compat_v1.py',
             'src/cad_recovery/raster_compatibility_pipeline_v1.py',
             'src/cad_recovery/raster_trailing_normalization_v2.py')
    proof = dict(schema_version='compatible-profile-execution/1.0',
        adapter=VERSION, base_profile_sha256=report['processing_profile']['sha256'],
        source_sha256=adaptation['sha256'], original_name=adaptation['original_name'],
        normalized=normalized, normalization=adaptation.get('compatibility_normalization'),
        original_strict_native_windows_available=adaptation.get(
            'native_windows_available', adaptation['source']['native_windows_available']),
        compatible_native_windows_available=adaptation.get('compatibility_native_windows_available', False),
        code=[dict(path=name, sha256=digest(root/name)) for name in files],
        CV_OCR_export_changed=False, Web_changed=False,
        full_resolution_quality_accepted=False)
    report['input_compatibility'] = proof
    report['processing_execution']['interface_contract'] = 'compatible-raster-profile/1.0'
    report['parameters']['input_adapter'] = VERSION
    if normalized:
        issue = dict(code='png_trailer_normalized', severity='warning',
            message='PNG обработан по проверенной копии без хвоста после IEND. Исходный файл сохранён.')
        for page in report['pages']:
            page['issues'].append(issue.copy())
            prefix = f"page-{page['page_index']+1:03d}"
            write_json(output/f'{prefix}-report.json', page)
    write_json(output/'input-compatibility.json', proof)
    write_json(output/'report.json', report)
    write_json(output/'profile-artifacts.json', dict(
        schema_version='profile-artifacts/compatible-input-1.0',
        base_profile_sha256=report['processing_profile']['sha256'],
        source_sha256=report['source_sha256'], input_compatibility=proof,
        files=[dict(name=p.name, sha256=digest(p), size_bytes=p.stat().st_size)
               for p in sorted(output.iterdir()) if p.is_file()
               and p.name != 'profile-artifacts.json' and p.suffix in ('.json', '.png', '.dxf')]))
    return report
