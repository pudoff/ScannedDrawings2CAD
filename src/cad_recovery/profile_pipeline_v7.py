"""Opt-in safety repair, retaining the complete V6 CV/OCR and input adapter."""
from pathlib import Path
from unittest.mock import patch

from . import profile_pipeline_v6 as previous
from .dataset_layout_v1 import digest, read
from .pipeline import write_json
from .profile_contract_v7 import ProcessingOptions, CANDIDATE, resolve_profile, validate_binding


def run_profile(source, output, options, *, models=None, expected_binding=None):
    options = ProcessingOptions.model_validate(options)
    if options.profile != CANDIDATE or (expected_binding and expected_binding.get('schema_version') != 'processing-profile/7.0'):
        return previous.run_profile(source, output, options.model_dump(), models=models, expected_binding=expected_binding)
    binding = validate_binding(options.profile, expected_binding) if expected_binding else resolve_profile(options.profile)
    from . import native_cad_layers_candidate_v1 as legacy_export
    from .native_cad_layers_candidate_v2 import export_candidate
    with patch.object(legacy_export, 'export_candidate', export_candidate):
        report = previous.run_profile(source, output, options.model_dump(), models=models,
                                      expected_binding=binding['core_binding'])
    output = Path(output)
    counts = []
    for page in report['pages']:
        prefix = f"page-{page['page_index']+1:03d}"
        annotation = read(output/f'{prefix}-native-annotations.json')
        proof = annotation['source_binding_guard']
        name = f'{prefix}-source-binding-guard.json'
        write_json(output/name, proof)
        page['artifacts']['source_binding_guard'] = name
        count = len(proof['withheld']); counts.append(count)
        page['demo_export']['source_binding_guard'] = {'version': '1.0', 'withheld_objects': count,
            'source_intervals_repaired': False, 'withheld_ink': 'residual'}
        if count:
            page['issues'].append({'code': 'unbound_geometry_retained', 'severity': 'warning',
                'message': 'Часть линий сохранена исходными штрихами: их привязку не удалось подтвердить. Проверьте эти участки.'})
        write_json(output/f'{prefix}-demo-export.json', page['demo_export'])
        write_json(output/f'{prefix}-report.json', page)
    report.update(processing_profile=binding, processing_options=options.model_dump(),
        export_safety={'version': 'source-binding/1.0', 'withheld_objects_by_page': counts,
                       'CV_OCR_changed': False, 'source_intervals_repaired': False})
    report['parameters']['profile_version'] = '7.0'
    from .profile_pipeline_v1 import validate_result
    validate_result(output, report, binding)
    write_json(output/'profile-contract.json', binding); write_json(output/'report.json', report)
    write_json(output/'profile-artifacts.json', {'schema_version': 'profile-artifacts/7.0',
        'profile_sha256': binding['sha256'], 'source_sha256': report['source_sha256'],
        'files': [{'name': p.name, 'sha256': digest(p), 'size_bytes': p.stat().st_size}
                  for p in sorted(output.iterdir()) if p.is_file() and p.name != 'profile-artifacts.json'
                  and p.suffix in ('.json', '.png', '.dxf')]})
    return report
