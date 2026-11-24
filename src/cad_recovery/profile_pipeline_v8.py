"""Use compatible input only for explicit v8 bindings; preserve old dispatch."""
from pathlib import Path

from . import profile_pipeline_v7 as previous
from .raster_compatibility_pipeline_v1 import run_profile as compatible
from .profile_contract_v8 import ProcessingOptions, resolve_profile, validate_binding
from .dataset_layout_v1 import digest
from .pipeline import write_json


def run_profile(source, output, options, *, models=None, expected_binding=None):
    options = ProcessingOptions.model_validate(options)
    if expected_binding and expected_binding.get('schema_version') != 'processing-profile/8.0':
        return previous.run_profile(source, output, options.model_dump(), models=models,
                                    expected_binding=expected_binding)
    binding = validate_binding(options.profile, expected_binding) if expected_binding else resolve_profile(options.profile)
    report = compatible(source, output, options.model_dump(), models=models,
                        expected_binding=binding['core_binding'])
    output = Path(output)
    report.update(processing_profile=binding, processing_options=options.model_dump())
    report['processing_execution']['interface_contract'] = 'CLI_and_worker_share_run_profile/8.0'
    report['parameters']['profile_version'] = '8.0'
    for page in report['pages']:
        for issue in page['issues']:
            if issue.get('code') == 'png_trailer_normalized':
                issue['message'] = 'Входной PNG подготовлен для обработки. Исходный файл сохранён.'
        write_json(output/f"page-{page['page_index']+1:03d}-report.json",page)
    from .profile_pipeline_v1 import validate_result
    validate_result(output, report, binding)
    write_json(output/'profile-contract.json',binding)
    write_json(output/'report.json',report)
    write_json(output/'profile-artifacts.json',dict(schema_version='profile-artifacts/8.0',
        profile_sha256=binding['sha256'],source_sha256=report['source_sha256'],
        input_compatibility=report['input_compatibility'],
        files=[dict(name=p.name,sha256=digest(p),size_bytes=p.stat().st_size)
               for p in sorted(output.iterdir()) if p.is_file()
               and p.name!='profile-artifacts.json' and p.suffix in ('.json','.png','.dxf')]))
    return report
