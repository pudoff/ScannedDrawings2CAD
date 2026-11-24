"""Passive reporting only; the v8 CV/OCR/export returns before new metadata."""
from pathlib import Path
from . import profile_pipeline_v8 as previous
from .profile_contract_v9 import ProcessingOptions,resolve_profile,validate_binding
from .search_completeness_v1 import capture,make,validate
from .dataset_layout_v1 import digest,read
from .pipeline import write_json


def run_profile(source,output,options,*,models=None,expected_binding=None):
    options=ProcessingOptions.model_validate(options)
    if expected_binding and expected_binding.get('schema_version')!='processing-profile/9.0':
        return previous.run_profile(source,output,options.model_dump(),models=models,expected_binding=expected_binding)
    binding=validate_binding(options.profile,expected_binding) if expected_binding else resolve_profile(options.profile)
    with capture() as records:
        report=previous.run_profile(source,output,options.model_dump(),models=models,expected_binding=binding['core_binding'])
    output=Path(output).resolve();report['processing_profile']=binding
    for page in report['pages']:
        prefix=f"page-{page['page_index']+1:03d}";folder=output/'private'/prefix/'experimental'
        gray=folder/'gray.png';graph=read(folder/'graph.json') if (folder/'graph.json').exists() else None
        value=make(report,page,records.get(str(folder)),gray_sha256=digest(gray) if gray.exists() else None,
                   graph=graph,binding_sha256=binding['search_reporting_registry_sha256'])
        validate(value,report,page,working_image_sha256=digest(gray) if gray.exists() else None)
        page['search_completeness']=value;name=prefix+'-search-completeness.json'
        page['artifacts']['search_completeness']=name;write_json(output/name,value)
        if value['domains']['head_search']['status'] in ('incomplete','unknown'):
            page['issues'].append(dict(code='head_search_'+value['domains']['head_search']['status'],severity='warning',
                message='Поиск наконечников ограничен или его полнота не записана. Проверьте размерные элементы; причины указаны в статусе поиска.'))
        write_json(output/(prefix+'-report.json'),page)
    report['processing_execution']['interface_contract']='CLI_and_worker_share_run_profile/9.0'
    report['parameters']['profile_version']='9.0'
    write_json(output/'profile-contract.json',binding);write_json(output/'report.json',report)
    write_json(output/'profile-artifacts.json',dict(schema_version='profile-artifacts/9.0',profile_sha256=binding['sha256'],
        source_sha256=report['source_sha256'],files=[dict(name=p.name,sha256=digest(p),size_bytes=p.stat().st_size)
        for p in sorted(output.iterdir()) if p.is_file() and p.name!='profile-artifacts.json' and p.suffix in ('.json','.png','.dxf')]))
    return report
