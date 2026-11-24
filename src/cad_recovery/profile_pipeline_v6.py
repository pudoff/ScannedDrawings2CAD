"""A source-preserving input adapter over the unchanged CV/OCR/CAD core."""
import math
from pathlib import Path
import shutil
import time

import numpy as np
from . import profile_pipeline_v5 as core
from .profile_contract_v6 import ProcessingOptions,resolve_profile,validate_binding,ProfileError
from .dataset_layout_v1 import digest,read
from .pipeline import write_json
from .raster_input_v2 import prepare


def run_profile(source,output,options,*,models=None,expected_binding=None,_core_runner=None):
    options=ProcessingOptions.model_validate(options)
    if expected_binding and expected_binding.get('schema_version')!='processing-profile/6.0':
        return core.run_profile(source,output,options.model_dump(),models=models,expected_binding=expected_binding)
    binding=validate_binding(options.profile,expected_binding) if expected_binding else resolve_profile(options.profile)
    source=Path(source).resolve();output=Path(output).resolve();started=time.perf_counter()
    if output.exists():raise ProfileError('output_exists','Choose a new output directory')
    prepared,adaptation=prepare(source,output.with_name(output.name+'-input-v2'))
    parameters=options.model_dump()
    if adaptation['mode']!='bounded_pdf_passthrough' and parameters['image_dpi'] is not None:
        dpi=parameters['image_dpi']*math.sqrt(math.prod(adaptation['scale_xy']));parameters['image_dpi']=dpi if dpi>=36 else None
    runner=_core_runner or core.run_profile
    try:
        report=runner(prepared,output,parameters,models=models,expected_binding=binding['core_binding'])
    except ProfileError as error:
        if (output/'report.json').exists():
            partial=read(output/'report.json')
            reasons=[p.get('experimental_cv',{}).get('reason') for p in partial.get('pages',[])]
            if 'Incomplete source paths' in reasons:raise ProfileError('cv_paths_incomplete','CV source paths are incomplete; raster intake succeeded') from error
        raise
    if digest(source)!=adaptation['sha256']:raise ProfileError('input_integrity','Original source changed')
    if adaptation['mode']!='bounded_pdf_passthrough':
        inner=report.get('input_adaptation',{});matrix=np.array(inner.get('source_pixel_to_analysis_px',np.eye(3)))@np.array(adaptation['source_pixel_to_analysis_px'])
        adaptation.update(source_pixel_to_analysis_px=matrix.tolist(),requested_image_dpi=options.image_dpi,effective_image_dpi=parameters['image_dpi'])
        report.update(source_sha256=adaptation['sha256'],analysis_source_sha256=inner.get('analysis_sha256',adaptation['analysis_sha256']),input_adaptation=adaptation)
        write_json(output/'input-adaptation.json',adaptation)
        shutil.copyfile(Path(adaptation['context_path']),output/'source-context.png')
        for page in report['pages']:
            prefix=f"page-{page['page_index']+1:03d}";graph_path=output/f'{prefix}-graph.json'
            graph=read(graph_path);graph['source'].update(path=str(source),sha256=adaptation['sha256'])
            graph.setdefault('provenance',{})['common_raster_input']=adaptation
            if 'demo_input_adaptation' in graph['provenance']:graph['provenance']['demo_input_adaptation']=adaptation
            write_json(graph_path,graph)
            exported=page.get('demo_export')
            if exported:
                exported.update(input_resized=adaptation['resized'],original_size_px=adaptation['original_size_px'],
                    full_resolution_analysis=False,native_source_windows_available=adaptation['native_windows_available'])
                write_json(output/f'{prefix}-demo-export.json',exported)
            page['source_input']=dict(original_sha256=adaptation['sha256'],original_size_px=adaptation['original_size_px'],
                native_windows_available=adaptation['native_windows_available'],full_resolution_analysis=False)
            write_json(output/f'{prefix}-report.json',page)
    report.update(processing_profile=binding,processing_options=options.model_dump(),common_input=adaptation,
        processing_execution=dict(interface_contract='CLI_and_worker_share_run_profile/6.0',source_name=source.name,
            elapsed_seconds=round(time.perf_counter()-started,4)),full_resolution_quality_accepted=False)
    report.setdefault('parameters',{}).update(profile_version='6.0',input_adapter='bounded-raster/2.0')
    report.setdefault('limitations',[]).append('Обзор уменьшен; приёмка деталей исходного разрешения и сшивки выполняется отдельно по10.4.3.')
    from .profile_pipeline_v1 import validate_result
    validate_result(output,report,binding)
    write_json(output/'report.json',report);write_json(output/'profile-contract.json',binding)
    write_json(output/'profile-artifacts.json',dict(schema_version='profile-artifacts/6.0',profile_sha256=binding['sha256'],
        source_sha256=adaptation['sha256'],files=[dict(name=p.name,sha256=digest(p),size_bytes=p.stat().st_size)
        for p in sorted(output.iterdir()) if p.is_file() and p.name!='profile-artifacts.json' and p.suffix in ('.json','.png','.dxf')]))
    return report
