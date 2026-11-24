"""One profile dispatcher for direct CLI and leased service runs."""
import time
from pathlib import Path

from .pipeline import run_pipeline, write_json
from .demo_pipeline_v1 import run_demo_pipeline
from .profile_contract_v1 import (ProcessingOptions, ProfileError, resolve_profile,
    validate_binding, verify_models, digest)


def validate_result(output, report, binding):
    import ezdxf
    required = ['source_preview','graph','dxf']
    if binding['definition']['geometry']:
        required += ['geometry_candidates','geometry_preview']
    if binding['definition']['ocr']:
        required += ['ocr_report','ocr_preview','arrow_report','arrow_preview',
                     'dimension_associations','dimension_associations_preview','dxf_preview','demo_export']
    if report['status'] == 'failed' or not report['pages']:
        raise ProfileError('processing_failed','Profile produced no successful page result')
    for page in report['pages']:
        for role in required:
            name = page['artifacts'].get(role)
            if not name or Path(name).name != name or not (output/name).is_file():
                raise ProfileError('profile_artifact_mismatch',f'Required profile artifact missing: {role}')
        if binding['definition']['ocr']:
            engine=page['ocr']['engine']
            expected = {m['filename']:m['sha256'] for m in binding['ocr']['models']}
            actual = {m['filename']:m['sha256'] for m in engine['models']}
            if (actual != expected or engine['provider'] != 'CPUExecutionProvider' or engine['offline'] is not True
                    or page['dimension_associations']['algorithm_version'] != binding['definition']['association']):
                raise ProfileError('profile_artifact_mismatch','Output used an unexpected OCR engine or association algorithm')
        saved = ezdxf.readfile(output/page['artifacts']['dxf'])
        audit = saved.audit()
        if audit.errors or audit.fixes:
            raise ProfileError('profile_artifact_mismatch','Exported DXF failed structural validation')


def run_profile(source, output, options, *, models=None, expected_binding=None):
    tick=time.perf_counter()
    options=ProcessingOptions.model_validate(options)
    binding=(validate_binding(options.profile,expected_binding) if expected_binding is not None
             else resolve_profile(options.profile))
    model_root=verify_models(binding,models)
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():
        raise ProfileError('output_exists','Output already exists; choose a new directory')
    source_hash=digest(source)
    if options.profile=='presentation_draft':
        report=run_demo_pipeline(source,output,ocr_models=model_root,render_dpi=options.render_dpi,
            image_dpi=options.image_dpi,rotation_deg=options.rotation_deg)
    else:
        report=run_pipeline(source,output,export=True,render_dpi=options.render_dpi,
            image_dpi=options.image_dpi,prepare=options.profile=='geometry_observations' or options.rotation_deg!=0,
            detect=options.profile=='geometry_observations',rotation_deg=options.rotation_deg,ocr=False,associate=False)
    if digest(source)!=source_hash:
        raise ProfileError('input_integrity','Source changed during profile processing')
    validate_result(output,report,binding)
    report.update(processing_profile=binding,processing_options=options.model_dump(),
        processing_execution={'interface_contract':'CLI_and_worker_share_run_profile/1.0',
            'source_name':source.name,'ocr_models_directory':str(model_root) if model_root else None,
            'elapsed_seconds':round(time.perf_counter()-tick,4)})
    write_json(output/'profile-contract.json',binding)
    write_json(output/'report.json',report)
    files=[]
    for path in sorted(output.iterdir()):
        if path.is_file() and path.suffix in ('.png','.json','.dxf'):
            files.append({'name':path.name,'sha256':digest(path),'size_bytes':path.stat().st_size})
    write_json(output/'profile-artifacts.json',{'schema_version':'profile-artifacts/1.0',
        'profile_sha256':binding['sha256'],'source_sha256':source_hash,'files':files})
    return report
