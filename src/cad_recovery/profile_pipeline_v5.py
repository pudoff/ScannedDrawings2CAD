"""Publish the qualified CAD/layer candidate through the common service contract."""
from collections import Counter
from pathlib import Path
import shutil
import time

from . import profile_pipeline_v4 as previous
from .profile_contract_v5 import ProcessingOptions, CANDIDATE, NATIVE, validate_binding, resolve_profile, ProfileError
from .dataset_layout_v1 import digest
from .pipeline import write_json

ROOT = previous.ROOT


def upgrade_page(output, page):
    # Optional CV modules are imported after the legacy producer initializes them.
    from .native_annotation_inputs_v1 import load_inputs
    from .native_cad_layers_candidate_v1 import export_candidate
    import ezdxf
    prefix = f"page-{page['page_index']+1:03d}"
    old = page['demo_export']
    if old['mode'] != 'native_annotations_draft':
        raise ProfileError('candidate_unavailable','Candidate requires raster CAD observations; use the legacy profile for vector PDF')
    observed = output/'private'/prefix/'experimental'
    packet, protection, annotations, _, _ = load_inputs(ROOT, observed, output/page['artifacts']['ocr_report'])
    target = output/'private'/prefix/'native-cad-v5'; target.mkdir()
    report = export_candidate(packet, annotations, target/'annotated.dxf', protected=protection)
    shutil.copyfile(target/'annotated.dxf', output/f'{prefix}.dxf')
    shutil.copyfile(target/'annotated.png', output/f'{prefix}-dxf.png')
    write_json(output/f'{prefix}-native-annotations.json',report)
    doc = ezdxf.readfile(output/f'{prefix}.dxf'); audit=doc.audit()
    if audit.errors or audit.fixes:
        raise ProfileError('candidate_invalid','Candidate DXF requires repair')
    exported = {**old, 'schema_version':'native-cad-web-export/5.0',
        'dxf_sha256':digest(output/f'{prefix}.dxf'), 'preview_sha256':digest(output/f'{prefix}-dxf.png'),
        'geometry_entity_count':report['geometry']['object_count'], 'modelspace_counts':report['counts'],
        'layer_counts':dict(Counter(e.dxf.layer for e in doc.modelspace())),
        'source_pixels':report['source_pixels'], 'residual_pixels':report['residual_pixels'],
        'annotation_replaced_pixels':report['annotation_replaced_pixels'],
        'candidate_revision':'5.0', 'frame_layer_assignments':len(report['frame_semantics']['assignments']['decisions']),
        'manual_CAD_verified':False, 'trueview_gui_verified':False, 'preview_rendered_from_dxf':True}
    page['demo_export']=exported
    write_json(output/f'{prefix}-demo-export.json',exported)
    write_json(output/f'{prefix}-report.json',page)
    return dict(page_index=page['page_index'],candidate_applied=True,
                frame_layer_assignments=exported['frame_layer_assignments'])


def run_profile(source, output, options, *, models=None, expected_binding=None):
    options=ProcessingOptions.model_validate(options)
    if options.profile != CANDIDATE:
        return previous.run_profile(source,output,options.model_dump(),models=models,expected_binding=expected_binding)
    binding=validate_binding(CANDIDATE,expected_binding) if expected_binding is not None else resolve_profile(CANDIDATE)
    started=time.perf_counter(); output=Path(output).resolve(); original_sha=digest(Path(source))
    report=previous.run_profile(source,output,{**options.model_dump(),'profile':NATIVE},models=models)
    pages=[upgrade_page(output,page) for page in report['pages']]
    if digest(Path(source)) != original_sha:
        raise ProfileError('input_integrity','Input changed')
    report.update(processing_profile=binding,processing_options=options.model_dump(),
        native_cad_integration=dict(revision='5.0',pages=pages),
        processing_execution=dict(interface_contract='CLI_and_worker_share_run_profile/5.0',
            elapsed_seconds=round(time.perf_counter()-started,4),source_name=Path(source).name))
    report['parameters'].update(profile=CANDIDATE,profile_version='5.0',raster_export='whole_primitives_OCR_FRAME_draft')
    report['limitations'].append('Экспериментальный профиль: улучшение целых примитивов и отдельных линий рамки; общая классификация слоёв ещё неполна.')
    from .profile_pipeline_v1 import validate_result
    validate_result(output,report,binding)
    write_json(output/'profile-contract.json',binding);write_json(output/'report.json',report)
    write_json(output/'profile-artifacts.json',dict(schema_version='profile-artifacts/5.0',
        profile_sha256=binding['sha256'],source_sha256=original_sha,
        files=[dict(name=p.name,sha256=digest(p),size_bytes=p.stat().st_size)
            for p in sorted(output.iterdir()) if p.is_file() and p.name!='profile-artifacts.json'
            and p.suffix in ('.png','.json','.dxf')]))
    return report
