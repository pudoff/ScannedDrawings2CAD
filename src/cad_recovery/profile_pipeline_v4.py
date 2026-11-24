"""Web/CLI integration of released native CAD assembly and annotations."""
from pathlib import Path
from collections import Counter
import shutil
import time
import ezdxf
import numpy as np
from PIL import Image
from . import profile_pipeline_v3 as previous
from .profile_contract_v4 import ProcessingOptions,resolve_profile,validate_binding,NATIVE_PROFILES,REVIEW,ProfileError
from .dataset_layout_v1 import checked,digest,read
from .pipeline import write_json

ROOT = Path(__file__).resolve().parents[2]


def upgrade_page(output,page,binding):
    prefix = f"page-{page['page_index']+1:03d}"; old = page['demo_export']
    observed = output/'private'/prefix/'experimental'
    if old['mode']=='pdf_vector_draft' or not (observed/'summary.json').is_file():
        return {'branch':'direct_vector_pdf' if old['mode']=='pdf_vector_draft' else 'previous_fallback',
                'native_upgrade_applied':False}
    # The previous raster producer initializes its optional CV runtime. Keep
    # module import and direct vector PDF independent of that runtime.
    from .grid_role_revision_v1 import sha
    from .native_annotation_inputs_v1 import load_inputs
    from .native_annotation_export_v1 import export_annotations
    review = None; review_matched = False
    if binding['name']==REVIEW:
        candidate = checked(ROOT,binding['review_example']); expected = read(candidate)
        pixels = np.array(Image.open(observed/'gray.png').convert('L'))
        review_matched = expected['source_pixel_sha256']==sha(pixels) and expected['size_px']==list(pixels.shape[::-1])
        if review_matched: review = candidate
        else:
            page['issues'].append({'code':'review_example_not_matching','severity':'warning',
                'message':'Пример размера не относится к этому изображению. Выполнен обычный CAD-экспорт без подстановки примера.'})
    ocr_name = page['artifacts']['ocr_report']
    packet,protected,annotations,inputs,_ = load_inputs(ROOT,observed,output/ocr_name,review_path=review)
    target = output/'private'/prefix/'native-cad'; target.mkdir()
    result = export_annotations(packet,annotations,target/'annotated.dxf',protected=protected)
    archive = target/'previous'; archive.mkdir()
    for name in (old['dxf'],old['preview']):
        path = output/name
        if path.exists(): path.replace(archive/path.name)
    shutil.copyfile(target/'annotated.dxf',output/f'{prefix}.dxf')
    shutil.copyfile(target/'annotated.png',output/f'{prefix}-dxf.png')
    write_json(output/f'{prefix}-native-annotations.json',result)
    size = list(packet['gray'].shape[::-1]); working_to_processed = np.linalg.inv(np.asarray(packet['summary']['input_to_working'])).tolist()
    auto = sum(r['type']=='DIMENSION' and r.get('evidence_mode')=='automatic_chain_draft' for r in result['annotations'])
    reviewed = result['native_dimension_count']-auto
    exported = {**old,'schema_version':'native-cad-web-export/1.0','mode':'native_annotations_draft',
        'dxf':f'{prefix}.dxf','preview':f'{prefix}-dxf.png','dxf_sha256':digest(output/f'{prefix}.dxf'),
        'preview_sha256':digest(output/f'{prefix}-dxf.png'),'empty':result['source_pixels']==0,
        'geometry_entity_count':result['geometry']['object_count'],'modelspace_counts':result['counts'],
        'native_text_count':result['native_text_count'],'native_mtext_count':result['native_mtext_count'],
        'visible_text_count':result['visible_text_count'],'review_text_count':result['native_text_count']+result['native_mtext_count']-result['visible_text_count'],
        'native_dimensions':result['native_dimension_count'],'automatic_dimensions':auto,'review_dimensions':reviewed,
        'native_leaders':result['native_leader_count'],'review_example_requested':binding['name']==REVIEW,
        'review_example_applied':review_matched,'working_size_px':size,'working_to_processed':working_to_processed,
        'units':'working_pixels_unitless','physical_geometry_approved':False,'trueview_gui_verified':False,
        'font_file':result['font_file'],'annotations_report':f'{prefix}-native-annotations.json',
        'source_pixels':result['source_pixels'],'annotation_replaced_pixels':result['annotation_replaced_pixels'],
        'residual_pixels':result['residual_pixels'],'preview_rendered_from_dxf':True}
    doc = ezdxf.readfile(output/f'{prefix}.dxf')
    exported['layer_counts'] = dict(Counter(e.dxf.layer for e in doc.modelspace()))
    exported['source_size_px'] = size
    page['demo_export'] = exported; page['artifacts']['native_annotations'] = exported['annotations_report']
    page['issues'].append({'code':'native_cad_draft','severity':'warning',
        'message':'Нативные CAD-объекты и текст требуют проверки. Неуверенные строки сохранены отдельно; размеры — в пикселях рабочей копии.'})
    if reviewed:
        page['issues'].append({'code':'prepared_review_dimensions','severity':'warning',
            'message':'Размер из демонстрационного примера подготовлен ассистентом по исходным опорам; это не автоматическое распознавание и не подтверждённые миллиметры.'})
    write_json(output/f'{prefix}-demo-export.json',exported); write_json(output/f'{prefix}-report.json',page)
    for record in inputs.values(): checked(ROOT,record)
    return {'native_upgrade_applied':True,'review_example_applied':review_matched}


def run_profile(source,output,options,*,models=None,expected_binding=None):
    options = ProcessingOptions.model_validate(options)
    binding = validate_binding(options.profile,expected_binding) if expected_binding is not None else resolve_profile(options.profile)
    if options.profile not in NATIVE_PROFILES:
        return previous.run_profile(source,output,options.model_dump(),models=models,expected_binding=binding)
    started = time.perf_counter(); output = Path(output).resolve(); original_sha = digest(Path(source))
    legacy = {**options.model_dump(),'profile':'experimental_draft'}
    report = previous.run_profile(source,output,legacy,models=models)
    pages = [upgrade_page(output,page,binding) for page in report['pages']]
    if digest(Path(source))!=original_sha: raise ProfileError('input_integrity','Input changed')
    report.update(processing_profile=binding,processing_options=options.model_dump(),
        native_cad_integration={'revision':'4.0','pages':pages},
        processing_execution={'interface_contract':'CLI_and_worker_share_run_profile/4.0',
            'elapsed_seconds':round(time.perf_counter()-started,4),'source_name':Path(source).name},
        limitations=['Демонстрационный CAD-экспорт; неполное распознавание и спорные участки требуют проверки.',
            'TEXT_DRAFT видим; TEXT_REVIEW сохранён выключенным вместе с исходными штрихами.',
            'Размеры unitless в px; подготовленные примеры явно отделены от автоматических связей.',
            'Крупный растр может быть уменьшен; полный исходник сохранён. Векторный PDF извлекается напрямую.',
            'Правки OCR в разделе «Ваши правки» меняют метаданные; нативный текст текущего DXF редактируется в CAD-редакторе.'])
    report['parameters'].update(profile=options.profile,profile_version='4.0',raster_export='native_CAD_objects_and_annotations')
    from .profile_pipeline_v1 import validate_result
    validate_result(output,report,binding)
    write_json(output/'profile-contract.json',binding); write_json(output/'report.json',report)
    write_json(output/'profile-artifacts.json',{'schema_version':'profile-artifacts/4.0','profile_sha256':binding['sha256'],
        'source_sha256':original_sha,'files':[{'name':p.name,'sha256':digest(p),'size_bytes':p.stat().st_size}
            for p in sorted(output.iterdir()) if p.is_file() and p.suffix in ('.png','.json','.dxf')]})
    return report
