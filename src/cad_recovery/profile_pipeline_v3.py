"""Shared CLI/worker demo pipeline, with current experiments in downloaded DXF."""
from pathlib import Path
import time
from copy import deepcopy
import numpy as np
from PIL import Image, ImageDraw
from .profile_contract_v3 import ProcessingOptions, EXPERIMENTAL, resolve_profile, validate_binding, verify_models, ProfileError
from .dataset_layout_v1 import digest
from .pipeline import write_json


def _working_ocr(ocr, summary):
    result = deepcopy(ocr)
    matrix = np.asarray(summary['input_to_working'])
    for line in result.get('lines', []):
        p = np.asarray(line['polygon_px'])
        line['polygon_px'] = (p @ matrix[:2,:2].T + matrix[:2,2]).tolist()
    return result


def run_profile(source, output, options, *, models=None, expected_binding=None):
    options = ProcessingOptions.model_validate(options)
    binding = validate_binding(options.profile, expected_binding) if expected_binding is not None else resolve_profile(options.profile)
    if options.profile != EXPERIMENTAL:
        from .profile_pipeline_v2 import run_profile as previous
        return previous(source, output, options.model_dump(), models=models, expected_binding=binding)
    from .demo_pipeline_v1 import run_demo_pipeline
    from .experimental_cv_v1 import analyze
    from .experimental_export_v1 import export
    from .profile_pipeline_v1 import validate_result
    tick = time.perf_counter()
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ProfileError('output_exists', 'Choose a new output directory')
    source_hash = digest(source)
    model_root = verify_models(binding, models)
    report = run_demo_pipeline(source, output, ocr_models=model_root,
        render_dpi=options.render_dpi, image_dpi=options.image_dpi, rotation_deg=options.rotation_deg)
    for page in report['pages']:
        prefix = f"page-{page['page_index']+1:03d}"
        # Native vector PDF extraction is preferable to rasterizing its CAD paths.
        if page['demo_export']['mode'] == 'pdf_vector_draft':
            page['experimental_cv'] = {'branch':'direct_vector_pdf', 'raster_revectorization':False}
            continue
        gray = output/page['preprocessing']['artifacts']['grayscale']
        try:
            packet = analyze(gray, output/'private'/prefix/'experimental')
        except ValueError as error:
            # Bounded research producers may abstain on unfamiliar/large sheets.
            # Keep the already produced faithful draft and make the fallback explicit.
            fallback = {'version':'experimental-cv-demo/1.0','fallback':'presentation_draft',
                        'reason':str(error)[:500],'physical_geometry_approved':False}
            page['experimental_cv'] = fallback
            page['demo_export']['experimental_fallback'] = fallback
            page['issues'].append({'code':'experimental_cv_fallback','severity':'warning',
                'message':'Экспериментальный CV отказался от листа. Сохранён штриховой DXF предыдущего профиля; причина доступна в отчёте.'})
            page['artifacts']['experimental_cv'] = f'{prefix}-experimental-cv.json'
            write_json(output/f'{prefix}-experimental-cv.json', fallback)
            write_json(output/f'{prefix}-demo-export.json', page['demo_export'])
            write_json(output/f'{prefix}-report.json', page)
            continue
        old = page['demo_export']
        archive = output/'private'/prefix/'legacy'
        archive.mkdir(parents=True)
        (output/old['dxf']).replace(archive/old['dxf'])
        (output/old['preview']).replace(archive/old['preview'])
        exported = export(packet, output/old['dxf'], ocr=_working_ocr(page['ocr'], packet['summary']))
        adapted = report.get('input_adaptation', {})
        exported.update(input_resized=adapted.get('resized', False) or packet['summary']['resized'],
            original_size_px=adapted.get('original_size_px') or packet['summary']['input_size_px'],
            analysis_size_px=packet['summary']['size_px'],
            requested_rotation_clockwise_deg=options.rotation_deg)
        page['demo_export'] = exported
        page['experimental_cv'] = packet['summary']
        page['artifacts'].update(experimental_cv=f'{prefix}-experimental-cv.json')
        write_json(output/f'{prefix}-experimental-cv.json', packet['summary'])
        write_json(output/f'{prefix}-experimental-geometry.json', {
            'coordinate_system':'experimental_working_raster_pixel_centres',
            'summary':packet['summary'], 'candidates':packet['geometry'], 'heads':packet['heads']})
        Image.fromarray(packet['roles']).save(output/f'{prefix}-cv-roles.png')
        from .curve_graph_v1 import sample
        preview=Image.fromarray(packet['gray']).convert('RGB')
        draw=ImageDraw.Draw(preview)
        for candidate in packet['geometry']:
            points=sample(candidate)
            if len(points)>1:
                draw.line([tuple(p) for p in points],fill='#168d90',width=1)
        preview.save(output/f'{prefix}-cv-preview.png')
        page['artifacts'].update(experimental_geometry=f'{prefix}-experimental-geometry.json',
            experimental_preview=f'{prefix}-cv-preview.png',experimental_roles=f'{prefix}-cv-roles.png')
        write_json(output/f'{prefix}-demo-export.json', exported)
        page['issues'].append({'code':'experimental_cv_demo', 'severity':'warning',
            'message':'Экспериментальный CV: геометрия и роли требуют проверки. Неуверенные штрихи сохранены отдельно; физический масштаб не подтверждён.'})
        if not packet['summary']['search_complete']:
            page['issues'].append({'code':'experimental_search_incomplete','severity':'warning',
                'message':'Поиск CV ограничен бюджетом или неоднозначностью. Сохранён доступный результат и нераспознанные штрихи.'})
        write_json(output/f'{prefix}-report.json', page)
    if digest(source) != source_hash:
        raise ProfileError('input_integrity', 'Input changed during processing')
    validate_result(output, report, binding)
    report.update(processing_profile=binding, processing_options=options.model_dump(),
        processing_execution={'interface_contract':'CLI_and_worker_share_run_profile/3.0',
            'elapsed_seconds':round(time.perf_counter()-tick,4), 'source_name':source.name},
        limitations=[
            'Демонстрационный экспериментальный профиль; общий допуск качества и ручная проверка TrueView открыты.',
            'CV-слои и геометрия — автоматические предложения. Размеры детали, масштаб, DIMENSION/HATCH не подтверждены.',
            'Нераспознанные штрихи сохранены отдельно; исходная штриховая подложка доступна в выключенном SOURCE_INK.',
            'OCR остаётся экспериментальным; распознанные TEXT находятся на выключенном OCR_OBSERVATIONS.',
            'Крупный растр может быть уменьшен до рабочей копии; оригинальный загруженный файл сохранён.',
            'Векторный PDF использует прямое извлечение объектов вместо повторной векторизации растра.'])
    report['parameters'].update(profile=EXPERIMENTAL, profile_version='3.0', raster_export='experimental_layered_native_geometry')
    write_json(output/'profile-contract.json', binding)
    write_json(output/'report.json', report)
    write_json(output/'profile-artifacts.json', {'schema_version':'profile-artifacts/3.0',
        'profile_sha256':binding['sha256'], 'source_sha256':source_hash,
        'files':[{'name':p.name,'sha256':digest(p),'size_bytes':p.stat().st_size}
                 for p in sorted(output.iterdir()) if p.is_file() and p.suffix in ('.png','.json','.dxf')]})
    return report
