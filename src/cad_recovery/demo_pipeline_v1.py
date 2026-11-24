"""Presentation profile: reuse measured CV/OCR, export an explicit review draft."""
from pathlib import Path
import math
import time

from .pipeline import run_pipeline, write_json
from .schema import DrawingGraph


def run_demo_pipeline(source, output, *, ocr_models, render_dpi=120., image_dpi=None, rotation_deg=0):
    from .arrow_detection import detect_arrows
    from .dimension_association_v3 import associate_dimensions_v3
    from .demo_export_v1 import export_review_dxf
    from .demo_input_v1 import prepare_demo_input

    tick = time.perf_counter()
    output = Path(output)
    analysis_source, adaptation = prepare_demo_input(source, output.with_name(output.name + '-input'))
    effective_dpi = image_dpi
    if adaptation['resized'] and image_dpi is not None:
        effective_dpi = image_dpi * math.sqrt(math.prod(adaptation['scale_xy']))
        if effective_dpi < 36:
            effective_dpi = None  # Native intake will explicitly mark uncalibrated paper coordinates.
    adaptation.update(requested_image_dpi=image_dpi, effective_image_dpi=effective_dpi)
    report = run_pipeline(analysis_source, output, export=False, render_dpi=render_dpi,
        image_dpi=effective_dpi, prepare=True, detect=True, ocr=True,
        ocr_models=Path(ocr_models), associate=False, rotation_deg=rotation_deg)
    try:
        report.update(source_sha256=adaptation['sha256'], analysis_source_sha256=adaptation['analysis_sha256'],
                      input_adaptation=adaptation)
        write_json(output / 'input-adaptation.json', adaptation)
        for page in report['pages']:
            prefix = f"page-{page['page_index'] + 1:03d}"
            gray = output / page['preprocessing']['artifacts']['grayscale']
            binary = output / page['preprocessing']['artifacts']['binary']
            geometry, ocr = page['geometry_detection'], page['ocr']
            arrows = detect_arrows(gray, output, prefix=prefix, geometry=geometry,
                text_polygons=[line['polygon_px'] for line in ocr['lines']])
            associations = associate_dimensions_v3(gray, output, prefix=prefix,
                geometry=geometry, ocr=ocr, arrows=arrows)
            page.update(arrows=arrows, dimension_associations=associations)
            page['artifacts'].update(arrow_report=arrows['artifacts']['report'],
                arrow_preview=arrows['artifacts']['preview'],
                dimension_associations=associations['artifacts']['report'],
                dimension_associations_preview=associations['artifacts']['preview'])
            graph = DrawingGraph.model_validate_json((output / f'{prefix}-graph.json').read_text('utf-8'))
            exported = export_review_dxf(graph, binary, output / f'{prefix}.dxf', geometry=geometry,
                ocr=ocr, rotation_deg=rotation_deg)
            exported.update(input_resized=adaptation['resized'],
                source_preview_resized=adaptation['resized'], original_size_px=adaptation['original_size_px'],
                analysis_size_px=adaptation['analysis_size_px'])
            page['demo_export'] = exported
            page['artifacts'].update(dxf=exported['dxf'], dxf_preview=exported['preview'],
                demo_export=f'{prefix}-demo-export.json')
            page['issues'] = [i for i in page['issues'] if i['code'] not in ('raster_unprocessed', 'preprocessing_only')]
            seen = {(i['code'], i['message']) for i in page['issues']}
            for stage in (geometry, ocr, arrows, associations):
                for item in stage.get('issues', []):
                    if (item['code'], item['message']) not in seen:
                        page['issues'].append({'code': item['code'], 'message': item['message'],
                                               'severity': item.get('severity', 'warning')})
                        seen.add((item['code'], item['message']))
            page['issues'].append({'code': 'demo_draft_unverified', 'severity': 'warning',
                'message': 'Демонстрационный DXF: штрихи или PDF-объекты. Физические размеры и семантика требуют проверки.'})
            if adaptation['resized']:
                original, reduced = adaptation['original_size_px'], adaptation['analysis_size_px']
                page['issues'].append({'code': 'demo_input_reduced', 'severity': 'warning',
                    'message': f'Для показа создана рабочая копия {original[0]}×{original[1]} → '
                               f'{reduced[0]}×{reduced[1]} px. Оригинал сохранён; тонкие детали могут быть потеряны.'})
            if exported['empty']:
                page['issues'].append({'code': 'empty_draft', 'severity': 'warning',
                    'message': 'На странице не найдены штрихи для DXF. Геометрия не восстановлена.'})
            page['status'] = 'needs_review'
            graph.provenance['demo_profile'] = {
                'version': 'presentation_draft/1.0', 'export_report': f'{prefix}-demo-export.json',
                'association_report': associations['artifacts']['report'],
                'association_algorithm': associations['algorithm_version'],
                'acceptance': 'observations_only; visible draft is separate from accepted graph geometry'}
            graph.provenance['demo_input_adaptation'] = adaptation
            (output / f'{prefix}-graph.json').write_text(graph.model_dump_json(indent=2), encoding='utf-8')
            write_json(output / f'{prefix}-demo-export.json', exported)
            write_json(output / f'{prefix}-report.json', page)
        report.update(status='needs_review', elapsed_seconds=round(time.perf_counter() - tick, 4))
        report['parameters'].update(profile='presentation_draft', profile_version='1.0',
            export_dxf=True, associate_dimensions=True, association_algorithm='contour-association-v3',
            raster_export='binary-mask-native-solid', physical_geometry_approved=False)
        report['limitations'] = [
            'Демонстрационный черновик: для сканов сохранены штрихи бинарной маски, для векторных PDF — извлечённые объекты.',
            'CV/OCR и размерные связи являются наблюдениями; точный масштаб детали и физические размеры не подтверждены.',
            'Предпросмотр отрисован из сохранённого DXF. Проверка в Autodesk TrueView выполняется отдельно.',
        ]
        write_json(output / 'report.json', report)
        return report
    except Exception as exc:
        write_json(output / 'report.json', {'status': 'failed', 'error': str(exc),
            'elapsed_seconds': round(time.perf_counter() - tick, 4), 'profile': 'presentation_draft'})
        raise
