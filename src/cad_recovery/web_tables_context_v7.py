"""Validate the stored v6/v7/v8 parent; retain pixel-centre registration rules."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset_layout_v1 import read, digest, encoded
from .demo_input_v1 import _orientation_matrix
from .web_tables_contract_v7 import binding
from .table_registration_v1 import pdf_geometry, orthogonal_view


def validate_parent_profile(report, manifest, document, run):
    from .profile_contract_v9 import validate_binding, ProcessingOptions
    profile = report.get('processing_profile')
    if not isinstance(profile, dict) or profile.get('schema_version') not in (
            'processing-profile/6.0','processing-profile/7.0','processing-profile/8.0','processing-profile/9.0'):
        raise ValueError('Unsupported parent processing profile')
    if str(document['id']) != str(run['job_id']):
        raise ValueError('Parent document differs')
    saved = dict(run['options'])
    if saved.pop('_profile_binding', None) != profile:
        raise ValueError('Stored parent binding differs from report')
    has_review = '_table_review' in saved
    review = saved.pop('_table_review', None)
    if has_review:
        if (not isinstance(review,dict) or set(review) !=
                {'id','parent_version','page','reviewer','automatic_quality_credit'}
                or review.get('automatic_quality_credit') is not False
                or type(review.get('parent_version')) is not int
                or type(review.get('page')) is not int or not 0 <= review['page'] < 20
                or review['parent_version'] != run['version']-1):
            raise ValueError('Published table parent provenance differs')
        pages = [p for p in report['pages'] if p['page_index']==review['page']]
        detail = pages[0].get('table_refinement',{}) if len(pages)==1 else {}
        if (detail.get('proposal_id') != review['id'] or detail.get('reviewer') != review['reviewer']
                or detail.get('automatic_quality_credit') is not False):
            raise ValueError('Published table report provenance differs')
    parsed = ProcessingOptions.model_validate(saved).model_dump()
    if parsed != report.get('processing_options') or parsed['profile'] != profile.get('name'):
        raise ValueError('Stored parent processing options differ')
    summary = manifest.get('summary')
    if (not isinstance(summary, dict) or summary != run.get('summary')
            or summary.get('processing_profile') != profile
            or summary.get('processing_options') != parsed):
        raise ValueError('Sealed and SQL parent summaries differ')
    validate_binding(profile['name'], profile)
    return profile


def describe(files, document, run, page_index):
    if type(page_index) is not int or page_index < 0:
        raise ValueError('Nonnegative integer page required')
    manifest_path, manifest = files.manifest(run)
    folder = manifest_path.parent/'output'
    report = read(folder/'report.json')
    profile = validate_parent_profile(report, manifest, document, run)
    pages = [p for p in report['pages'] if p['page_index'] == page_index]
    if len(pages) != 1: raise ValueError('Page not found')
    page = pages[0]; prefix = f'page-{page_index+1:03d}'
    source = files.path(document['input_path'])
    if digest(source) != document['input_sha256'] or report['source_sha256'] != document['input_sha256']:
        raise ValueError('Original source changed')
    gray = folder/'private'/prefix/'experimental/gray.png'
    if not gray.exists(): gray = folder/f'{prefix}-processed.png'
    data = dict(schema_version='web-table-context/7.0', job_id=str(run['job_id']), version=run['version'],
        page_index=page_index, source_manifest_sha256=run['manifest']['sha256'],
        source_sha256=document['input_sha256'], binding_sha256=binding()['sha256'],
        enabled=False, reason=None, working_size_px=None, native_size_px=None,
        outside_table_native_resolution=False, automatic_routing=False)
    if not gray.exists():
        data['reason'] = 'В этой версии нет растрового листа для разметки таблицы.'
        return _sealed(data), None
    with Image.open(gray) as image: size = list(image.size)
    data.update(working_size_px=size, working_image_sha256=digest(gray))
    exp = page.get('demo_export', {})
    dxf = folder/f'{prefix}.dxf'
    if not dxf.exists() or digest(dxf) != exp.get('dxf_sha256'):
        raise ValueError('Baseline DXF changed')
    data['baseline_dxf_sha256'] = digest(dxf)
    data.update(parent_profile_sha256=profile['sha256'],
        parent_profile_schema=profile['schema_version'],parent_report_sha256=digest(folder/'report.json'),
        parent_options_sha256=hashlib.sha256(encoded(run['options'])).hexdigest())
    common = report.get('common_input', {})
    experimental = page.get('experimental_cv', {})
    try:
        if source.suffix.lower() == '.pdf':
            geometry = pdf_geometry(source, page_index)
            graph = read(folder/f'{prefix}-graph.json')
            if (graph['source']['page_index'] != page_index or graph['source']['sha256'] != document['input_sha256']
                    or graph['provenance']['pdf_rotation_deg'] != geometry['rotation_deg']
                    or not np.allclose(graph['provenance']['pdf_bbox_pt'], geometry['pdfium_bbox_pt'], atol=1e-4, rtol=0)
                    or not np.allclose([graph['page']['width_mm'], graph['page']['height_mm']],
                        np.asarray(geometry['visible_size_pt'])*25.4/72, atol=1e-4, rtol=0)):
                raise ValueError('PDF source page and baseline frame disagree')
            source_preview = folder/f'{prefix}-source.png'
            with Image.open(source_preview) as preview: preview_size = list(preview.size)
            prep = page['preprocessing']
            if preview_size != prep['input_size_px'] or digest(source_preview) != prep['source_sha256']:
                raise ValueError('PDF baseline source preview changed')
            width_pt, height_pt = geometry['visible_size_pt']
            input_to_paper = np.asarray(prep['page_mapping']['input_pixel_to_paper_mm'])
            expected = [[width_pt*25.4/72/preview_size[0], 0, width_pt*25.4/72/preview_size[0]/2],
                [0, -height_pt*25.4/72/preview_size[1], height_pt*25.4/72-height_pt*25.4/72/preview_size[1]/2], [0, 0, 1]]
            if not np.allclose(input_to_paper, expected, atol=1e-8, rtol=0): raise ValueError('PDF paper/preview frame differs')
            display_to_paper = np.array([[25.4/72, 0, 0], [0, -25.4/72, height_pt*25.4/72], [0, 0, 1]])
            native_to_paper = display_to_paper @ np.linalg.inv(geometry['pdf_display_points_to_native'])
            matrix = np.asarray(prep['transforms']['input_to_processed']) @ np.linalg.inv(input_to_paper) @ native_to_paper
            native_size = geometry['size_px']
            data['pdf_registration'] = geometry
        else:
            raw_size = common.get('original_size_px')
            orientation = common.get('exif_orientation', 1)
            if not raw_size: raise ValueError('Native raster registration missing')
            native_size = raw_size[::-1] if orientation >= 5 else raw_size
            matrix = np.array(page['preprocessing']['transforms']['input_to_processed']) @ np.array(common['source_pixel_to_analysis_px'])
            matrix = matrix @ np.linalg.inv(np.asarray(_orientation_matrix(orientation, *raw_size)))
        matrix = np.asarray(experimental.get('input_to_working', np.eye(3))) @ matrix
        data.update(source_native_size_px=native_size, **orthogonal_view(matrix, native_size))
    except (ValueError, KeyError) as error:
        data['reason'] = 'Регистрация страницы не подтверждена; прежний DXF сохранён. '+str(error)
        return _sealed(data), gray
    export_size = exp.get('working_size_px', exp.get('source_size_px'))
    if export_size != size: raise ValueError('Preview and CAD frames differ')
    data.update(enabled=True,
        baseline_core_search_complete=bool(experimental.get('search_complete', exp.get('mode') == 'raster_ink_draft')))
    return _sealed(data), gray


def _sealed(data):
    return {**data, 'context_sha256': hashlib.sha256(encoded(data)).hexdigest()}


def validate_edit(context, edit):
    if not context['enabled']: raise ValueError(context['reason'])
    for key in ('context_sha256', 'source_manifest_sha256', 'source_sha256', 'binding_sha256'):
        if getattr(edit, key) != context[key]: raise ValueError('Документ, версия или регистрация изменились. Обновите лист.')
    w, h = context['working_size_px']
    for r in edit.regions:
        l, t, rr, b = r.box_px
        if not 0 <= l < rr <= w or not 0 <= t < b <= h: raise ValueError('Область выходит за лист.')
