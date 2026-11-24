"""Read-only registration to the exact preview/CAD frame of a sealed version."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset_layout_v1 import read, digest, encoded
from .demo_input_v1 import _orientation_matrix
from .web_tables_contract_v1 import binding


def describe(files, document, run, page_index):
    manifest_path, manifest = files.manifest(run)
    folder = manifest_path.parent/'output'
    report = read(folder/'report.json')
    pages = [p for p in report['pages'] if p['page_index'] == page_index]
    if len(pages) != 1: raise ValueError('Page not found')
    page = pages[0]; prefix = f'page-{page_index+1:03d}'
    source = files.path(document['input_path'])
    if digest(source) != document['input_sha256'] or report['source_sha256'] != document['input_sha256']:
        raise ValueError('Original source changed')
    gray = folder/'private'/prefix/'experimental/gray.png'
    if not gray.exists(): gray = folder/f'{prefix}-processed.png'
    data = dict(schema_version='web-table-context/1.0', job_id=str(run['job_id']), version=run['version'],
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
    if source.suffix.lower() == '.pdf':
        data['reason'] = 'Таблицы PDF требуют отдельной проверенной регистрации. Текущий DXF сохранён; используйте PNG/JPEG для этого ручного режима.'
        return _sealed(data), gray
    from .profile_contract_v6 import resolve_profile
    profile = report['processing_profile']
    if profile != resolve_profile(profile['name']): raise ValueError('Parent component binding changed')
    common = report.get('common_input', {})
    raw_size = common.get('original_size_px')
    orientation = common.get('exif_orientation', 1)
    if not raw_size: raise ValueError('Native raster registration missing')
    native_size = raw_size[::-1] if orientation >= 5 else raw_size
    matrix = np.array(page['preprocessing']['transforms']['input_to_processed']) @ np.array(common['source_pixel_to_analysis_px'])
    experimental = page.get('experimental_cv', {})
    matrix = np.asarray(experimental.get('input_to_working', np.eye(3))) @ matrix
    matrix = matrix @ np.linalg.inv(np.asarray(_orientation_matrix(orientation, *raw_size)))
    export_size = exp.get('working_size_px', exp.get('source_size_px'))
    if export_size != size: raise ValueError('Preview and CAD frames differ')
    data.update(native_size_px=native_size, native_to_working=matrix.tolist(),
                baseline_core_search_complete=bool(experimental.get('search_complete', exp.get('mode') == 'raster_ink_draft')))
    if (not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0
            or abs(matrix[0, 1])+abs(matrix[1, 0]) > 1e-9):
        data['reason'] = 'Для этого поворота/перспективы табличная регистрация ещё не проверена. Текущий результат сохранён.'
    else: data['enabled'] = True
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
