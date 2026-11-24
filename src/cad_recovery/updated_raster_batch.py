"""Prepare a pinned update of RGB and strictly opaque RGBA rasters."""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import time
import zlib

from .clear_pairs import contained, digest, read_json
from .large_raster import prepare_large_png
from .object_review_case import _pinned, _write_json
from .opaque_rgba import prepare_opaque_rgba


def _header(path):
    with path.open('rb') as stream:
        raw = stream.read(33)
    if (len(raw) != 33 or raw[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
            or zlib.crc32(raw[12:29]) & 0xffffffff != struct.unpack('>I', raw[29:33])[0]):
        raise ValueError('Invalid PNG signature or IHDR CRC')
    width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', raw[16:29])
    if (not 0 < width <= 32768 or not 0 < height <= 32768 or width*height > 300_000_000
            or depth != 8 or color not in (2, 6) or compression or filtering or interlace):
        raise ValueError('Unsupported PNG format or source dimensions')
    return width, height, color


def _implementation():
    names = ['updated_raster_batch.py', 'opaque_rgba.py', 'large_raster.py', 'png_stream.py']
    return {name: digest(Path(__file__).with_name(name)) for name in names}


def _peak_memory():
    result = {'peak_process_working_set_bytes': None, 'measurement_method': 'unavailable',
              'measurement_scope': 'current_process_lifetime_including_imports_and_all_pages'}
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        *[(n, ctypes.c_size_t) for n in ['PeakWorkingSetSize', 'WorkingSetSize',
                          'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                          'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]]
        kernel, psapi = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            result.update(peak_process_working_set_bytes=int(counters.PeakWorkingSetSize),
                          measurement_method='Windows_GetProcessMemoryInfo_PeakWorkingSetSize')
    elif platform.system() == 'Linux':
        import resource
        result.update(peak_process_working_set_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      measurement_method='Linux_getrusage_ru_maxrss_KiB_to_bytes')
    return result


def _checkpoint(output, report):
    temporary = output/'.report-writing.json'
    _write_json(temporary, report)
    temporary.replace(output/'report.json')


def _finish_index(output, report):
    cards, artifacts = [], {}
    for item in report['items']:
        name = html.escape(item['input_path'])
        if item['preparation_report']:
            relative = item['preparation_report']
            artifacts[relative] = item['preparation_report_sha256']
            metadata = read_json(output/relative)
            if item['status'] == 'prepared':
                picture = (Path(relative).parent/metadata['overview']['path']).as_posix()
                cards.append(f'<section><p>{name}</p><a href="{relative}">Происхождение и фрагменты</a>'
                             f'<a href="{picture}"><img loading="lazy" src="{picture}" alt="{name}"></a></section>')
                continue
        cards.append(f'<section><p>{name}</p><p>Не подготовлен: '+html.escape(str(item.get('error')))+ '</p></section>')
    (output/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>Подготовленные растры обновления</title><style>body{font:16px sans-serif;margin:2em}'
        'section{display:inline-block;vertical-align:top;width:24em;padding:1em}img{width:100%;display:block}</style>'
        '<h1>Подготовка добавленных чертежей</h1><p>Обзоры служат для навигации; точные пиксели хранятся '
        'во фрагментах исходного разрешения. Соответствие DXF и обучающие метки ещё не утверждены.</p>'
        '<p>Состояние: '+html.escape(report['status'])+'</p><a href="report.json">Отчёт</a> · '
        '<a href="config.json">Параметры запуска</a>'+''.join(cards)+'</html>', encoding='utf-8')
    artifacts.update({name: digest(output/name) for name in ['report.json', 'config.json', 'index.html']})
    _write_json(output/'manifest.json', {'schema_version': '1.0', 'status': report['status'],
                                       'artifact_sha256': artifacts, 'training_eligible': False})


def prepare_updated_dataset(workspace: Path, config_path: Path, output: Path, *, limit=None) -> dict:
    started = time.perf_counter()
    root, config_path, output = Path(workspace).resolve(), Path(config_path).resolve(), Path(output).resolve()
    if not config_path.is_relative_to(root) or not output.is_relative_to(root) or output == root:
        raise ValueError('Preparation paths must remain within workspace')
    if output.exists():
        raise FileExistsError('Choose a new preparation output directory')
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError('Diagnostic limit must be a positive integer')
    config = read_json(config_path)
    if config.get('schema_version') != '1.0':
        raise ValueError('Unknown update preparation schema')
    dataset = contained(root, config['dataset'])
    if output.is_relative_to(dataset):
        raise ValueError('Preparation output must be outside dataset')
    intake_path = contained(root, config['intake']['path'])
    update_path = contained(root, config['update_report']['path'])
    _pinned(intake_path, config['intake']['sha256'])
    _pinned(update_path, config['update_report']['sha256'])
    intake, update = read_json(intake_path), read_json(update_path)
    if (intake.get('schema_version') != '1.0' or intake.get('purpose') != 'training_intake_not_split'
            or intake.get('training_eligible') is not False or update.get('status') != 'registered'
            or update.get('training_eligible') is not False):
        raise ValueError('Expected registered unapproved update intake')
    items = intake['items']
    if (not isinstance(items, list) or not 1 <= len(items) <= 10000
            or len(items) != config['expected_count'] or len(items) != update['summary']['pending_rasters']):
        raise ValueError('Unexpected update source inventory')
    parameters = config['parameters']
    if set(parameters) != {'tile_size', 'overlap', 'overview_max_side', 'strip_rows'}:
        raise ValueError('Unexpected preparation parameters')
    for name, low, high in [('tile_size', 128, 2048), ('overlap', 0, 1024),
                            ('overview_max_side', 128, 4096), ('strip_rows', 1, 256)]:
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError('Invalid preparation parameter: '+name)
    if parameters['overlap'] > parameters['tile_size']//2:
        raise ValueError('Overlap exceeds half tile size')
    checked, names, sources = [], set(), set()
    for item in items:
        name = item['input_path']
        source = contained(dataset, name)
        if name in names or source in sources or source.suffix.lower() != '.png':
            raise ValueError('Duplicate or non-PNG input')
        names.add(name)
        sources.add(source)
        if item.get('training_eligible') is not False or item.get('split') is not None:
            raise ValueError('Preparation intake must not carry assigned training splits')
        _pinned(source, item['input_sha256'])
        if source.stat().st_size != item['input_size_bytes']:
            raise ValueError('Source size differs from intake')
        width, height, color = _header(source)
        header = item['png_ihdr']
        if [width, height, color] != [header['width_px'], header['height_px'], header['color_type']]:
            raise ValueError('Source header differs from intake')
        target = item.get('candidate_target')
        if target:
            _pinned(contained(dataset, target['path']), target['sha256'])
        checked.append((item, source, color))
    checked.sort(key=lambda r: r[0]['input_path'])
    selected = checked if limit is None else checked[:limit]
    code = _implementation()
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output/'config.json', config)
    report = {'schema_version': '1.0', 'date': config['date'], 'status': 'running',
              'purpose': 'raster_preparation_only', 'intake_sha256': config['intake']['sha256'],
              'update_report_sha256': config['update_report']['sha256'], 'config_sha256': digest(config_path),
              'implementation': code, 'parameters': {**parameters, 'limit': limit},
              'runtime': {'python': platform.python_version(), 'Pillow': importlib.metadata.version('Pillow'),
                          'numpy': importlib.metadata.version('numpy')},
              'counts': {'intake_total': len(items), 'selected': len(selected), 'prepared': 0, 'failed': 0,
                         'not_selected': len(items)-len(selected), 'rgb_prepared': 0, 'opaque_rgba_prepared': 0},
              'items': [], 'training_eligible': False, 'annotation_approved': False,
              'source_modified': False, 'split_modified': False,
              'limitations': ['Prepared pixels do not verify PNG/DXF correspondence or approve labels.',
                              'Opaque RGBA only: any non-255 alpha is refused, never composited.',
                              'Old preparation IDs and directories remain separate and unchanged.']}
    _checkpoint(output, report)
    for ordinal, (item, source, color) in enumerate(selected, 1):
        identity = hashlib.sha256(item['input_path'].encode('utf-8')).hexdigest()[:12]
        directory = f'source-{ordinal:04d}-{identity}'
        record = {'input_path': item['input_path'], 'input_sha256': item['input_sha256'],
                  'output_directory': directory, 'preparation_report': None, 'status': 'failed',
                  'input_color_type': color}
        try:
            _pinned(source, item['input_sha256'])
            prepare = prepare_large_png if color == 2 else prepare_opaque_rgba
            result = prepare(source, output/directory, expected_sha256=item['input_sha256'], **parameters)
            if result.get('status') != 'prepared' or result.get('source_sha256') != item['input_sha256']:
                raise ValueError('Preparation did not validate the original source')
            relative = f'{directory}/preparation.json'
            record.update(status='prepared', preparation_report=relative,
                          preparation_report_sha256=digest(output/relative), source_size_px=result['source_size_px'],
                          tile_count=len(result['tiles']))
            report['counts']['prepared'] += 1
            report['counts']['rgb_prepared' if color == 2 else 'opaque_rgba_prepared'] += 1
        except Exception as error:
            record['error'] = {'type': type(error).__name__, 'message': str(error).replace(str(dataset), 'dataset')}
            if (output/directory/'preparation.json').is_file():
                record['preparation_report'] = f'{directory}/preparation.json'
                record['preparation_report_sha256'] = digest(output/record['preparation_report'])
            report['counts']['failed'] += 1
        report['items'].append(record)
        report['elapsed_seconds'] = round(time.perf_counter()-started, 4)
        _checkpoint(output, report)
        print(json.dumps({'completed': ordinal, 'selected': len(selected), 'status': record['status'],
                          'input_path': item['input_path']}, ensure_ascii=True), flush=True)
    report['status'] = ('complete' if report['counts']['prepared'] == len(items) else
                        'failed' if report['counts']['prepared'] == 0 else 'partial')
    if _implementation() != code:
        report['status'] = 'failed'
        report['implementation_changed_during_run'] = True
    report['elapsed_seconds'] = round(time.perf_counter()-started, 4)
    try:
        report['resources'] = _peak_memory()
    except (OSError, AttributeError):
        report['resources'] = {'peak_process_working_set_bytes': None, 'measurement_method': 'unavailable'}
    _checkpoint(output, report)
    _finish_index(output, report)
    return report
