"""Small public projection of owned attempt diagnostics; no paths, PIDs or raw errors."""
from pathlib import Path
from .dataset_layout_v1 import read

PHASES={'document_conversion':'Конвертация документа','table_build':'Распознавание таблиц',
 'table_structure_OCR':'Сетка и текст таблицы','table_text_choice':'Подготовка выбранного текста',
 'window_cv':'Геометрия','window_ocr':'Чтение текста','window_semantic_CAD':'Сборка объектов',
 'source_prepare':'Подготовка исходника','page_context':'Анализ листа','page_stitch_export':'Сборка DXF'}
CAUSES={'processing_timeout':'Обработка остановлена по времени','memory_limit':'Достигнут лимит памяти',
 'output_limit':'Достигнут лимит результата','budget_monitor_failed':'Недоступен контроль ресурсов',
 'lease_lost_or_cancelled':'Обработка отменена или передана другому исполнителю',
 'processing_incomplete':'Обработка не завершилась','processing_failed':'Ошибка обработки',
 'table_processing_failed':'Не удалось подготовить таблицу'}


def public(value):
    progress=value.get('progress') or {};active=progress.get('active_phases',[])
    name=active[-1].get('name') if active else (progress.get('last_completed') or {}).get('name')
    cause=(value.get('first_cause') or {}).get('code')
    context=next((r.get('context',{}) for r in reversed(active) if 'window' in r.get('context',{})),{})
    return dict(available=True,outcome=value.get('outcome'),
        reason=CAUSES.get(cause,'Обработка остановлена') if cause else None,
        phase=PHASES.get(name,'Обработка') if name else None,
        window=context.get('window') if type(context.get('window')) is int else None,
        elapsed_seconds=value.get('walltime_seconds'),peak_memory_MiB=value.get('peak_rss_MiB'),
        memory_is_observed_estimate=True,progress_available=bool(progress))


def latest(files,relative):
    base=files.path(relative)
    paths=[]
    if base.exists():
        for folder in base.iterdir():
            if folder.is_dir() and not folder.is_symlink() and not getattr(folder,'is_junction',lambda:False)():
                path=files.path((folder/'execution-budget.json').relative_to(files.root).as_posix())
                if path.exists():paths.append(path)
    if not paths:return dict(available=False,phase=None,reason=None,progress_available=False)
    path=max(paths,key=lambda p:p.stat().st_mtime_ns)
    try:
        if path.stat().st_size>1024**2:raise ValueError('Oversized diagnostic')
        return public(read(path))
    except (OSError,ValueError):return dict(available=False,phase=None,reason=None,progress_available=False)
