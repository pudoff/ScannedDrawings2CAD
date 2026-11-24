"""Public table execution preparation shared with future service adapters."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TableLimits:
    timeout_seconds: float
    max_output_bytes: int = 1536*1024**2
    max_output_files: int = 12000


@dataclass(frozen=True)
class TableExecution:
    request: dict
    limits: TableLimits
    memory_MiB: int


def prepare_table_execution(row: dict, parent: dict, document: dict, *, files, store,
                            factories: tuple[Callable[[], dict], ...], models: Path) -> TableExecution:
    if parent['manifest']['sha256'] != row['request']['source_manifest_sha256']:
        raise ValueError('Parent manifest changed')
    bindings = [factory() for factory in factories]
    selected = next((b for b in bindings if b['sha256'] == row['request']['binding_sha256']), None)
    if selected is None:
        raise ValueError('Unsupported table extension')
    request = dict(storage=str(files.root),
        parent={k: str(parent[k]) if k in ('id', 'job_id') else parent[k]
                for k in ('id', 'job_id', 'version', 'status', 'options', 'manifest', 'summary')},
        document={k: str(document[k]) if k == 'id' else document[k]
                  for k in ('id', 'input_path', 'input_sha256')},
        page=row['page_index'], edit=row['request'], table_binding=selected,
        proposal_id=str(row['id']), models=str(models))
    choice = row['request'].get('text_selection')
    if choice:
        if choice.get('source_kind') == 'published':
            if (choice['proposal_id'] != str(parent['id'])
                    or choice['preview_sha256'] != parent['manifest']['sha256']):
                raise ValueError('Published text version changed')
            source_manifest, source_summary = parent['manifest'], parent['summary']
        else:
            source = store.get(row['job_id'], row['parent_version'], row['page_index'], choice['proposal_id'])
            if source['status'] != 'prepared' or source['preview_manifest']['sha256'] != choice['preview_sha256']:
                raise ValueError('Text source proposal changed')
            source_manifest, source_summary = source['preview_manifest'], source['summary']
        request.update(text_source_manifest=source_manifest, text_source_summary=source_summary)
    text_only = selected['sha256'] == bindings[-1]['sha256']
    return TableExecution(request, TableLimits(300. if text_only else 900.), 3072 if text_only else 6144)
