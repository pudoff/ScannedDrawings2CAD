"""Cumulative choices on sealed evidence; independent of table geometry."""
from copy import deepcopy
from pathlib import Path
from .dataset_layout_v1 import read, encoded, digest
from .web_text_selection_v1 import evidence, packet as old_packet


def state_path(output, page):
    return Path(output)/f'page-{page+1:03d}-text-publication.json'


def initial_state(document, parent, page, baseline):
    return dict(schema_version='web-text-publication/1.0',
        basis=dict(job_id=str(parent['job_id']), version=parent['version'], page_index=page,
                   source_sha256=document['input_sha256'], baseline_sha256=digest(baseline)),
        selections=[], associations=[], reviewers=[], handle_mapping=[])


def load_state(output, page, job_id, source_sha256):
    state = read(state_path(output, page))
    if (state['schema_version'] != 'web-text-publication/1.0'
            or state['basis']['job_id'] != str(job_id) or state['basis']['page_index'] != page
            or state['basis']['source_sha256'] != source_sha256
            or digest(evidence(output, page)/'baseline.dxf') != state['basis']['baseline_sha256']):
        raise ValueError('Text evidence belongs to another source/document/page')
    return state


def merge(state, selection):
    value = deepcopy(state)
    cells = {(r['region_id'], r['cell_id']): r for r in state['selections']}
    associations = {r['handle']: r for r in state['associations']}
    for row in selection.get('clear_cells', []):
        key = (row['region_id'], row['cell_id'])
        if key not in cells:
            raise ValueError('Only a previously selected cell can be cleared')
        del cells[key]
    for handle in selection.get('clear_associations', []):
        if handle not in associations:
            raise ValueError('Only a previously associated text can be cleared')
        del associations[handle]
    for row in selection['selections']:
        cells[(row['region_id'], row['cell_id'])] = deepcopy(row)
    for row in selection['associations']:
        associations[row['handle']] = deepcopy(row)
    if len(cells) > 128 or len(associations) > 64:
        raise ValueError('Cumulative choice limit exceeded')
    value['selections'] = [cells[k] for k in sorted(cells)]
    value['associations'] = [associations[k] for k in sorted(associations)]
    return value


def packet(output, page, offset=0, limit=40):
    value = old_packet(output, page, offset, limit)
    path = state_path(output, page)
    if not value.get('enabled') or not path.exists():
        return dict(enabled=False, reason='Подготовьте новый просмотр таблицы для отдельного выбора текста.')
    state = read(path)
    value.update(previous_choice={k: state[k] for k in ('selections', 'associations')},
                 previous_reviewer=state['reviewers'][-1] if state['reviewers'] else '',
                 text_only=True, publication_state_sha256=digest(path))
    return value
