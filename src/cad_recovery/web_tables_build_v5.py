"""Preserve current pages; publish only cumulative, explicitly selected text."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import shutil

import ezdxf
from .dataset_layout_v1 import read, encoded, digest
from .service_profiles_v9.storage import Storage
from .web_tables_contract_v5 import binding, parse_request
from .web_tables_context_v5 import describe, validate_edit
from .web_text_selection_v1 import evidence
from .web_text_publication_v1 import initial_state, load_state, merge, state_path
from .table_text_publish_v1 import make_request, publish, geometry_snapshot, verify_geometry
from .web_text_lineage_v1 import attach_lineage,validate_proposal


def write(path, value):
    Path(path).write_bytes(encoded(value))


def copy_output(files, manifest, target):
    path, value = files.manifest(dict(status='completed', manifest=manifest))
    target.mkdir()
    for item in value['inventory']:
        name = item['path']
        if name == 'output/evidence-inventory.json':
            continue
        if (not name.startswith('output/') or '\\' in name or ':' in name
                or any(part in ('', '.', '..') for part in name.split('/'))):
            raise ValueError('Invalid parent output inventory')
        source = files.path((path.parent/name).relative_to(files.root).as_posix())
        if digest(source) != item['sha256']:
            raise ValueError('Parent output changed')
        dest = target/Path(name).relative_to('output')
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)


def finalize(request, summary, state, publication):
    output = Path(request['folder'])/'output'; index = request['page']; prefix = f'page-{index+1:03d}'
    from .annotation_style_v1 import render_review_dxf
    render_review_dxf(output/f'{prefix}.dxf', output/f'{prefix}-dxf.png')
    report = read(output/'report.json'); page = next(p for p in report['pages'] if p['page_index'] == index)
    detail = dict(page.get('table_refinement', {}), schema_version='manual-table-text/2.0',
        proposal_id=request['proposal_id'], reviewer=request['edit']['reviewer'],
        extension_binding=request['table_binding'], text_review_available=True,
        mode='baseline_geometry_explicit_cumulative_text', candidate_geometry_applied=False,
        selected_texts=len(state['handle_mapping']), automatic_quality_credit=False,
        overlap_review_required=bool(state['handle_mapping']))
    page['table_refinement'] = detail; report['table_refinement'] = detail
    doc = ezdxf.readfile(output/f'{prefix}.dxf')
    page['demo_export'].update(dxf_sha256=digest(output/f'{prefix}.dxf'),
        preview_sha256=digest(output/f'{prefix}-dxf.png'), table_refinement=detail,
        modelspace_counts=dict(Counter(e.dxftype() for e in doc.modelspace())),
        layer_counts=dict(Counter(e.dxf.layer for e in doc.modelspace())))
    state['published_dxf_sha256'] = digest(output/f'{prefix}.dxf')
    write(state_path(output, index), state)
    write(output/f'{prefix}-text-choice.json', dict(choice={k: state[k] for k in ('selections', 'associations')},
        reviewer=request['edit']['reviewer'], source_preview=request.get('text_source_manifest'),
        parent_version=request['parent']['version'], automatic_quality_credit=False))
    write(output/f'{prefix}-text-publication-review.json', publication)
    write(output/f'{prefix}-demo-export.json', page['demo_export']); write(output/f'{prefix}-report.json', page)
    report['processing_execution'] = dict(parent=report.get('processing_execution'),
        interface_contract='baseline_geometry_web_text/1.0', extension_binding=request['table_binding'])
    write(output/'report.json', report)
    summary.update(has_changes=True, table_refinement=detail,
        pages=[{k: v for k, v in p.items() if k in ('page_index', 'status', 'issues', 'demo_export', 'table_refinement')}
               for p in report['pages']])
    write(output/'profile-artifacts.json', dict(schema_version='profile-artifacts/text-only/1.0',
        profile_sha256=report['processing_profile']['sha256'], extension_binding_sha256=request['table_binding']['sha256'],
        files=[dict(name=p.name, sha256=digest(p), size_bytes=p.stat().st_size) for p in sorted(output.iterdir())
               if p.is_file() and p.name != 'profile-artifacts.json' and p.suffix in ('.json', '.png', '.dxf')]))
    write(Path(request['folder'])/'completion.json', summary)
    return summary


def build(request, *, engine=None, progress=None):
    if request['table_binding'] != binding():
        raise ValueError('Text-only extension changed')
    edit = parse_request(request['edit']); page = request['page']; prefix = f'page-{page+1:03d}'
    files = Storage(SimpleNamespace(root=Path(request['storage']).resolve()))
    folder = Path(request['folder']).resolve(); files.path(folder.relative_to(files.root).as_posix())
    output = folder/'output'; document = request['document']; parent = request['parent']
    context, _ = describe(files, document, parent, page); validate_edit(context, edit)
    parent_path, _ = files.manifest(parent); current = parent_path.parent/'output'
    source = files.path(document['input_path'])
    if edit.text_selection is None:
        # Retain the qualified OCR/evidence producer, but its composed geometry
        # is never the preview/main CAD in this extension.
        from .web_tables_build_v3 import build as prepare
        from .web_tables_contract_v3 import binding as old_binding
        from .web_tables_context_v3 import describe as old_context
        old, _ = old_context(files, document, parent, page)
        delegated = deepcopy(request); delegated['table_binding'] = old_binding()
        for key in ('context_sha256', 'binding_sha256'):
            delegated['edit'][key] = old[key]
        summary = prepare(delegated, engine=engine, progress=progress)
        slot = evidence(output, page)
        if not (slot/'review.json').exists():
            return summary  # UNKNOWN remains unmodified and is not applicable.
        shutil.copyfile(current/f'{prefix}.dxf', output/f'{prefix}.dxf')
        shutil.copyfile(current/f'{prefix}.dxf', slot/'baseline.dxf')
        state = initial_state(document, parent, page, slot/'baseline.dxf')
        state.update(regions=[r.model_dump(mode='json') for r in edit.regions], page_kind=edit.page_kind)
        state=attach_lineage(state,request)
        # The old exporter aliases describe discarded geometry. All original
        # entities are actually retained in this text-only proposal.
        doc = ezdxf.readfile(output/f'{prefix}.dxf')
        write(output/f'{prefix}-table-aliases.json', dict(source_manifest_sha256=parent['manifest']['sha256'],
            aliases={e.dxf.handle: dict(status='retained', target_handles=[e.dxf.handle]) for e in doc.modelspace()},
            new_entities=[], old_observation_graph_is_parent_evidence=True))
        return finalize(request, summary, state, dict(mode='evidence_only', selected_native_texts=0,
            candidate_geometry_applied=False, new_CV_OCR_calls='table producer only'))

    selection = edit.text_selection.model_dump(mode='json')
    source_path, _ = files.manifest(dict(status='completed', manifest=request['text_source_manifest']))
    original = source_path.parent/'output'
    if request['text_source_manifest']['sha256'] != selection['preview_sha256']:
        raise ValueError('Text source manifest changed')
    state = load_state(original, page, parent['job_id'], document['input_sha256'])
    if digest(original/f'{prefix}.dxf') != state['published_dxf_sha256']:
        raise ValueError('Text source CAD changed')
    if digest(evidence(original, page)/'review.json') != selection['review_sha256']:
        raise ValueError('Text source review changed')
    if selection['source_kind'] == 'published' and request['text_source_manifest'] != parent['manifest']:
        raise ValueError('Published text must use the exact current version')
    copy_output(files, parent['manifest'], output)
    # The manifest and evidence remain immutable; only explicitly selected text
    # may differ from the exact parent in a chained proposal.
    if selection['source_kind'] == 'proposal':
        validate_proposal(state,parent,document,page,original/f'{prefix}.dxf',current/f'{prefix}.dxf')
        verify_geometry(geometry_snapshot(ezdxf.readfile(evidence(original,page)/'baseline.dxf')),
                        ezdxf.readfile(original/f'{prefix}.dxf'))
        shutil.copytree(evidence(original, page), evidence(output, page), dirs_exist_ok=True)
        for suffix in ('-tables.json', '-table-review.json'):
            shutil.copyfile(original/(prefix+suffix), output/(prefix+suffix))
    before_state = deepcopy(state); state = merge(state, selection)
    state['reviewers'].append(edit.reviewer)
    state=attach_lineage(state,request,previous_output=original)
    basis = evidence(output, page)/'baseline.dxf'
    verify_geometry(geometry_snapshot(ezdxf.readfile(basis)), ezdxf.readfile(current/f'{prefix}.dxf'))
    private = folder/'private'; private.mkdir(); target = private/'selected.dxf'
    if state['selections'] or state['associations']:
        choice = make_request(basis, evidence(output, page), source, document_id=str(parent['job_id']),
            version=state['basis']['version'], page_index=page, reviewer=edit.reviewer,
            selections=state['selections'], associations=state['associations'])
        publication = publish(basis, evidence(output, page), source, target, request=choice,
            document_id=str(parent['job_id']), version=state['basis']['version'], page_index=page)
        state['handle_mapping'] = publication['handle_mapping']
    else:
        shutil.copyfile(basis, target); state['handle_mapping'] = []
        publication = dict(mode='explicit_clear_all_choices', new_CV_OCR_calls=0,
                           candidate_geometry_applied=False)
    shutil.copyfile(target, output/f'{prefix}.dxf')
    # Alias previously published TEXT by its evidence identity; unrelated CAD
    # entities keep their original handles. Removed choices retire explicitly.
    after = {(r['origin'], r['evidence_handle']): r['published_handle'] for r in state['handle_mapping']}
    aliases = {e.dxf.handle: dict(status='retained', target_handles=[e.dxf.handle])
               for e in ezdxf.readfile(basis).modelspace()}
    for row in before_state['handle_mapping']:
        handle = row['published_handle']; mapped = after.get((row['origin'], row['evidence_handle']))
        if handle not in aliases:
            aliases[handle] = (dict(status='retained_with_text_decision', target_handles=[mapped]) if mapped else
                               dict(status='retired_by_explicit_text_choice', target_handles=[]))
    write(output/f'{prefix}-table-aliases.json', dict(source_manifest_sha256=parent['manifest']['sha256'],
        aliases=aliases, new_entities=state['handle_mapping'], old_observation_graph_is_parent_evidence=True))
    # Use source table metadata but every other page/file from the current parent.
    report = read(output/'report.json'); old = read(original/'report.json')
    current_page = next(p for p in report['pages'] if p['page_index'] == page)
    current_page['table_refinement'] = next(p for p in old['pages'] if p['page_index'] == page)['table_refinement']
    write(output/'report.json', report)
    summary = deepcopy(request['parent']['summary'])
    result = finalize(request, summary, state, publication)
    if digest(source) != document['input_sha256']:
        raise ValueError('Original source changed')
    return result
