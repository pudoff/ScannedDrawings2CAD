"""Private table candidate producer; only the queue can publish its sealed output."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import shutil

import ezdxf
import numpy as np

from .dataset_layout_v1 import read, encoded, digest
from .service_profiles_v9.storage import Storage
from .web_tables_contract_v2 import binding, parse_request
from .web_tables_context_v2 import describe, validate_edit
from .native_sheet_source_v1 import NativePage
from .table_registration_v1 import prepare_page, view_record
from .native_table_structure_v1 import analyze
from .native_table_export_v1 import export


def _write(path, value): path.write_bytes(encoded(value))


def build(request, *, engine=None):
    root = Path(request['storage']).resolve(); folder = Path(request['folder']).resolve()
    files = Storage(SimpleNamespace(root=root))
    if files.path(folder.relative_to(root).as_posix()) != folder: raise ValueError('Attempt outside storage')
    if binding() != request['table_binding']: raise ValueError('Table code binding changed')
    parent = request['parent']; document = request['document']; page_index = request['page']
    edit = parse_request(deepcopy(request['edit']))
    context, image = describe(files, document, parent, page_index); validate_edit(context, edit)
    parent_path, manifest = files.manifest(parent)
    baseline = parent_path.parent/'output'; source = files.path(document['input_path'])
    output = folder/'output'; private = folder/'private'
    output.mkdir(); private.mkdir()
    journal = dict(parent_manifest=parent['manifest'], context=context,
        explicit_review=edit.model_dump(mode='json'), table_binding=request['table_binding'],
        no_automatic_routing=True)
    # A complete copy of the parent version keeps all pages/history compatible.
    for item in manifest.get('inventory', manifest['files']):
        rel = Path(item['path'])
        if rel.as_posix() == 'output/evidence-inventory.json': continue
        if rel.parts[0] != 'output': raise ValueError('Parent inventory path outside output')
        src = files.path((parent_path.parent/rel).relative_to(root).as_posix())
        if digest(src) != item['sha256']: raise ValueError('Parent evidence changed')
        dest = output/rel.relative_to('output'); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    # Persist the exact working ROIs/matrix before OCR; scratch cleanup must not
    # discard the operator's audit trail, including after another refinement.
    _write(output/f'page-{page_index+1:03d}-table-review.json', journal)
    record = prepare_page(source, private/'source', page_index, context.get('pdf_registration'))
    native = NativePage(view_record(record, context))
    prefix = f'page-{page_index+1:03d}'
    try:
        if native.record['size_px'] != context['native_size_px']: raise ValueError('Native frame changed')
        matrix = np.asarray(context['native_to_working']); inverse = np.linalg.inv(matrix)
        regions = []
        for row in edit.regions:
            l, t, r, b = row.box_px
            p = np.array([[l, t, 1], [r, b, 1]]) @ inverse.T
            low, high = np.floor(p[:, :2].min(0)).astype(int), np.ceil(p[:, :2].max(0)).astype(int)
            regions.append(dict(id=row.id, kind=row.kind, box_px=[max(0, int(low[0])), max(0, int(low[1])),
                min(native.width, int(high[0])), min(native.height, int(high[1]))]))
        review = dict(schema_version='native-table-review/1.0', source_sha256=document['input_sha256'],
            source_surface_sha256=native.record['surface_sha256'], size_px=native.record['size_px'],
            page_index=page_index, page_kind=edit.page_kind, reviewer=edit.reviewer, regions=regions)
        journal['native_surface'] = {k: v for k, v in native.record.items() if k != 'surface'}
        journal['native_review'] = review
        _write(output/f'{prefix}-table-review.json', journal)
        if engine is None:
            from .ocr_engine import LocalOCREngine
            engine = LocalOCREngine(Path(request['models']))
        graph = analyze(native, review, document['input_sha256'], private/'analysis', engine=engine)
        positive = [t for t in graph['tables'] if t['cells'] and t['rules']]
        unknown = [t['id'] for t in graph['tables'] if t not in positive]
        states = []
        for table in graph['tables']:
            counts = {}
            for cell in table['cells']: counts[cell['state']] = counts.get(cell['state'], 0)+1
            states.append(dict(id=table['id'], kind=table['kind'], status=table['structure']['status'],
                               cells=len(table['cells']), states=counts))
        export_report = None
        if positive:
            produced = private/'result'; produced.mkdir()
            selected = {**graph, 'tables': positive}
            export_report = export(native, selected, baseline/f'{prefix}.dxf', produced/f'{prefix}.dxf',
                                   matrix, context['working_size_px'])
            shutil.copyfile(produced/f'{prefix}.dxf', output/f'{prefix}.dxf')
        # UNKNOWN regions keep the parent's exact CAD, never a rectangle erasure.
        public_graph = deepcopy(graph)
        for table in public_graph['tables']:
            table['masks'] = {k: {f: v for f, v in ref.items() if f != 'path'} for k, ref in table['masks'].items()}
        public_graph['review'] = {k: v for k, v in review.items() if k != 'source_surface_sha256'}
        _write(output/f'{prefix}-tables.json', public_graph)
        before = ezdxf.readfile(baseline/f'{prefix}.dxf'); after = ezdxf.readfile(output/f'{prefix}.dxf')
        removed = set(export_report['removed_baseline_handles'] if export_report else [])
        text_review = set(export_report['legacy_text_retained_for_review'] if export_report else [])
        aliases = {}
        for entity in before.modelspace():
            handle = entity.dxf.handle
            aliases[handle] = (dict(status='retired_by_explicit_table_replacement', target_handles=[])
                if handle in removed else dict(status='retained', target_handles=[handle]))
            if handle in text_review:
                aliases[handle] = dict(status='retained_on_review_layer', target_handles=[handle],
                    original_layer=entity.dxf.layer, review_layer='TABLE_LEGACY_TEXT_REVIEW')
        if any(after.entitydb.get(h) is None for a in aliases.values() for h in a['target_handles']):
            raise ValueError('Dangling retained CAD alias')
        _write(output/f'{prefix}-table-aliases.json', dict(source_manifest_sha256=parent['manifest']['sha256'],
            aliases=aliases, new_entities=export_report['emitted'] if export_report else [],
            old_observation_graph_is_parent_evidence=True))
        from .annotation_style_v1 import render_review_dxf
        if positive: render_review_dxf(output/f'{prefix}.dxf', output/f'{prefix}-dxf.png')
        report = read(output/'report.json')
        page = next(p for p in report['pages'] if p['page_index'] == page_index)
        detail = dict(schema_version='manual-table-refinement/1.0', proposal_id=request['proposal_id'],
            reviewer=edit.reviewer, accepted_regions=[t['id'] for t in positive],
            legacy_text_moved_to_review_layer=len(text_review),
            unknown_regions_kept_baseline=unknown, regions=states, ocr_calls=graph['ocr_calls'],
            outside_table_native_resolution=context['outside_table_native_resolution'],
            automatic_quality_credit=False, extension_binding=request['table_binding'],
            overlap_review_required=bool(export_report and export_report['retained_primitive_and_residual_overlap_possible']))
        page['table_refinement'] = detail
        exported = page['demo_export']
        from collections import Counter
        exported.update(dxf_sha256=digest(output/f'{prefix}.dxf'),
            preview_sha256=digest(output/f'{prefix}-dxf.png'),
            modelspace_counts=dict(Counter(e.dxftype() for e in after.modelspace())),
            layer_counts=dict(Counter(e.dxf.layer for e in after.modelspace())), table_refinement=detail)
        _write(output/f'{prefix}-demo-export.json', exported); _write(output/f'{prefix}-report.json', page)
        report['table_refinement'] = detail
        report['processing_execution'] = dict(parent=report.get('processing_execution'),
            interface_contract='leased_manual_table_extension/1.0', extension_binding=request['table_binding'])
        report.setdefault('limitations', []).append('Табличные области уточнены вручную; OCR и смысловые роли требуют проверки. Неизвестные области сохранили прежний CAD.')
        _write(output/'report.json', report)
        _write(output/'profile-artifacts.json', dict(schema_version='profile-artifacts/table-review/1.0',
            profile_sha256=report['processing_profile']['sha256'], extension_binding_sha256=request['table_binding']['sha256'],
            files=[dict(name=p.name, sha256=digest(p), size_bytes=p.stat().st_size) for p in sorted(output.iterdir())
                   if p.is_file() and p.name != 'profile-artifacts.json' and p.suffix in ('.json', '.png', '.dxf')]))
        summary = dict(status='needs_review', has_changes=bool(positive), page_count=report['page_count'],
            processing_profile=report['processing_profile'], processing_options=report['processing_options'],
            limitations=report['limitations'], pages=[{k: v for k, v in p.items() if k in
                ('page_index', 'status', 'issues', 'demo_export', 'table_refinement')} for p in report['pages']],
            table_refinement=detail)
        _write(folder/'completion.json', summary)
        if digest(source) != document['input_sha256']: raise ValueError('Original changed during table job')
        return summary
    finally:
        native.close()
        # Masks/native surfaces are private scratch, not dependencies of the DXF.
        # All paths are generated children of this fenced attempt directory.
        if private.exists() and private.resolve().is_relative_to(folder) and not private.is_symlink():
            if any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in private.rglob('*')):
                raise ValueError('Linked private table cache; cleanup refused')
            shutil.rmtree(private)
