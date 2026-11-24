"""Publish explicitly selected table text onto baseline CAD, never its geometry.

The immutable evidence packet retains all unselected OCR alternatives. A selected
candidate's handle is local to that packet; the journal records its new handle.
This component does not infer text, table ownership, roles, or physical units.
"""
from collections import Counter
from pathlib import Path
import hashlib
import shutil
import tempfile

import ezdxf
import numpy as np

from .dataset_layout_v1 import digest, encoded, read
from .refinement_contract_v1 import tag_hash
from .table_text_reconcile_v1 import provenance, selection_binding, visible
from .web_text_selection_v1 import box_working, relevant_unassigned

SELECTED = 'TABLE_TEXT_PUBLISHED_V1'
LEGACY = 'TABLE_TEXT_PRIOR_V1'
POLICY = dict(version='baseline-table-text-publication/1.0', max_choices=50000,
              geometry_origin='baseline only, including grid and linked entities',
              publication='explicit source-bound choices only',
              unselected_candidates='immutable evidence packet, not active CAD',
              new_CV_OCR_calls=0, automatic_quality_credit=False,
              production_admitted=False)


def _hash(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def _clean(path):
    doc = ezdxf.readfile(path)
    audit = doc.audit()
    if audit.errors or audit.fixes or doc.units != 0:
        raise ValueError('Clean unitless CAD required')
    # audit() can normalize linked owners in memory without reporting a fix.
    # Validate a disposable instance and return the original on-disk structure.
    return ezdxf.readfile(path)


def _tree(entity):
    """Include linked VERTEX/ATTRIB/SEQEND, not just the parent's DXF tags."""
    children = []
    if entity.dxftype() == 'POLYLINE':
        children = list(entity.vertices)
    elif entity.dxftype() == 'INSERT':
        children = list(entity.attribs)
    seqend = getattr(entity, 'seqend', None)
    if seqend is not None:
        children.append(seqend)
    return [tag_hash(entity), *[_tree(e) for e in children]]


def geometry_snapshot(doc):
    """All layout/block geometry and existing CAD resources must remain exact."""
    return dict(blocks={block.name: [(e.dxf.handle, _tree(e)) for e in block
                                    if e.dxftype() not in ('TEXT', 'MTEXT')]
                        for block in doc.blocks},
                resources={name: {e.dxf.name: tag_hash(e) for e in getattr(doc, name)}
                           for name in ('layers', 'linetypes', 'styles', 'dimstyles', 'appids')})


def verify_geometry(before, doc):
    after = geometry_snapshot(doc)
    if before['blocks'] != after['blocks']:
        raise ValueError('Baseline geometry changed or candidate geometry imported')
    for table, entries in before['resources'].items():
        for name, value in entries.items():
            if after['resources'][table].get(name) != value:
                raise ValueError('Baseline resource changed: '+table+':'+name)
    return sum(len(rows) for rows in before['blocks'].values())


def load_packet(baseline, evidence, source, *, document_id, version, page_index):
    folder = Path(evidence)
    if (not isinstance(document_id, str) or not document_id.strip() or len(document_id) > 128
            or type(version) is not int or version < 1 or type(page_index) is not int or page_index < 0):
        raise ValueError('Document, positive version and page identity required')
    graph, registration, review = (read(folder / name) for name in
                                   ('graph.json', 'registration.json', 'review.json'))
    if digest(source) != graph['source_sha256']:
        raise ValueError('Source changed')
    matrix = np.asarray(registration['native_to_DXF'], float)
    working = np.asarray(registration['native_to_working'], float)
    for value in (matrix, working):
        if (value.shape != (3, 3) or not np.isfinite(value).all()
                or not np.array_equal(value[2], [0, 0, 1])
                or abs(np.linalg.det(value[:2, :2])) < 1e-12):
            raise ValueError('Finite invertible registration required')
    original_binding = selection_binding(baseline, folder/'candidate.dxf', graph, matrix)
    if original_binding != review['binding']:
        raise ValueError('Evidence baseline/candidate/graph/frame changed')
    binding = dict(**original_binding, registration_sha256=digest(folder/'registration.json'),
                   review_sha256=digest(folder/'review.json'), document_id=document_id,
                   version=version, page_index=page_index)
    return graph, registration, review, binding


def make_request(baseline, evidence, source, *, document_id, version, page_index,
                 reviewer, selections, associations=None):
    *_, binding = load_packet(baseline, evidence, source, document_id=document_id,
                              version=version, page_index=page_index)
    return dict(schema_version='baseline-table-text-choice/1.0', binding=binding,
                reviewer=reviewer, selections=selections, associations=associations or [])


def _layer(doc, name, active):
    if name in doc.layers:
        layer = doc.layers.get(name)
        if (layer.is_off() or layer.is_frozen() or not layer.dxf.get('plot', 1)) != (not active):
            raise ValueError('Reserved publication layer visibility differs')
        if active and (layer.is_locked() or layer.is_frozen()):
            raise ValueError('Reserved publication layer unavailable')
    else:
        layer = doc.layers.new(name, dxfattribs={'color': 7, 'lineweight': 13})
        if not active:
            layer.off(); layer.dxf.plot = 0


def _style(source, target, name):
    entry = source.styles.get(name)
    if entry.has_extension_dict or entry.reactors:
        raise ValueError('Linked candidate text style requires explicit migration')
    attrs = {k: v for k, v in entry.dxfattribs().items() if k not in ('handle', 'owner', 'name')}
    if name in target.styles:
        current = {k: v for k, v in target.styles.get(name).dxfattribs().items()
                   if k not in ('handle', 'owner', 'name')}
        if current == attrs:
            return name
        name = 'TABLE_TEXT_' + _hash(attrs)[:20]
    if name in target.styles:
        current = {k: v for k, v in target.styles.get(name).dxfattribs().items()
                   if k not in ('handle', 'owner', 'name')}
        if current != attrs:
            raise ValueError('Candidate text style identity collision')
    else:
        target.styles.new(name, dxfattribs=attrs)
    return name


def _copy_text(entity, source_doc, doc):
    if entity.dxftype() not in ('TEXT', 'MTEXT') or entity.has_extension_dict or entity.reactors:
        raise ValueError('Only unlinked TEXT/MTEXT candidates supported')
    if entity.dxf.get('linetype', 'BYLAYER').upper() not in ('BYLAYER', 'CONTINUOUS'):
        raise ValueError('Candidate text carries unsupported line resource')
    if entity.dxf.get('invisible', 0):
        raise ValueError('Invisible candidate text cannot be activated by layer')
    style = _style(source_doc, doc, entity.dxf.get('style', 'Standard'))
    copied = entity.copy()
    copied.dxf.layer = SELECTED
    copied.dxf.style = style
    if copied.xdata:
        for appid in copied.xdata.data:
            if appid not in doc.appids:
                doc.appids.new(appid)
    doc.modelspace().add_entity(copied)
    return copied


def publish(baseline, evidence, source, target, *, request, document_id, version, page_index):
    """New version only; fail before publishing on stale or unsupported choices."""
    target = Path(target)
    if target.suffix.lower() != '.dxf':
        raise ValueError('DXF output required')
    journal = target.with_suffix('.json')
    if target.exists() or journal.exists():
        raise FileExistsError(target)
    graph, registration, review, binding = load_packet(baseline, evidence, source,
        document_id=document_id, version=version, page_index=page_index)
    if (request.get('schema_version') != 'baseline-table-text-choice/1.0'
            or request.get('binding') != binding or not isinstance(request.get('reviewer'), str)
            or not request['reviewer'].strip() or len(request['reviewer']) > 256):
        raise ValueError('Current source/version/page bound request and named reviewer required')
    selections = request.get('selections'); associations = request.get('associations')
    if (not isinstance(selections, list) or not isinstance(associations, list)
            or not 1 <= len(selections)+len(associations) <= POLICY['max_choices']):
        raise ValueError('Explicit bounded nonempty choices required')
    doc = _clean(baseline); candidate = _clean(Path(evidence)/'candidate.dxf')
    before = geometry_snapshot(doc)
    old_texts = {e.dxf.handle: (tag_hash(e, text_layer=True), e.dxf.layer, visible(doc, e))
                 for e in doc.modelspace().query('TEXT MTEXT')}
    decisions = {(r['region_id'], r['cell_id']): r for r in review['decisions']}
    if len(decisions) != len(review['decisions']):
        raise ValueError('Duplicate evidence cell')
    seen = set(); used = set(); operations = []
    for selection in selections:
        key = (selection['region_id'], selection['cell_id'])
        if key in seen or key not in decisions or selection.get('origin') not in ('baseline', 'candidate'):
            raise ValueError('Unknown/duplicate cell or origin')
        seen.add(key); decision = decisions[key]
        if decision['ownership_ambiguous']:
            raise ValueError('Ambiguous cell requires separate area association')
        origin = selection['origin']; pool = {r['handle']: r for r in decision[origin]}
        handles = selection.get('handles')
        if (not isinstance(handles, list) or not handles or any(not isinstance(h, str) for h in handles)
                or len(set(handles)) != len(handles) or not set(handles).issubset(pool)):
            raise ValueError('Explicit unique handles from selected cell/origin required')
        for handle in handles:
            identity = (origin, handle)
            if identity in used:
                raise ValueError('Duplicate selected text')
            used.add(identity)
            entity = (doc if origin == 'baseline' else candidate).entitydb.get(handle)
            if (entity is None or entity.dxftype() not in ('TEXT', 'MTEXT')
                    or entity.dxf.owner != (doc if origin == 'baseline' else candidate).modelspace().layout_key
                    or entity.plain_text() != pool[handle]['raw']):
                raise ValueError('Evidence selected text changed')
            if origin == 'candidate':
                prov = provenance(entity)
                if (doc.entitydb.get(handle) is not None or not prov or prov.get('kind') != 'ocr_draft'
                        or prov.get('source_sha256') != graph['source_sha256']
                        or (prov.get('region_id'), prov.get('cell_id')) != key):
                    raise ValueError('Candidate source/cell provenance differs')
        operations.append(dict(kind='cell', origin=origin, handles=handles,
                               prior_handles=[r['handle'] for r in decision['baseline']],
                               region_id=key[0], cell_id=key[1]))
    available = {r['handle']: r for r in relevant_unassigned(review, graph)}
    w, h = registration['working_size_px']
    for selection in associations:
        handle = selection['handle']; box = selection['box_px']
        if handle not in available or ('baseline', handle) in used:
            raise ValueError('Unknown/duplicate unassigned baseline text')
        used.add(('baseline', handle))
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError('Integer working area required')
        l, t, r, b = box
        glyph = box_working(available[handle]['glyph_box_px'], registration['native_to_working'])
        entity = doc.entitydb.get(handle)
        if (not 0 <= l < r <= w or not 0 <= t < b <= h
                or not (l <= glyph[0] and t <= glyph[1] and glyph[2] <= r and glyph[3] <= b)
                or entity is None or entity.dxftype() not in ('TEXT', 'MTEXT')
                or entity.plain_text() != available[handle]['raw']):
            raise ValueError('Area must contain the complete original text')
        operations.append(dict(kind='explicit_text_area', origin='baseline', handles=[handle],
                               prior_handles=[], box_px=box, cell_structure_confirmed=False))
    _layer(doc, SELECTED, True); _layer(doc, LEGACY, False)
    mapping = []; allowed_layers = set()
    for operation in operations:
        for handle in operation['prior_handles']:
            if handle not in old_texts:
                raise ValueError('Evidence original handle missing')
            doc.entitydb[handle].dxf.layer = LEGACY; allowed_layers.add(handle)
        for handle in operation['handles']:
            if operation['origin'] == 'baseline':
                entity = doc.entitydb[handle]; entity.dxf.layer = SELECTED; allowed_layers.add(handle)
            else:
                entity = _copy_text(candidate.entitydb[handle], candidate, doc)
            mapping.append(dict(origin=operation['origin'], evidence_handle=handle,
                                published_handle=entity.dxf.handle, raw=entity.plain_text(),
                                operation=operation))
    # Every existing text outside the explicit operation keeps its visibility.
    for handle, (content, layer, active) in old_texts.items():
        entity = doc.entitydb.get(handle)
        if entity is None or tag_hash(entity, text_layer=True) != content:
            raise ValueError('Baseline text content/geometry changed')
        if handle not in allowed_layers and (entity.dxf.layer != layer or visible(doc, entity) != active):
            raise ValueError('Unselected baseline text visibility changed')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.text-publication-', dir=target.parent) as scratch:
        path = Path(scratch)/'published.dxf'; doc.saveas(path)
        result = _clean(path); exact = verify_geometry(before, result)
        for row in mapping:
            entity = result.entitydb[row['published_handle']]
            if not visible(result, entity) or entity.plain_text() != row['raw']:
                raise ValueError('Published text is not visible/exact')
        report = dict(schema_version='baseline-table-text-publication/1.0', policy=POLICY,
                      binding=binding, request=request, handle_mapping=mapping,
                      baseline_text_handles_retained=len(old_texts),
                      baseline_nontext_entities_exact=exact,
                      imported_candidate_nontext_entities=0,
                      actions=dict(Counter(r['origin'] for r in mapping)),
                      unselected_alternatives_retained_in_bound_packet=True,
                      dxf_sha256=digest(path), audit_errors=0, audit_fixes=0,
                      new_CV_OCR_calls=0, automatic_quality_credit=False, production_admitted=False)
        created = []
        try:
            with target.open('xb') as output:
                created.append(target)
                with path.open('rb') as stream:
                    shutil.copyfileobj(stream, output)
            with journal.open('xb') as output:
                created.append(journal); output.write(encoded(report))
        except BaseException:
            for item in reversed(created):
                item.unlink()
            raise
    return report
