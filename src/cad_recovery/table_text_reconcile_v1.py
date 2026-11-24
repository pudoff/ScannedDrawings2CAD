"""Source-bound, lossless text decisions after native table composition.

Geometry and source ink come from the existing table exporter. This extension
only resolves TEXT/MTEXT visibility; alternatives and original handles remain.
No confidence threshold, vocabulary, target CAD or drawing name selects text.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import re
import unicodedata

import ezdxf
from ezdxf import bbox
import numpy as np

from .dataset_layout_v1 import digest, encoded
from .native_table_export_v1 import APPID
from .table_cad_export_v1 import _snapshot

CANDIDATES = 'TABLE_TEXT_CANDIDATES_REVIEW'
LEGACY = 'TABLE_LEGACY_TEXT_REVIEW'
SELECTED = 'TABLE_TEXT_SELECTED'
POLICY = dict(version='table-text-reconcile/1.0',
    automatic_replace='unique-cell complete observed reading with exact normalized agreement',
    conflict_default='restore original text and visibility; hide new candidate',
    unreadable_default='retain baseline; empty_observed is not permission to erase',
    normalization='Unicode NFKC and whitespace only; no case or symbol substitution',
    explicit_choices='source/baseline/composition/graph/matrix SHA and named reviewer',
    max_cells=65536, max_texts=50000, max_text_cell_pairs=2000000,
    production_admitted=False)


def normalized(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', value)).strip()


def selection_binding(baseline, composed, graph, native_to_DXF):
    return dict(source_sha256=graph['source_sha256'], baseline_sha256=digest(baseline),
        composition_sha256=digest(composed), graph_sha256=hashlib.sha256(encoded(graph)).hexdigest(),
        native_to_DXF_sha256=hashlib.sha256(encoded(np.asarray(native_to_DXF, float).tolist())).hexdigest())


def visible(doc, entity):
    layer = doc.layers.get(entity.dxf.layer)
    return not layer.is_off() and not layer.is_frozen() and bool(layer.dxf.get('plot', 1)) and not entity.dxf.get('invisible', 0)


def provenance(entity):
    if not entity.has_xdata(APPID):
        return None
    return json.loads(''.join(v for code, v in entity.get_xdata(APPID) if code == 1000))


def reconcile(baseline, composed, target, *, graph, native_to_DXF, choices=None):
    """Produce a NEW DXF and decision journal; no input is modified.

    Explicit selections may activate a subset of existing baseline/candidate
    handles in one unambiguous cell. Empty selections and stale bindings fail.
    """
    target = Path(target)
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError(target)
    matrix = np.asarray(native_to_DXF, float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
        not np.allclose(matrix[2], [0, 0, 1]) or abs(np.linalg.det(matrix[:2, :2])) < 1e-12):
        raise ValueError('Finite invertible native-to-CAD registration required')
    inverse = np.linalg.inv(matrix)
    binding = selection_binding(baseline, composed, graph, matrix)
    chosen = {}
    if choices is not None:
        if (choices.get('schema_version') != 'table-text-choices/1.0' or choices.get('binding') != binding
                or not isinstance(choices.get('reviewer'), str) or not choices['reviewer'].strip()):
            raise ValueError('Source-bound choices and named reviewer required')
        for row in choices.get('selections', []):
            key = (row['region_id'], row['cell_id'])
            if key in chosen or row.get('origin') not in ('baseline', 'candidate'):
                raise ValueError('Unique cell and supported selection origin required')
            handles = row.get('handles')
            if handles is not None and (not isinstance(handles, list) or not handles or
                    any(not isinstance(h, str) for h in handles) or len(set(handles)) != len(handles)):
                raise ValueError('Nonempty unique handle selection required')
            chosen[key] = row
    old, doc = ezdxf.readfile(baseline), ezdxf.readfile(composed)
    for source in (old, doc):
        audit = source.audit()
        if audit.errors or audit.fixes or source.units != 0:
            raise ValueError('Clean unitless CAD inputs required')
    for name in (CANDIDATES, LEGACY):
        if name in old.layers and not old.layers.get(name).is_off():
            raise ValueError('Existing review layer must remain hidden; refusing implicit visibility change')
    if SELECTED in old.layers:
        layer = old.layers.get(SELECTED)
        if layer.is_off() or layer.is_frozen() or not layer.dxf.get('plot', 1):
            raise ValueError('Existing selection layer visibility differs; explicit migration required')
    for name in (CANDIDATES, LEGACY, SELECTED):
        if name not in doc.layers:
            doc.layers.new(name, dxfattribs={'color': 7, 'lineweight': 13})
    for name in (CANDIDATES, LEGACY):
        doc.layers.get(name).off(); doc.layers.get(name).dxf.plot = 0
    doc.layers.get(SELECTED).on(); doc.layers.get(SELECTED).thaw(); doc.layers.get(SELECTED).dxf.plot = 1
    cells = {}
    for table in graph['tables']:
        for cell in table['cells']:
            key = (table['id'], cell['id'])
            if key in cells: raise ValueError('Duplicate cell identity')
            cells[key] = dict(cell=cell, old=[], new=[], ambiguous=False)
    if len(cells) > POLICY['max_cells']: raise ValueError('Cell decision budget')
    if not set(chosen).issubset(cells): raise ValueError('Selection refers to unknown cell')

    def record(entity, document):
        bounds = bbox.extents([entity], fast=False)
        if not bounds.has_data: raise ValueError('Missing glyph bounds; refusing text replacement')
        center = inverse @ [bounds.center.x, bounds.center.y, 1]
        corners = np.array([[x, y, 1] for x in (bounds.extmin.x, bounds.extmax.x)
                            for y in (bounds.extmin.y, bounds.extmax.y)]) @ inverse.T
        return dict(handle=entity.dxf.handle, raw=entity.plain_text(), layer=entity.dxf.layer,
            active=visible(document, entity), position_px=center[:2].tolist(),
            glyph_box_px=[*corners[:, :2].min(0).tolist(), *corners[:, :2].max(0).tolist()])

    def owners(item):
        x, y = item['position_px']
        return [key for key, group in cells.items() if group['cell']['box_px'][0] <= x < group['cell']['box_px'][2]
                and group['cell']['box_px'][1] <= y < group['cell']['box_px'][3]]

    untouched, original_records = [], {}
    texts = list(old.modelspace().query('TEXT MTEXT'))
    if len(texts) > POLICY['max_texts']: raise ValueError('Baseline text budget')
    if len(texts)*len(cells) > POLICY['max_text_cell_pairs']:
        raise ValueError('Text-to-cell association budget')
    for entity in texts:
        copied = doc.entitydb.get(entity.dxf.handle)
        if copied is None or copied.dxftype() != entity.dxftype():
            raise ValueError('Original text handle missing from composition')
        before = _snapshot(entity)
        # Frozen V1 only relayers legacy text; reject any other untracked change.
        a = [(k,v) for k,v in before if k != 8]
        b = [(k,v) for k,v in _snapshot(copied) if k != 8]
        if a != b: raise ValueError('Composition altered original text contents/geometry')
        copied.dxf.layer = entity.dxf.layer
        item = record(entity, old); original_records[entity.dxf.handle] = item
        owned = owners(item)
        if len(owned) == 1:
            cells[owned[0]]['old'].append(item)
        else:
            untouched.append(item)
            # An unassigned glyph overlapping a cell blocks silent substitution.
            l,t,r,b = item['glyph_box_px']
            for key, group in cells.items():
                x,y,xx,yy = group['cell']['box_px']
                if max(l,x)<min(r,xx) and max(t,y)<min(b,yy):group['ambiguous'] = True
    new_texts = [e for e in doc.modelspace().query('TEXT MTEXT') if old.entitydb.get(e.dxf.handle) is None]
    if len(new_texts) > POLICY['max_texts']: raise ValueError('Candidate text budget')
    if (len(texts)+len(new_texts))*len(cells) > POLICY['max_text_cell_pairs']:
        raise ValueError('Text-to-cell association budget')
    for entity in new_texts:
        prov = provenance(entity)
        if not prov or prov.get('kind') != 'ocr_draft' or prov.get('source_sha256') != graph['source_sha256']:
            raise ValueError('Candidate text lacks registered table provenance')
        key = (prov['region_id'], prov['cell_id'])
        if key not in cells: raise ValueError('Candidate points to missing cell')
        item = record(entity, doc)
        if owners(item) != [key]: cells[key]['ambiguous'] = True
        cells[key]['new'].append(item)

    def aggregate(items):
        return normalized(' '.join(item['raw'] for item in sorted(items,
            key=lambda item:(round(item['position_px'][1], 3), item['position_px'][0], item['handle']))))

    decisions = []
    for key, group in cells.items():
        cell, prior, incoming = group['cell'], group['old'], group['new']
        decision = dict(region_id=key[0], cell_id=key[1], state=cell['state'],
            ocr_status=cell.get('ocr_status'), baseline=prior, candidate=incoming,
            baseline_raw=aggregate(prior), candidate_raw=aggregate(incoming),
            ownership_ambiguous=group['ambiguous'])
        decisions.append(decision)
        # All new strings start as alternatives. They are activated only below.
        for item in incoming: doc.entitydb[item['handle']].dxf.layer = CANDIDATES
        complete = (cell['state']=='text_draft' and cell.get('ocr_status')=='observed'
            and not cell.get('ocr_conflicts') and not group['ambiguous'] and bool(incoming)
            and aggregate(incoming)==normalized(' '.join(token['raw'] for token in cell['tokens'])))
        selection = chosen.get(key)
        if selection:
            if group['ambiguous']: raise ValueError('Explicit choice requires unambiguous cell ownership')
            pool = prior if selection['origin']=='baseline' else incoming
            available = {v['handle'] for v in pool}
            wanted = selection.get('handles', sorted(available))
            if not wanted or not set(wanted).issubset(available):
                raise ValueError('Selected text does not belong to requested origin/cell')
            for item in prior: doc.entitydb[item['handle']].dxf.layer = LEGACY
            for handle in wanted: doc.entitydb[handle].dxf.layer = SELECTED
            decision.update(action='explicit_'+selection['origin'], selected_handles=wanted,
                            reviewer=choices['reviewer'])
        elif complete and (not prior or decision['baseline_raw']==decision['candidate_raw']):
            for item in prior: doc.entitydb[item['handle']].dxf.layer = LEGACY
            for item in incoming: doc.entitydb[item['handle']].dxf.layer = SELECTED
            decision.update(action='candidate_new' if not prior else 'agreement',
                            selected_handles=[i['handle'] for i in incoming])
        else:
            if group['ambiguous']: reason = 'ambiguous_ownership'
            elif cell['state']=='empty_observed': reason = 'empty_observed_keep_prior'
            elif not incoming: reason = 'unreadable_or_missing_keep_prior'
            elif not complete: reason = 'incomplete_or_conflicting_OCR'
            else: reason = 'different_readings'
            decision.update(action='baseline_fallback' if not incoming else 'conflict_keep_baseline',
                            reason=reason, selected_handles=[i['handle'] for i in prior if i['active']])
    target.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(target)
    result = ezdxf.readfile(target); audit = result.audit()
    if audit.errors or audit.fixes: raise ValueError('Reconciled CAD requires repair')
    # Only text layer attributes may differ from the previous composition.
    before_composed = ezdxf.readfile(composed); exact = text_layer_changes = 0
    for entity in before_composed.modelspace():
        actual = result.entitydb.get(entity.dxf.handle)
        if actual is None: raise ValueError('Composed entity deleted')
        a,b = _snapshot(entity),_snapshot(actual)
        if entity.dxftype() in ('TEXT','MTEXT'):
            if [(k,v) for k,v in a if k!=8] != [(k,v) for k,v in b if k!=8]:
                raise ValueError('Text content/geometry changed')
            text_layer_changes += a!=b
        else:
            if a!=b: raise ValueError('Non-text table geometry changed')
            exact += 1
    report = dict(schema_version='table-text-reconciliation/1.0',policy=POLICY,binding=binding,
        choices=choices,decisions=decisions,unassigned_baseline=untouched,
        actions=dict(Counter(d['action'] for d in decisions)),
        nontext_composed_entities_exact=exact,text_layer_changes=text_layer_changes,
        baseline_text_handles_retained=len(texts),candidate_text_handles_retained=len(new_texts),
        dxf_sha256=digest(target),audit_errors=0,audit_fixes=0,
        production_admitted=False,all_strings_retained=True,new_CV_OCR_calls=0)
    report['role_mapping'] = {SELECTED: 6, CANDIDATES: 'review_only', LEGACY: 'review_only'}
    target.with_suffix('.json').write_bytes(encoded(report))
    return report
