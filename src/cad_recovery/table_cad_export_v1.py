"""Reviewed table substitution over a frozen pixel-unit CAD result.

Only entities wholly within reviewed boxes may be replaced. All boundary
crossers and all outside entities retain their handles, attributes and geometry.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import ezdxf
from ezdxf import bbox
from ezdxf.enums import TextEntityAlignment
import numpy as np
from PIL import Image

from .dataset_layout_v1 import digest, encoded
from .experimental_export_v1 import _solids
from .native_annotation_export_v1 import FONT_FILE
from .table_structure_v1 import analyze

APPID = 'CADRECOVERY_TABLE_V1'
LAYERS = ('TABLE_GRID', 'TITLE_GRID', 'TEXT', 'TABLE_UNRESOLVED_INK',
          'TABLE_TEXT_SOURCE_REVIEW', 'TABLE_SOURCE_REVIEW')


def _snapshot(entity):
    # Tags include vertex, text, block reference and xdata, not just bbox.
    from ezdxf.lldxf.tagwriter import TagCollector
    return [(t.code, str(t.value)) for t in TagCollector.dxftags(entity)]


def _inside(extents, box):
    l, t, r, b = box
    return l-.5 <= extents[0] and t-.5 <= extents[1] and extents[2] <= r-.5 and extents[3] <= b-.5


def _intersects(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def export_reviewed(gray, review, source_sha, baseline, target, *, recognizer=None,
                    source_to_working=None, render=False, ocr_provenance=None):
    target = Path(target)
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError(target)
    if not target.parent.is_dir():
        raise ValueError('Output directory must exist')
    graph, masks = analyze(gray, review, source_sha, recognizer=recognizer)
    h, w = gray.shape
    matrix = np.asarray(source_to_working, float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError('Invertible source-to-working transform required')
    graph.update(source_to_working=matrix.tolist(), working_to_source=np.linalg.inv(matrix).tolist(),
                 working_to_dxf=[[1, 0, 0], [0, -1, h-1], [0, 0, 1]],
                 ocr_provenance=ocr_provenance,
                 role_mapping={'TABLE_GRID': 8, 'TITLE_GRID': 8, 'TEXT': 6,
                               'TABLE_UNRESOLVED_INK': 255},
                 baseline_sha256=digest(baseline))
    doc = ezdxf.readfile(baseline)
    if doc.units != 0:
        raise ValueError('Only unitless working-pixel CAD baselines are supported')
    if doc.audit().errors:
        raise ValueError('Invalid baseline DXF')
    model = doc.modelspace()
    preserved, removed, crossing, unbounded = {}, [], [], []
    for entity in list(model):
        # The original complete source and notes are retained independently.
        layer = entity.dxf.layer
        bounds = bbox.extents([entity], fast=False)
        if not bounds.has_data:
            preserved[entity.dxf.handle] = _snapshot(entity)
            unbounded.append(entity.dxf.handle)
            continue
        left, bottom, _ = bounds.extmin
        right, top, _ = bounds.extmax
        extent = [left, h-1-top, right, h-1-bottom]
        owned = any(_inside(extent, r['box_px']) for r in review['regions'])
        if owned and layer not in ('SOURCE_INK', 'REVIEW_NOTES'):
            removed.append(dict(handle=entity.dxf.handle, type=entity.dxftype(), layer=layer))
            model.delete_entity(entity)
        else:
            preserved[entity.dxf.handle] = _snapshot(entity)
            if any(_intersects(extent, r['box_px']) for r in review['regions']):
                crossing.append(entity.dxf.handle)
    for name in LAYERS:
        if name not in doc.layers:
            doc.layers.new(name, dxfattribs=dict(color=7, lineweight=13))
    for name in ('TABLE_SOURCE_REVIEW', 'TABLE_TEXT_SOURCE_REVIEW'):
        doc.layers.get(name).off()
        doc.layers.get(name).dxf.plot = 0
    if APPID not in doc.appids:
        doc.appids.new(APPID)
    if 'TABLE_TEXT_V1' not in doc.styles:
        doc.styles.new('TABLE_TEXT_V1', dxfattribs=dict(font=FONT_FILE))
    point = lambda p: (float(p[0]), float(h-1-p[1]))
    emitted = []

    def record(entity, region, kind, cell=None):
        payload = dict(region_id=region['id'], cell_id=cell, kind=kind, source_sha256=source_sha,
                       working_pixel_sha256=graph['working_pixel_sha256'])
        serialized = json.dumps(payload, ensure_ascii=True, separators=(',', ':'))
        entity.set_xdata(APPID, [(1000, serialized[i:i+200]) for i in range(0, len(serialized), 200)])
        emitted.append(dict(handle=entity.dxf.handle, type=entity.dxftype(), layer=entity.dxf.layer,
                            entity_tags_sha256=sha256(encoded(_snapshot(entity))).hexdigest(), **payload))

    for table in graph['tables']:
        layer = 'TITLE_GRID' if table['kind'] == 'title' else 'TABLE_GRID'
        for rule in table['rules']:
            record(model.add_line(*map(point, rule['points_px']), dxfattribs={'layer': layer}), table, 'rule')
        for cell in table['cells']:
            for token in cell['tokens']:
                polygon = np.asarray(token['polygon_px'])
                direction = polygon[1]-polygon[0]
                # OCR text is a draft; no numeric/engineering interpretation.
                angle = -float(np.degrees(np.arctan2(direction[1], direction[0])))
                polygon_height = float(np.linalg.norm(polygon[3]-polygon[0]))
                polygon_width = float(np.linalg.norm(direction))
                if min(polygon_height, polygon_width) < 1:
                    continue
                literal = token['raw'].replace('%%', '% %')
                if '\n' in literal or '\r' in literal or '\\' in literal or '{' in literal or '}' in literal:
                    # TEXT preserves literals; multiple lines remain separate
                    # entities rather than executable MTEXT formatting codes.
                    lines = literal.splitlines() or [literal]
                else:
                    lines = [literal]
                height = polygon_height*.8/len(lines)
                down = (polygon[3]-polygon[0])/polygon_height
                along = direction/polygon_width
                for i, line in enumerate(lines):
                    anchor = polygon[0]+down*((i+.85)*polygon_height/len(lines))
                    entity = model.add_text(line, dxfattribs=dict(layer='TEXT', style='TABLE_TEXT_V1',
                        height=height, rotation=angle, insert=point(anchor)))
                    entity.set_placement(point(anchor), point(anchor+along*polygon_width),
                                         align=TextEntityAlignment.FIT)
                    if entity.dxf.rotation == 0:
                        entity.dxf.discard('rotation')  # DXF writer omits the default, including -0.0.
                    if entity.dxf.valign == 0:
                        entity.dxf.discard('valign')
                    record(entity, table, 'ocr_draft', cell['id'])
    _solids(model, masks['residual'], h, 'TABLE_UNRESOLVED_INK')
    _solids(model, masks['text'], h, 'TABLE_TEXT_SOURCE_REVIEW')
    _solids(model, masks['ink'] & masks['selected'], h, 'TABLE_SOURCE_REVIEW')
    doc.saveas(target)
    reopened = ezdxf.readfile(target)
    audit = reopened.audit()
    if audit.errors or audit.fixes:
        raise ValueError('Table DXF requires repair')
    for handle, tags in preserved.items():
        if _snapshot(reopened.entitydb[handle]) != tags:
            raise ValueError('Retained baseline entity changed')
    for value in emitted:
        entity = reopened.entitydb[value['handle']]
        if entity.dxftype() != value['type'] or entity.dxf.layer != value['layer'] or not entity.has_xdata(APPID):
            raise ValueError('Table entity/role round trip failed')
        if sha256(encoded(_snapshot(entity))).hexdigest() != value['entity_tags_sha256']:
            raise ValueError('New table geometry/text/provenance changed in round trip')
    graph.update(dxf=target.name, dxf_sha256=digest(target),
                 baseline_complete=review.get('baseline_complete'),
                 preserved_baseline_entities=len(preserved), preserved_baseline_entities_exact=True,
                 replaced_baseline_entities=removed, boundary_or_reference_entities_retained=crossing,
                 unbounded_entities_retained=unbounded, emitted=emitted,
                 counts=dict(Counter(e.dxftype() for e in reopened.modelspace())),
                 layer_counts=dict(Counter(e.dxf.layer for e in reopened.modelspace())),
                 new_mechanical_entities=0, audit_errors=0, audit_fixes=0,
                 units='working_pixels_unitless', physical_geometry_approved=False,
                 manual_CAD_verified=False, font_installation_verified=False,
                 boundary_duplicates_possible=bool(crossing))
    for key in ('grid', 'text', 'residual', 'gridless'):
        a = masks[key]
        if a.dtype == bool:
            a = a.astype(np.uint8)*255
        Image.fromarray(a).save(target.with_name(target.stem+'-'+key+'.png'))
    target.with_suffix('.json').write_bytes(encoded(graph))
    if render:
        from .annotation_style_v1 import render_review_dxf
        render_review_dxf(target, target.with_suffix('.png'))
    return graph
