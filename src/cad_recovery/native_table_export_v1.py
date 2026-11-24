"""Native table evidence composed with an immutable CAD baseline.

Outside entities retain exact tags. Unknown native primitives inside cells are
retained, not silently converted into text or erased with a table bbox.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import cv2
import ezdxf
from ezdxf import bbox
from ezdxf.enums import TextEntityAlignment
import numpy as np

from .dataset_layout_v1 import digest, encoded
from .demo_export_v1 import merged_runs
from .native_annotation_export_v1 import FONT_FILE
from .table_cad_export_v1 import _snapshot

APPID = 'CADRECOVERY_NATIVE_TABLE'
SOURCE = 'NATIVE_TABLE_SOURCE_REVIEW'
TEXT_SOURCE = 'NATIVE_TABLE_TEXT_SOURCE_REVIEW'
RESIDUAL = 'NATIVE_TABLE_UNRESOLVED_INK'


def subtract(rect, cut):
    l, t, r, b = rect; x, y, xx, yy = cut
    x, y, xx, yy = max(l, x), max(t, y), min(r, xx), min(b, yy)
    if x >= xx or y >= yy: return [rect]
    return [a for a in [(l, t, x, b), (xx, t, r, b), (x, t, xx, y), (x, yy, xx, b)]
            if a[0] < a[2] and a[1] < a[3]]


def export(page, graph, baseline, target, native_to_working, working_size):
    target = Path(target)
    if target.exists() or target.with_suffix('.json').exists(): raise FileExistsError(target)
    matrix = np.asarray(native_to_working, float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[2], [0, 0, 1]) or matrix[0, 1] != 0 or matrix[1, 0] != 0
            or min(matrix[0, 0], matrix[1, 1]) <= 0):
        raise ValueError('Native table exporter requires an explicit positive axis-aligned registration')
    h = working_size[1]
    cad_matrix = np.array([[1, 0, 0], [0, -1, h-1], [0, 0, 1]]) @ matrix
    inverse = np.linalg.inv(cad_matrix)
    def points(p): return np.column_stack([np.asarray(p, float), np.ones(len(p))]) @ cad_matrix.T
    def native(p): return (np.column_stack([np.asarray(p, float), np.ones(len(p))]) @ inverse.T)[:, :2]
    doc = ezdxf.readfile(baseline)
    audit = doc.audit()
    if doc.units != 0 or audit.errors or audit.fixes: raise ValueError('Clean pixel-unit baseline required')
    model = doc.modelspace(); tables = graph['tables']
    opened = []
    try:
        for table in tables:
            table['_masks'] = {}
            for name, ref in table['masks'].items():
                if digest(ref['path']) != ref['sha256']: raise ValueError('Native table mask changed')
                mask = np.memmap(ref['path'], np.uint8, 'r', shape=tuple(ref['shape']))
                opened.append(mask); table['_masks'][name] = mask
        for layer in ('TABLE_GRID', 'TITLE_GRID', 'TEXT', SOURCE, TEXT_SOURCE, RESIDUAL, 'TABLE_LEGACY_TEXT_REVIEW'):
            if layer not in doc.layers: doc.layers.new(layer, dxfattribs={'color': 7, 'lineweight': 13})
        for layer in (SOURCE, TEXT_SOURCE, 'TABLE_LEGACY_TEXT_REVIEW'):
            doc.layers.get(layer).off(); doc.layers.get(layer).dxf.plot = 0
        if APPID not in doc.appids: doc.appids.new(APPID)
        if 'NATIVE_TABLE_TEXT_V1' not in doc.styles:
            doc.styles.new('NATIVE_TABLE_TEXT_V1', dxfattribs={'font': FONT_FILE})
        cuts = []
        for table in tables:
            l, t, r, b = table['box_px']
            p = points([[l-.5, t-.5], [r-.5, b-.5]])
            cuts.append((float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())))
        token_boxes = []
        for table in tables:
            for cell in table['cells']:
                for token in cell['tokens']:
                    p = points(token['polygon_px'])
                    token_boxes.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
        preserved, removed, split, text_review, inside_native = {}, [], [], [], []
        reference_layers = {'SOURCE_INK', 'NATIVE_SOURCE_INK', 'REVIEW_NOTES', 'REVIEW_NOTICE'}
        for entity in list(model):
            handle = entity.dxf.handle
            bounds = bbox.extents([entity], fast=False)
            if not bounds.has_data:
                preserved[handle] = _snapshot(entity); continue
            rect = [bounds.extmin.x, bounds.extmin.y, bounds.extmax.x, bounds.extmax.y]
            intersect = [i for i, c in enumerate(cuts) if rect[0] <= c[2] and c[0] <= rect[2] and rect[1] <= c[3] and c[1] <= rect[3]]
            if not intersect or entity.dxf.layer in reference_layers:
                preserved[handle] = _snapshot(entity); continue
            if entity.dxftype() == 'SOLID':
                vertices = np.array([list(p)[:2] for p in entity.wcs_vertices()])
                if len(np.unique(vertices[:, 0])) != 2 or len(np.unique(vertices[:, 1])) != 2:
                    preserved[handle] = _snapshot(entity); inside_native.append(handle); continue
                pieces = [rect]
                for i in intersect: pieces = [p for piece in pieces for p in subtract(piece, cuts[i])]
                attrs = {k: v for k, v in entity.dxf.all_existing_dxf_attribs().items()
                         if k not in ('handle', 'owner', 'vtx0', 'vtx1', 'vtx2', 'vtx3')}
                for l, t, r, b in pieces:
                    model.add_solid([(l, t), (r, t), (l, b), (r, b)], dxfattribs=attrs)
                removed.append(handle)
                if pieces: split.append(dict(handle=handle, original_box=rect, remaining_boxes=pieces))
                model.delete_entity(entity)
                continue
            whole = [i for i in intersect if cuts[i][0] <= rect[0] and rect[2] <= cuts[i][2]
                     and cuts[i][1] <= rect[1] and rect[3] <= cuts[i][3]]
            grid_owned = False
            if whole and entity.dxftype() in ('LINE', 'LWPOLYLINE') and not (
                    entity.dxftype() == 'LWPOLYLINE' and entity.has_arc):
                vertices = ([list(entity.dxf.start)[:2], list(entity.dxf.end)[:2]] if entity.dxftype() == 'LINE'
                            else [list(p) for p in entity.get_points('xy')])
                if entity.dxftype() == 'LWPOLYLINE' and entity.closed and vertices: vertices.append(vertices[0])
                p = native(vertices); segments = []
                for a, b in zip(p, p[1:]):
                    n = math_ceil_distance(a, b)
                    if n > 100000: segments = []; break
                    segments.append(np.linspace(a, b, n))
                if segments:
                    q = np.rint(np.concatenate(segments)).astype(int)
                    for i in whole:
                        table = tables[i]; l, t, r, b = table['box_px']
                        supported = np.zeros(len(q), bool)
                        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                            x, y = q[:, 0]+dx-l, q[:, 1]+dy-t
                            valid = (x >= 0) & (x < r-l) & (y >= 0) & (y < b-t)
                            supported[valid] |= table['_masks']['grid'][y[valid], x[valid]] > 0
                        if supported.all(): grid_owned = True; break
            if grid_owned:
                removed.append(handle); model.delete_entity(entity)
            elif whole and entity.dxftype() in ('TEXT', 'MTEXT') and any(
                max(rect[0], c[0]) < min(rect[2], c[2]) and max(rect[1], c[1]) < min(rect[3], c[3]) for c in token_boxes):
                entity.dxf.layer = 'TABLE_LEGACY_TEXT_REVIEW'; text_review.append(handle)
            else:
                preserved[handle] = _snapshot(entity); inside_native.append(handle)

        emitted = []; totals = Counter()
        def record(entity, table, kind, cell=None):
            value = dict(source_sha256=graph['source_sha256'], region_id=table['id'], cell_id=cell, kind=kind)
            raw = json.dumps(value, ensure_ascii=True, separators=(',', ':'))
            entity.set_xdata(APPID, [(1000, raw[i:i+200]) for i in range(0, len(raw), 200)])
            emitted.append(dict(handle=entity.dxf.handle, type=entity.dxftype(), layer=entity.dxf.layer,
                                tags_sha256=sha256(encoded(_snapshot(entity))).hexdigest(), **value))
        def solids(mask, x, y, layer):
            for l, t, r, b in merged_runs(mask):
                p = points([[x+l-.5, y+t-.5], [x+r-.5, y+t-.5],
                            [x+l-.5, y+b-.5], [x+r-.5, y+b-.5]])
                model.add_solid([tuple(v[:2]) for v in p], dxfattribs={'layer': layer})
        for table in tables:
            layer = 'TITLE_GRID' if table['kind'] == 'title' else 'TABLE_GRID'
            for rule in table['rules']:
                p = points(rule['points_px'])
                e = model.add_line(tuple(p[0, :2]), tuple(p[1, :2]), dxfattribs={'layer': layer})
                record(e, table, 'rule')
            for cell in table['cells']:
                for token in cell['tokens']:
                    p = points(token['polygon_px'])[:, :2]
                    height = float(np.linalg.norm(p[3]-p[0]))
                    direction = p[1]-p[0]
                    width = float(np.linalg.norm(direction))
                    if min(width, height) < .1: continue
                    lines = token['raw'].replace('%%', '% %').splitlines()
                    if not lines: continue
                    down = (p[3]-p[0])/height; along = direction/width
                    for i, line in enumerate(lines):
                        anchor = p[0]+down*((i+.85)*height/len(lines))
                        e = model.add_text(line, dxfattribs=dict(layer='TEXT', style='NATIVE_TABLE_TEXT_V1', height=height*.8/len(lines)))
                        e.set_placement(tuple(anchor), tuple(anchor+along*width), align=TextEntityAlignment.FIT)
                        if e.dxf.valign == 0: e.dxf.discard('valign')
                        if e.dxf.rotation == 0: e.dxf.discard('rotation')
                        record(e, table, 'ocr_draft', cell['id'])
            l, t, r, b = table['box_px']
            for y in range(t, b, 64):
                bottom = min(b, y+64)
                gray = page.crop([l, y, r, bottom]); ink = gray < 240
                grid = (table['_masks']['grid'][y-t:bottom-t] > 0) & ink
                text = (table['_masks']['text'][y-t:bottom-t] > 0) & ink & ~grid
                residual = ink & ~grid & ~text
                if not np.array_equal(grid | text | residual, ink): raise ValueError('Native table partition failed')
                solids(ink, l, y, SOURCE); solids(text, l, y, TEXT_SOURCE); solids(residual, l, y, RESIDUAL)
                totals.update(source_pixels=int(ink.sum()), grid_pixels=int(grid.sum()),
                              text_pixels=int(text.sum()), residual_pixels=int(residual.sum()))
        doc.saveas(target); reopened = ezdxf.readfile(target); audit = reopened.audit()
        if audit.errors or audit.fixes: raise ValueError('Native table DXF requires repair')
        for handle, tags in preserved.items():
            if _snapshot(reopened.entitydb[handle]) != tags: raise ValueError('Preserved baseline entity changed')
        for item in emitted:
            if sha256(encoded(_snapshot(reopened.entitydb[item['handle']]))).hexdigest() != item['tags_sha256']:
                raise ValueError('Native table entity roundtrip changed')
        report = dict(schema_version='native-table-export/1.0', source_sha256=graph['source_sha256'],
            baseline_sha256=digest(baseline), dxf_sha256=digest(target),
            native_to_working=matrix.tolist(), native_to_DXF=cad_matrix.tolist(), working_size_px=working_size,
            source_partition_exact=True, pixel_counts=dict(totals),
            preserved_baseline_entities=len(preserved), preserved_baseline_tags_exact=True,
            removed_baseline_handles=removed, boundary_solid_splits=split,
            legacy_text_retained_for_review=text_review, inside_or_boundary_primitives_retained=inside_native,
            retained_primitive_and_residual_overlap_possible=bool(inside_native),
            emitted=emitted, counts=dict(Counter(e.dxftype() for e in reopened.modelspace())),
            layer_counts=dict(Counter(e.dxf.layer for e in reopened.modelspace())),
            role_mapping={'TABLE_GRID': 8, 'TITLE_GRID': 8, 'TEXT': 6, RESIDUAL: 255},
            audit_errors=0, audit_fixes=0, new_mechanical_entities=0,
            physical_geometry_approved=False, manual_CAD_verified=False, production_admitted=False)
        target.with_suffix('.json').write_bytes(encoded(report))
        return report
    finally:
        for table in tables: table.pop('_masks', None)
        for mask in opened: mask._mmap.close()


def math_ceil_distance(a, b):
    return max(2, int(np.ceil(np.linalg.norm(b-a)))+1)
