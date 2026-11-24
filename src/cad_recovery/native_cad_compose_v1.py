"""Owned source cores, native CAD imports and type/layer constrained seams."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math

import cv2
import ezdxf
from ezdxf.math import Matrix44
from ezdxf.addons import Importer
import numpy as np

from .dataset_layout_v1 import encoded, digest
from .experimental_export_v1 import _solids
from .demo_export_v1 import merged_runs
from .native_sheet_stitch_v1 import stitch
from .native_merge_v1 import merge, conflicts

APPID = 'CADRECOVERY_NATIVE_PAGE'


def foreground(gray):
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
    return otsu | (gray < 240), otsu


def clip_path(points, box):
    """Clip straight polyline edges to disjoint owned pixel cells; no joining exits."""
    l, t, r, b = [v-.5 for v in box]
    pieces, current = [], []
    for a, z in zip(points, points[1:]):
        a, z = np.asarray(a, float), np.asarray(z, float)
        d = z-a
        low, high = 0., 1.
        for p, q in zip([-d[0], d[0], -d[1], d[1]], [a[0]-l, r-a[0], a[1]-t, b-a[1]]):
            if p == 0:
                if q < 0: low, high = 1., 0.; break
            elif p < 0: low = max(low, q/p)
            else: high = min(high, q/p)
        if low >= high:
            if current: pieces.append(current); current = []
            continue
        start, end = (a+low*d).tolist(), (a+high*d).tolist()
        if current and np.allclose(current[-1], start, atol=1e-9, rtol=0):
            current.append(end)
        else:
            if current: pieces.append(current)
            current = [start, end]
        if high < 1:
            pieces.append(current); current = []
    if current: pieces.append(current)
    return pieces


def _solids_to_core(local, model, tile, h, H):
    x, y, _, _ = tile['box']
    cl, ct, cr, cb = tile['core']
    layers = Counter()
    for entity in local.modelspace().query('SOLID'):
        if entity.dxf.layer == 'SOURCE_INK':
            continue
        a = np.asarray([list(p)[:2] for p in entity.wcs_vertices()])
        # Frozen residual SOLIDs are exact axis-aligned pixel rectangles.
        left, right = int(round(a[:, 0].min()+.5))+x, int(round(a[:, 0].max()+.5))+x
        top, bottom = int(round(h-.5-a[:, 1].max()))+y, int(round(h-.5-a[:, 1].min()))+y
        left, right, top, bottom = max(left, cl), min(right, cr), max(top, ct), min(bottom, cb)
        if left >= right or top >= bottom:
            continue
        layer = entity.dxf.layer
        model.add_solid([(left-.5, H-.5-bottom), (right-.5, H-.5-bottom),
                         (left-.5, H-.5-top), (right-.5, H-.5-top)], dxfattribs={'layer': layer})
        layers[layer] += 1
    return dict(layers)


def _mask_solids(model, mask, tile, h, H, layer):
    """Small owned mask only, translated to global source pixel centres."""
    x, y, _, _ = tile['box']
    l, t, r, b = tile['core']
    cropped = mask[t-y:b-y, l-x:r-x]
    # _solids uses local pixel centres; emit then translate without a full-page allocation.
    for left, top, right, bottom in merged_runs(cropped):
        x0, x1 = left+l-.5, right+l-.5
        y0, y1 = H-.5-(bottom+t), H-.5-(top+t)
        model.add_solid([(x0, y0), (x1, y0), (x0, y1), (x1, y1)], dxfattribs={'layer': layer})


class Composer:
    def __init__(self, size, source_sha):
        self.width, self.height = size
        self.source_sha = source_sha
        self.doc = ezdxf.new('R2007', setup=True)
        self.doc.units = 0
        self.doc.appids.new(APPID)
        self.model = self.doc.modelspace()
        for layer in ('NATIVE_SOURCE_INK', 'NATIVE_UNRESOLVED_INK'):
            self.doc.layers.new(layer, dxfattribs={'color': 7, 'lineweight': 13})
        self.doc.layers.get('NATIVE_SOURCE_INK').off()
        self.doc.layers.get('NATIVE_SOURCE_INK').dxf.plot = 0
        self.geometry = defaultdict(list)
        self.texts = defaultdict(list)
        self.aliases = {}
        self.windows = []
        self.unresolved = []
        self.compound = {}

    def _reference(self, entity, uid, tile):
        self.aliases[uid] = entity.dxf.handle
        payload = json.dumps(dict(source_sha256=self.source_sha, window=tile['index'],
            local_uid=uid, window_box=tile['box'], owned_core=tile['core']), separators=(',', ':'))
        entity.set_xdata(APPID, [(1000, payload[i:i+200]) for i in range(0, len(payload), 200)])

    def add(self, gray, tile, report):
        x, y, _, _ = tile['box']
        h, w = gray.shape
        H = self.height
        ink, otsu = foreground(gray)
        l, t, r, b = tile['core']
        count = int(ink[t-y:b-y, l-x:r-x].sum())
        _mask_solids(self.model, ink, tile, h, H, 'NATIVE_SOURCE_INK')
        if report.get('CAD') is None:
            _mask_solids(self.model, ink, tile, h, H, 'NATIVE_UNRESOLVED_INK')
            self.windows.append(dict(index=tile['index'], source_pixels=count,
                                     fallback_pixels=count, reason=report.get('reason')))
            return
        path = Path(report['CAD']['path'])
        if digest(path) != report['CAD']['sha256']:
            raise ValueError('Window CAD changed before composition')
        local = ezdxf.readfile(path)
        importer = Importer(local, self.doc)
        importer.import_tables(('layers', 'linetypes', 'styles', 'dimstyles'))
        imported = []
        for old in local.modelspace():
            if old.dxftype() == 'SOLID' or old.dxf.layer == 'REVIEW_NOTES':
                continue
            before = len(self.model.entity_space.entities)
            importer.import_entity(old, self.model)
            if len(self.model.entity_space.entities) != before+1:
                raise ValueError('Missing imported native CAD entity')
            imported.append((old, self.model.entity_space.entities[-1]))
        # Keep imported references to anonymous DIMENSION blocks/resources.
        importer.finalize()
        # Capture each imported entity at insertion; Importer.handle_mapping
        # covers resources, not ordinary modelspace entity handles.
        for old, entity in imported:
            entity.transform(Matrix44.translate(x, H-h-y, 0))
            uid = f"tile-{tile['index']:04d}/{old.dxf.handle}"
            self._reference(entity, uid, tile)
            kind = entity.dxftype()
            attrs = (entity.dxf.layer, entity.dxf.linetype, entity.dxf.color, entity.dxf.lineweight)
            if kind in ('LINE', 'CIRCLE', 'ARC'):
                row = dict(uid=uid, tile_ids=[tile['index']], handle=entity.dxf.handle)
                if kind == 'LINE':
                    row.update(kind='line', points_px=[[float(p[0]), H-1-float(p[1])]
                        for p in (entity.dxf.start, entity.dxf.end)])
                else:
                    p = entity.dxf.center
                    row.update(kind='circle' if kind == 'CIRCLE' else 'arc',
                               center_px=[float(p[0]), H-1-float(p[1])], radius_px=float(entity.dxf.radius))
                    if kind == 'ARC':
                        row.update(start_angle_deg=(-entity.dxf.end_angle) % 360,
                                   sweep_angle_deg=(entity.dxf.end_angle-entity.dxf.start_angle) % 360)
                self.geometry[attrs].append(row)
            elif kind == 'LWPOLYLINE':
                points = list(entity.get_points('xyb'))
                if any(p[2] for p in points):
                    self.unresolved.append(dict(uid=uid, reason='bulged_polyline_requires_review'))
                    continue
                source = [[float(p[0]), H-1-float(p[1])] for p in points]
                if entity.closed and source: source.append(source[0])
                pieces = clip_path(source, tile['core'])
                self.model.delete_entity(entity)
                handles = []
                for piece in pieces:
                    created = self.model.add_lwpolyline([(p[0], H-1-p[1]) for p in piece],
                        dxfattribs=dict(layer=attrs[0], linetype=attrs[1], color=attrs[2], lineweight=attrs[3]))
                    self._reference(created, uid, tile)
                    handles.append(created.dxf.handle)
                self.aliases[uid] = handles
            elif kind in ('TEXT', 'MTEXT'):
                from .native_annotation_export_v1 import provenance
                record = provenance(old)
                p = np.asarray(record['polygon_px'], float)+[x, y]
                self.texts[attrs].append(dict(uid=uid, raw=record['raw'],
                    confidence=record.get('confidence') or 0., polygon_px=p.tolist(), handle=entity.dxf.handle,
                    edge_margin_px=float(min(p[:, 0].min()-x, p[:, 1].min()-y,
                                             x+w-p[:, 0].max(), y+h-p[:, 1].max()))))
            elif kind in ('DIMENSION', 'LEADER', 'POINT'):
                # Compound geometry is never exploded into unrelated strokes.
                # Exact source-space signatures deduplicate complete overlaps.
                from .native_annotation_export_v1 import provenance
                if kind == 'POINT':
                    key = (kind, attrs, tuple(entity.dxf.location))
                else:
                    record = provenance(old)
                    key = (kind, attrs, record['kind'], tuple(np.round(
                        (np.asarray(record['points_px'])+[x, y]).ravel(), 6)),
                        tuple(v['raw'] for v in record.get('observed_texts', [])))
                if key in self.compound:
                    self.aliases[uid] = self.compound[key]
                    self.model.delete_entity(entity)
                else: self.compound[key] = entity.dxf.handle
            else:
                self.unresolved.append(dict(uid=uid, reason='unsupported_native_entity', kind=kind))
        _solids_to_core(local, self.model, tile, h, H)
        # Weak source ink omitted by Otsu must not vanish at native resolution.
        _mask_solids(self.model, ink & ~otsu, tile, h, H, 'NATIVE_UNRESOLVED_INK')
        self.windows.append(dict(index=tile['index'], source_pixels=count,
                                 fallback_pixels=0, source_partition_retained=True))

    def finish(self, page, target):
        joins, texts = [], []
        for attrs, rows in self.geometry.items():
            objects, proof = stitch(rows, page)
            handles = {r['uid']: r['handle'] for r in rows}
            retained = set()
            for obj in objects:
                entity = self.doc.entitydb[obj['handle']]
                members = obj['observation_uids']
                retained.add(entity.dxf.handle)
                if obj['kind'] == 'line':
                    entity.dxf.start, entity.dxf.end = [(p[0], self.height-1-p[1], 0) for p in obj['points_px']]
                elif obj['kind'] == 'circle' and entity.dxftype() == 'ARC':
                    replacement = self.model.add_circle((obj['center_px'][0], self.height-1-obj['center_px'][1]),
                        obj['radius_px'], dxfattribs=dict(layer=attrs[0], linetype=attrs[1], color=attrs[2], lineweight=attrs[3]))
                    retained.remove(entity.dxf.handle)
                    entity = replacement
                    retained.add(entity.dxf.handle)
                elif obj['kind'] == 'arc':
                    entity.dxf.start_angle = -(obj['start_angle_deg']+obj['sweep_angle_deg'])
                    entity.dxf.end_angle = -obj['start_angle_deg']
                for uid in members: self.aliases[uid] = entity.dxf.handle
                payload = json.dumps(dict(source_sha256=self.source_sha, observation_uids=members,
                                           global_pixel_geometry={k:obj[k] for k in
                                               ('kind', 'points_px', 'center_px', 'radius_px', 'start_angle_deg', 'sweep_angle_deg') if k in obj}),
                                     separators=(',', ':'))
                entity.set_xdata(APPID, [(1000, payload[i:i+200]) for i in range(0, len(payload), 200)])
            for handle in handles.values():
                if handle not in retained: self.model.delete_entity(self.doc.entitydb[handle])
            joins.append(dict(layer=attrs[0], observations=len(rows), objects=len(objects), **proof))
        for attrs, rows in self.texts.items():
            merged, aliases, comparisons = merge(rows, 'text')
            lookup = {r['id']: r for r in merged}
            retained = {r['handle'] for r in merged}
            for row in rows:
                self.aliases[row['uid']] = lookup[aliases[row['uid']]]['handle']
                if row['handle'] not in retained:
                    self.model.delete_entity(self.doc.entitydb[row['handle']])
            texts.append(dict(layer=attrs[0], observations=len(rows), retained=len(merged),
                              comparisons=comparisons, conflicts=conflicts(merged)))
        for value in self.aliases.values():
            for handle in value if isinstance(value, list) else [value]:
                if self.doc.entitydb.get(handle) is None:
                    raise ValueError('Dangling native observation alias')
        self.doc.header['$LWDISPLAY'] = True
        self.doc.set_modelspace_vport(height=max(self.width, self.height)*1.1,
            center=(self.width/2, self.height/2))
        self.doc.saveas(target)
        reopened = ezdxf.readfile(target)
        audit = reopened.audit()
        if audit.errors or audit.fixes:
            raise ValueError('Composed native CAD requires repair')
        report = dict(source_sha256=self.source_sha, size_px=[self.width, self.height],
            source_windows=self.windows, aliases=self.aliases, geometry_seams=joins, text_seams=texts,
            unresolved=self.unresolved, counts=dict(Counter(e.dxftype() for e in reopened.modelspace())),
            layers=dict(Counter(e.dxf.layer for e in reopened.modelspace())),
            audit_errors=0, audit_fixes=0, dxf_sha256=digest(target),
            units='oriented_source_pixels_unitless', physical_geometry_approved=False,
            local_polylines_clipped_to_owned_cores=True, compound_dimensions_not_exploded=True,
            semantic_quality_accepted=False, production_admitted=False)
        Path(target).with_suffix('.json').write_bytes(encoded(report))
        return report
