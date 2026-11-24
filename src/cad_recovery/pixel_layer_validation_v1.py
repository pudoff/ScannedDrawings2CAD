"""Read-only audit of reviewed pixel DXF against source, observations and review.

No exporter, snapshot or exporter report is used. Required IDs are a declared
coverage lock, not evidence that the automatic classifier is correct.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from PIL import Image

from .cad_validation_geometry_v1 import raw_geometry, compare_geometry

APP = 'CAD_RECOVERY_PIXEL_V1'
FRAME = 'processed_pixel_centres_xright_ydown'
SCHEMA = {'part_contour': ('MAIN', 50), 'thin': ('THIN', 13),
          'dimension_line': ('DIMS', 13), 'extension_line': ('EXTENSIONS', 13),
          'leader_line': ('LEADERS', 13), 'arrowhead': ('ARROWHEADS', 13),
          'axis_construction': ('AXIS', 13), 'sheet_frame': ('FRAME', 18),
          'title_grid': ('TITLE_GRID', 13), 'hatching': ('HATCH_STROKES', 13),
          'text': ('TEXT_INK', 13)}
WEIGHTS = dict(SCHEMA.values()) | {'SOURCE_INK': 0, 'UNKNOWN_INK': 0, 'UNKNOWN_POINTS': 0}
NOTE = 'REVIEW DRAFT / PIXEL UNITS / PHYSICAL DIMENSIONS UNCONFIRMED'


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rle(spans, shape):
    require(isinstance(spans, list) and len(spans) <= 2000000, 'span budget')
    mask = np.zeros(shape, bool)
    visits = 0
    for row in spans:
        require(isinstance(row, list) and len(row) == 3 and all(type(v) is int for v in row), 'integer spans')
        y, a, b = row
        require(0 <= y < shape[0] and 0 <= a < b <= shape[1], 'span bounds')
        visits += b-a
        require(visits <= 2000000 and not mask[y, a:b].any(), 'span overlap/budget')
        mask[y, a:b] = True
    return mask


def planar(entity):
    for name, default in (('extrusion', (0., 0., 1.)),):
        require(np.allclose(tuple(entity.dxf.get(name, default)), default, rtol=0, atol=1e-10), 'extrusion')
    require(abs(entity.dxf.get('thickness', 0)) <= 1e-10, 'thickness')
    require(not entity.dxf.get('invisible', 0), 'invisible entity')
    require(entity.dxf.get('color', 256) == 256 and entity.dxf.get('true_color') is None, 'entity color must be ByLayer')
    require(entity.dxf.get('lineweight', -1) == -1 and entity.dxf.get('linetype', 'BYLAYER') == 'BYLAYER', 'entity style must be ByLayer')
    require(entity.dxf.get('transparency') is None, 'entity transparency')


def provenance(entity):
    require(entity.has_xdata(APP), 'missing provenance')
    fields, numbers, pieces = {}, [], []
    for tag in entity.get_xdata(APP):
        if tag.code == 1000:
            require('=' in tag.value, 'malformed provenance')
            key, value = tag.value.split('=', 1)
            require(key not in fields, 'duplicate provenance key')
            fields[key] = value
        elif tag.code == 1040:
            require(math.isfinite(tag.value), 'nonfinite provenance error')
            numbers.append(float(tag.value))
        elif tag.code == 1070:
            pieces.append(tag.value)
        else:
            raise ValueError('unsupported provenance tag')
    require(set(fields) == {'source', 'observation', 'role', 'review', 'path', 'observation_sha256', 'source_height', 'frame'}, 'provenance keys')
    require(len(numbers) == 2 and len(pieces) == 1 and pieces[0] >= 0, 'provenance numeric fields')
    return fields, numbers, pieces[0]


def points(c, shape):
    kind = c['kind']
    if kind in ('circle', 'arc'):
        center = np.asarray(c['center_px'], float); radius = float(c['radius_px'])
        start = float(c.get('start_angle_deg', 0)); sweep = 360. if kind == 'circle' else float(c['sweep_angle_deg'])
        require(center.shape == (2,) and np.isfinite([*center, radius, start, sweep]).all()
                and radius > 0 and 0 < sweep <= 360 and (kind != 'arc' or sweep < 360), 'curve values')
        count = max(8, math.ceil(radius*math.radians(sweep)/.25)+1)
        require(count <= 250000, 'curve sample budget')
        angles = np.linspace(math.radians(start), math.radians(start+sweep), count)
        p = center + radius*np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        require(kind in ('line', 'polyline', 'arrow_open', 'arrow_filled'), 'primitive kind')
        p = np.asarray(c['points_px'], float)
        require(p.ndim == 2 and p.shape[1] == 2 and 2 <= len(p) <= 8192, 'primitive vertices')
        require(kind != 'line' or len(p) == 2, 'line vertex count')
        require(not kind.startswith('arrow_') or len(p) == 3, 'head vertex count')
        require(kind != 'polyline' or not c.get('closed') or len(p) >= 3, 'closed vertex count')
        require(np.all(np.linalg.norm(np.diff(p, axis=0), axis=1) >= 1e-7), 'degenerate segment')
        if kind.startswith('arrow_'):
            a, b = p[1]-p[0], p[2]-p[0]
            require(abs(a[0]*b[1]-a[1]*b[0]) >= .5, 'degenerate head')
    require(np.isfinite(p).all() and (p >= 0).all() and (p[:, 0] < shape[1]).all()
            and (p[:, 1] < shape[0]).all(), 'primitive bounds')
    return p


def coverage(c, p, shape):
    result = np.zeros(shape, np.uint8)
    q = np.rint(p).astype(np.int32)
    require((q >= 0).all() and (q[:, 0] < shape[1]).all() and (q[:, 1] < shape[0]).all(), 'rounded bounds')
    if c['kind'] == 'arrow_filled':
        cv2.fillPoly(result, [q], 1)
    elif c['kind'] == 'arrow_open':
        for endpoint in q[1:]:
            cv2.line(result, tuple(q[0]), tuple(endpoint), 1)
    else:
        cv2.polylines(result, [q], c['kind'] == 'circle' or bool(c.get('closed')), 1)
    return result.astype(bool)


def native_expectation(c, height):
    def flip(p): return (float(p[0]), height-1-float(p[1]))
    kind = c['kind']
    if kind == 'line':
        return [{'type': 'LINE', 'start': flip(c['points_px'][0]), 'end': flip(c['points_px'][1])}]
    if kind == 'arrow_open':
        return [{'type': 'LINE', 'start': flip(c['points_px'][0]), 'end': flip(p)} for p in c['points_px'][1:]]
    if kind == 'arrow_filled':
        p = [flip(p) for p in c['points_px']]
        return [{'type': 'SOLID', 'vertices': p+[p[-1]]}]
    if kind == 'polyline':
        return [{'type': 'LWPOLYLINE', 'closed': bool(c.get('closed')), 'vertices': [(*flip(p), 0.) for p in c['points_px']]}]
    result = {'type': kind.upper(), 'center': flip(c['center_px']), 'radius': float(c['radius_px'])}
    if kind == 'arc':
        result.update(start_angle=(-(c['start_angle_deg']+c['sweep_angle_deg'])) % 360, end_angle=(-c['start_angle_deg']) % 360)
    return [result]


def rectangle(entity, target, height):
    require(entity.dxftype() == 'SOLID', 'raster entity must be SOLID')
    p = np.asarray([tuple(entity.dxf.get('vtx'+str(i))) for i in range(4)], float)
    require(np.isfinite(p).all() and np.max(np.abs(p[:, 2])) <= 1e-8, 'raster planarity')
    x0, x1 = p[:, 0].min(), p[:, 0].max()
    y0, y1 = p[:, 1].min(), p[:, 1].max()
    bounds = np.array([x0+.5, x1+.5, height-.5-y1, height-.5-y0])
    require(np.max(np.abs(bounds-np.rint(bounds))) <= 1e-8, 'raster pixel boundaries')
    a, b, t, u = np.rint(bounds).astype(int)
    require(0 <= a < b <= target.shape[1] and 0 <= t < u <= height, 'raster bounds')
    expected = np.array([[x0, y0, 0], [x1, y0, 0], [x0, y1, 0], [x1, y1, 0]])
    require(np.max(np.abs(p-expected)) <= 1e-8, 'raster rectangle corners')
    require(not target[t:u, a:b].any(), 'duplicate/overlapping raster entity')
    target[t:u, a:b] = True


def same_regions(before, after, connectivity):
    na, la = cv2.connectedComponents(before.astype(np.uint8), connectivity=connectivity)
    nb, lb = cv2.connectedComponents(after.astype(np.uint8), connectivity=connectivity)
    common = before & after
    pairs = np.unique(np.column_stack((la[common], lb[common])), axis=0)
    return na == nb and len(pairs) == na-1 and len(np.unique(pairs[:, 0])) == na-1 and len(np.unique(pairs[:, 1])) == nb-1


def audit(binary, observations, review, dxf, *, required_ids, display=None, weight_overrides=None):
    """Return bounded diagnostics; failures never grant engineering approval."""
    cv2.setNumThreads(1)
    errors, changed = [], set()
    result = {'schema_version': 'pixel-export-qa/1.0', 'valid_review_draft': False,
              'physical_geometry_approved': False, 'whole_sheet_automatic_semantics_admitted': False,
              'manual_CAD_checked': False, 'human_edit_seconds': None, 'production_changed': False}

    def check(condition, code, detail=''):
        if not condition and len(errors) < 200:
            errors.append({'code': code, 'detail': str(detail)[:256]})

    try:
        require(Path(dxf).stat().st_size <= 64*1024**2, 'DXF byte budget')
        with Image.open(binary) as im:
            require(im.mode == 'L' and 0 < im.width*im.height <= 4194304, 'bounded grayscale binary required')
            image = np.array(im)
        require(np.isin(image, [0, 255]).all(), 'binary values')
        ink = image == 0; h, w = ink.shape
        sha = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
        require(observations['schema_version'] == 'pixel-observations/1.0' and review['schema_version'] == 'pixel-role-review/1.0', 'input schema')
        require(observations['source_sha256'] == review['source_sha256'] == sha
                and observations['size_px'] == review['size_px'] == [w, h]
                and observations['coordinate_space'] == FRAME and review['observations_sha256'] == object_digest(observations), 'source/review binding')
        require(review.get('physical_geometry_approved') is False, 'unapproved physical contract')
        oi, ri = observations['items'], review['items']
        require(len(oi) <= 4096 and len(ri) <= 4096, 'object budget')
        obs = {c['id']: c for c in oi}; rows = {r['observation_id']: r for r in ri}
        require(len(obs) == len(oi) and len(rows) == len(ri) and len({r['id'] for r in ri}) == len(ri), 'duplicate input IDs')
        require(isinstance(required_ids, list) and len(required_ids) == len(set(required_ids))
                and set(required_ids) <= obs.keys() & rows.keys(), 'coverage contract')
        protect = rle(review.get('protected_spans', []), ink.shape)
        text_protect = rle(review.get('text_protected_spans', []), ink.shape)
        require(not ((protect | text_protect) & ~ink).any(), 'protection outside source ink')
        removed = np.zeros_like(ink); glyph = np.zeros_like(ink); paint_all = np.zeros_like(ink)
        ownership = np.zeros(ink.shape, np.uint16); paint_owners = np.zeros_like(ownership)
        selected = {}; visits = samples = work = 0
        for key in sorted(required_ids):
            c, r = obs[key], rows[key]
            require(r.get('status') == 'reviewed_pixel_role' and r.get('reviewer') and r.get('review_ref')
                    and r['observation_sha256'] == object_digest(c) and r['role'] in SCHEMA, 'selected role not approved: '+key)
            require(all(isinstance(s, str) and 0 < len(s.strip()) <= 128 for s in
                        [key, c.get('path_id', key), r['id'], r['reviewer'], r['review_ref']]), 'provenance identifier')
            text = c['kind'] == 'ink_region'
            require(text == (r['role'] == 'text') and c['kind'].startswith('arrow_') == (r['role'] == 'arrowhead'), 'role/type conflict')
            owned = rle(r['owned_spans'], ink.shape)
            visits += int(owned.sum()); work += ink.size*(2 if text else 6)
            require(visits <= 2000000 and work <= 40000000, 'ownership/raster work budget')
            require(owned.any() and not (owned & ~ink).any(), 'ownership outside ink/empty')
            shared = np.zeros_like(ink); contacts = r.get('shared_contacts', [])
            require(len(contacts) <= 2, 'shared endpoint budget')
            p = None if text else points(c, ink.shape)
            samples += 0 if p is None else len(p)
            require(samples <= 250000, 'global curve budget')
            ends = [] if text else [p[0], p[-1]] if c['kind'] in ('line', 'polyline') and not c.get('closed') else []
            if c['kind'].startswith('arrow_'): ends = [p[0], (p[1]+p[2])/2]
            for contact in contacts:
                require(isinstance(contact, list) and len(contact) == 2 and all(type(v) is int for v in contact), 'integer contact')
                x, y = contact
                require(0 <= x < w and 0 <= y < h and ink[y, x] and any(math.dist(contact, q) <= .75 for q in ends), 'shared endpoint location')
                shared[y, x] = True
            require(not (shared & text_protect).any(), 'shared text')
            effective = owned & ~shared
            require(effective.any() and not (effective & (removed | glyph)).any(), 'exclusive ownership conflict')
            ownership[effective] += 1
            bound = float(r.get('uncertainty_px', 1.5))
            require(math.isfinite(bound) and 0 < bound <= 1.5, 'uncertainty bound')
            if text:
                glyph |= effective; expected_error = 0.; native = []
            else:
                paint = coverage(c, p, ink.shape)
                require(not ((owned | paint) & (protect | text_protect) & ~shared).any(), 'protected erasure/intersection')
                require(not (paint & ink & ~owned & ~shared).any(), 'unowned source intersection')
                outward = cv2.distanceTransform((~(owned | shared)).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                inward = cv2.distanceTransform((~paint).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                expected_error = max(float(outward[paint].max()), float(inward[owned].max()))
                require(expected_error <= bound and cv2.connectedComponents((owned | shared).astype(np.uint8), connectivity=8)[0] == 2, 'disconnected/unsupported source')
                removed |= effective; paint_all |= paint; paint_owners[paint & ~shared] += 1
                native = native_expectation(c, h)
            selected[key] = {'observation': c, 'review': r, 'error': expected_error,
                             'bound': 0. if text else bound, 'native': native, 'glyph': effective if text else None}
        require(ownership.max(initial=0) <= 1 and paint_owners.max(initial=0) <= 1, 'overlapping claims')
        residual = ink & ~removed & ~glyph
        counts, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        singleton_labels = stats[:, cv2.CC_STAT_AREA] == 1
        singleton_labels[0] = False
        isolated = singleton_labels[labels]
        expected_points = isolated & residual
        masks = {n: np.zeros_like(ink) for n in ('SOURCE_INK', 'UNKNOWN_INK', 'UNKNOWN_POINTS', 'TEXT_INK')}
        glyphs = {k: np.zeros_like(ink) for k, s in selected.items() if s['glyph'] is not None}
        require(set(weight_overrides or {}) <= WEIGHTS.keys(), 'weight override layers')
        expected_display = {n: n != 'SOURCE_INK' for n in WEIGHTS}
        if display is not None:
            require(set(display) == set(WEIGHTS) and all(type(v) is bool for v in display.values()), 'display contract')
            expected_display = display
        doc = ezdxf.readfile(dxf)
        entities = list(doc.modelspace()); require(len(entities) <= 50000, 'entity budget')
        audit_result = doc.audit()
        check(not audit_result.errors and not audit_result.fixes, 'CAD_AUDIT', (len(audit_result.errors), len(audit_result.fixes)))
        for name, weight in WEIGHTS.items():
            layer = doc.layers.get(name)
            check(layer.is_off() == (not expected_display[name]) and not layer.is_frozen(), 'LAYER_VISIBILITY', name)
            check(layer.dxf.lineweight == (weight_overrides or {}).get(name, weight)
                  and layer.dxf.linetype.upper() == 'CONTINUOUS', 'LAYER_STYLE', name)
            if name == 'SOURCE_INK': check(layer.is_locked() and layer.dxf.plot == 0, 'SOURCE_LAYER_POLICY')
        for name, expected in {'$INSUNITS': 0, '$LWDISPLAY': 1, '$FILLMODE': 1, '$TILEMODE': 1, '$PDMODE': 2, '$PDSIZE': .7}.items():
            check(doc.header.get(name) == expected, 'HEADER', name)
        note_height = max(w, h)/130
        center = (w/2-.5, (h-1-4*note_height)/2)
        view_height = max(h+4*note_height, w*281/194)*1.06
        active = doc.viewports.get_config('*Active')[0]
        check(math.dist(tuple(active.dxf.center)[:2], center) <= 1e-8 and abs(active.dxf.height-view_height) <= 1e-8
              and abs(active.dxf.view_twist) <= 1e-10, 'MODEL_VIEW')
        viewport = [e for e in doc.layouts.get('A4_PREVIEW').query('VIEWPORT') if e.dxf.id != 1]
        check(len(viewport) == 1, 'PAPER_VIEW_COUNT')
        if len(viewport) == 1:
            vp = viewport[0]
            check(math.dist(tuple(vp.dxf.view_center_point)[:2], center) <= 1e-8 and abs(vp.dxf.view_height-view_height) <= 1e-8
                  and abs(vp.dxf.width-194) <= 1e-8 and abs(vp.dxf.height-281) <= 1e-8 and bool(vp.dxf.flags & 16384), 'PAPER_VIEW')
        seen, raster_ids = Counter(), set(); note_count = 0
        actual_types = Counter(); native_count = 0
        for entity in entities:
            key = None
            try:
                planar(entity)
                layer, kind = entity.dxf.layer, entity.dxftype()
                actual_types[layer+':'+kind] += 1
                if layer == 'REVIEW_NOTES':
                    require(kind == 'TEXT' and entity.dxf.text == NOTE and entity.dxf.style == 'DEMO_ARIAL'
                            and abs(entity.dxf.height-note_height) <= 1e-8 and not entity.has_xdata(APP), 'draft note')
                    note_count += 1; continue
                f, numbers, piece = provenance(entity); key = f['observation']
                require(f['source'] == sha and f['source_height'] == str(h) and f['frame'] == FRAME, 'source/frame provenance')
                if layer in ('SOURCE_INK', 'UNKNOWN_INK', 'UNKNOWN_POINTS'):
                    require(f['role'] == ('source_ink' if layer == 'SOURCE_INK' else 'unknown') and not f['review']
                            and not f['observation_sha256'] and f['path'] == key and numbers == [0., 0.] and piece == 0, 'residual provenance')
                    require((layer, key) not in raster_ids, 'duplicate raster ID'); raster_ids.add((layer, key))
                    if layer == 'UNKNOWN_POINTS':
                        require(kind == 'POINT', 'singleton must be POINT')
                        p = tuple(entity.dxf.location)
                        require(all(math.isfinite(v) for v in p) and abs(p[2]) <= 1e-8, 'point planarity')
                        x, y = int(round(p[0])), int(round(h-1-p[1]))
                        require(0 <= x < w and 0 <= y < h and math.dist(p[:2], (x, h-1-y)) <= 1e-8
                                and key == f'pixel-{x}-{y}' and not masks[layer][y, x], 'point location/identity')
                        masks[layer][y, x] = True
                    else:
                        require(key.startswith(layer+'-run-') and key[len(layer)+5:].isdigit(), 'raster ID')
                        rectangle(entity, masks[layer], h)
                    continue
                require(key in selected, 'unapproved/extra observation')
                s = selected[key]; c, r = s['observation'], s['review']
                require(layer == SCHEMA[r['role']][0] and f['role'] == r['role'] and f['review'] == r['review_ref']
                        and f['path'] == c.get('path_id', key) and f['observation_sha256'] == object_digest(c), 'review/role/path provenance')
                require(abs(numbers[0]-s['error']) <= 1e-7 and abs(numbers[1]-s['bound']) <= 1e-10, 'support error provenance')
                if s['glyph'] is not None:
                    require(piece == 0 and numbers == [0., 0.], 'glyph provenance')
                    rectangle(entity, glyphs[key], h)
                    rectangle(entity, masks['TEXT_INK'], h)
                else:
                    require(piece < len(s['native']) and not seen[(key, piece)], 'duplicate/extra native piece')
                    seen[(key, piece)] += 1
                    expected = s['native'][piece]
                    if expected['type'] == 'SOLID':
                        require(kind == 'SOLID', 'head must be SOLID')
                        actual = [tuple(entity.dxf.get('vtx'+str(i))) for i in range(4)]
                        require(all(abs(p[2]) <= 1e-8 and math.dist(p[:2], q) <= 1e-8 for p, q in zip(actual, expected['vertices'])), 'head geometry')
                    else:
                        require(not compare_geometry(expected, raw_geometry(entity), 1e-8)[0], 'native geometry')
                    native_count += 1
            except (ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
                check(False, 'ENTITY', str(entity.dxf.handle)+': '+str(exc))
                if key in selected: changed.add(key)
        check(note_count == 1, 'DRAFT_NOTE_COUNT', note_count)
        for key, s in selected.items():
            if s['glyph'] is not None:
                good = np.array_equal(glyphs[key], s['glyph'])
            else:
                good = all(seen[(key, i)] == 1 for i in range(len(s['native'])))
            check(good, 'OBJECT_COVERAGE', key)
            if not good: changed.add(key)
        for name, expected in [('SOURCE_INK', ink), ('UNKNOWN_INK', residual & ~expected_points),
                               ('UNKNOWN_POINTS', expected_points), ('TEXT_INK', glyph)]:
            check(np.array_equal(masks[name], expected), 'PIXEL_PARTITION', name)
        check(not (masks['TEXT_INK'] & (masks['UNKNOWN_INK'] | masks['UNKNOWN_POINTS'])).any(), 'DUPLICATE_VISIBLE_INK')
        check(not (removed & (protect | text_protect)).any(), 'PROTECTED_ERASURE')
        changed_raster = residual | glyph | paint_all
        check(same_regions(ink, changed_raster, 8) and same_regions(np.pad(~ink, 1, constant_values=True),
              np.pad(~changed_raster, 1, constant_values=True), 4), 'SOURCE_TOPOLOGY')
        result.update(source_sha256=sha, input_ink_pixels=int(ink.sum()), replaced_ink_pixels=int(removed.sum()),
                      classified_text_pixels=int(glyph.sum()), unknown_ink_pixels=int((residual & ~expected_points).sum()),
                      unknown_points=int(expected_points.sum()), required_ids=sorted(required_ids), native_entities=native_count,
                      actual_types=dict(sorted(actual_types.items())), changed_observation_ids=sorted(changed),
                      diagnostic_changed_objects=len(changed), diagnostic_is_minimal_edit_distance=False)
    except (ValueError, KeyError, TypeError, AttributeError, IndexError, OSError, OverflowError, ezdxf.DXFError) as exc:
        check(False, 'INPUT_OR_DOCUMENT', str(exc))
    result['errors'] = errors
    result['valid_review_draft'] = not errors
    result['status'] = 'review_draft' if not errors else 'requires_review'
    return result
