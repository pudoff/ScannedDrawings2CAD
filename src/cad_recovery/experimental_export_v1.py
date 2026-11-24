"""Editable experimental geometry plus explicit source ink and residual layers."""
from collections import Counter
from pathlib import Path
import hashlib
import math
import cv2
import ezdxf
import numpy as np
from .curve_graph_v1 import sample
from .demo_export_v1 import merged_runs, _configure_view, _observations
from .annotation_style_v1 import render_review_dxf

ROLE_LAYERS = {1: 'PART_CV', 2: 'DIMENSION_LINES_CV', 3: 'EXTENSION_LINES_CV',
    4: 'LEADERS_CV', 5: 'AXES_CV', 6: 'TEXT_INK', 7: 'FRAME_CV', 8: 'TITLE_GRID_CV', 9: 'HATCH_STROKES_CV'}
THIN = {2, 3, 4, 5, 7, 8, 9}


def _solids(model, mask, height, layer):
    for left, top, right, bottom in merged_runs(mask.astype(bool)):
        x0, x1 = left-.5, right-.5
        y0, y1 = height-.5-bottom, height-.5-top
        model.add_solid([(x0,y0),(x1,y0),(x0,y1),(x1,y1)], dxfattribs={'layer': layer})


def export(packet, target, *, ocr=None):
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    gray, roles = packet['gray'], packet['roles']
    height, width = gray.shape
    # Preserve even tiny components not used by the experimental detector.
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1] > 0
    remaining = ink.copy()
    doc = ezdxf.new('R2007', setup=True)
    doc.units = 0
    doc.header['$LWDISPLAY'] = True
    doc.styles.new('DEMO_ARIAL', dxfattribs={'font': 'arial.ttf'})
    layers = [*ROLE_LAYERS.values(), 'GEOMETRY_CV', 'THIN_GEOMETRY_CV', 'ARROWHEADS_CV', 'UNRESOLVED_INK', 'SOURCE_INK']
    for name in layers:
        doc.layers.new(name, dxfattribs={'color': 7, 'lineweight': 18 if name == 'THIN_GEOMETRY_CV' else
            13 if name in [ROLE_LAYERS[k] for k in THIN] else 35})
    doc.layers.get('SOURCE_INK').off()
    doc.layers.get('SOURCE_INK').dxf.plot = 0
    model = doc.modelspace()
    point = lambda p: (float(p[0]), height-1-float(p[1]))
    coverage = np.zeros(gray.shape, np.uint8)
    native_count = 0
    text = roles == 6
    # Keep observed text ink intact. OCR is separately switchable editable TEXT.
    for candidate in packet['geometry']:
        support = np.asarray(candidate.get('support_points', candidate.get('points_px', [])), float)
        if len(support) < 2 or not np.isfinite(support).all():
            continue
        q = np.rint(support).astype(int)
        valid = (q >= 0).all(1) & (q[:,0] < width) & (q[:,1] < height)
        q = q[valid]
        if len(q) < 2:
            continue
        classes = roles[q[:,1], q[:,0]]
        if np.mean(classes == 6) >= .15:
            continue
        known = classes[(classes >= 1) & (classes <= 9)]
        role = int(np.bincount(known).argmax()) if len(known) >= .5*len(classes) else 255
        if role == 6:
            continue
        pen = float(candidate.get('local_width', {}).get('width_px') or 1.)
        layer = ROLE_LAYERS.get(role, 'GEOMETRY_CV' if pen >= 2.5 else 'THIN_GEOMETRY_CV')
        attrs = {'layer': layer}  # BYLAYER lets CAD change both visibility and pens.
        kind = candidate['kind']
        if kind == 'line':
            p = candidate['points_px']
            model.add_line(point(p[0]), point(p[-1]), dxfattribs=attrs)
        elif kind == 'circle':
            model.add_circle(point(candidate['center_px']), candidate['radius_px'], dxfattribs=attrs)
        elif kind == 'arc':
            start, sweep = candidate['start_angle_deg'], candidate['sweep_angle_deg']
            a, b = (-(start+sweep), -start) if sweep >= 0 else (-start, -(start+sweep))
            model.add_arc(point(candidate['center_px']), candidate['radius_px'], a, b, dxfattribs=attrs)
        elif kind == 'polyline':
            model.add_lwpolyline([point(p) for p in candidate['points_px']],
                                close=candidate.get('closed', False), dxfattribs=attrs)
        else:
            continue
        native_count += 1
        # Ownership only near this actual observed source path, never its bbox.
        local = np.zeros_like(coverage)
        cv2.polylines(local, [q.astype(np.int32)], False, 255, max(1, math.ceil(pen+1)))
        coverage |= local
    heads = 0
    for head in packet['heads']:
        polygon = np.asarray(head.get('polygon_px', []), float)
        if polygon.shape != (3,2) or not np.isfinite(polygon).all():
            continue
        if np.any(polygon < 0) or np.any(polygon[:,0] >= width) or np.any(polygon[:,1] >= height):
            continue
        # Only source-supported triangles; no growth into a blank candidate bbox.
        mask = np.zeros_like(coverage)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 255)
        if not np.any(mask) or np.mean(ink[mask > 0]) < .7 or np.any(text & (mask > 0)):
            continue
        model.add_solid([point(p) for p in polygon], dxfattribs={'layer': 'ARROWHEADS_CV'})
        coverage |= mask
        heads += 1
    remaining &= ~((coverage > 0) & ~text)
    # Every unmodelled source pixel is retained in exactly one visible ink layer.
    for role, layer in ROLE_LAYERS.items():
        selected = remaining & (roles == role)
        _solids(model, selected, height, layer)
        remaining[selected] = False
    _solids(model, remaining, height, 'UNRESOLVED_INK')
    _solids(model, ink, height, 'SOURCE_INK')
    _observations(doc, height, None, ocr)
    # Return geometry to the preprocessing coordinate frame used by the viewer.
    original_width, original_height = packet['summary']['input_size_px']
    sx, sy = original_width/width, original_height/height
    if sx != 1 or sy != 1:
        from ezdxf.math import Matrix44
        from ezdxf.transform import copies
        transform = Matrix44.scale(sx, sy, 1) @ Matrix44.translate((sx-1)/2, (sy-1)/2, 0)
        logger, transformed = copies(list(model), transform)
        if len(logger):
            raise ValueError('Experimental DXF coordinate transform failed')
        for entity in list(model):
            model.delete_entity(entity)
        for entity in transformed:
            model.add_entity(entity)
    _configure_view(doc, (-.5,-.5,original_width-.5,original_height-.5), pixel_units=True)
    doc.saveas(target)
    reopened = ezdxf.readfile(target)
    audit = reopened.audit()
    if audit.errors or audit.fixes:
        raise ValueError('Experimental DXF requires structural repair')
    if reopened.modelspace().query('IMAGE UNDERLAY DIMENSION'):
        raise ValueError('Unexpected external raster or unverified DIMENSION')
    preview = target.with_name(target.stem+'-dxf.png')
    render_review_dxf(target, preview)
    counts = dict(Counter(e.dxftype() for e in reopened.modelspace()))
    layer_counts = dict(Counter(e.dxf.layer for e in reopened.modelspace()))
    return {'schema_version': 'experimental-export/1.0', 'mode': 'experimental_cv_draft',
        'status': 'needs_review', 'dxf': target.name, 'preview': preview.name,
        'dxf_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'preview_sha256': hashlib.sha256(preview.read_bytes()).hexdigest(),
        'empty': not bool(ink.any()), 'geometry_entity_count': native_count,
        'arrowhead_count': heads, 'modelspace_counts': counts, 'layer_counts': layer_counts,
        'source_size_px': [original_width,original_height], 'working_size_px': [width,height],
        'foreground_pixels': int(ink.sum()), 'residual_pixels': int(remaining.sum()),
        'unmodeled_source_pixels_dropped': 0,
        'source_reference_layer': 'SOURCE_INK', 'source_reference_initially_hidden': True,
        'source_reference_resolution': 'experimental_working_raster',
        'unresolved_ink_preserved': True, 'preview_rendered_from_dxf': True,
        'physical_geometry_approved': False, 'native_dimensions': 0, 'trueview_gui_verified': False,
        'audit_errors': 0, 'audit_fixes': 0, 'external_images': 0, 'units': 'processed_pixels_unitless',
        'dxf_version': reopened.dxfversion, 'plot_layout': 'A4_PREVIEW',
        'requested_rotation_clockwise_deg': 0, 'experimental_cv': packet['summary']}
