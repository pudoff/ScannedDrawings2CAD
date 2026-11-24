"""Visible, self-contained review DXF; never claims physical reconstruction.

Raster ink is encoded losslessly relative to the supplied binary mask. CV/OCR
observations have separate, initially hidden layers and cannot replace the ink.
"""
from collections import Counter
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import ezdxf
from ezdxf import bbox
import numpy as np
from PIL import Image

MAX_RECTANGLES = 250_000
MAX_PIXELS = 25_000_000


def merged_runs(mask, *, limit=MAX_RECTANGLES):
    """Coalesce identical horizontal runs; preserve holes and isolated pixels."""
    if mask.ndim != 2 or mask.dtype != bool or not 0 < mask.size <= MAX_PIXELS:
        raise ValueError('Expected a bounded boolean ink mask')
    active, rectangles = {}, []
    for y, row in enumerate(mask):
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        current = set(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        for left, right in active.keys() - current:
            rectangles.append((int(left), active.pop((left, right)), int(right), y))
        for key in current:
            active.setdefault(key, y)
        if len(rectangles) + len(active) > limit:
            raise ValueError(f'Raster draft exceeds {limit} vector rectangles; use a smaller page region')
    rectangles.extend((int(left), top, int(right), mask.shape[0]) for (left, right), top in active.items())
    return sorted(rectangles)


def _layer(doc, name, *, color=7, hidden=False):
    if name not in doc.layers:
        doc.layers.new(name, dxfattribs={'color': color})
    if hidden:
        doc.layers.get(name).off()


def _observations(doc, height, geometry, ocr):
    """Store only measured pixel observations; never create DIMENSIONs."""
    model = doc.modelspace()
    _layer(doc, 'CV_CANDIDATES', color=3, hidden=True)
    _layer(doc, 'OCR_OBSERVATIONS', color=1, hidden=True)
    point = lambda p: (float(p[0]), height - 1 - float(p[1]))
    for item in (geometry or {}).get('candidates', []):
        attrs = {'layer': 'CV_CANDIDATES'}
        if item['kind'] == 'line':
            model.add_line(point(item['points_px'][0]), point(item['points_px'][1]), dxfattribs=attrs)
        elif item['kind'] == 'circle':
            model.add_circle(point(item['center_px']), float(item['radius_px']), dxfattribs=attrs)
        elif item['kind'] == 'arc':
            start, sweep = float(item['start_angle_deg']), float(item['sweep_angle_deg'])
            model.add_arc(point(item['center_px']), float(item['radius_px']),
                          -(start + sweep), -start, dxfattribs=attrs)
    for item in (ocr or {}).get('lines', []):
        raw = item.get('raw', item.get('text', ''))
        if not raw:
            continue
        polygon = np.asarray(item['polygon_px'], dtype=float)
        delta = polygon[1] - polygon[0]
        h = float(np.linalg.norm(polygon[3] - polygon[0]))
        if h <= 0:
            continue
        model.add_text(raw, dxfattribs={'layer': 'OCR_OBSERVATIONS', 'style': 'DEMO_ARIAL',
            'insert': point(polygon[3]), 'height': h * .8,
            'rotation': -math.degrees(math.atan2(delta[1], delta[0]))})


def _configure_view(doc, bounds, *, pixel_units):
    left, bottom, right, top = bounds
    width, height = max(1., right - left), max(1., top - bottom)
    _layer(doc, 'REVIEW_NOTES')
    if 'DEMO_ARIAL' not in doc.styles:
        doc.styles.new('DEMO_ARIAL', dxfattribs={'font': 'arial.ttf'})
    note_height = max(width, height) / 130
    units = 'PIXEL UNITS' if pixel_units else 'PDF PAPER COORDINATES'
    doc.modelspace().add_text(f'REVIEW DRAFT / {units} / PHYSICAL DIMENSIONS UNCONFIRMED',
        dxfattribs={'layer': 'REVIEW_NOTES', 'style': 'DEMO_ARIAL',
                   'insert': (left, bottom - 3 * note_height), 'height': note_height})
    bottom -= 4 * note_height
    center = ((left + right) / 2, (bottom + top) / 2)
    view_height = max(top - bottom, width * 281 / 194) * 1.06
    doc.header['$FILLMODE'] = 1
    doc.header['$LWDISPLAY'] = False
    doc.header['$TILEMODE'] = 1
    doc.header['$EXTMIN'] = (left, bottom, 0)
    doc.header['$EXTMAX'] = (right, top, 0)
    # ezdxf refreshes header extents from the model layout on save.
    doc.modelspace().reset_extents((left, bottom, 0), (right, top, 0))
    doc.header['$LIMMIN'] = (left, bottom)
    doc.header['$LIMMAX'] = (right, top)
    doc.modelspace().reset_limits((left, bottom), (right, top))
    doc.set_modelspace_vport(height=view_height, center=center)
    sheet = doc.layouts.new('A4_PREVIEW')
    sheet.page_setup(size=(210, 297), margins=(8, 8, 8, 8), units='mm', device='DWG To PDF.pc3')
    sheet.reset_extents((0, 0, 0), (194, 281, 0))
    _layer(doc, 'DEMO_VIEWPORTS', color=8)
    doc.layers.get('DEMO_VIEWPORTS').dxf.plot = 0
    vp = sheet.add_viewport(center=(97, 140.5), size=(194, 281),
        view_center_point=center, view_height=view_height, dxfattribs={'layer': 'DEMO_VIEWPORTS'})
    vp.dxf.flags = int(vp.dxf.flags) | 16384


def render_dxf(path, target):
    """Render the reopened DXF, not the input raster or an observation overlay."""
    from ezdxf.addons.drawing import Frontend, RenderContext, layout
    from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy
    from ezdxf.addons.drawing.svg import SVGBackend
    import fitz
    doc = ezdxf.readfile(path)
    backend = SVGBackend()
    Frontend(RenderContext(doc), backend, config=Configuration(
        background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK,
    )).draw_layout(doc.modelspace())
    svg = backend.get_string(layout.Page(210, 297))
    tree = ET.fromstring(svg)
    # MuPDF does not consistently apply the SVG backend's CSS classes.
    rules = {key: value for e in tree.iter() if e.tag.endswith('style')
             for key, value in re.findall(r'\.([A-Za-z0-9_-]+)\s*\{([^}]+)\}', e.text or '')}
    for element in tree.iter():
        for cls in element.get('class', '').split():
            for part in rules.get(cls, '').split(';'):
                if ':' in part:
                    key, value = part.split(':', 1)
                    element.set(key.strip(), value.strip())
    with fitz.open(stream=ET.tostring(tree), filetype='svg') as rendered:
        pix = rendered[0].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        Path(target).write_bytes(pix.tobytes('png'))


def export_review_dxf(graph, binary_path, output, *, geometry=None, ocr=None, rotation_deg=0):
    """Create one immutable draft and return its verified content contract."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if rotation_deg not in (0, 90, 180, 270):
        raise ValueError('Rotation must be a quarter turn')
    raster = graph.page.kind != 'vector'
    report = {'schema_version': 'demo-export/1.0', 'status': 'needs_review',
        'mode': 'raster_ink_draft' if raster else 'pdf_vector_draft',
        'physical_geometry_approved': False, 'native_dimensions': 0,
        'trueview_gui_verified': False, 'preview_rendered_from_dxf': True,
        'units': 'processed_pixels_unitless' if raster else 'pdf_paper_mm_not_part_scale'}
    if raster:
        with Image.open(binary_path) as im:
            if im.width * im.height > MAX_PIXELS:
                raise ValueError('Raster draft exceeds 25 megapixels')
            binary = np.asarray(im.convert('L'))
        if not np.isin(binary, (0, 255)).all():
            raise ValueError('Raster draft requires the recorded binary mask')
        mask = binary == 0
        height, width = mask.shape
        rectangles = merged_runs(mask)
        doc = ezdxf.new('R2007', setup=True)
        doc.units = 0
        doc.styles.new('DEMO_ARIAL', dxfattribs={'font': 'arial.ttf'})
        _layer(doc, 'SCAN_INK_DRAFT')
        for left, top, right, bottom in rectangles:
            x0, x1 = left - .5, right - .5
            y0, y1 = height - .5 - bottom, height - .5 - top
            doc.modelspace().add_solid([(x0, y0), (x1, y0), (x0, y1), (x1, y1)],
                                      dxfattribs={'layer': 'SCAN_INK_DRAFT'})
        _observations(doc, height, geometry, ocr)
        bounds = (-.5, -.5, width - .5, height - .5)
        count = len(rectangles)
        report.update(foreground_pixels=int(mask.sum()), mask_sha256=hashlib.sha256(
            Path(binary_path).read_bytes()).hexdigest(), source_size_px=[width, height],
            rectangle_limit=MAX_RECTANGLES, hidden_observation_layers=['CV_CANDIDATES', 'OCR_OBSERVATIONS'])
    else:
        from .exporter import export_dxf
        baseline = export_dxf(graph, output)
        if baseline['status'] == 'failed':
            raise ValueError('PDF vector export failed validation')
        doc = ezdxf.readfile(output)
        if rotation_deg:
            from ezdxf.math import Matrix44
            transform = Matrix44.z_rotate(math.radians(-rotation_deg))
            for entity in doc.modelspace():
                entity.transform(transform)
        count = len(doc.modelspace())
        extent = bbox.extents(doc.modelspace(), fast=True)
        bounds = ((extent.extmin.x, extent.extmin.y, extent.extmax.x, extent.extmax.y)
                  if extent.has_data else (0, 0, graph.page.width_mm, graph.page.height_mm))
        # Standard font improves reproducibility for Cyrillic extracted PDF text.
        doc.styles.get('Standard').dxf.font = 'arial.ttf'
    _configure_view(doc, bounds, pixel_units=raster)
    doc.saveas(output)
    reopened = ezdxf.readfile(output)
    auditor = reopened.audit()
    if auditor.errors or auditor.fixes:
        raise ValueError('Saved demo DXF requires repairs')
    if reopened.modelspace().query('IMAGE UNDERLAY PDFUNDERLAY DWFUNDERLAY DGNUNDERLAY DIMENSION'):
        raise ValueError('Unexpected external image, underlay or accepted dimension in review draft')
    if raster:
        restored = np.zeros(mask.shape, dtype=bool)
        for solid in reopened.modelspace().query('SOLID[layer=="SCAN_INK_DRAFT"]'):
            points = [solid.dxf.get(f'vtx{i}') for i in range(4)]
            left, right = round(min(p.x for p in points) + .5), round(max(p.x for p in points) + .5)
            top = round(height - .5 - max(p.y for p in points))
            bottom = round(height - .5 - min(p.y for p in points))
            restored[top:bottom, left:right] = True
        if not np.array_equal(restored, mask):
            raise ValueError('Raster ink changed during DXF round trip')
        report['mask_roundtrip_exact'] = True
    preview = output.with_name(output.stem + '-dxf.png')
    render_dxf(output, preview)
    report.update(modelspace_counts=dict(Counter(e.dxftype() for e in reopened.modelspace())),
        geometry_entity_count=count, empty=count == 0, dxf_version=reopened.dxfversion,
        requested_rotation_clockwise_deg=rotation_deg,
        audit_errors=0, audit_fixes=0, external_images=0,
        dxf_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        preview_sha256=hashlib.sha256(preview.read_bytes()).hexdigest(),
        dxf=output.name, preview=preview.name, plot_layout='A4_PREVIEW')
    return report
