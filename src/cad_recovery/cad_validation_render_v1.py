"""Deterministic bounded review previews; these do not constitute CAD-GUI proof."""
from pathlib import Path
import random
import re
import xml.etree.ElementTree as ET

import ezdxf
import fitz
from ezdxf.addons.drawing import Frontend, RenderContext, layout
from ezdxf.addons.drawing.config import Configuration, BackgroundPolicy, ColorPolicy
from ezdxf.addons.drawing.svg import SVGBackend
from ezdxf.math import BoundingBox2d

from .cad_validation_v1 import validate_cad_v1
from .graph_v2 import read_graph_v2


def render_validated_cad_v1(graph, dxf_path, output, *, binding_report=None, workspace_root=None):
    """Revalidate bytes and render a fresh in-memory copy without modifying DXF."""
    graph = read_graph_v2(graph)
    report = validate_cad_v1(graph, dxf_path, binding_report=binding_report, workspace_root=workspace_root)
    if report['status'] == 'failed': raise ValueError('Rendering requires a structurally matching validated DXF')
    if max(graph.page.width_mm, graph.page.height_mm) > 2000: raise ValueError('Preview paper size budget exceeded')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in ('sheet', 'geometry'):
        svg_path, png_path = output/(name+'.svg'), output/(name+'.png')
        if svg_path.exists() or png_path.exists(): raise FileExistsError('Preserve previous review previews')
        doc = ezdxf.readfile(dxf_path)
        for block in doc.blocks:
            if len(block) > 2000: raise ValueError('Preview block budget exceeded')
            for entity in block:
                if entity.dxftype() == 'INSERT':
                    referenced = doc.blocks.get(entity.dxf.name)
                    if referenced is None or len(referenced) > 100 or any(e.dxftype() not in ('SOLID', 'LINE', 'ARC', 'CIRCLE', 'LWPOLYLINE') for e in referenced):
                        raise ValueError('Unsupported nested block resource in preview')
        if name == 'geometry':
            model = doc.modelspace()
            for entity in list(model):
                if entity.dxftype() in ('DIMENSION', 'TEXT', 'MTEXT', 'HATCH'): model.delete_entity(entity)
        backend = SVGBackend()
        state = random.getstate()
        try:
            random.seed(0)
            config = Configuration(background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK)
            Frontend(RenderContext(doc), backend, config=config).draw_layout(doc.layouts.get('Sheet'))
        finally: random.setstate(state)
        svg = backend.get_string(layout.Page(graph.page.width_mm, graph.page.height_mm),
            settings=layout.Settings(fit_page=False, scale=1),
            render_box=BoundingBox2d([(0, 0), (graph.page.width_mm, graph.page.height_mm)]))
        if len(svg) > 16*1024*1024: raise ValueError('Generated preview budget exceeded')
        with svg_path.open('x', encoding='utf-8', newline='\n') as stream: stream.write(svg)
        tree = ET.fromstring(svg)
        rules = {key: value for e in tree.iter() if e.tag.endswith('style')
                 for key, value in re.findall(r'\.([A-Za-z0-9_-]+)\s*\{([^}]+)\}', e.text or '')}
        for element in tree.iter():
            for cls in element.get('class', '').split():
                for key, value in [part.split(':', 1) for part in rules.get(cls, '').split(';') if ':' in part]:
                    element.set(key.strip(), value.strip())
        with fitz.open(stream=ET.tostring(tree), filetype='svg') as document:
            raster = document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            with png_path.open('xb') as stream: stream.write(raster.tobytes('png'))
            result[name] = {'width_px': raster.width, 'height_px': raster.height}
    return {'files': ['sheet.svg', 'sheet.png', 'geometry.svg', 'geometry.png'], 'previews': result,
            'hatch_renderer_seed': 0, 'target_cad_gui_verified': False}
