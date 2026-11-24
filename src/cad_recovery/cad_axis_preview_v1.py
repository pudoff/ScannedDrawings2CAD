"""Render Autodesk A-alignment in a temporary in-memory view of the actual DXF."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import ezdxf
import numpy as np
from .cad_object_export_v2 import APPID,read_provenance


def render(path,target):
    from ezdxf.addons.drawing import Frontend,RenderContext,layout
    from ezdxf.addons.drawing.config import Configuration,BackgroundPolicy,ColorPolicy,LineweightPolicy
    from ezdxf.addons.drawing.svg import SVGBackend
    from ezdxf.math import BoundingBox2d
    import fitz
    doc=ezdxf.readfile(path)
    for entity in list(doc.modelspace().query('LINE')):
        if not entity.has_xdata(APPID):continue
        proof=read_provenance(entity);evidence=proof.get('axis_pattern_evidence')
        if evidence is None:continue
        start=np.array(entity.dxf.start);end=np.array(entity.dxf.end);u=end-start;u/=np.linalg.norm(u)
        attrs={'layer':entity.dxf.layer,'color':256,'lineweight':-1,'linetype':'CONTINUOUS'}
        for low,high in evidence['A_aligned_runs']:
            doc.modelspace().add_line(start+(low-evidence['lo'])*u,start+(high-evidence['lo'])*u,dxfattribs=attrs)
        doc.modelspace().delete_entity(entity)
    backend=SVGBackend()
    Frontend(RenderContext(doc),backend,config=Configuration(background_policy=BackgroundPolicy.WHITE,
        color_policy=ColorPolicy.BLACK,lineweight_policy=LineweightPolicy.ABSOLUTE)).draw_layout(doc.modelspace())
    bounds=BoundingBox2d([tuple(doc.header['$EXTMIN'])[:2],tuple(doc.header['$EXTMAX'])[:2]])
    tree=ET.fromstring(backend.get_string(layout.Page(210,297),render_box=bounds))
    rules={key:value for e in tree.iter() if e.tag.endswith('style') for key,value in re.findall(r'\.([A-Za-z0-9_-]+)\s*\{([^}]+)\}',e.text or '')}
    for element in tree.iter():
        for cls in element.get('class','').split():
            for part in rules.get(cls,'').split(';'):
                if ':' in part:
                    key,value=part.split(':',1);element.set(key.strip(),value.strip())
    with fitz.open(stream=ET.tostring(tree),filetype='svg') as rendered:
        Path(target).write_bytes(rendered[0].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).tobytes('png'))
