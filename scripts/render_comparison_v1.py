"""Bounded child rendering of a published DXF; no OCR or conversion pipeline."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def render(request):
    import ezdxf
    from ezdxf.addons.drawing import Frontend,RenderContext,layout
    from ezdxf.addons.drawing.config import Configuration,ColorPolicy,BackgroundPolicy
    from ezdxf.addons.drawing.svg import SVGBackend
    from ezdxf.math import BoundingBox2d
    path=Path(request['path']);data=path.read_bytes()
    if len(data)>16*1024**2 or hashlib.sha256(data).hexdigest()!=request['sha256']:raise ValueError('DXF integrity/size check failed')
    doc=ezdxf.readfile(path);model=doc.modelspace()
    if any(any(c in style.dxf.font for c in ('/','\\',':')) for style in doc.styles):raise ValueError('External font paths unsupported')
    if len(model)>100000 or sum(len(block) for block in doc.blocks)>200000:raise ValueError('DXF object limit')
    if model.query('IMAGE PDFUNDERLAY DWFUNDERLAY DGNUNDERLAY'):raise ValueError('External graphic references unsupported')
    counts=Counter(e.dxftype() for e in model)
    if any(block.block.dxf.flags&4 for block in doc.blocks):raise ValueError('External block references unsupported')
    audit=doc.audit()
    if audit.errors or audit.fixes:raise ValueError('DXF requires repairs')
    common={'modelspace_counts':dict(counts),'audit_errors':0,'audit_fixes':0,
        'hidden_layers':[l.dxf.name for l in doc.layers if l.is_off() or l.is_frozen()],
        'visible_entity_count':sum(not doc.layers.get(e.dxf.layer).is_off() and not doc.layers.get(e.dxf.layer).is_frozen() for e in model),
        'dxf_sha256':request['sha256'],'physical_geometry_approved':False}
    if request['render_box'] is None:return common
    l,b,r,t=request['render_box']
    if not 0<r-l<=1e6 or not 0<t-b<=1e6:raise ValueError('Invalid rendering frame')
    backend=SVGBackend();random.seed(181)
    Frontend(RenderContext(doc),backend,config=Configuration(background_policy=BackgroundPolicy.OFF,
        color_policy=ColorPolicy.BLACK,hatching_timeout=3.)).draw_layout(model)
    # Fixed geometry frame, not independent fit to each drawing's content bbox.
    text=backend.get_string(layout.Page(r-l,t-b),render_box=BoundingBox2d([(l,b),(r,t)]))
    if len(text.encode())>16*1024**2:raise ValueError('SVG byte budget')
    tree=ET.fromstring(text);view=list(map(float,tree.attrib['viewBox'].split()))
    if view[:2]!=[0,0]:raise ValueError('Unexpected SVG origin')
    # SVGBackend emits glyph paths. Refuse scripts, images, foreign content or external URLs.
    allowed={'svg','g','path','defs','style','clipPath','rect','polygon','polyline','line','circle','ellipse'}
    for e in tree.iter():
        if e.tag.split('}')[-1] not in allowed:raise ValueError('Unsupported SVG element')
        for k,v in e.attrib.items():
            if k.lower().startswith('on') or (k.endswith('href') and not v.startswith('#')) or re.search(r'url\(\s*["\x27]?(?:https?:|data:|//)',v,re.I):raise ValueError('External SVG content')
    return {**common,'svg':text,'svg_size':view[2:]}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=render(json.loads(Path(args.request).read_text(encoding='utf-8')))
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
