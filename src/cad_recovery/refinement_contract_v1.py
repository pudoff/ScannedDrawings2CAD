"""Ready CAD is authoritative; optional source proposals cannot replace it."""
from pathlib import Path
import hashlib
import ezdxf
from ezdxf.lldxf.tagwriter import TagCollector
from .dataset_layout_v1 import encoded, digest

POLICY = dict(version='bounded-source-review/1.0', maximum_rois=2, maximum_core_side=1024,
              halo=128, geometry_seconds=60, table_seconds=300, memory_MiB=3072,
              output_bytes=4*1024**3, output_files=60000,
              geometry_publication='separate review DXF; no automatic replacement',
              outside_ROIs='ready baseline; native search not claimed',
              automatic_page_routing=False, production_admitted=False)
GEOMETRY = {'LINE','ARC','CIRCLE','LWPOLYLINE','POLYLINE','SPLINE','ELLIPSE','DIMENSION','HATCH','INSERT'}
GRID = {'TITLE_GRID_CV','TABLE_GRID','TITLE_GRID'}


def tag_hash(entity, *, text_layer=False):
    tags=[(t.code,t.value) for t in TagCollector.dxftags(entity) if not(text_layer and t.code==8)]
    return hashlib.sha256(encoded(tags)).hexdigest()


def snapshot(path):
    doc=ezdxf.readfile(path);audit=doc.audit()
    if audit.errors or audit.fixes or doc.units!=0:
        raise ValueError('Clean unitless ready CAD required')
    protected={};texts={}
    for e in doc.modelspace():
        kind=e.dxftype()
        if kind in ('TEXT','MTEXT'):
            texts[e.dxf.handle]=tag_hash(e,text_layer=True)
        elif kind in GEOMETRY:
            # Grid rules may be replaced by the existing explicit table route.
            # A patterned LINE is always protected, even on a grid layer.
            lt=e.dxf.get('linetype','BYLAYER')
            if lt.upper()=='BYLAYER':lt=doc.layers.get(e.dxf.layer).dxf.linetype
            replaceable_grid=e.dxf.layer in GRID and lt.upper()=='CONTINUOUS' and kind in ('LINE','LWPOLYLINE') and not getattr(e,'has_arc',False)
            if not replaceable_grid:protected[e.dxf.handle]=tag_hash(e)
    tables={name:{entry.dxf.name:tag_hash(entry) for entry in getattr(doc,name)}
            for name in ('layers','linetypes','styles','dimstyles')}
    blocks={block.name:[tag_hash(e) for e in block] for block in doc.blocks
            if not block.name.upper().startswith(('*MODEL_SPACE','*PAPER_SPACE'))}
    return dict(source_CAD_sha256=digest(path), protected=protected,texts=texts,tables=tables,blocks=blocks)


def verify_preservation(before, target):
    """No lost global primitive, CENTER pattern/resource, text content or block."""
    doc=ezdxf.readfile(target);audit=doc.audit();errors=[]
    if audit.errors or audit.fixes or doc.units!=0:errors.append('CAD_audit_or_units')
    for handle, expected in before['protected'].items():
        e=doc.entitydb.get(handle)
        if e is None or not e.is_alive or tag_hash(e)!=expected:errors.append('primitive:'+handle)
    for handle, expected in before['texts'].items():
        e=doc.entitydb.get(handle)
        if e is None or not e.is_alive or tag_hash(e,text_layer=True)!=expected:errors.append('text:'+handle)
    for name, entries in before['tables'].items():
        table=getattr(doc,name)
        for key, expected in entries.items():
            if key not in table or tag_hash(table.get(key))!=expected:errors.append(name+':'+key)
    for name, expected in before['blocks'].items():
        if name not in doc.blocks or [tag_hash(e) for e in doc.blocks[name]]!=expected:errors.append('block:'+name)
    return dict(accepted=not errors,errors=errors,protected_entities=len(before['protected']),
                retained_texts=len(before['texts']),source_CAD_sha256=before['source_CAD_sha256'],
                target_sha256=digest(target),automatic_geometry_credit=False)


def validate_rois(rows, pages):
    if not isinstance(rows,list) or not 1<=len(rows)<=POLICY['maximum_rois']:
        raise ValueError('One or two explicit ROIs required')
    result=[];ids=set()
    for row in rows:
        if set(row)!= {'id','page_index','core'} or not isinstance(row['id'],str) or not row['id'].isascii() or not row['id'].replace('-','').isalnum() or len(row['id'])>40 or row['id'] in ids:
            raise ValueError('Unique bounded ASCII ROI ID required')
        index=row['page_index'];box=row['core']
        if type(index)is not int or not 0<=index<len(pages):raise ValueError('Source page required')
        w,h=pages[index]['size_px']
        if not isinstance(box,list) or len(box)!=4 or any(type(x)is not int for x in box):raise ValueError('Integer source ROI required')
        l,t,r,b=box
        if not 0<=l<r<=w or not 0<=t<b<=h or max(r-l,b-t)>POLICY['maximum_core_side']:
            raise ValueError('ROI exceeds source or fixed core budget')
        ids.add(row['id']);halo=POLICY['halo']
        result.append({**row,'box':[max(0,l-halo),max(0,t-halo),min(w,r+halo),min(h,b+halo)]})
    return result
