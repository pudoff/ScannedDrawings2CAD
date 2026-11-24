"""Service kernel: bounded additional seam contexts, immutable v1 producers."""
from pathlib import Path
from .dataset_layout_v1 import read,encoded,digest
from .family_catalog_v2 import require
from .native_tiled_v1 import window_boxes,window_id,retry_boxes,analyze_window,assemble

LIMITS={'tile_size':1536,'overlap':256,'max_windows':256,'max_retry_windows':16,'max_source_pixels':300_000_000}

def run(prepared,output,engine):
    prepared=Path(prepared);output=Path(output);report=read(prepared/'preparation.json')
    width,height=report['source_size_px']
    require(type(width) is int and type(height) is int and 0<width<=32768 and 0<height<=32768 and width*height<=300_000_000,'Invalid native source dimensions')
    require(report['status']=='prepared' and report['source_integrity_validated'] is True,'Incomplete native preparation')
    region=[0,0,width,height];boxes=window_boxes(region);require(len(boxes)+16<=256,'Native window budget exceeded')
    output.mkdir(parents=True,exist_ok=False);rows=[];seen=set()
    for box in boxes:
        rows.append(analyze_window(prepared,box,output/window_id(box),engine));seen.add(tuple(box))
    retries=0
    while retries<16:
        proposed,_=retry_boxes(rows,region,seen)
        batch=proposed[:16-retries]
        if not batch:break
        for box in batch:
            rows.append(analyze_window(prepared,box,output/window_id(box),engine));seen.add(tuple(box));retries+=1
    pending,extra=retry_boxes(rows,region,seen);result=assemble(rows,report['source_sha256'],region,deferred_retries=len(pending)+extra)
    result.update(status='needs_review',source_size_px=[width,height],base_window_count=len(boxes),retry_window_count=retries,
                  preparation_sha256=digest(prepared/'preparation.json'),kernel_version='native-profile-kernel/2.0',limits=LIMITS)
    (output/'observations.json').write_bytes(encoded(result));return result
