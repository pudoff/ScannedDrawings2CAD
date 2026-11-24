"""Prepare all remaining sources for relationship review, without OCR or labels."""
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image, ImageDraw

from cad_recovery.clear_pairs import digest
from cad_recovery.label_release_v1 import checked_pin
from cad_recovery.coallocation_review_v1 import source_regions_v1
from cad_recovery.prepared_batch_validation import _rgba_pixels
from cad_recovery.raster_regions import read_prepared_region

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/ocr-control-scope-probes-v3'
PARENT={'path':'artifacts/ocr-control-relations-final-v2.json','sha256':'05e2332182c0dcc0c1822978509181c7d8a43eb076c839afbda6b278dee1a53b'}
SCREEN={'path':'artifacts/ocr-control-relations-v2/manifest.json','sha256':'6ddb15361a9266d67e3475f8c6d2198391282a1c8b49c6948b91941304c2ba93'}
PREPARED={'path':'artifacts/prepared-raster-catalog-v2.json','sha256':'f972a535ef21787447e3e25263d74a44172817fcfe2673013c32a8595e87da33'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8')
def pin(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)}


def main():
    assert not OUT.exists()
    checked_pin(ROOT,PARENT)
    screen=read(checked_pin(ROOT,SCREEN));base=ROOT/'artifacts/ocr-control-relations-v2'
    for record in screen['files']:checked_pin(base,record)
    pending=[r for r in read(base/'dispositions.json') if r['status']=='pending_relationship_review']
    assert len(pending)==25 and len({r['coallocation_group_id'] for r in pending})==4
    prepared={r['input_path']:r for r in read(checked_pin(ROOT,PREPARED))['entries']}
    protocol={'schema_version':'ocr-control-scope-review-protocol/3.0','task':'P14.1b.7b',
        'parent':PARENT,'screening':SCREEN,'prepared_catalog':PREPARED,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'prior_representative_reviews_exist':True,'all_25_sources_fixed_before_new_review':True,
        'source_order':'inherited dispositions order; no quality selection',
        'exact_regions':'bottom-left and bottom-right min(3200,width) x min(1500,height)',
        'review_scope':['all 25 overviews','both bottom regions of each source',
            'additional exact regions when an overview reveals relevant references',
            'all catalog filenames and exact file hashes','previous saved literal relationship readings'],
        'unresolved_relationships':'retain pending; do not silently approve independence',
        'native_DXF_geometry_or_text_used':False,'OCR_run':False,'new_OCR_labels':0,
        'validation_test_admitted':False,'independent_evaluation':False,
        'control_protocol':'A distinct attainable protocol is written after this relationship review and before line annotation or model inference.'}
    protocol_path=ROOT/'configs/reviews/ocr-control-scope-review-v3.json'
    assert not protocol_path.exists();protocol_path.write_bytes(encoded(protocol))
    OUT.mkdir();sources=[]
    for i,row in enumerate(pending):
        binding=prepared[row['input_path']];source=checked_pin(ROOT,{'path':'dataset/'+row['input_path'],'sha256':row['input_sha256']})
        prep_path=checked_pin(ROOT,binding['preparation']);prep=read(prep_path)
        w,h=binding['source_size_px'];rw,rh=min(w,3200),min(h,1500)
        boxes={'left':[0,h-rh,rw,h],'right':[w-rw,h-rh,w,h]}
        if 'rgba_conversion' in prep:
            rgb,conversion=_rgba_pixels(source,{'sha256':row['input_sha256'],'size_px':[w,h],'size_bytes':source.stat().st_size},prep,prep_path.parent)
            regions={key:rgb[b[1]:b[3],b[0]:b[2]] for key,b in boxes.items()}
        else:regions=source_regions_v1(source,row['input_sha256'],boxes);conversion=None
        records=[]
        for name,box in boxes.items():
            tiled,provenance=read_prepared_region(prep_path.parent,box,max_pixels=16_777_216)
            assert np.array_equal(tiled,regions[name])
            path=OUT/f'{i:02d}-{name}.png';Image.fromarray(regions[name]).save(path)
            records.append({'name':name,'box_source_px':box,'image':pin(path),'tile_provenance':provenance,'original_pixels_equal':True})
        overview=prep_path.parent/prep['overview']['path'];assert digest(overview)==prep['overview']['sha256']
        sources.append({'index':i,'id':row['id'],'input_path':row['input_path'],'input_sha256':row['input_sha256'],
            'group_id':row['coallocation_group_id'],'source_size_px':[w,h],'preparation':binding['preparation'],
            'overview':pin(overview),'regions':records,'RGBA_conversion':conversion})
        print(json.dumps({'source':i,'total':25}),flush=True)
    contacts=[]
    for start in range(0,len(sources),5):
        canvas=Image.new('RGB',(1600,2100),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(sources[start:start+5]):
            x=(j%2)*800;y=(j//2)*700
            draw.text((x+10,y+5),f"{row['index']:02d}  {Path(row['input_path']).name}",fill='black')
            with Image.open(ROOT/row['overview']['path']) as im:
                im=im.convert('RGB');im.thumbnail((780,650));canvas.paste(im,(x+(800-im.width)//2,y+30))
        path=OUT/f'contact-{start:02d}.png';canvas.save(path);contacts.append(pin(path))
    manifest={'schema_version':'ocr-control-scope-probes/3.0','protocol':pin(protocol_path),
        'extractor':pin(Path(__file__)),'parent':PARENT,'screening':SCREEN,'prepared_catalog':PREPARED,
        'source_count':25,'original_pixel_regions_replayed':50,'sources':sources,'contacts':contacts,
        'OCR_run':False,'annotation_approved':False,'training_eligible':False}
    path=OUT/'manifest.json';path.write_bytes(encoded(manifest));print(json.dumps(pin(path)))


if __name__=='__main__':main()
