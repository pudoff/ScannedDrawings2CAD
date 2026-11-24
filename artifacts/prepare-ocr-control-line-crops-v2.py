"""Exact crop proposals for every fixed title field; no OCR or label decisions."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from cad_recovery.clear_pairs import digest
from cad_recovery.label_release_v1 import checked_pin

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/ocr-control-line-crops-v2'
PROTOCOL={'path':'configs/reviews/ocr-control-lines-protocol-v2.json','sha256':'bea086abe02ece838162b2bb840a89e174eb742e723b1ad1ab18bfcbafbdf010'}


def pin(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)}
def spans(values):return [(int(g[0]),int(g[-1])) for g in np.split(values,np.where(np.diff(values)>1)[0]+1) if len(g)]


def cells(index,a):
    if index>=9:
        mask=(a[:,:,0]<80)&(a[:,:,2]>220)&(a[:,:,1]>70)&(a[:,:,1]<200)
        cols=spans(np.flatnonzero(mask[950:1400].sum(axis=0)>200))
        left,name_end,right=cols[-5],cols[-4],cols[-1]
        return {'designation':[left[1]+3,925,right[0]-3,1048],
                'part_name':[left[1]+3,1063,name_end[0]-3,1278], 'adjacent_reference':None}
    mask=(a[:,:,1]<80)&(a[:,:,0]>220)&(a[:,:,2]>220) if index<7 else np.all(a<100,axis=2)
    cols=spans(np.flatnonzero(mask.sum(axis=0)>a.shape[0]*.45))
    left,name_end,reference,right=cols if index<7 else cols[1:5]
    rows=spans(np.flatnonzero(mask.sum(axis=1)>a.shape[1]*.45))
    code_bottom,name_bottom=(rows[0][0],rows[1][0]) if index in (3,7) else (rows[1][0],rows[2][0])
    code_top=0 if index in (3,7) else rows[0][1]+3
    return {'designation':[left[1]+3,code_top,reference[0]-3,code_bottom-3],
            'part_name':[left[1]+3,code_bottom+8,name_end[0]-3,name_bottom-3],
            'adjacent_reference':[reference[1]+3,code_top,right[0]-3,code_bottom-3]}


def main():
    assert not OUT.exists();protocol=json.loads(checked_pin(ROOT,PROTOCOL).read_text(encoding='utf-8'))
    OUT.mkdir();sources=[];present=[]
    for source in protocol['sources']:
        index=source['index'];kind='left' if index in (16,17,18,21) else 'right'
        parent=source['parents'][kind]
        with Image.open(checked_pin(ROOT,parent['image'])) as im:
            assert im.mode=='RGB';a=np.asarray(im).copy()
        boxes=cells(index,a);fields=[]
        for field in protocol['fields']:
            identifier='ctl2-'+source['id'].removeprefix('ocr-control-')+'-'+field.replace('_','-')
            cell=boxes[field]
            record={'id':identifier,'field':field,'parent_kind':kind,'cell_box_parent_xyxy':cell,
                'source_box_xyxy':None,'rotation_cw_degrees':0,'raw_crop_pin':None,'image_pin':None}
            if cell is not None:
                x1,y1,x2,y2=cell;piece=a[y1:y2,x1:x2]
                yy,xx=np.where(np.any(piece<200,axis=2));assert len(xx)>0,(index,field)
                # Keep original white pixels only. A narrow cell limits padding.
                box=[max(x1,x1+int(xx.min())-12),max(y1,y1+int(yy.min())-12),
                     min(x2,x1+int(xx.max())+13),min(y2,y1+int(yy.max())+13)]
                raw=a[box[1]:box[3],box[0]:box[2]].copy()
                px,py=parent['box_source_xyxy'][:2]
                record['source_box_xyxy']=[box[0]+px,box[1]+py,box[2]+px,box[3]+py]
                path=OUT/(identifier+'.png');Image.fromarray(raw).save(path)
                record.update(raw_crop_pin=pin(path),image_pin=pin(path))
                present.append((index,field,path))
            fields.append(record)
        sources.append({'source_id':source['id'],'index':index,'fields':fields})
    contacts=[]
    for start in range(0,len(present),12):
        canvas=Image.new('RGB',(2280,1040),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for j,(index,field,path) in enumerate(present[start:start+12]):
            x=j%3*760;y=j//3*260;draw.text((x+8,y+5),f'{index:02d} / {field}',fill='black')
            with Image.open(path) as im:
                im.thumbnail((740,220));canvas.paste(im,(x+(760-im.width)//2,y+30+(220-im.height)//2))
        path=OUT/f'contact-{start:02d}.png';canvas.save(path);contacts.append(pin(path))
    manifest={'schema_version':'ocr-control-line-crops/2.0','protocol':PROTOCOL,'extractor':pin(Path(__file__)),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'sources':sources,'contacts':contacts,
        'candidate_count':sum(len(s['fields']) for s in sources),'present_field_count':len(present),
        'method':'frame-guided field-cell geometry, diagnostic ink bounds and up to 12 original white pixels; no OCR',
        'OCR_run':False,'annotation_approved':False,'original_source_pixel_replay_performed':False}
    path=OUT/'manifest.json';path.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(pin(path)))


if __name__=='__main__':main()
