"""Map manually selected navigation boxes to authenticated original RGB crops."""
from pathlib import Path
from math import floor,ceil
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import read,encoded,pin
from cad_recovery.dataset_preparation_v1 import crop_source

root=Path.cwd();selection=read(root/'configs/reviews/ocr-fresh-holdout-selection-v1.json')
readings=read(root/'configs/reviews/ocr-fresh-holdout-readings-v1.json')
out=root/'artifacts/ocr-fresh-holdout-localized-v1';out.mkdir(exist_ok=True)
ledger=[]
for spec,source in zip(readings['sources'],selection['selected_sources'],strict=True):
    index=spec['index'];w,h=source['source_size_px']
    with Image.open(root/f'artifacts/ocr-fresh-holdout-navigation-v1/nav-{index:02d}.png') as nav:nw,nh=nav.size
    for field in spec['fields']:
        slot={'slot_id':f'fresh-{index:02d}-{field["field"]}','index':index,'field':field['field'],
              'source_path':source['input'],'source_sha256':source['input_sha256'],'group_id':source['group_id'],
              'split':'test','status':field.get('status','accepted'),'reason':field.get('reason'), 'layout':spec['layout'],
              'name_is_identifier_placeholder':field.get('name_is_identifier_placeholder',False),'lines':[]}
        boxes=[]
        for j,line in enumerate(field['lines'],1):
            x0,y0,x1,y1=line['box'];box=[floor(x0*w/nw),floor(y0*h/nh),ceil(x1*w/nw),ceil(y1*h/nh)]
            rgb,proof=crop_source(root,source,box);rotation=line.get('rotation',0)
            if rotation:rgb=np.rot90(rgb,-rotation//90)
            target=out/f'{slot["slot_id"]}-{j}.png';Image.fromarray(rgb).save(target)
            slot['lines'].append({'text':line['text'],'box_xyxy':box,'rotation_clockwise':rotation,
                                 'image':pin(root,target.relative_to(root).as_posix()),'pixel_provenance':proof})
            boxes.append(box)
        box=[max(0,min(b[0] for b in boxes)-40),max(0,min(b[1] for b in boxes)-40),
             min(w,max(b[2] for b in boxes)+40),min(h,max(b[3] for b in boxes)+40)]
        rgb,proof=crop_source(root,source,box);target=out/f'{slot["slot_id"]}-context.png';Image.fromarray(rgb).save(target)
        slot['context']={'box_xyxy':box,'image':pin(root,target.relative_to(root).as_posix()),'pixel_provenance':proof}
        if slot['status']!='accepted':slot['lines']=[]
        ledger.append(slot)
(out/'labels.json').write_bytes(encoded({'selection':pin(root,'configs/reviews/ocr-fresh-holdout-selection-v1.json'),
    'readings':pin(root,'configs/reviews/ocr-fresh-holdout-readings-v1.json'),'slots':ledger}))
print(len(ledger),sum(len(s['lines']) for s in ledger))
