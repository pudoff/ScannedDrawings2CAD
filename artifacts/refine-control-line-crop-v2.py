"""Preserve the first crop proposal and extend one clipped crop by two pixels."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cad_recovery.clear_pairs import digest
from cad_recovery.label_release_v1 import checked_pin

ROOT=Path(__file__).resolve().parents[1]
PREVIOUS={'path':'artifacts/ocr-control-line-crops-v2/manifest.json','sha256':'c3f874708a286fca9f82fa6279415b4b7221087c19bc587b262437a7124e5c04'}
OUT=ROOT/'artifacts/ocr-control-line-crop-fix-v2'


def pin(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)}


def main():
    assert not OUT.exists();OUT.mkdir()
    manifest=json.loads(checked_pin(ROOT,PREVIOUS).read_text(encoding='utf-8'))
    protocol=json.loads(checked_pin(ROOT,manifest['protocol']).read_text(encoding='utf-8'))
    source=next(r for r in manifest['sources'] if r['index']==13)
    field=next(r for r in source['fields'] if r['field']=='part_name');before=json.loads(json.dumps(field))
    binding=next(r for r in protocol['sources'] if r['index']==13)
    parent=binding['parents'][field['parent_kind']];box=list(field['source_box_xyxy']);box[0]-=2
    px,py=parent['box_source_xyxy'][:2]
    with Image.open(checked_pin(ROOT,parent['image'])) as im:
        raw=np.array(im)[box[1]-py:box[3]-py,box[0]-px:box[2]-px]
    yy,xx=np.where(np.any(raw<200,axis=2))
    margins=[int(xx.min()),int(yy.min()),raw.shape[1]-int(xx.max())-1,raw.shape[0]-int(yy.max())-1]
    assert margins==[1,12,12,12]
    image=OUT/(field['id']+'.png');Image.fromarray(raw).save(image)
    field.update(source_box_xyxy=box,raw_crop_pin=pin(image),image_pin=pin(image))
    field['cell_box_parent_xyxy'][0]-=2
    manifest.update(previous_proposals=PREVIOUS,extractor=pin(Path(__file__)),
        crop_amendments=[{'source_index':13,'field':'part_name','before':before,'after':field,
            'reason':'Initial cell inset clipped one glyph column. Extend left by two original pixels, excluding the adjacent blue frame; retain one original white column.'}])
    output=OUT/'manifest.json';output.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(pin(output)))


if __name__=='__main__':main()
