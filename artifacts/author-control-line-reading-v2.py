"""Save manual decisions after all final crops were viewed, without OCR."""
from datetime import datetime, timezone
import json
from pathlib import Path

from cad_recovery.clear_pairs import digest
from cad_recovery.label_release_v1 import checked_pin
from cad_recovery.ocr_control_line_schema_v2 import ControlReviewV2

ROOT=Path(__file__).resolve().parents[1]
CROPS={'path':'artifacts/ocr-control-line-crop-fix-v2/manifest.json','sha256':'eae02d95fd46913c455437a8767a11df3d28e4a8ce6fcb648aa86ca74a30fbbb'}
OUTPUT=ROOT/'artifacts/ocr-control-line-reading-v2.json'
NAMES={0:'Планка',1:'Планка',2:'Ось',3:'Планка',4:'Проушина',5:'Лист',6:'Лист',7:'Планка',
    9:'Втулка',10:'Лист',11:'Ребро',12:'Ребро',13:'Боковое ребро',14:'Лист боковой',15:'Ребро',
    16:'Полоса',17:'Ребро',18:'Ребро',19:'Лист',20:'Лист',21:'Капролон',22:'Лист',23:'Лист'}
DESIGNATIONS={0:'1539140.01',1:'1539141.01',2:'1539142.01',3:'1539145.01',
    4:'1539151.01',5:'1539151.02',6:'1539181.01',7:'626183.01'}


def pin(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)}


def main():
    assert not OUTPUT.exists()
    proposals=json.loads(checked_pin(ROOT,CROPS).read_text(encoding='utf-8'))
    protocol=json.loads(checked_pin(ROOT,proposals['protocol']).read_text(encoding='utf-8'))
    sources=[]
    for source,crop in zip(protocol['sources'],proposals['sources'],strict=True):
        assert source['id']==crop['source_id'];fields=[];index=source['index']
        for candidate in crop['fields']:
            field=candidate['field'];text=None
            if field=='adjacent_reference' and index>=9:
                decision='absent';reason='У данного штампа нет отдельной соседней ячейки ссылки; сохранена отсутствующая позиция.'
            elif field in source['inherited_uncertain_fields']:
                decision='unresolved';reason=('Сохранена прежняя неоднозначность З/3 в проектном префиксе; текст не восстанавливается по имени файла.'
                    if field=='designation' else 'Сохранена прежняя неопределённость буквенной записи ссылки; префикс не дописывается, OCR-цель не выпускается.')
            else:
                decision='accepted_for_ocr_transcription';text=NAMES[index] if field=='part_name' else DESIGNATIONS[index]
                reason='Полная строка прочитана на конечной точной вырезке; рамки вне вырезки, пиксели не изменялись.'
            record={k:candidate[k] for k in ('id','field','parent_kind','source_box_xyxy','rotation_cw_degrees','raw_crop_pin','image_pin')}
            record.update(decision=decision,text=text,reason=reason,
                normalization='Буквальный текст. Русские слова записаны кириллицей; числа и точка сохранены без замены. Неуверенные поля без OCR-цели.',
                visual_checks={'full_text_visible':decision!='absent','text_matches':decision=='accepted_for_ocr_transcription',
                    'no_line_carrier':decision!='absent','exact_rotation':True})
            fields.append(record)
        sources.append({**{k:source[k] for k in ('id','input_path','input_sha256','preparation','source_size_px')},
            'fields':fields,'scope_notes':['Все три позиции учтены. Источники, группы и split закреплены до разметки.',
                'Все конечные вырезки просмотрены на контактных листах; исправленная строка «Боковое ребро» дополнительно просмотрена отдельно.',
                'Материал и физический смысл не исправляют видимое название. Подтверждения человеком нет.']})
    packet={'schema_version':'ocr-control-line-reading/2.0','protocol':proposals['protocol'],'crop_proposals':CROPS,
        'reviewer':'assistant','human_verified':False,'ocr_run':False,'native_cad_opened':False,'training_started':False,
        'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'viewed_contacts':proposals['contacts'],
        'additional_viewed_images':[proposals['crop_amendments'][0]['after']['image_pin']],'sources':sources}
    ControlReviewV2.model_validate(packet)
    OUTPUT.write_text(json.dumps(packet,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(pin(OUTPUT)))


if __name__=='__main__':main()
