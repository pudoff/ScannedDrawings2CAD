"""Save literal assistant observations after viewing all 25 sources; no OCR labels."""
from datetime import datetime, timezone
import json
from pathlib import Path
from cad_recovery.clear_pairs import digest

ROOT=Path(__file__).resolve().parents[1]
PROBES=ROOT/'artifacts/ocr-control-scope-probes-v3/manifest.json'
OUTPUT=ROOT/'artifacts/ocr-control-scope-readings-v3.json'
VALUES=[
    (0,'1539140.01','Планка',['2959СО']),
    (1,'1539141.01','Планка',['2959СО']),
    (2,'1539142.01','Ось',['2959СО']),
    (3,'1539145.01','Планка',['2959СО']),
    (4,'1539151.01','Проушина',['2959СО']),
    (5,'1539151.02','Лист',['2959СО']),
    (6,'1539181.01','Лист',['2959СО']),
    (7,'626183.01','Планка',['362.03.2959СО']),
    (8,None,None,[]),
    (9,'ЗЗП.000.001.09.01','Втулка',[]),
    (10,'ЗЗП.000.001.09.02','Лист',[]),
    (11,'ЗЗП.000.001.10.01','Ребро',[]),
    (12,'ЗЗП.000.001.10.02','Ребро',[]),
    (13,'ЗЗП.000.001.10.03','Боковое ребро',[]),
    (14,'ЗЗП.000.001.11.01','Лист боковой',[]),
    (15,'ЗЗП.000.001.11.02','Ребро',[]),
    (16,'ЗЗП.000.001.11.05','Полоса',[]),
    (17,'ЗЗП.000.001.11.06','Ребро',[]),
    (18,'ЗЗП.000.001.11.07','Ребро',[]),
    (19,'ЗЗП.000.001.13.01','Лист',[]),
    (20,'ЗЗП.000.001.13.02','Лист',[]),
    (21,'ЗЗП.000.001.13.03','Капролон',[]),
    (22,'ЗЗП.000.001.14.01','Лист',[]),
    (23,'ЗЗП.000.001.15.01','Лист',[]),
    (24,None,None,[]),
]


def pin(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path)}


def main():
    assert not OUTPUT.exists()
    manifest=json.loads(PROBES.read_text(encoding='utf-8'));sources=[]
    assert len(VALUES)==len(manifest['sources'])==25
    for index,designation,name,related in VALUES:
        row=manifest['sources'][index];assert row['index']==index
        notes=['Просмотрены общий обзор на контактном листе и обе точные нижние области. '
               'Обозначения и название прочитаны из растра; имя файла не заменяет чтение.',
               'Новых строк OCR-разметки и утверждения независимости нет. Отсутствие найденной '
               'связи ограничено явно просмотренными областями и сохранённым реестром чтений.']
        uncertainty=[]
        if index<8:
            notes+=['Соседняя ячейка содержит ссылку 2959. Все восемь документов сохраняются '
                    'в одной дополнительной группе. Назначение ячейки и равенство короткого '
                    'и полного кодов не доказаны.',
                    'Видны общие с прежними числовыми чертежами оформление, организация ИГХ РУС '
                    'и фамилии разработчиков. Независимость по организации/стилю не утверждается.']
            uncertainty=['С/О записаны кириллицей по прежней конвенции; кодовые точки шрифта неизвестны.',
                         'У семи документов префикс ссылки не напечатан и не восстанавливается.']
        elif 9<=index<=23:
            notes+=['В штампе виден общий проектный префикс и надпись ZavodZ; проект '
                    'сохраняется целиком. В просмотренных областях нет конкретной ссылки '
                    '2963СО/2965СО или 2959СО. Общие слова «Лист»/«Ребро» не устанавливают '
                    'тождество деталей с прежними данными.']
            uncertainty=['Начертания первых двух символов З/3 неоднозначны; запись ЗЗП является '
                         'конвенцией для поиска связей, а не утверждённой точной OCR-целью.']
            if index==21:
                notes+=['Название прочитано как «Капролон»; строка материала содержит '
                        '«Фторопласт-4». Несогласованность сохранена без исправления.']
        elif index==8:
            notes+=['Справочная схема болтов, гаек и шайб систем HR/HV; видны ссылки '
                    'ГОСТ 32484.3–6-2013. Обычного штампа и обозначения документа нет. '
                    'Происхождение и связь с прежним крепежом не установлены; оставить pending.']
        else:
            notes+=['Фрагмент втулочной детали с отверстиями и пазом; штампа, обозначения '
                    'документа и источника публикации нет. Название «Муфта» известно из '
                    'имени файла, а не прочитано на изображении. Происхождение оставить pending.']
        sources.append({'index':index,'id':row['id'],'input_path':row['input_path'],
            'input_sha256':row['input_sha256'],'group_id':row['group_id'],
            'title_designation':designation,'title_name':name,'related_designations':related,
            'document_kind':'reference_diagram' if index==8 else 'unidentified_detail_fragment' if index==24 else 'detail_drawing',
            'decision':'needs_further_review' if index in (8,24) else 'no_known_cross_split_link_in_reviewed_scope',
            'visual_checks':{'overview_seen':True,'left_seen':True,'right_seen':True},
            'viewed_images':[row['overview'],*[r['image'] for r in row['regions']]],
            'overview_contact':manifest['contacts'][index//5],
            'uncertainties':uncertainty,'notes':notes})
    packet={'schema_version':'ocr-control-scope-readings/3.0','probes':pin(PROBES),
        'protocol':manifest['protocol'],'reviewer':'assistant','human_verified':False,
        'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'OCR_run':False,
        'new_OCR_labels':0,'training_started':False,'independent_evaluation':False,'sources':sources}
    OUTPUT.write_text(json.dumps(packet,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(pin(OUTPUT)))


if __name__=='__main__':main()
