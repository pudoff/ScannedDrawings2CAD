"""Preregistered OCR baseline: validation then test, without parameter changes."""
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image

from .clear_pairs import contained, digest
from .family_catalog_v2 import stable_digest
from .label_release_v1 import checked_pin
from .ocr_baseline_metrics_v3 import build_control_metadata_v3, score_control_v3, SPLITS
from .ocr_baseline_v1 import PROTOCOL as OLD_PROTOCOL, combine_detections_v1, _make_engine, _effective_engine_settings
from .ocr_baseline_v2 import (_required_pins as previous_pins, _environment, _check_engine,
    _predictions as previous_predictions, _runtime as previous_runtime)
from .ocr_control_release_v2 import verify_ocr_control_v2, CODE as RELEASE_CODE
from .ocr_release_v3 import _package, read, encoded, pin, require
from .updated_raster_batch import _peak_memory

RELEASE={'path':'artifacts/ocr-control-release-v2/manifest.json','sha256':'0fc692a65a742b659cbc8d83919df1bbc960226f05c31eee51691964bdd97e03'}
PARENT={'path':'artifacts/ocr-control-release-final-v2.json','sha256':'5a0744c433dfcd808b29acb8967b20fa765c4991f81caa1b15179b9c10b22d33'}
PREVIOUS={'path':'artifacts/ocr-baseline-v2/manifest.json','sha256':'685e428102c03d3cd2b088f136b8a9a058b8964ecbe8d6e783caa197cb6ca3e3'}
CODE_PATHS=['src/cad_recovery/'+name+'.py' for name in ('ocr_baseline_v3','ocr_baseline_metrics_v3','updated_raster_batch')]+['scripts/evaluate_ocr_baseline_v3.py']
PROTOCOL={**deepcopy(OLD_PROTOCOL),'scope':'limited_title_control_two_groups_validation_then_test',
    'sample_count':31,'split_counts':{'validation':15,'test':16},'split_order':list(SPLITS),
    'sample_order':'split_order_then_id_ascending','settings':'exact_previous_engine_models_environment_and_effective_settings',
    'test_gate':'same frozen config, input image hashes, code/model pins and effective settings checked after validation',
    'test_engine_state':'same instance after validation; no weight or parameter updates',
    'strata':'fixed release metadata before inference; split, field role, source and coallocation group',
    'coverage':'all 69 slots; unscored uncertainty/absence shown separately',
    'duplicates':'primary retains all lines; secondary uses smallest frozen ID per identical image hash',
    'memory':'process-lifetime peak at checkpoints, including release verification; not isolated OCR or GPU memory',
    'combined_validation_test_score':None}
MEMORY_SCOPE='process lifetime up to checkpoint including release verification; not isolated OCR or GPU memory'
RUNTIME_V2_FIELDS={'schema_version','started_utc','initialization_seconds','rows','decode_total_seconds',
    'recognize_total_seconds','hardware','python','packages','cold_first_sample_id','warmup_calls'}


def same(a,b):return stable_digest(a)==stable_digest(b)
def now():return datetime.now(timezone.utc).isoformat()
def timestamp(value):
    t=datetime.fromisoformat(value);require(t.tzinfo is not None,'Missing timezone');return t


def _required_pins(root):
    values={r['path']:r for r in previous_pins(root)}
    for path in sorted(set(CODE_PATHS+RELEASE_CODE)):values[path]=pin(root,path)
    return [values[path] for path in sorted(values)]


def _inventory(root,rows):
    return [{**{k:r[k] for k in ('id','source_id','source_sha256','group_id','image_sha256','text','split')},
        'image_path':Path(r['image_path']).relative_to(root).as_posix()} for r in rows]


def _control_order(rows):
    require(len(rows)==PROTOCOL['sample_count'] and {s:sum(r['split']==s for r in rows) for s in SPLITS}
        ==PROTOCOL['split_counts'],'Incomplete preregistered split inventory')
    require([r['id'] for r in rows]==[r['id'] for r in sorted(rows,key=lambda r:(SPLITS.index(r['split']),r['id']))],
        'Control order must be validation then test, ascending ID within each')


def _bundle(root):
    parent=read(checked_pin(root,PARENT));require(RELEASE in parent['manifests'],'Wrong released control parent')
    report,parts,base=verify_ocr_control_v2(root,RELEASE)
    require(report['task_split_counts']=={'train':90,'validation':15,'test':16},'Changed limited control split counts')
    rows=[{**r,'image_path':str(base/r['image'])} for split in SPLITS for r in sorted(parts[split],key=lambda r:r['id'])]
    metadata=build_control_metadata_v3(read(base/'labels.json'),read(base/'task-splits.json'),rows)
    previous,_=_package(root,PREVIOUS);old_report=read(previous.parent/'report.json');runtime=read(previous.parent/'runtime.json')
    require(metadata['field_slot_count']==69 and len(metadata['strata'])==31,'Changed field or sample coverage')
    return rows,metadata,old_report,{k:runtime[k] for k in ('python','packages')}


def prepare_config_v3(workspace,output_path):
    root=Path(workspace).resolve();output=contained(root,output_path)
    require(output.parent==root/'configs/reviews' and not output.exists(),'Use a new experiment configuration')
    rows,metadata,previous,environment=_bundle(root)
    _control_order(rows)
    require(same(_environment(),environment),'Environment differs from previous OCR experiment')
    config={'schema_version':'ocr-baseline-config/3.0','task':'P16.1c','created_before_inference_utc':now(),
        'release':RELEASE,'parent_final':PARENT,'previous_baseline':PREVIOUS,'protocol':PROTOCOL,
        'model_dir':'artifacts/models/ppocr-v5','pins':_required_pins(root),'inventory':_inventory(root,rows),
        'metadata':metadata,'expected_engine':previous['engine'],'expected_settings':previous['effective_settings'],
        'expected_environment':environment}
    output.write_bytes(encoded(config));return pin(root,output_path)


def _verify_inputs(root,config_pin):
    config=read(checked_pin(root,config_pin))
    require(set(config)=={'schema_version','task','created_before_inference_utc','release','parent_final',
        'previous_baseline','protocol','model_dir','pins','inventory','metadata','expected_engine','expected_settings','expected_environment'},'Unexpected experiment config')
    require(config['schema_version']=='ocr-baseline-config/3.0' and config['task']=='P16.1c'
        and config['release']==RELEASE and config['parent_final']==PARENT and config['previous_baseline']==PREVIOUS
        and config['model_dir']=='artifacts/models/ppocr-v5' and same(config['protocol'],PROTOCOL),'Changed preregistered scope')
    timestamp(config['created_before_inference_utc'])
    require(config['pins']==_required_pins(root),'Code/model pins changed')
    rows,metadata,previous,environment=_bundle(root)
    _control_order(rows)
    require(same(config['inventory'],_inventory(root,rows)) and same(config['metadata'],metadata),'Control inventory or metadata changed')
    require(same(config['expected_engine'],previous['engine']) and same(config['expected_settings'],previous['effective_settings'])
        and same(config['expected_environment'],environment) and same(_environment(),environment),'Frozen previous runtime settings changed')
    return config,rows,contained(root,config['model_dir'])


def _phase_gate(root,record,config,engine):
    checked_pin(root,record)
    require(config['pins']==_required_pins(root) and same(_environment(),config['expected_environment']),'Inputs changed before test')
    for row in config['inventory']:checked_pin(root,{'path':row['image_path'],'sha256':row['image_sha256']})
    settings=_effective_engine_settings(engine);_check_engine(root,config,engine.provenance,settings)
    return {'checked_utc':now(),'config':record,'settings_content_sha256':stable_digest(settings),
        'validation_sample_count':config['protocol']['split_counts']['validation'],'configuration_changed':False}


def _memory():
    value=_peak_memory();value['measurement_scope']=MEMORY_SCOPE;return value


def _one(engine,row):
    started=time.perf_counter();decode=recognize=0.0;stage='decode';error=None
    detections=[];combined={'prediction':'','order':[]}
    try:
        with Image.open(row['image_path']) as image:rgb=np.array(image.convert('RGB'))
        bgr=np.ascontiguousarray(rgb[:,:,::-1]);decode=time.perf_counter()-started
        stage='recognize';before=time.perf_counter()
        try:detections=engine.recognize(bgr)
        finally:recognize=time.perf_counter()-before
        stage='combine';combined=combine_detections_v1(detections)
    except Exception as exc:
        if stage=='decode':decode=time.perf_counter()-started
        error={'stage':stage,'type':type(exc).__name__,'message':str(exc)}
        detections=[];combined={'prediction':'','order':[]}
    prediction={'id':row['id'],'source_id':row['source_id'],'reference':row['text'],'split':row['split'],
        'image':'images/'+row['id']+'.png','image_sha256':row['image_sha256'],'prediction':combined['prediction'],
        'status':'error' if error else 'ok' if combined['prediction'] else 'empty',
        'error':error,'detections':detections,'order':combined['order']}
    timing={'id':row['id'],'decode_seconds':decode,'recognize_seconds':recognize,'sample_wall_seconds':time.perf_counter()-started}
    return prediction,timing


def _summary(scores):return {k:scores[k] for k in ('sample_count','status_counts','raw','normalized','by_source')}


def _documents(config_pin,config,predictions,provenance,settings):
    scores=score_control_v3(predictions,config['metadata'])
    parts={split:{'overall':_summary(value['overall']),
        'by_role':{k:_summary(v) for k,v in value['by_role'].items()},
        'by_group':{k:_summary(v) for k,v in value['by_group'].items()},
        'unique_image_diagnostic':_summary(value['unique_image_diagnostic'])} for split,value in scores['by_split'].items()}
    report={'schema_version':'ocr-baseline-report/3.0','task':'P16.1c','config':config_pin,'scope':PROTOCOL['scope'],
        'by_split':parts,'coverage_by_split':scores['coverage_by_split'],'coverage_by_source':scores['coverage_by_source'],
        'duplicates':scores['duplicates'],'combined_validation_test_score':None,
        'engine':provenance,'effective_settings':settings,'training_started':False,'parameter_selection':False,
        'source_modified':False,'independent_evaluation':False,
        'limitations':['One conservative group per split; only accepted title fields are scored.',
            'Validation contains names; test contains names and numeric designations. Report roles separately.',
            'Primary scores retain repeated crops; secondary unique-image diagnostics use frozen representatives.',
            'Uncertain/absent fields are coverage, not OCR failures or silently accepted targets.',
            'Shared style/organization and unseen family links remain possible; no global independence is established.',
            'Full det/cls/rec on oriented line crops; not full-page detection, dimension OCR or CAD accuracy.',
            'Validation is first/cold; test reuses the unchanged warm engine. Latencies are observations, not a speed comparison.',
            'Peak memory includes release authentication before OCR; it is not isolated model or GPU memory.',
            'Saved-evidence verification invokes no OCR and relies on trusted manifests for execution provenance.']}
    html=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR: ограниченный контроль</title>',
        '<style>body{font:17px system-ui;max-width:1150px;margin:2em auto}img{max-width:100%}pre{white-space:pre-wrap}section{margin:2em 0}table{border-collapse:collapse}td,th{padding:.5em;border:1px solid #bbb}</style>',
        '<h1>OCR: validation и test</h1><p>Две группы, по одной на split. Поля штампа; глобальная независимость не подтверждена. Обучения и подбора по test не было.</p>',
        '<p><a href="report.json">Отчёт</a> · <a href="metrics.json">Выравнивания и метрики</a> · <a href="metadata.json">Полнота полей</a> · <a href="runtime.json">Время и RAM</a></p>',
        '<table><tr><th>Split</th><th>Строк</th><th>Полное совпадение raw</th><th>CER raw</th><th>Совпадение normalized</th><th>CER normalized</th></tr>']
    for split,part in parts.items():
        r,n=part['overall']['raw'],part['overall']['normalized']
        html.append(f'<tr><td>{split}</td><td>{r["sample_count"]}</td><td>{r["exact_matches"]}</td><td>{r["distance"]}/{r["reference_characters"]} = {100*r["cer"]:.2f}%</td><td>{n["exact_matches"]}</td><td>{n["distance"]}/{n["reference_characters"]} = {100*n["cer"]:.2f}%</td></tr>')
    html+=['</table><p>Пустые ответы и ошибки сохранены в знаменателях. Неопределённые и отсутствующие поля показаны в отчёте полноты.</p>',
        '<label>Выборка <select id="split" onchange="filterRows()"><option value="">Все</option><option>validation</option><option>test</option></select></label> ',
        '<label>Поле <select id="role" onchange="filterRows()"><option value="">Все</option><option>part_name</option><option>designation</option></select></label>']
    strata={r['id']:r for r in config['metadata']['strata']}
    for row in predictions:
        role=strata[row['id']]['role']
        html.append('<section data-split="'+escape(row['split'],quote=True)+'" data-role="'+escape(role,quote=True)+'"><h2>'+escape(row['id'])+'</h2><p>'+escape(row['split']+' / '+role+' / '+row['source_id'])+'</p><img loading="lazy" src="'+escape(row['image'],quote=True)+'" alt="Исходная строка"><pre>Эталон: '+escape(row['reference'])+'\nOCR: '+escape(row['prediction'])+'\nСтатус: '+escape(row['status'])+'</pre></section>')
    html.append('<script>function filterRows(){const s=document.getElementById("split").value,r=document.getElementById("role").value;document.querySelectorAll("section[data-split]").forEach(e=>e.hidden=!!((s&&e.dataset.split!==s)||(r&&e.dataset.role!==r)));}</script></html>\n')
    files={'predictions.json':encoded(predictions),'metrics.json':encoded(scores),'report.json':encoded(report),
        'metadata.json':encoded(config['metadata']),'index.html':''.join(html).encode('utf-8')}
    return files,report


def run_baseline_v3(workspace,config_pin,output_dir):
    root=Path(workspace).resolve();output=contained(root,output_dir)
    require(output.parent==root/'artifacts' and not output.exists(),'Use a new direct artifacts directory')
    config,rows,model_dir=_verify_inputs(root,config_pin)
    started=now();require(timestamp(started)>=timestamp(config['created_before_inference_utc']),'Inference predates preregistration')
    memory={'before_engine':_memory()};before=time.perf_counter();engine=_make_engine(model_dir);init=time.perf_counter()-before
    settings,provenance=_effective_engine_settings(engine),engine.provenance;_check_engine(root,config,provenance,settings)
    memory['after_engine']=_memory();predictions=[];timings=[];phases={};gate=None
    for split in SPLITS:
        if split=='test':gate=_phase_gate(root,config_pin,config,engine)
        part=[r for r in rows if r['split']==split];begin=now()
        for row in part:
            prediction,timing=_one(engine,row);predictions.append(prediction);timings.append(timing)
        phases[split]={'started_utc':begin,'finished_utc':now(),'sample_ids':[r['id'] for r in part]}
        memory['after_'+split]=_memory()
    _check_engine(root,config,engine.provenance,_effective_engine_settings(engine))
    finished=now();_verify_inputs(root,config_pin)
    files,report=_documents(config_pin,config,predictions,provenance,settings)
    runtime={'schema_version':'ocr-baseline-runtime/3.0','started_utc':started,'finished_utc':finished,
        'initialization_seconds':init,'rows':timings,'decode_total_seconds':sum(r['decode_seconds'] for r in timings),
        'recognize_total_seconds':sum(r['recognize_seconds'] for r in timings),'phases':phases,'test_gate':gate,'memory':memory,
        'hardware':{'platform':platform.platform(),'processor':platform.processor(),'machine':platform.machine(),'logical_cpus':os.cpu_count()},
        **_environment(),'cold_first_sample_id':rows[0]['id'],'warmup_calls':0}
    _runtime(config,rows,runtime);files['runtime.json']=encoded(runtime)
    for row in rows:
        path=Path(row['image_path']);require(digest(path)==row['image_sha256'],'Input image changed before publishing')
        files['images/'+row['id']+'.png']=path.read_bytes()
    output.mkdir()
    for name,data in files.items():
        path=contained(output,name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    manifest={'schema_version':'ocr-baseline-manifest/3.0','config':config_pin,'files':[pin(output,p) for p in sorted(files)]}
    (output/'manifest.json').write_bytes(encoded(manifest));return report


def _runtime(config,rows,runtime):
    require(set(runtime)==RUNTIME_V2_FIELDS|{'finished_utc','phases','test_gate','memory'}
        and runtime['schema_version']=='ocr-baseline-runtime/3.0','Unexpected control runtime')
    old={k:runtime[k] for k in RUNTIME_V2_FIELDS};old['schema_version']='ocr-baseline-runtime/2.0'
    previous_runtime(config,rows,old)
    phases=runtime['phases'];require(set(phases)==set(SPLITS),'Missing inference phase')
    for split in SPLITS:
        require(set(phases[split])=={'started_utc','finished_utc','sample_ids'}
            and phases[split]['sample_ids']==[r['id'] for r in rows if r['split']==split],'Changed phase inventory')
    gate=runtime['test_gate']
    require(set(gate)=={'checked_utc','config','settings_content_sha256','validation_sample_count','configuration_changed'}
        and gate['settings_content_sha256']==stable_digest(config['expected_settings'])
        and type(gate['validation_sample_count']) is int and gate['validation_sample_count']==config['protocol']['split_counts']['validation']
        and gate['configuration_changed'] is False,'Invalid frozen test gate')
    times=[runtime['started_utc'],phases['validation']['started_utc'],phases['validation']['finished_utc'],gate['checked_utc'],
           phases['test']['started_utc'],phases['test']['finished_utc'],runtime['finished_utc']]
    require([timestamp(v) for v in times]==sorted(timestamp(v) for v in times),'Validation/test chronology differs')
    memory=runtime['memory'];require(set(memory)=={'before_engine','after_engine','after_validation','after_test'},'Missing memory checkpoints')
    peaks=[]
    for key in ('before_engine','after_engine','after_validation','after_test'):
        value=memory[key]
        require(set(value)=={'peak_process_working_set_bytes','measurement_method','measurement_scope'}
            and value['measurement_scope']==MEMORY_SCOPE,'Invalid memory measurement scope')
        peak=value['peak_process_working_set_bytes']
        require((peak is None and value['measurement_method']=='unavailable') or
            (type(peak) is int and peak>0 and value['measurement_method'] in
                ('Windows_GetProcessMemoryInfo_PeakWorkingSetSize','Linux_getrusage_ru_maxrss_KiB_to_bytes')),'Invalid peak memory')
        if peak is not None:peaks.append(peak)
    require(peaks==sorted(peaks),'Process-lifetime memory peak cannot decrease')


def verify_baseline_v3(workspace,manifest_pin):
    root=Path(workspace).resolve();path,manifest=_package(root,manifest_pin);base=path.parent
    require(set(manifest)=={'schema_version','config','files'} and manifest['schema_version']=='ocr-baseline-manifest/3.0','Wrong baseline manifest')
    config,rows,_=_verify_inputs(root,manifest['config']);predictions=read(base/'predictions.json')
    require(isinstance(predictions,list) and len(predictions)==len(rows),'Missing control predictions')
    for record,row in zip(predictions,rows,strict=True):
        require('split' in record and record['split']==row['split'],'Changed prediction split')
    previous_predictions(rows,[{k:v for k,v in r.items() if k!='split'} for r in predictions],base)
    report=read(base/'report.json');_check_engine(root,config,report['engine'],report['effective_settings'])
    files,expected=_documents(manifest['config'],config,predictions,report['engine'],report['effective_settings'])
    runtime=read(base/'runtime.json');_runtime(config,rows,runtime)
    require(runtime['test_gate']['config']==manifest['config'],'Test gate uses a different configuration')
    files['runtime.json']=encoded(runtime)
    for row in rows:files['images/'+row['id']+'.png']=Path(row['image_path']).read_bytes()
    require(manifest['files']==[pin(base,p) for p in sorted(files)],'Saved inventory differs')
    require(all((base/p).read_bytes()==data for p,data in files.items()),'Saved diagnostic differs from evidence')
    _verify_inputs(root,manifest['config']);return expected
