"""Replay detector results against frozen labels; never execute a neural model."""
from copy import deepcopy
from pathlib import Path
import math
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from .dataset_layout_v1 import checked,encoded,pin,read
from .family_catalog_v2 import require
from .yolox_protocol_v1 import POLICY,BUDGETS,verify_prepared,geometry,learning_rate,detector_decision
from .yolox_metrics_v1 import (decode,native_predictions,detection_metrics,average_precision,
                              context_coverage,decision_input)

FOLDER='artifacts/yolox-comparison-v2'
PROTOCOL={'path':'configs/reviews/yolox-experiment-v1.json',
          'sha256':'67e98548fafeb76c50d29046f2b4a7f2761341e28413903d5fb3a7bec5262c6f'}


def output_run(root,item):
    folder=root/FOLDER/item['id'];config=read(checked(root,item['config']))
    require(config['policy']==POLICY and config['budgets']==BUDGETS,'Worker parameters rebound')
    for p in config['code']:checked(root,p)
    process=read(folder/'process.json')
    command=process['command'];require(command[command.index('--config')+1]==item['config']['path']
        and command[command.index('--sha256')+1]==item['config']['sha256']
        and command[command.index('--output')+1]==f"{FOLDER}/{item['id']}/result",'Worker command/config mismatch')
    report_path=folder/'result/report.json'
    report=read(report_path) if report_path.exists() else None
    if report:require(report['config']==item['config'],'Rebound worker report')
    return config,process,report


def training_audit(root,execution):
    rows=[];equal=True;resource=True
    require([r['id'] for r in execution['training']]==[f'train-{a}-{i}' for a in ('rgb','gray') for i in range(2)],'Training inventory changed')
    for item in execution['training']:
        config,process,report=output_run(root,item)
        protocol=read(checked(root,PROTOCOL))
        require(config['training']==protocol['inputs']['train.json'] and config['code']==execution['common']['code'],
                'Training request/code differs from registration')
        require(config['qualified_initial']==read(checked(root,protocol['qualification']))['results'][0]['initial'],
                'Qualified model initialization rebound')
        require(report is not None and process['budget_pass'],'Training did not finish; no final comparison can be released')
        request=read(checked(root,config['training']))
        require(report['geometries']=={c['id']:geometry(c['size_px']) for c in request['cases']},'Training geometry changed')
        require(report['updates']==300 and len(report['history'])==300 and report['validation_training_calls']==0
            and report['inference_calls']==0 and report['initial']==config['qualified_initial']
            and report['final']!=report['initial'] and report['checkpoint_selection']==POLICY['checkpoint_selection'],
                'Invalid training scope/initialization')
        require(report['arm']==config['arm'] and report['repeat']==config['repeat'],'Training arm rebound')
        require(report['checkpoint']['path']==f"{FOLDER}/{item['id']}/result/final.pt",'Unexpected checkpoint destination')
        checked(root,report['checkpoint'])
        for index,row in enumerate(report['history']):
            require(row['update']==index+1 and row['case_ids']==request['schedule'][index]
                and row['learning_rate']==learning_rate(index),'Changed update schedule/LR')
            require(math.isfinite(row['gradient_norm']) and row['gradient_norm']>=0
                and all(math.isfinite(v) for v in row['losses'].values()),'Nonfinite loss/gradient history')
        resource &= (process['budget_pass'] and process['budget_seconds']==1200 and process['budget_rss_MiB']==6144
            and process['walltime_seconds']<=1200 and process['peak_rss_MiB']<=6144
            and report['allocated_peak_MiB']<=4096 and report['reserved_peak_MiB']<=6144)
        rows.append({'id':item['id'],'config':item['config'],'process':process,'report':report})
    for first,second in [(rows[0],rows[1]),(rows[2],rows[3])]:
        a,b=first['report'],second['report']
        equal &= all(a[k]==b[k] for k in ('initial','final','history','geometries','transfer'))
    return rows,bool(equal),bool(resource)


def collect_inference(root,item,ledger):
    config,process,report=output_run(root,item);request=read(checked(root,config['inference']))
    protocol=read(checked(root,PROTOCOL))
    split='validation' if '-validation' in item['id'] else 'train'
    require(config['inference']==protocol['inputs'][f'inference-{split}.json'],'Inference request differs from assigned split')
    if config['method']=='yolox':
        suffix=item['id'].removeprefix('infer-').split('-');arm,repeat=suffix[:2]
        trained=read(root/FOLDER/f'train-{arm}-{repeat}/result/report.json')
        require(config['checkpoint']==trained['checkpoint'] and config['trained_final']==trained['final']
                and config['arm']==arm and config['device']==('cpu' if item['id'].endswith('-cpu') else 'cuda'),
                'Inference checkpoint/profile differs from assigned training')
    else:require(item['id'].startswith(config['method']+'-'),'CV baseline method rebound')
    expected={c['id']:c for c in request['cases']}
    actual={};failed=not process['budget_pass'] or report is None
    if report:
        require(report['method']==config['method'] and report['ground_truth_read'] is False
                and report['device']==config.get('device','cpu'),'Wrong inference method/authority')
        require(len({r['id'] for r in report['rows']})==len(report['rows'])
                and {r['id'] for r in report['rows']}<=set(expected),'Invented/duplicate inference case')
        actual={r['id']:r for r in report['rows']}
    operating={};ap={};details=[];resource=not failed
    for identifier,case in expected.items():
        row=actual.get(identifier)
        if failed or row is None:
            operating[identifier]=[];ap[identifier]=[];resource=False
            details.append({'id':identifier,'failure':'missing_or_failed_assigned_output'});continue
        require(row['image']==case['image'] and row['size_px']==case['size_px'],'Inference image/grid rebound')
        require(math.isfinite(row['page_seconds']) and row['page_seconds']>=0,'Invalid page timing')
        begin=time.perf_counter()
        if config['method']=='yolox':
            require(row['geometry']==geometry(case['size_px']),'Inference edge transform changed')
            require(row['raw']['path']==f"{FOLDER}/{item['id']}/result/{identifier}.npy",'Raw output rebound')
            raw=np.load(checked(root,row['raw']),allow_pickle=False)
            require(raw.shape==(8400,10),'Lost detector anchors/classes')
            decoded=decode(raw,case['size_px'])
            operating[identifier],d1=native_predictions(decoded['operating'],ledger[identifier])
            ap[identifier],d2=native_predictions(decoded['ap'],ledger[identifier])
            details.append({'id':identifier,'raw':row['raw'],'decode':decoded['audit'],
                'native_padding_discarded_operating':d1,'native_padding_discarded_ap':d2,
                'postprocess_seconds':time.perf_counter()-begin,'worker_page_seconds':row['page_seconds'],
                'compute_seconds':row['compute_seconds']})
            limit=BUDGETS['cpu_inference_page_seconds'] if config['device']=='cpu' else BUDGETS['inference_page_seconds']
            resource &= row['page_seconds']+details[-1]['postprocess_seconds']<=limit
        else:
            require(len(row['predictions'])<=24,'CV candidate budget exceeded')
            require(all(p['role'] in ('title_block','table','view_or_detail','unknown') for p in row['predictions']),
                    'Legacy CV acquired unsupported typed semantics')
            values=[{'id':f'cv-{i:02d}','role':p['role'],'box':p['box']} for i,p in enumerate(row['predictions'])]
            operating[identifier],discarded=native_predictions(values,ledger[identifier]);ap[identifier]=[]
            details.append({'id':identifier,'worker_page_seconds':row['page_seconds'],
                'native_padding_discarded':discarded,'original_predictions':row['predictions']})
    if config['method']=='yolox':
        expected_limit=180;expected_ram=6144
        if config['device']=='cuda' and report:
            resource &= report['allocated_peak_MiB']<=4096 and report['reserved_peak_MiB']<=6144
    else:expected_limit=90;expected_ram=1024
    resource &= (process['budget_seconds']==expected_limit and process['budget_rss_MiB']==expected_ram
        and process['walltime_seconds']<=expected_limit and process['peak_rss_MiB']<=expected_ram)
    return {'id':item['id'],'method':config['method'],'config':item['config'],'operating':operating,'ap':ap,
        'details':details,'resources_pass':bool(resource),'complete':not failed and set(actual)==set(expected),
        'process':process,'initialization_seconds':report['initialization_seconds'] if report else None}


def inference_equal(one,two):
    if not one['complete'] or not two['complete']:return False
    if one['operating']!=two['operating'] or one['ap']!=two['ap']:return False
    for a,b in zip(one['details'],two['details']):
        if 'raw' in a and a['raw']['sha256']!=b['raw']['sha256']:return False
        if 'original_predictions' in a and a['original_predictions']!=b['original_predictions']:return False
    return True


def score_run(targets,run):
    selected=[t for t in targets if t['id'] in run['operating']]
    require(len(selected)==len(run['operating']),'Inference/evaluator page mismatch')
    typed=detection_metrics(selected,run['operating'],typed=True)
    coarsened=detection_metrics(selected,run['operating'],typed=False)
    return {'typed':typed,'coarse':coarsened,
        'AP':average_precision(selected,run['ap']) if run['method']=='yolox' else None,
        'contexts':context_coverage(selected,run['operating'],typed)}


def evaluate(root,*,write=True):
    verify_prepared(root,PROTOCOL);folder=root/FOLDER;execution=read(folder/'execution.json')
    for p in execution['common']['code']:checked(root,p)
    registration=read(folder/'evaluation-registration.json')
    for p in registration['code']:checked(root,p)
    protocol=read(checked(root,PROTOCOL))
    targets=read(checked(root,protocol['inputs']['targets.json']))
    ledger={r['id']:r for r in read(checked(root,protocol['inputs']['lineage.json']))}
    trains,train_equal,train_resources=training_audit(root,execution)
    tasks=read(folder/'inference-tasks.json')
    expected=[f'infer-{a}-{i}-{s}' for a in ('rgb','gray') for i in range(2) for s in ('train','validation')]+['infer-rgb-0-validation-cpu']
    require([t['id'] for t in tasks]==expected,'Assigned inference inventory changed')
    runs={item['id']:collect_inference(root,item,ledger) for item in tasks+execution['baselines']}
    scores={key:score_run(targets,value) for key,value in runs.items()}
    repeats={}
    for a in ('rgb','gray'):
        for s in ('train','validation'):repeats[f'{a}-{s}']=inference_equal(runs[f'infer-{a}-0-{s}'],runs[f'infer-{a}-1-{s}'])
    for method in ('legacy_grid','context_cv_rejected_candidate'):
        for s in ('train','validation'):repeats[f'{method}-{s}']=inference_equal(runs[f'{method}-{s}-0'],runs[f'{method}-{s}-1'])
    primary=scores['infer-rgb-0-validation'];control=scores['infer-gray-0-validation']
    baselines={m:scores[f'{m}-validation-0'] for m in ('legacy_grid','context_cv_rejected_candidate')}
    baseline_geometry=set().union(*(set(s['coarse']['geometric_matched_ids']) for s in baselines.values()))
    candidate=decision_input(primary['typed'],primary['coarse'],baseline_geometry)
    gray=decision_input(control['typed'],control['coarse'],baseline_geometry)
    cv={m:decision_input(s['typed'],s['coarse'],[]) for m,s in baselines.items()}
    resources=train_resources and all(r['resources_pass'] for r in runs.values())
    decision=detector_decision(candidate,cv,gray,repeat_equal=train_equal and all(repeats.values()),
        resources_pass=bool(resources),context_pass=primary['contexts']['passed'],downstream_pass=False)
    result={'schema_version':'yolox-comparison/1.0','protocol':PROTOCOL,'training_equal':train_equal,
        'prediction_repeats':repeats,'training':[{'id':r['id'],'initial':r['report']['initial'],'final':r['report']['final'],
            'updates':300,'first_loss':r['report']['history'][0]['losses']['total_loss'],
            'last_loss':r['report']['history'][-1]['losses']['total_loss'],'seconds':r['report']['training_seconds'],
            'allocated_peak_MiB':r['report']['allocated_peak_MiB'],'reserved_peak_MiB':r['report']['reserved_peak_MiB'],
            'process':r['process']} for r in trains],
        'scores':scores,'decision_inputs':{'candidate':candidate,'gray':gray,'baselines':cv},'decision':decision,
        'resources_pass':bool(resources),'training_updates':1200,'training_image_presentations':2400,
        'neural_inference_calls':44,'cv_inference_calls':40,'successful_training_runs':4,
        'technical_pre_image_failed_workers':4,'downstream_OCR_quality_evaluated':False,
        'physical_geometry_approved':False,'independent_evaluation':False,'production_admitted':False}
    if write:
        require(not (folder/'report.json').exists(),'Do not overwrite the evaluated result')
        (folder/'report.json').write_bytes(encoded(result))
        (folder/'decoded-predictions.json').write_bytes(encoded({k:{'operating':v['operating'],'ap':v['ap']} for k,v in runs.items()}))
        (folder/'runtime-details.json').write_bytes(encoded({k:{field:v[field] for field in ('details','process','resources_pass','complete','initialization_seconds')} for k,v in runs.items()}))
    return result,runs,targets,ledger


def review_artifacts(root,runs,targets,ledger):
    """Visible overlays and bounded ink-retention proxies, never pixel-role GT."""
    folder=root/FOLDER/'review';require(not folder.exists(),'Choose a fresh review folder');folder.mkdir()
    protocol=read(checked(root,PROTOCOL));cases={}
    for split in ('train','validation'):
        cases.update({c['id']:c for c in read(checked(root,protocol['inputs'][f'inference-{split}.json']))['cases']})
    methods={'YOLO RGB':'infer-rgb-0-','YOLO gray':'infer-gray-0-',
        'CV legacy':'legacy_grid-','CV context (rejected)':'context_cv_rejected_candidate-'}
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);images=[];proxies=[]
    for target in targets:
        identifier=target['id'];line=ledger[identifier];split=target['split']
        with Image.open(checked(root,cases[identifier]['image'])) as image:original=image.convert('RGB')
        ink=np.any(np.asarray(original)<245,axis=2)
        for index,(method,prefix) in enumerate(methods.items()):
            key=prefix+split+('-0' if method.startswith('CV') else '')
            run=runs[key];canvas=original.copy();draw=ImageDraw.Draw(canvas)
            for row in target['regions']:
                box=np.array(row['bbox'],float);factor=line['overview_to_native_edges'][0][0];box/=factor
                draw.rectangle(box.tolist(),outline='#888888',width=1)
            for row in run['operating'][identifier]:
                box=np.array(row['box'],float)/line['overview_to_native_edges'][0][0]
                colour='#cf1414' if method.startswith('YOLO') else '#0055c9'
                draw.rectangle(box.tolist(),outline=colour,width=3)
                text=row['role']+(f" {row['score']:.2f}" if 'score' in row else '')
                draw.text((box[0],box[1]),text,font=font,fill=colour,stroke_width=1,stroke_fill='white')
            name=f'{identifier}-{index}.png';canvas.save(folder/name)
            images.append({'page':identifier,'source_id':line['source_id'],'split':split,'method':method,
                'image':pin(root,(folder/name).relative_to(root).as_posix())})
            score=detection_metrics([target],{identifier:run['operating'][identifier]},typed=True)
            by_id={p['id']:p for p in run['operating'][identifier]}
            for match in score['pages'][0]['matches']:
                label=next(r for r in target['regions'] if r['id']==match['label_id'])
                factor=line['overview_to_native_edges'][0][0];x,y,r,b=np.array(label['bbox'])/factor
                left,top,right,bottom=map(int,[math.floor(x),math.floor(y),math.ceil(r),math.ceil(b)])
                patch=ink[top:bottom,left:right];total=int(patch.sum());retained=0
                if match['matched']:
                    a,c,d,e=np.array(by_id[match['candidate_id']]['box'])/factor
                    columns=np.arange(left,right)+.5;rows=np.arange(top,bottom)+.5
                    retained=int((patch & (rows[:,None]>=c)&(rows[:,None]<e)&(columns[None]>=a)&(columns[None]<d)).sum())
                proxies.append({'page':identifier,'method':method,'target_id':label['id'],'envelope_ink_pixels':total,
                    'retained_envelope_ink_pixels':retained,'lost_fraction_proxy':1-retained/total if total else 0.,
                    'pixel_role_ground_truth':False})
    (folder/'images.json').write_bytes(encoded(images));(folder/'ink-proxies.json').write_bytes(encoded(proxies))
    from html import escape
    html=['<!doctype html><meta charset="utf-8"><title>YOLOX / CV diagnostic review</title>',
        '<style>body{font:16px Arial;margin:24px}section{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}img{width:100%;border:1px solid #ccc}figure{margin:0}h2{grid-column:1/-1}</style>',
        '<h1>YOLOX / CV: diagnostic layout review</h1><p>Gray boxes: reviewed targets. Red/blue: predictions. No OCR or physical acceptance.</p>']
    for target in targets:
        html.append(f"<section><h2>{escape(target['id'])} / {escape(target['split'])}</h2>")
        for row in images:
            if row['page']==target['id']:
                name=Path(row['image']['path']).name;html.append(f'<figure><figcaption>{escape(row["method"])}</figcaption><a href="{name}"><img src="{name}"></a></figure>')
        html.append('</section>')
    (folder/'index.html').write_text('\n'.join(html),encoding='utf-8')
    return {'overlays':len(images),'ink_proxies':len(proxies)}
