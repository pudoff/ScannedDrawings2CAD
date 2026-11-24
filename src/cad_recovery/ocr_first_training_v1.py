"""Audit fixed training, qualify selected export, then evaluate final test once."""
from datetime import datetime,timezone
from html import escape
import json
import math
import os
import hashlib
from pathlib import Path
import time
import numpy as np
from .dataset_layout_v1 import checked,encoded,inside,pin,read
from .family_catalog_v2 import require
from .ocr_training_experiment_v1 import verify_experiment
from .ocr_validation_expansion_v1 import verify_release
from .ocr_recognition_baseline_v1 import _worker,EXPORT_ENV
from .ocr_recognition_core_v1 import TRAINING,EXPORT_BUDGET,selection_score,improves,compare_export,decode_ctc
from .ocr_recognition_metrics_v1 import score_recognition
from .ocr_baseline_metrics_v1 import aggregate_scores_v1
from .ocr_release_v3 import _package

EXPERIMENT={'path':'configs/reviews/ocr-first-training-experiment-v1.json','sha256':'bb17efd77c70de2b073d18791154e8c996234083503bc662de74739a9a3829f4'}
PARENT={'path':'artifacts/ocr-recognition-final-v1.json','sha256':'e81e7daea1ca3e316edecbb61e6a06f165bffbccc88f3ec384a4208856c05813'}
CODE=['src/cad_recovery/ocr_first_training_v1.py','scripts/infer_selected_ocr_v1.py','scripts/evaluate_first_ocr_training_v1.py']
TEST_POLICY={'split':'test','count':19,'selection_after_test':False,'selected_model_evaluations_per_crop':1,
    'arms':['selected_rec','epoch_zero_rec','saved_legacy_pipeline'],
    'epoch_zero':'paired baseline once; reuse selected probabilities if ONNX bytes identical',
    'preprocess':'original pinned Python/Pillow/OpenCV worker; exact frozen tensor_rgb',
    'failure':'retain error/empty in denominator; no rerun or model changes after starting test',
    'production_model_updated':False,'export_budget':EXPORT_BUDGET}


def replay_training(c,summary,epochs):
    require(c['parameters']==TRAINING and len(epochs)==20,'Wrong training protocol or epoch count')
    require(summary['schema_version']=='ocr-first-training-result/1.0' and summary['epochs_completed']==20
        and summary['train_rows_consumed_total']==2420 and summary['dataset_training_started'] is True
        and summary['test_evaluated'] is False and summary['production_model_updated'] is False,'Wrong training scope')
    baseline=selection_score(c['validation_rows'],c['epoch_zero_predictions'])
    require(summary['baseline_selection']==baseline==c['epoch_zero_selection_score'],'Changed epoch zero')
    best=baseline;best_epoch=0;accepted=[];previous_seconds=0
    for number,epoch in enumerate(epochs,1):
        require(epoch['epoch']==number and epoch['consumed_rows']==121 and epoch['batch_sizes']==[8]*15+[1]
            and math.isfinite(epoch['mean_batch_loss']) and epoch['elapsed_seconds']>=previous_seconds,'Incomplete/non-finite training history')
        previous_seconds=epoch['elapsed_seconds']
        # Complete metrics validate statuses, labels and the full denominator too.
        score_recognition(c['validation_rows'],epoch['predictions'])
        score=selection_score(c['validation_rows'],epoch['predictions']);accept=improves(score,best,baseline)
        require(epoch['selection']==score and epoch['new_best'] is accept,'Checkpoint selection does not replay')
        if accept:best=score;best_epoch=number;accepted.append(number)
    require(summary['selected_epoch']==best_epoch and summary['selection']==best
        and summary['selected_checkpoint']==f'epoch-{best_epoch:02d}.pdparams'
        and summary['training_and_validation_seconds']>=previous_seconds,'Selected checkpoint differs from fixed rule')
    return {'epochs':20,'optimizer_steps':320,'train_presentations':2420,'selected_epoch':best_epoch,
        'improving_epochs':accepted,'selection':best,'improved_over_epoch_zero':best_epoch!=0}


def training_output(root,c,training):
    base=inside(root,training);require(base.parent==root/'artifacts','Training output must be an artifacts directory')
    summary=read(base/'selected.json');require(summary['experiment_sha256']==EXPERIMENT['sha256'],'Wrong training experiment')
    epochs=[read(base/f'epoch-{i:02d}.json') for i in range(1,21)];replay=replay_training(c,summary,epochs)
    checkpoint=pin(root,(base/summary['selected_checkpoint']).relative_to(root).as_posix())
    require(checkpoint['sha256']==summary['selected_checkpoint_sha256'],'Selected weights changed')
    for i in [0]+replay['improving_epochs']:
        require((base/f'epoch-{i:02d}.pdparams').is_file(),'Missing saved weights')
        if i:require((base/f'epoch-{i:02d}.pdopt').is_file(),'Missing saved optimizer')
    require(summary['initial_state_sha256']=='7f1ada1d7f6e4c07700c03a50a978cbc6002d65f9494170264e918c834426156','Wrong initial model')
    return summary,epochs,checkpoint,replay


def inventory(root,base):
    return [pin(root,p.relative_to(root).as_posix()) for p in sorted(base.rglob('*')) if p.is_file()]


def package(root,target,schema,**extra):
    files=[{'path':p.relative_to(target).as_posix(),'sha256':pin(root,p.relative_to(root).as_posix())['sha256']}
        for p in sorted(target.rglob('*')) if p.is_file() and p.name!='manifest.json']
    (target/'manifest.json').write_bytes(encoded({'schema_version':schema,'files':files,**extra}))
    return pin(root,(target/'manifest.json').relative_to(root).as_posix())


def ort_session(path):
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1;options.log_severity_level=3
    return ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])


def alphabet(root):return read(root/'artifacts/ocr-training-preparation-v1/inference-alphabet.json')['characters']


def prediction(identifier,prob,chars):
    text,confidence=decode_ctc(prob,chars)
    return {'id':identifier,'prediction':text,'confidence':confidence,'status':'ok' if text else 'empty'}


def verify_tensors(base,results,samples):
    require([r['id'] for r in results]==[r['id'] for r in samples],'Changed prepared tensor inventory/order')
    for r in results:
        x=np.load(base/'tensors'/f"{r['id']}.npy",allow_pickle=False)
        require(x.shape==(3,48,320) and x.dtype==np.float32 and np.isfinite(x).all()
            and hashlib.sha256(x.tobytes()).hexdigest()==r['preprocessing']['tensor_sha256'],'Saved tensor/proof changed')


def qualify(workspace,training,output):
    root=Path(workspace).resolve();checked(root,PARENT);verify_experiment(root,EXPERIMENT);c=read(checked(root,EXPERIMENT))
    summary,epochs,checkpoint,replay=training_output(root,c,training)
    target=inside(root,output);require(target.parent==root/'artifacts' and not target.exists(),'Choose a new qualification directory');target.mkdir()
    pins={p['path']:p for p in c['inputs']}
    for q in inventory(root,inside(root,training))+[pin(root,n) for n in CODE]+[EXPERIMENT,PARENT]:pins[q['path']]=q
    config={'schema_version':'selected-ocr-validation-config/1.0','created_utc':datetime.now(timezone.utc).isoformat(),
        'experiment':EXPERIMENT,'mode':'validation','training':training,'checkpoint':checkpoint,'runtime_config':c['runtime_config'],
        'pins':[pins[k] for k in sorted(pins)],'export_cases':{'selected':checkpoint},
        'samples':[{'id':r['id'],'image':r['image_pin']} for r in c['validation_rows']],
        'test_policy':TEST_POLICY,'export_budget':EXPORT_BUDGET}
    (target/'config.json').write_bytes(encoded(config));config_pin=pin(root,(target/'config.json').relative_to(root).as_posix())
    _worker(root,config_pin,target/'export',EXPORT_ENV+'/Scripts/python.exe','scripts/export_native_recognition_v1.py')
    runtime=read(checked(root,c['runtime_config']))
    native=_worker(root,config_pin,target/'native',runtime['python'],'scripts/infer_selected_ocr_v1.py')
    session=ort_session(target/'export/selected.onnx');chars=alphabet(root);exported=[];equivalence=[];timings=[]
    (target/'onnx-probabilities').mkdir()
    for r,native_prediction in zip(c['validation_rows'],native['results'],strict=True):
        identifier=r['id'];require(identifier==native_prediction['id'],'Wrong validation order')
        x=np.load(target/'native/tensors'/f'{identifier}.npy',allow_pickle=False)[None]
        reference=np.load(target/'native/probabilities'/f'{identifier}.npy',allow_pickle=False);tick=time.perf_counter()
        prob=session.run(None,{session.get_inputs()[0].name:x})[0][0];timings.append(time.perf_counter()-tick)
        p=prediction(identifier,prob,chars);exported.append(p);np.save(target/'onnx-probabilities'/f'{identifier}.npy',prob,allow_pickle=False)
        equivalence.append({'id':identifier,**compare_export(reference,prob),'text_equal':p['prediction']==native_prediction['prediction']})
    expected=c['epoch_zero_predictions'] if summary['selected_epoch']==0 else epochs[summary['selected_epoch']-1]['predictions']
    actual=[{k:r[k] for k in ('id','prediction','confidence','status')} for r in native['results']]
    require(actual==expected,'Reloaded selected model differs from its validation epoch')
    require(all(r['allclose'] and r['text_equal'] for r in equivalence),'Selected model export failed fixed numeric budget; test remains closed')
    for q in config['pins']:checked(root,q)
    report={'schema_version':'first-ocr-training-qualification/1.0','task':'P16.2b','training_replay':replay,
        'checkpoint':checkpoint,'native_validation':score_recognition(c['validation_rows'],actual),
        'exported_validation':score_recognition(c['validation_rows'],exported),'equivalence':equivalence,
        'export_budget':EXPORT_BUDGET,'test_eligible':True,'test_used':False,'production_model_updated':False,
        'training_memory':summary['memory'],'training_and_validation_seconds':summary['training_and_validation_seconds'],
        'onnx_validation_seconds':sum(timings),'limitations':['Fixed tiny corpus; no independent family accuracy claim.',
        'Export engineering budget was revised in P16.2a.4; original strict results retained.']}
    (target/'predictions.json').write_bytes(encoded({'native':actual,'exported':exported}))
    (target/'report.json').write_bytes(encoded(report))
    return package(root,target,'first-ocr-training-qualification-manifest/1.0',config=config_pin)


def verify_qualification(workspace,manifest_pin):
    root=Path(workspace).resolve();path,m=_package(root,manifest_pin)
    require(m['schema_version']=='first-ocr-training-qualification-manifest/1.0','Wrong qualification manifest')
    config=read(checked(root,m['config']));c=read(checked(root,EXPERIMENT))
    require(config['experiment']==EXPERIMENT and config['mode']=='validation' and config['test_policy']==TEST_POLICY
        and config['export_budget']==EXPORT_BUDGET,'Changed qualification protocol')
    for q in config['pins']:checked(root,q)
    require(set(CODE).issubset({q['path'] for q in config['pins']}),'Missing qualification code pins')
    require(config['samples']==[{'id':r['id'],'image':r['image_pin']} for r in c['validation_rows']],'Changed validation pixels')
    summary,epochs,checkpoint,replay=training_output(root,c,config['training']);report=read(path.parent/'report.json')
    require(report['training_replay']==replay and report['checkpoint']==checkpoint==config['checkpoint']
        and report['test_eligible'] is True and report['test_used'] is False,'Changed checkpoint or qualification')
    values=read(path.parent/'predictions.json');native=read(path.parent/'native/result.json');chars=alphabet(root)
    expected=c['epoch_zero_predictions'] if summary['selected_epoch']==0 else epochs[summary['selected_epoch']-1]['predictions']
    require(values['native']==expected,'Qualification does not reproduce selected epoch')
    require(report['native_validation']==score_recognition(c['validation_rows'],values['native'])
        and report['exported_validation']==score_recognition(c['validation_rows'],values['exported']),'Validation metrics changed')
    require(len(native['results'])==len(report['equivalence'])==64 and native['model_state_unchanged']
        and native['gradient_steps']==native['test_model_calls']==0,'Wrong native inference scope')
    verify_tensors(path.parent/'native',native['results'],config['samples'])
    export=read(path.parent/'export/result.json')
    require(len(export['exports'])==1 and export['exports'][0]['checkpoint']==checkpoint
        and export['exports'][0]['onnx_sha256']==pin(root,(path.parent/'export/selected.onnx').relative_to(root).as_posix())['sha256'],
        'Selected export checkpoint/model binding changed')
    for index,row in enumerate(c['validation_rows']):
        identifier=row['id'];arrays=[]
        for directory,key in [('native/probabilities','native'),('onnx-probabilities','exported')]:
            prob=np.load(path.parent/directory/f'{identifier}.npy',allow_pickle=False);arrays.append(prob)
            require(prediction(identifier,prob,chars)==values[key][index],'Saved validation decode changed')
        compare={'id':identifier,**compare_export(*arrays),'text_equal':values['native'][index]['prediction']==values['exported'][index]['prediction']}
        require(compare==report['equivalence'][index] and compare['allclose'] and compare['text_equal'],'Invalid export equivalence')
    return path.parent,config,report


def test_scores(rows,predictions):
    by={p['id']:p for p in predictions}
    require(len(by)==len(predictions)==len(rows) and set(by)=={r['id'] for r in rows},'Incomplete or duplicated test results')
    def score(part):
        return aggregate_scores_v1([{'id':r['id'],'source_id':r['source_id'],'reference':r['text'],
            'prediction':by[r['id']]['prediction'],'status':by[r['id']]['status']} for r in part])
    unique={}
    for r in sorted(rows,key=lambda r:r['id']):unique.setdefault(r['image_sha256'],r)
    return {'overall':score(rows),'unique_images':score(list(unique.values())),
        'by_field':{k:score([r for r in rows if r['field']==k]) for k in sorted({r['field'] for r in rows})},
        'by_group':{k:score([r for r in rows if r['group_id']==k]) for k in sorted({r['group_id'] for r in rows})}}


def paired_test_delta(metrics):
    selected=metrics['selected_rec']['overall']['raw'];zero=metrics['epoch_zero_rec']['overall']['raw']
    return {'exact_matches_selected_minus_zero':selected['exact_matches']-zero['exact_matches'],
            'raw_CER_selected_minus_zero':selected['cer']-zero['cer'],
            'scope':'paired final reporting only; never used to change checkpoint selection'}


def journal_append(path,event):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event,ensure_ascii=True,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def reserve_test(root,config_pin,qualification_pin):
    marker=inside(root,'artifacts/ocr-first-training-test-once-v1.json')
    with marker.open('x',encoding='utf-8') as stream:
        stream.write(encoded({'experiment':EXPERIMENT,'test_config':config_pin,'qualification':qualification_pin,
            'started_utc':datetime.now(timezone.utc).isoformat(),'policy':TEST_POLICY}).decode('utf-8'));stream.flush();os.fsync(stream.fileno())
    return marker


def call_test_once(session,x,identifier,chars,probability_path,journal,arm):
    journal_append(journal,{'stage':'started','arm':arm,'id':identifier})
    tick=time.perf_counter()
    try:
        prob=session.run(None,{session.get_inputs()[0].name:x})[0][0]
        result=prediction(identifier,prob,chars)
        np.save(probability_path,prob,allow_pickle=False)
    except Exception as error:
        result={'id':identifier,'prediction':'','confidence':0.0,'status':'error',
                'error':{'type':type(error).__name__,'message':str(error)}}
    journal_append(journal,{'stage':'completed','arm':arm,'id':identifier,'seconds':time.perf_counter()-tick,'result':result})
    return result


def validate_call_journal(events,identifiers,identical):
    expected=[]
    for identifier in identifiers:
        for arm in ('selected_rec','epoch_zero_rec'):
            stages=('reused',) if arm=='epoch_zero_rec' and identical else ('started','completed')
            expected.extend((stage,arm,identifier) for stage in stages)
    require([(e['stage'],e['arm'],e['id']) for e in events]==expected,'Test journal has extra, missing or reordered calls')


def evaluate_test(workspace,qualification_pin,output):
    root=Path(workspace).resolve();qbase,qconfig,qualification=verify_qualification(root,qualification_pin)
    require(not (root/'artifacts/ocr-first-training-test-once-v1.json').exists(),'Final test already reserved; verify saved results without inference')
    c=read(checked(root,EXPERIMENT));_,records,base=verify_release(root,c['release'])
    rows=[{**r,'image_pin':pin(base,r['image'])} for r in sorted(records,key=lambda r:r['id']) if r['split']=='test']
    for r in rows:r['image_pin']['path']=(base/r['image']).relative_to(root).as_posix()
    require(len(rows)==19 and len({r['image_sha256'] for r in rows})==18,'Changed accepted test scope')
    for key in ('image_sha256','source_sha256','group_id'):
        require(not ({r[key] for r in rows}&{r[key] for r in c['train_rows']+c['validation_rows']}),'Test overlap')
    target=inside(root,output);require(target.parent==root/'artifacts' and not target.exists(),'Choose a new final test output');target.mkdir()
    zero_base=checked(root,c['baseline_manifest']).parent
    selected=pin(root,(qbase/'export/selected.onnx').relative_to(root).as_posix())
    zero=pin(root,(zero_base/'export/pretrained.onnx').relative_to(root).as_posix())
    old_predictions=pin(root,'artifacts/ocr-baseline-v4/predictions.json')
    old=read(checked(root,old_predictions));old_test=[p for p in old if p['split']=='test']
    require(len(old_test)==19 and {p['id'] for p in old_test}=={r['id'] for r in rows},'Historical test scope differs')
    for p in old_test:
        row=next(r for r in rows if r['id']==p['id'])
        require(p['reference']==row['text'] and p['image_sha256']==row['image_sha256'],'Historical test target/pixels differ')
    pins={q['path']:q for q in qconfig['pins']}
    for q in inventory(root,qbase)+[r['image_pin'] for r in rows]+[zero,old_predictions,qualification_pin]:pins[q['path']]=q
    config={'schema_version':'first-ocr-final-test-config/1.0','created_before_test_utc':datetime.now(timezone.utc).isoformat(),
        'experiment':EXPERIMENT,'qualification':qualification_pin,'policy':TEST_POLICY,'mode':'prepare_test',
        'runtime_config':c['runtime_config'],'selected_checkpoint':qconfig['checkpoint'],'selected_onnx':selected,'epoch_zero_onnx':zero,
        'saved_legacy_predictions':old_predictions,'rows':rows,'samples':[{'id':r['id'],'image':r['image_pin']} for r in rows],
        'pins':[pins[k] for k in sorted(pins)],'one_shot_marker':'artifacts/ocr-first-training-test-once-v1.json'}
    (target/'config.json').write_bytes(encoded(config));config_pin=pin(root,(target/'config.json').relative_to(root).as_posix())
    runtime=read(checked(root,c['runtime_config']))
    prepared=_worker(root,config_pin,target/'prepared',runtime['python'],'scripts/infer_selected_ocr_v1.py')
    require(prepared['mode']=='prepare_test' and prepared['test_model_calls']==0,'Test preparation ran a model')
    sessions={'selected_rec':ort_session(checked(root,selected))};identical=selected['sha256']==zero['sha256']
    if not identical:sessions['epoch_zero_rec']=ort_session(checked(root,zero))
    for name in TEST_POLICY['arms'][:2]:(target/name).mkdir()
    marker=reserve_test(root,config_pin,qualification_pin)
    # Once reserved, never rerun a test crop/model, including after a failed call.
    arms={'selected_rec':[],'epoch_zero_rec':[]};chars=alphabet(root);journal=target/'calls.jsonl'
    for row in rows:
        identifier=row['id'];x=np.load(target/'prepared/tensors'/f'{identifier}.npy',allow_pickle=False)[None]
        for arm in ('selected_rec','epoch_zero_rec'):
            if arm=='epoch_zero_rec' and identical:
                p=dict(arms['selected_rec'][-1]);journal_append(journal,{'stage':'reused','arm':arm,'id':identifier,'reason':'identical ONNX bytes'})
            else:p=call_test_once(sessions[arm],x,identifier,chars,target/arm/f'{identifier}.npy',journal,arm)
            arms[arm].append(p)
    arms['saved_legacy_pipeline']=old_test
    metrics={name:test_scores(rows,p) for name,p in arms.items()}
    report={'schema_version':'first-ocr-final-test-report/1.0','task':'P16.2b','selected_epoch':qualification['training_replay']['selected_epoch'],
        'arms':metrics,'test_policy':TEST_POLICY,'actual_model_calls':{'selected_rec':19,'epoch_zero_rec':0 if identical else 19},
        'identical_epoch_zero_onnx_reused':identical,'checkpoint_selection_changed_after_test':False,
        'fine_tuning_improved_validation':qualification['training_replay']['improved_over_epoch_zero'],
        'paired_test_delta':paired_test_delta(metrics),
        'production_model_updated':False,'independent_evaluation':False,'unresolved_test_fields':2,
        'test_pending_sources_preserved':112,'limitations':['Only 19 accepted title rows from seven sources, not full test drawings or dimension OCR.',
            'Existing saved pipeline differs in detection/preprocessing; its delta is not a fine-tuning effect.']}
    (target/'predictions.json').write_bytes(encoded(arms));(target/'report.json').write_bytes(encoded(report))
    (target/'one-shot-marker.json').write_bytes(marker.read_bytes())
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Первое OCR-обучение</title><h1>Итоговый test: 19 принятых строк</h1>']
    for row in rows:
        page.append('<section><h3>'+escape(row['id'])+'</h3><b>'+escape(row['text'])+'</b>')
        for name,values in arms.items():page.append('<p>'+escape(name+': '+next(p['prediction'] for p in values if p['id']==row['id']))+'</p>')
        page.append('</section>')
    (target/'index.html').write_text(''.join(page)+'</html>',encoding='utf-8')
    for q in config['pins']:checked(root,q)
    return package(root,target,'first-ocr-final-test-manifest/1.0',config=config_pin,marker=pin(root,config['one_shot_marker']))


def verify_test(workspace,manifest_pin):
    root=Path(workspace).resolve();path,m=_package(root,manifest_pin)
    require(m['schema_version']=='first-ocr-final-test-manifest/1.0','Wrong final test manifest')
    config=read(checked(root,m['config']));require(config['experiment']==EXPERIMENT and config['policy']==TEST_POLICY,'Changed test policy')
    c=read(checked(root,EXPERIMENT));zero_base=checked(root,c['baseline_manifest']).parent
    require(config['epoch_zero_onnx']==pin(root,(zero_base/'export/pretrained.onnx').relative_to(root).as_posix()),'Epoch-zero baseline changed')
    qbase,qconfig,qualification=verify_qualification(root,config['qualification'])
    require(config['selected_checkpoint']==qconfig['checkpoint'] and config['selected_onnx']==pin(root,(qbase/'export/selected.onnx').relative_to(root).as_posix()),'Test selected model changed')
    marker=checked(root,m['marker']);require(marker.read_bytes()==(path.parent/'one-shot-marker.json').read_bytes(),'Test marker changed')
    require(m['marker']['path']==config['one_shot_marker']=='artifacts/ocr-first-training-test-once-v1.json'
        and read(marker)['test_config']==m['config'] and read(marker)['experiment']==EXPERIMENT,'One-shot marker belongs to another evaluation')
    for q in config['pins']:checked(root,q)
    _,records,base=verify_release(root,c['release'])
    rows=[{**r,'image_pin':pin(base,r['image'])} for r in sorted(records,key=lambda r:r['id']) if r['split']=='test']
    for r in rows:r['image_pin']['path']=(base/r['image']).relative_to(root).as_posix()
    require(config['rows']==rows and config['samples']==[{'id':r['id'],'image':r['image_pin']} for r in rows],'Test targets/pixels changed')
    prepared=read(path.parent/'prepared/result.json')
    require(prepared['mode']=='prepare_test' and prepared['test_model_calls']==0 and prepared['state_sha256'] is None,'Test preparation ran a model')
    verify_tensors(path.parent/'prepared',prepared['results'],config['samples'])
    arms=read(path.parent/'predictions.json');report=read(path.parent/'report.json');chars=alphabet(root)
    require(set(arms)==set(TEST_POLICY['arms']) and report['test_policy']==TEST_POLICY,'Wrong test arms')
    require(report['arms']=={arm:test_scores(rows,p) for arm,p in arms.items()},'Test metrics changed')
    require(report['paired_test_delta']==paired_test_delta(report['arms'])
        and report['fine_tuning_improved_validation'] is qualification['training_replay']['improved_over_epoch_zero'],
        'Reported fine-tuning effect changed')
    require(report['selected_epoch']==qualification['training_replay']['selected_epoch'] and report['checkpoint_selection_changed_after_test'] is False
        and report['production_model_updated'] is False,'Test altered model selection')
    identical=config['selected_onnx']['sha256']==config['epoch_zero_onnx']['sha256']
    require(report['identical_epoch_zero_onnx_reused'] is identical,'Wrong model reuse policy')
    events=[json.loads(line) for line in (path.parent/'calls.jsonl').read_text('utf-8').splitlines()]
    validate_call_journal(events,[r['id'] for r in rows],identical)
    for arm in TEST_POLICY['arms'][:2]:
        started=[e['id'] for e in events if e['stage']=='started' and e['arm']==arm]
        completed=[e for e in events if e['stage']=='completed' and e['arm']==arm]
        expected=[] if arm=='epoch_zero_rec' and identical else [r['id'] for r in rows]
        require(started==expected and [e['id'] for e in completed]==expected and report['actual_model_calls'][arm]==len(expected),'Test calls missing or repeated')
        for index,row in enumerate(rows):
            identifier=row['id'];saved=arms[arm][index];require(saved['id']==identifier,'Wrong test order')
            if arm=='epoch_zero_rec' and identical:
                require(saved==arms['selected_rec'][index],'Reused baseline differs');continue
            require(completed[index]['result']==saved,'Test journal/result mismatch')
            if saved['status']!='error':
                prob=np.load(path.parent/arm/f'{identifier}.npy',allow_pickle=False)
                require(saved==prediction(identifier,prob,chars),'Test decode differs from saved probabilities')
    require(arms['saved_legacy_pipeline']==[p for p in read(checked(root,config['saved_legacy_predictions'])) if p['split']=='test'],'Historical test predictions changed')
    return report


def verify_all(workspace,qualification_pin,test_pin):
    root=Path(workspace).resolve();verify_experiment(root,EXPERIMENT)
    qbase,config,qreport=verify_qualification(root,qualification_pin);test=verify_test(root,test_pin)
    require(read(checked(root,read(checked(root,test_pin))['config']))['qualification']==qualification_pin,'Test uses another qualification')
    return {'task':'P16.2b','training_replay':qreport['training_replay'],'selected_validation':qreport['native_validation']['overall']['raw'],
        'test':{arm:{mode:metrics['overall'][mode] for mode in ('raw','normalized')} for arm,metrics in test['arms'].items()},
        'qualification':qualification_pin,'test_manifest':test_pin,'test_model_calls':test['actual_model_calls'],
        'training_memory':qreport['training_memory'],'training_seconds':qreport['training_and_validation_seconds'],
        'production_model_updated':False,'independent_evaluation':False,'dataset_training_started':True}
