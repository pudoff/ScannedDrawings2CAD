"""Execute and verify the sealed development-only frozen-BN experiment."""
import math
from pathlib import Path
import numpy as np
from .dataset_layout_v1 import checked,encoded,inside,pin,read
from .family_catalog_v2 import require
from .ocr_frozen_bn_experiment_v1 import PARAMETERS,verify_protocol,validate_scope
from .ocr_first_training_v1 import inventory,package,ort_session,prediction,verify_tensors
from .ocr_recognition_baseline_v1 import _worker,EXPORT_ENV
from .ocr_recognition_core_v1 import selection_score,improves,compare_export,EXPORT_BUDGET
from .ocr_recognition_metrics_v1 import score_recognition
from .ocr_release_v3 import _package

EXPERIMENT={'path':'configs/reviews/ocr-frozen-bn-experiment-v1.json','sha256':'32b1c072c9d81f4ccd965d305c2f3fb10d12046858df7e2474f7dbbc3fbc2c44'}
PARENT={'path':'artifacts/ocr-regression-final-v1.json','sha256':'9bc1ca43cd2959894f872aa21688d4465ea3ba0518ea39c9ca784d78b3ae3719'}
CODE=['src/cad_recovery/ocr_frozen_bn_run_v1.py','scripts/audit_frozen_bn_checkpoint_v1.py','scripts/evaluate_ocr_frozen_bn_v1.py']


def replay(c,summary,epochs):
    validate_scope(c)
    require(summary['schema_version']=='ocr-frozen-bn-training-result/1.0' and summary['experiment_sha256']==EXPERIMENT['sha256']
        and summary['epochs_completed']==20 and summary['train_rows_consumed_total']==2420 and len(epochs)==20
        and summary['bn_stats_unchanged'] is True and summary['test_evaluated'] is False
        and summary['test_inference_allowed'] is False and summary['production_model_updated'] is False,'Changed training scope')
    baseline=selection_score(c['validation_rows'],c['epoch_zero_predictions']);best=baseline;best_epoch=0;accepted=[];elapsed=0
    require(summary['baseline_selection']==baseline==c['epoch_zero_selection_score'],'Changed epoch zero')
    for i,epoch in enumerate(epochs,1):
        require(epoch['epoch']==i and epoch['consumed_rows']==121 and epoch['batch_sizes']==[8]*15+[1]
            and epoch['bn_stats_unchanged'] is True and 0<epoch['changed_non_buffer_state_entries']<=666
            and math.isfinite(epoch['mean_batch_loss']) and epoch['elapsed_seconds']>=elapsed,'Invalid or incomplete epoch')
        elapsed=epoch['elapsed_seconds'];metrics=score_recognition(c['validation_rows'],epoch['predictions']);score=metrics['selection_score']
        take=improves(score,best,baseline)
        require(epoch['selection']==score and epoch['new_best'] is take,'Checkpoint rule did not replay')
        if take:best=score;best_epoch=i;accepted.append(i)
    require(summary['selected_epoch']==best_epoch and summary['selected_checkpoint']==f'epoch-{best_epoch:02d}.pdparams'
        and summary['selection']==best and summary['training_and_validation_seconds']>=elapsed,'Wrong selected checkpoint')
    require(summary['initial_state_sha256']=='7f1ada1d7f6e4c07700c03a50a978cbc6002d65f9494170264e918c834426156'
        and summary['final_state_sha256']!=summary['initial_state_sha256'],'Wrong or unchanged final model')
    return {'epochs':20,'optimizer_steps':320,'train_presentations':2420,'selected_epoch':best_epoch,
        'improving_epochs':accepted,'selection':best,'improved_over_epoch_zero':best_epoch!=0,'test_model_calls':0}


def load_training(root,c,training):
    base=inside(root,training);require(base.parent==root/'artifacts','Wrong training path')
    summary=read(base/'selected.json');epochs=[read(base/f'epoch-{i:02d}.json') for i in range(1,21)];result=replay(c,summary,epochs)
    selected=pin(root,(base/summary['selected_checkpoint']).relative_to(root).as_posix())
    final=pin(root,(base/'final-epoch.pdparams').relative_to(root).as_posix())
    require(selected['sha256']==summary['selected_checkpoint_sha256'] and final['sha256']==summary['final_checkpoint_sha256'],'Weights changed')
    model_names={f'epoch-{i:02d}.pdparams' for i in [0]+result['improving_epochs']}|{'final-epoch.pdparams'}
    opt_names={f'epoch-{i:02d}.pdopt' for i in result['improving_epochs']}|{'final-epoch.pdopt'}
    require({p.name for p in base.glob('*.pdparams')}==model_names and {p.name for p in base.glob('*.pdopt')}==opt_names,'Wrong checkpoint/optimizer inventory')
    return summary,epochs,selected,result


def summarize(c,summary,epochs,result,values,equivalence,audit):
    require(len(equivalence)==64 and all(e['allclose'] and e['text_equal'] for e in equivalence),'Selected export failed frozen numeric budget')
    expected=c['epoch_zero_predictions'] if result['selected_epoch']==0 else epochs[result['selected_epoch']-1]['predictions']
    require(values['native']==expected,'Selected reload differs from recorded validation')
    require(audit['all_BN_buffers_unchanged'] and audit['final_state_sha256']==summary['final_state_sha256']
        and audit['model_forward_calls']==audit['test_model_calls']==0,'Checkpoint state audit failed')
    return {'schema_version':'ocr-frozen-bn-run-report/1.0','task':'P16.2d','training_replay':result,
        'native_validation':score_recognition(c['validation_rows'],values['native']),
        'exported_validation':score_recognition(c['validation_rows'],values['exported']),
        'equivalence':equivalence,'export_budget':EXPORT_BUDGET,'checkpoint_audit':audit,
        'training_seconds':summary['training_and_validation_seconds'],'training_memory':summary['memory'],
        'test_model_calls':0,'test_inference_allowed':False,'new_final_test_allowed':False,
        'validation_reused_for_development':True,'independent_evaluation':False,'production_model_updated':False}


def qualify(workspace,training,output):
    root=Path(workspace).resolve();checked(root,PARENT);verify_protocol(root,EXPERIMENT);c=read(checked(root,EXPERIMENT))
    summary,epochs,selected,result=load_training(root,c,training)
    target=inside(root,output);require(target.parent==root/'artifacts' and not target.exists(),'Choose a new qualification output');target.mkdir()
    pins={q['path']:q for q in c['inputs']}
    for q in inventory(root,inside(root,training))+[EXPERIMENT,PARENT]+[pin(root,n) for n in CODE]:pins[q['path']]=q
    from datetime import datetime,timezone
    config={'schema_version':'ocr-frozen-bn-qualification-config/1.0','created_utc':datetime.now(timezone.utc).isoformat(),
        'experiment':EXPERIMENT,'training':training,'runtime_config':c['runtime_config'],'mode':'validation','checkpoint':selected,
        'pins':[pins[n] for n in sorted(pins)],'export_cases':{'selected':selected},'export_budget':EXPORT_BUDGET,
        'samples':[{'id':r['id'],'image':r['image_pin']} for r in c['validation_rows']],
        'models':[q for q in inventory(root,inside(root,training)) if q['path'].endswith('.pdparams')],
        'optimizers':[q for q in inventory(root,inside(root,training)) if q['path'].endswith('.pdopt')],
        'test_inference_allowed':False}
    (target/'config.json').write_bytes(encoded(config));config_pin=pin(root,(target/'config.json').relative_to(root).as_posix())
    runtime=read(checked(root,c['runtime_config']))
    audit=_worker(root,config_pin,target/'checkpoint-audit',runtime['python'],'scripts/audit_frozen_bn_checkpoint_v1.py')
    _worker(root,config_pin,target/'export',EXPORT_ENV+'/Scripts/python.exe','scripts/export_native_recognition_v1.py')
    native=_worker(root,config_pin,target/'native',runtime['python'],'scripts/infer_selected_ocr_v1.py')
    require(native['state_sha256']==audit['selected_state_sha256'],'Selected state/reload hash mismatch')
    chars=read(root/'artifacts/ocr-training-preparation-v1/inference-alphabet.json')['characters'];session=ort_session(target/'export/selected.onnx')
    values={'native':[{k:r[k] for k in ('id','prediction','confidence','status')} for r in native['results']],'exported':[]};equivalence=[]
    (target/'onnx-probabilities').mkdir()
    for row,p in zip(c['validation_rows'],values['native'],strict=True):
        identifier=row['id'];require(identifier==p['id'],'Wrong validation order')
        x=np.load(target/'native/tensors'/f'{identifier}.npy',allow_pickle=False)[None]
        before=np.load(target/'native/probabilities'/f'{identifier}.npy',allow_pickle=False)
        prob=session.run(None,{session.get_inputs()[0].name:x})[0][0];item=prediction(identifier,prob,chars);values['exported'].append(item)
        np.save(target/'onnx-probabilities'/f'{identifier}.npy',prob,allow_pickle=False)
        equivalence.append({'id':identifier,**compare_export(before,prob),'text_equal':p['prediction']==item['prediction']})
    report=summarize(c,summary,epochs,result,values,equivalence,audit)
    (target/'predictions.json').write_bytes(encoded(values));(target/'report.json').write_bytes(encoded(report))
    for q in config['pins']:checked(root,q)
    return package(root,target,'ocr-frozen-bn-run-manifest/1.0',config=config_pin)


def verify(workspace,manifest_pin):
    root=Path(workspace).resolve();path,m=_package(root,manifest_pin);require(m['schema_version']=='ocr-frozen-bn-run-manifest/1.0','Wrong run manifest')
    config=read(checked(root,m['config']));c=read(checked(root,EXPERIMENT));verify_protocol(root,EXPERIMENT)
    require(config['experiment']==EXPERIMENT and config['mode']=='validation' and config['test_inference_allowed'] is False
        and config['export_budget']==EXPORT_BUDGET and set(CODE).issubset({q['path'] for q in config['pins']}),'Changed qualification protocol')
    for q in config['pins']:checked(root,q)
    summary,epochs,selected,result=load_training(root,c,config['training']);values=read(path.parent/'predictions.json')
    report=read(path.parent/'report.json');audit=read(path.parent/'checkpoint-audit/result.json')
    require(report==summarize(c,summary,epochs,result,values,report['equivalence'],audit),'Changed run metrics')
    require(config['checkpoint']==selected and config['samples']==[{'id':r['id'],'image':r['image_pin']} for r in c['validation_rows']],'Changed evaluation pixels/model')
    native=read(path.parent/'native/result.json');verify_tensors(path.parent/'native',native['results'],config['samples'])
    require(native['state_sha256']==audit['selected_state_sha256'] and native['test_model_calls']==native['gradient_steps']==0,'Wrong native inference scope')
    chars=read(root/'artifacts/ocr-training-preparation-v1/inference-alphabet.json')['characters']
    for index,row in enumerate(c['validation_rows']):
        arrays=[]
        for key,directory in [('native','native/probabilities'),('exported','onnx-probabilities')]:
            prob=np.load(path.parent/directory/f"{row['id']}.npy",allow_pickle=False);arrays.append(prob)
            require(values[key][index]==prediction(row['id'],prob,chars),'Saved CTC output changed')
        e={'id':row['id'],**compare_export(*arrays),'text_equal':values['native'][index]['prediction']==values['exported'][index]['prediction']}
        require(e==report['equivalence'][index],'Changed numerical export evidence')
    export=read(path.parent/'export/result.json')
    require(export['exports'][0]['checkpoint']==selected and export['exports'][0]['onnx_sha256']==pin(root,(path.parent/'export/selected.onnx').relative_to(root).as_posix())['sha256'],'Export model/checkpoint binding changed')
    return report
