"""Validation-only regression analysis, causal BN state probe and next protocol."""
from collections import Counter
from datetime import datetime,timezone
from html import escape
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset_layout_v1 import checked,encoded,inside,pin,read
from .family_catalog_v2 import require
from .ocr_training_experiment_v1 import verify_experiment
from .ocr_first_training_v1 import EXPERIMENT,training_output,package,verify_tensors
from .ocr_recognition_baseline_v1 import _worker
from .ocr_recognition_core_v1 import TRAINING,KINDS,selection_score,decode_ctc
from .ocr_recognition_metrics_v1 import score_recognition,assembly_diagnostics
from .ocr_baseline_metrics_v1 import score_text_v1
from .ocr_release_v3 import _package

PARENT={'path':'artifacts/ocr-first-training-final-v1.json','sha256':'769839dcec904f9175bd52c92fb49dfb46f5ab67213a1d1578a68060fb8dcc68'}
QUALIFICATION={'path':'artifacts/ocr-first-training-qualification-v1/manifest.json','sha256':'8545778d84b105376b7d54a351ca94fc2c14e9f13bbdf2ccd76a2a983c1a3167'}
MARKER='artifacts/ocr-first-training-test-once-v1.json'
CODE=['src/cad_recovery/ocr_regression_v1.py','src/cad_recovery/ocr_batchnorm_policy_v1.py',
      'scripts/probe_ocr_batchnorm_v1.py','scripts/analyze_ocr_regression_v1.py']
PROBE={'seed':20261108,'train_passes_per_arm':1,'batch_size':8,'train_order':'same first-epoch default_rng permutation as experiment v1',
    'arms':['baseline','bn_live','restored','bn_frozen'],'gradient_steps_on_customer_data':0,
    'bn_live':'model.train, no_grad, one full train pass; then eval validation',
    'restored':'restore only original BN mean/variance, then eval validation',
    'bn_frozen':'reset RNG, model.train except BatchNorm.eval, no_grad same train pass; then eval validation',
    'synthetic_smoke':'one Adam step on synthetic tensors with BN.eval; affine gradients enabled, BN buffers unchanged',
    'test_model_calls':0,'causal_scope':'isolates effect of one pass of BN running-state updates in the pretrained model; does not identify the cause of the 20-epoch parameter-updating run'}


def regression_analysis(c,epochs,baseline_arms):
    rows=c['validation_rows'];zero={p['id']:p for p in c['epoch_zero_predictions']}
    require(len(epochs)==20,'Incomplete history')
    transitions=[];by_row={r['id']:{'lost_epochs':[],'repaired_epochs':[]} for r in rows}
    for number,epoch in enumerate(epochs,1):
        require(epoch['epoch']==number,'Changed epoch order')
        p={v['id']:v for v in epoch['predictions']};metrics=score_recognition(rows,epoch['predictions'])
        require(metrics['selection_score']==epoch['selection'],'Changed history metrics')
        counts={}
        for kind in KINDS:
            lost=[];fixed=[]
            for r in rows:
                if r['text_kind']!=kind:continue
                before=zero[r['id']]['status']=='ok' and zero[r['id']]['prediction']==r['text']
                after=p[r['id']]['status']=='ok' and p[r['id']]['prediction']==r['text']
                if before and not after:lost.append(r['id']);by_row[r['id']]['lost_epochs'].append(number)
                if not before and after:fixed.append(r['id']);by_row[r['id']]['repaired_epochs'].append(number)
            counts[kind]={'lost':lost,'repaired':fixed}
        transitions.append({'epoch':number,'loss':epoch['mean_batch_loss'],'raw':metrics['overall']['raw'],
            'selection':metrics['selection_score'],'by_kind':counts})
    chars_train=Counter(''.join(r['text'] for r in c['train_rows']));chars_val=Counter(''.join(r['text'] for r in rows))
    missing={ch:chars_val[ch] for ch in sorted(chars_val.keys()-chars_train.keys())}
    overlap=sorted({r['text'] for r in c['train_rows']}&{r['text'] for r in rows})
    examples=[]
    for row in rows:
        fate=by_row[row['id']]
        if fate['lost_epochs'] or fate['repaired_epochs']:
            examples.append({'id':row['id'],'text_kind':row['text_kind'],'reference':row['text'],'baseline':zero[row['id']]['prediction'],
                **fate,'epoch15':next(p['prediction'] for p in epochs[14]['predictions'] if p['id']==row['id']),
                'epoch20':next(p['prediction'] for p in epochs[19]['predictions'] if p['id']==row['id'])})
    examples.sort(key=lambda r:(-len(r['lost_epochs']),r['id']))
    return {'epochs':transitions,'examples':examples,'train':{'rows':121,'unique_images':len({r['image_sha256'] for r in c['train_rows']}),
        'sources':len({r['source_sha256'] for r in c['train_rows']}),'groups':len({r['group_id'] for r in c['train_rows']}),
        'unique_literals':len({r['text'] for r in c['train_rows']}),'characters':len(chars_train),
        'fields':dict(sorted(Counter(r['field'] for r in c['train_rows']).items()))},
        'validation':{'rows':64,'unique_images':len({r['image_sha256'] for r in rows}),'characters':len(chars_val),
            'characters_absent_from_train':missing,'rows_with_characters_absent_from_train':sum(any(ch in missing for ch in r['text']) for r in rows)},
        'literal_target_overlap':overlap,'baseline_selection':c['epoch_zero_selection_score'],
        'best_trained_epoch_by_macro':min(range(1,21),key=lambda i:epochs[i-1]['selection']['macro_raw_cer']),
        'pipeline_diagnostics':assembly_diagnostics(rows,baseline_arms['legacy_pipeline'],baseline_arms['legacy_rec']),
        'limitations':['Epoch frequencies describe repeated observations of the same 64 rows, not independent observations.',
            'Unseen train characters remain supported by the pretrained inference alphabet; not an OOV or label-repair claim.',
            'Literal overlap does not imply image/source/group leakage.',
            'Rejected v1 checkpoints were not saved, so their actual BN buffers/weights cannot be inspected retrospectively.']}


def freeze(workspace,output):
    root=Path(workspace).resolve();checked(root,PARENT);verify_experiment(root,EXPERIMENT);c=read(checked(root,EXPERIMENT))
    qpath,qmanifest=_package(root,QUALIFICATION);qc=read(checked(root,qmanifest['config']))
    for q in qc['pins']:checked(root,q)
    summary,epochs,checkpoint,replay=training_output(root,c,qc['training'])
    names={p['path']:p for p in qc['pins']}
    for q in [PARENT,QUALIFICATION,pin(root,MARKER)]+[pin(root,n) for n in CODE]:names[q['path']]=q
    cfg={'schema_version':'ocr-regression-probe-config/1.0','created_before_probe_utc':datetime.now(timezone.utc).isoformat(),
        'experiment':EXPERIMENT,'parent':PARENT,'qualification':QUALIFICATION,'runtime_config':c['runtime_config'],
        'protocol':PROBE,'training':qc['training'],'old_test_marker':pin(root,MARKER),'pins':[names[n] for n in sorted(names)],
        'train_samples':[{'id':r['id'],'image':r['image_pin'],'text':r['text']} for r in c['train_rows']],
        'validation_samples':[{'id':r['id'],'image':r['image_pin']} for r in c['validation_rows']]}
    require(len(cfg['train_samples'])==121 and len(cfg['validation_samples'])==64,'Wrong probe scope')
    target=inside(root,output);require(not target.exists(),'Choose a new probe config');target.write_bytes(encoded(cfg));return pin(root,output)


def authenticate(root,config_pin):
    cfg=read(checked(root,config_pin));require(cfg['schema_version']=='ocr-regression-probe-config/1.0' and cfg['protocol']==PROBE
        and cfg['experiment']==EXPERIMENT and cfg['parent']==PARENT,'Changed diagnostic protocol')
    for q in cfg['pins']:checked(root,q)
    checked(root,cfg['old_test_marker']);c=read(checked(root,EXPERIMENT))
    require(cfg['train_samples']==[{'id':r['id'],'image':r['image_pin'],'text':r['text']} for r in c['train_rows']]
        and cfg['validation_samples']==[{'id':r['id'],'image':r['image_pin']} for r in c['validation_rows']],'Diagnostic used different samples')
    require(set(CODE).issubset({p['path'] for p in cfg['pins']}),'Incomplete probe code pins')
    return cfg,c


def result_report(root,cfg,c,native):
    epochs=[read(root/cfg['training']/f'epoch-{i:02d}.json') for i in range(1,21)]
    baseline=read(checked(root,c['baseline_manifest']).parent/'predictions.json')
    analysis=regression_analysis(c,epochs,baseline)
    require(native['arms']['baseline']==c['epoch_zero_predictions'],'Probe initial model differs from sealed baseline')
    require(native['arms']['restored']==native['arms']['bn_frozen']==native['arms']['baseline'],'Restoration/frozen BN control failed')
    require(native['customer_gradient_steps']==native['test_model_calls']==0 and native['final_customer_state_restored']
        and not native['live_changes']['other_changed'] and native['live_changes']['batchnorm_changed']
        and not native['frozen_changes']['batchnorm_changed'] and not native['frozen_changes']['other_changed'], 'BN effect not isolated')
    require(native['synthetic']['steps']==1 and native['synthetic']['bn_affine_gradient_parameters']>0
        and not native['synthetic']['changes']['batchnorm_changed'] and native['synthetic']['changes']['other_changed'],'Frozen BN backward smoke failed')
    metrics={name:score_recognition(c['validation_rows'],p) for name,p in native['arms'].items()}
    chars=set(read(root/'artifacts/ocr-training-preparation-v1/inference-alphabet.json')['characters'])|{' '}
    require(all(ch in chars for r in c['validation_rows'] for ch in r['text']),'Unsupported validation alphabet')
    return {'schema_version':'ocr-regression-report/1.0','task':'P16.2c','analysis':analysis,'bn_probe':metrics,
        'bn_layer_count':native['batchnorm_layers'],'bn_buffer_count':len(native['batchnorm_buffers']),
        'bn_changed_buffer_count':len(native['live_changes']['batchnorm_changed']),
        'synthetic_backward':native['synthetic'],'causal_scope':PROBE['causal_scope'],
        'customer_gradient_steps':0,'test_model_calls':0,'old_test_marker':cfg['old_test_marker'],
        'test_inference_allowed':False,'independent_evaluation':False,'production_model_updated':False}


def run(workspace,config_pin,output):
    root=Path(workspace).resolve();cfg,c=authenticate(root,config_pin)
    target=inside(root,output);require(target.parent==root/'artifacts' and not target.exists(),'Choose a new diagnostic output');target.mkdir()
    runtime=read(checked(root,cfg['runtime_config']))
    native=_worker(root,config_pin,target/'native',runtime['python'],'scripts/probe_ocr_batchnorm_v1.py')
    authenticate(root,config_pin);report=result_report(root,cfg,c,native)
    (target/'report.json').write_bytes(encoded(report))
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Разбор OCR-регрессии</title><h1>Validation, без test</h1>']
    for row in report['analysis']['examples']:
        page.append('<section><h3>'+escape(row['id'])+'</h3><p>'+escape(row['reference'])+'</p><p>'+escape(row['baseline'])+'</p><p>'+escape(row['epoch20'])+'</p></section>')
    (target/'index.html').write_text(''.join(page)+'</html>',encoding='utf-8')
    return package(root,target,'ocr-regression-manifest/1.0',config=config_pin)


def verify(workspace,manifest_pin):
    root=Path(workspace).resolve();path,m=_package(root,manifest_pin);require(m['schema_version']=='ocr-regression-manifest/1.0','Wrong diagnostic manifest')
    cfg,c=authenticate(root,m['config']);native=read(path.parent/'native/result.json');report=result_report(root,cfg,c,native)
    require(report==read(path.parent/'report.json'),'Diagnostic report changed')
    chars=read(root/'artifacts/ocr-training-preparation-v1/inference-alphabet.json')['characters']
    for arm,predictions in native['arms'].items():
        require([p['id'] for p in predictions]==[r['id'] for r in c['validation_rows']],'Wrong diagnostic predictions')
        for p in predictions:
            probability=np.load(path.parent/'native'/arm/f"{p['id']}.npy",allow_pickle=False)
            text,confidence=decode_ctc(probability,chars);require(p['prediction']==text and p['confidence']==confidence,'Saved CTC decode differs')
    return report
