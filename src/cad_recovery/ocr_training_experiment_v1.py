"""Seal the first OCR fine-tuning experiment after recognition/export qualification."""
from pathlib import Path
from datetime import datetime,timezone
from .dataset_layout_v1 import checked,encoded,inside,pin,read
from .family_catalog_v2 import require
from .ocr_recognition_baseline_v1 import verify,authenticate,RELEASE,RUNTIME
from .ocr_recognition_core_v1 import TRAINING,encode_targets,selection_score
from .ocr_validation_expansion_v1 import verify_release

CODE=['src/cad_recovery/ocr_training_experiment_v1.py','scripts/train_native_recognition_v1.py','scripts/seal_ocr_training_experiment_v1.py']


def seal(workspace,baseline_manifest,output):
    root=Path(workspace).resolve();report=verify(root,baseline_manifest)
    manifest=read(checked(root,baseline_manifest));baseline=read(checked(root,manifest['config']))
    _,records,base=verify_release(root,RELEASE)
    train=[{**r,'image_pin':pin(base,r['image'])} for r in records if r['split']=='train']
    for r in train:r['image_pin']['path']=(base/r['image']).relative_to(root).as_posix()
    require(len(train)==121 and len({r['image_sha256'] for r in train})==120,'Changed accepted train corpus')
    chars=read(root/'artifacts/ocr-training-preparation-v1/inference-alphabet.json')['characters']
    encode_targets([r['text'] for r in train],chars)
    val=baseline['rows'];require(len(val)==64,'Wrong validation size')
    for k in ('image_sha256','source_sha256','group_id'):
        require(not ({r[k] for r in train}&{r[k] for r in val}),'Train/validation overlap')
    predictions=read(checked(root,baseline_manifest).parent/'predictions.json')['native_rec']
    zero=[{k:r[k] for k in ('id','prediction','confidence','status')} for r in predictions]
    score=selection_score(val,zero)
    require(score==report['arms']['native_rec']['selection_score'] and len(report['export_equivalence'])==64
            and all(r['allclose'] and r['text_equal'] for r in report['export_equivalence'])
            and len(report['updated_checkpoint_cross_backend_export'])==2,'Incomplete baseline/export qualification')
    pins={q['path']:q for q in baseline['pins']}
    for name in CODE+[baseline_manifest['path'],manifest['config']['path'],
                     (checked(root,baseline_manifest).parent/'predictions.json').relative_to(root).as_posix(),
                     (checked(root,baseline_manifest).parent/'report.json').relative_to(root).as_posix()]:pins[name]=pin(root,name)
    for r in train:pins[r['image_pin']['path']]=r['image_pin']
    experiment={'schema_version':'ocr-first-training-experiment/1.0','task':'P16.2a.4',
        'sealed_before_training_utc':datetime.now(timezone.utc).isoformat(),'baseline_manifest':baseline_manifest,
        'baseline_config':manifest['config'],'release':RELEASE,'runtime_config':RUNTIME,'parameters':TRAINING,
        'train_rows':train,'validation_rows':val,'epoch_zero_predictions':zero,'epoch_zero_selection_score':score,
        'inputs':[pins[n] for n in sorted(pins)],'training_ready':True,'dataset_training_started':False,
        'test_used_for_selection':False,'automatic_production_update':False,'independent_evaluation':False,
        'scope':'one CPU recognition fine-tuning experiment only; other task/CAD/VLM gates remain open',
        'after_training':['verify selected checkpoint/export numerical equivalence','one final evaluation on 19 accepted test rows',
                          'retain existing production model unless improvement is measured and reviewed']}
    path=inside(root,output);require(not path.exists(),'Choose a new experiment config');path.write_bytes(encoded(experiment));return pin(root,output)


def verify_experiment(workspace,experiment_pin):
    root=Path(workspace).resolve();c=read(checked(root,experiment_pin))
    require(c['schema_version']=='ocr-first-training-experiment/1.0' and c['parameters']==TRAINING
            and c['release']==RELEASE and c['runtime_config']==RUNTIME and c['training_ready'] is True
            and c['dataset_training_started'] is False and c['test_used_for_selection'] is False
            and c['automatic_production_update'] is False,'Changed sealed training scope')
    for p in c['inputs']:checked(root,p)
    require(set(CODE).issubset({p['path'] for p in c['inputs']}) and len(c['inputs'])==len({p['path'] for p in c['inputs']}), 'Incomplete experiment code pins')
    baseline=authenticate(root,c['baseline_config'],full=True)
    require(c['validation_rows']==baseline['rows'] and len(c['train_rows'])==121,'Changed experiment data')
    report=verify(root,c['baseline_manifest'])
    _,registry,base=verify_release(root,RELEASE)
    expected=[{**r,'image_pin':pin(base,r['image'])} for r in registry if r['split']=='train']
    for r in expected:r['image_pin']['path']=(base/r['image']).relative_to(root).as_posix()
    require(c['train_rows']==expected,'Training targets/source bindings changed')
    baseline_path=checked(root,c['baseline_manifest'])
    zero=read(baseline_path.parent/'predictions.json')['native_rec']
    zero=[{k:r[k] for k in ('id','prediction','confidence','status')} for r in zero]
    require(c['epoch_zero_predictions']==zero,'Epoch-zero raw predictions changed')
    require(c['epoch_zero_selection_score']==selection_score(c['validation_rows'],c['epoch_zero_predictions'])
            and c['epoch_zero_selection_score']==report['arms']['native_rec']['selection_score'],'Changed epoch zero')
    require(all(r['split']=='train' for r in c['train_rows']) and all(r['split']=='validation' for r in c['validation_rows']), 'Changed experiment split')
    return {'task':'P16.2a.4','training_ready':True,'scope':c['scope'],'train_rows':121,'validation_rows':64,
            'epochs':TRAINING['epochs'],'test_used':False,'dataset_training_started':False}
