"""A single next experiment: native BN.eval during train; validation only."""
from datetime import datetime,timezone
from pathlib import Path
from .dataset_layout_v1 import checked,encoded,inside,pin,read
from .family_catalog_v2 import require
from .ocr_recognition_core_v1 import TRAINING,selection_score

PARAMETERS={**TRAINING,'parameters':'all native trainable parameters; BatchNorm.eval after each model.train; affine gradients enabled',
    'batchnorm':'global running mean/variance fixed at pretrained values throughout training',
    'checkpoint_retention':'epoch zero, strict improvements with optimizer, final weights and optimizer even if rejected',
    'test_policy':'validation-only development experiment; no test model calls, no automatic final evaluation'}
HOLDOUT={'existing_test_v1':'consumed; never rerun for this experiment',
    'new_final_test_allowed':False,'release':None,
    'requirements':['new source/label selection protocol before new inference',
        'new reviewed accepted OCR release and explicit provenance/group quarantine checks',
        'exclude all train/validation sources/groups and already evaluated test-v1 sources/groups',
        'lock selected checkpoint/export before a separately authorized final evaluation'],
    'independent_evaluation':False,'family_independence':'unproven; directory separation alone cannot approve independence'}
CODE=['src/cad_recovery/ocr_frozen_bn_experiment_v1.py','scripts/train_ocr_frozen_bn_v1.py','scripts/seal_ocr_frozen_bn_v1.py']


def validate_scope(c):
    require(c['schema_version']=='ocr-frozen-bn-experiment/1.0' and c['parameters']==PARAMETERS
        and c['training_ready'] is True and c['test_inference_allowed'] is False and c['final_holdout']==HOLDOUT,
        'Changed frozen-BN experiment or holdout gate')
    require(c['dataset_gradient_training_started'] is False and c['validation_reused_for_development'] is True
        and c['automatic_production_update'] is False and c['independent_evaluation'] is False,'Changed development-only scope')
    train=c['train_rows'];val=c['validation_rows']
    require(len(train)==121 and len(val)==64 and all(r['split']=='train' for r in train)
        and all(r['split']=='validation' for r in val) and len({r['id'] for r in train+val})==185,'Wrong data routing')
    for key in ('image_sha256','source_sha256','group_id'):
        require(not ({r[key] for r in train}&{r[key] for r in val}),'Train/validation leakage')
    require(c['epoch_zero_selection_score']==selection_score(val,c['epoch_zero_predictions']), 'Changed epoch-zero selection')


def seal(workspace,diagnosis_pin,output):
    from .ocr_first_training_v1 import EXPERIMENT,inventory
    from .ocr_regression_v1 import verify
    root=Path(workspace).resolve();diagnosis=verify(root,diagnosis_pin);parent=read(checked(root,EXPERIMENT))
    manifest=read(checked(root,diagnosis_pin));config=read(checked(root,manifest['config']))
    pins={q['path']:q for q in config['pins']}
    for q in inventory(root,checked(root,diagnosis_pin).parent)+[pin(root,n) for n in CODE]+[diagnosis_pin,manifest['config']]:pins[q['path']]=q
    c={'schema_version':'ocr-frozen-bn-experiment/1.0','task':'P16.2c','sealed_before_training_utc':datetime.now(timezone.utc).isoformat(),
        'previous_experiment':EXPERIMENT,'diagnosis_manifest':diagnosis_pin,'parameters':PARAMETERS,'final_holdout':HOLDOUT,
        'runtime_config':parent['runtime_config'],'release':parent['release'],'train_rows':parent['train_rows'],
        'validation_rows':parent['validation_rows'],'epoch_zero_predictions':parent['epoch_zero_predictions'],
        'epoch_zero_selection_score':parent['epoch_zero_selection_score'],'inputs':[pins[n] for n in sorted(pins)],
        'training_ready':True,'dataset_gradient_training_started':False,'test_inference_allowed':False,
        'validation_reused_for_development':True,'automatic_production_update':False,'independent_evaluation':False,
        'scope':'one CPU validation-only controlled BN-mode variant; other training parameters unchanged',
        'evidence_limit':'BN-only probe supports sensitivity to statistics; it does not apportion v1 regression between BN and learned parameters'}
    validate_scope(c);path=inside(root,output);require(not path.exists(),'Choose a new experiment config');path.write_bytes(encoded(c));return pin(root,output)


def verify_protocol(workspace,config_pin):
    from .ocr_first_training_v1 import EXPERIMENT
    from .ocr_regression_v1 import verify
    root=Path(workspace).resolve();c=read(checked(root,config_pin));validate_scope(c)
    for q in c['inputs']:checked(root,q)
    require(set(CODE).issubset({q['path'] for q in c['inputs']}) and len(c['inputs'])==len({q['path'] for q in c['inputs']}),'Missing or duplicate next-experiment pins')
    verify(root,c['diagnosis_manifest']);old=read(checked(root,EXPERIMENT))
    require(c['previous_experiment']==EXPERIMENT and c['runtime_config']==old['runtime_config'] and c['release']==old['release']
        and c['train_rows']==old['train_rows'] and c['validation_rows']==old['validation_rows']
        and c['epoch_zero_predictions']==old['epoch_zero_predictions'],'Changed paired experiment inputs')
    return {'training_ready':True,'scope':c['scope'],'train':121,'validation':64,'test_inference_allowed':False,
        'new_final_test_allowed':False,'dataset_gradient_training_started':False,'validation_reused_for_development':True}
