"""Inspect saved model/optimizer arrays; zero forward calls or training steps."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from cad_recovery.ocr_recognition_core_v1 import native_environment,native_model
from cad_recovery.ocr_batchnorm_policy_v1 import batchnorm_buffer_keys,snapshot,state_changes


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def optimizer_arrays(value):
    if isinstance(value,dict):return sum(optimizer_arrays(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(optimizer_arrays(v) for v in value)
    if hasattr(value,'numpy') or isinstance(value,np.ndarray):
        array=value.numpy() if hasattr(value,'numpy') else value
        if not np.isfinite(array).all():raise ValueError('Non-finite optimizer tensor')
        return 1
    if isinstance(value,(int,float,np.number)) and not bool(np.isfinite(value)):raise ValueError('Non-finite optimizer scalar')
    return 0


def main():
    from smoke_ocr_training_native_v1 import state_digest
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--config-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert digest(a.config)==a.config_sha256;c=json.loads(a.config.read_text('utf-8'))
    for q in c['pins']:assert digest(ROOT/q['path'])==q['sha256'],q['path']
    runtime=json.loads((ROOT/c['runtime_config']['path']).read_text('utf-8'));native_environment(ROOT,runtime['source_root'])
    import paddle
    assert paddle.__version__=='3.2.2';paddle.set_device('cpu');paddle.seed(20261108)
    model,chars,_=native_model(ROOT,runtime);initial=snapshot(model.state_dict());buffers=batchnorm_buffer_keys(model,paddle)
    initial_digest=state_digest(initial);records=[];selected_digest=None;final_digest=None
    trainable={name for name,param in model.named_parameters() if not param.stop_gradient}
    for q in c['models']:
        weights=paddle.load(str(ROOT/q['path']));weights.pop('StructuredToParameterName@@',None)
        arrays={k:v.numpy() if hasattr(v,'numpy') else v for k,v in weights.items()};changes=state_changes(initial,arrays,buffers)
        assert not changes['batchnorm_changed'],q['path']
        value_digest=state_digest(arrays)
        if q==c['checkpoint']:selected_digest=value_digest
        if Path(q['path']).name=='final-epoch.pdparams':final_digest=value_digest;assert changes['other_changed']
        if Path(q['path']).name=='epoch-00.pdparams':assert value_digest==initial_digest and not changes['other_changed']
        records.append({'checkpoint':q,'state_sha256':value_digest,'BN_changed':0,'other_changed':len(changes['other_changed']),
            'changed_trainable_entries':sum(name in trainable for name in changes['other_changed']),
            'changed_nontrainable_nonBN_entries':[name for name in changes['other_changed'] if name not in trainable]})
    assert selected_digest and final_digest and len(buffers)==302 and len(initial)==968
    optimizer_records=[]
    for q in c['optimizers']:
        state=paddle.load(str(ROOT/q['path']));count=optimizer_arrays(state);assert count>0
        optimizer_records.append({'optimizer':q,'finite_tensor_count':count})
    assert optimizer_records
    for q in c['pins']:assert digest(ROOT/q['path'])==q['sha256'],q['path']
    target=a.output.resolve();assert not target.exists() and target.is_relative_to(ROOT/'artifacts');target.mkdir()
    result={'schema_version':'ocr-frozen-bn-checkpoint-audit/1.0','config_sha256':a.config_sha256,'state_entries':len(initial),
        'BN_buffer_count':len(buffers),'trainable_parameter_entries':len(trainable),'all_BN_buffers_unchanged':True,'models':records,'optimizers':optimizer_records,
        'initial_state_sha256':initial_digest,'selected_state_sha256':selected_digest,'final_state_sha256':final_digest,
        'model_forward_calls':0,'test_model_calls':0,'gradient_steps':0}
    (target/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'models':len(records),'optimizers':len(optimizer_records),'all_BN_buffers_unchanged':True,'model_forward_calls':0}))


if __name__=='__main__':main()
