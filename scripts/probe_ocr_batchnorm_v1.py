"""Predeclared train-only BN state probe; validation reading, no test model calls."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from cad_recovery.ocr_recognition_core_v1 import native_environment,native_model,tensor_rgb,decode_ctc,encode_targets,epoch_batches
from cad_recovery.ocr_batchnorm_policy_v1 import batchnorm_layers,batchnorm_buffer_keys,train_with_frozen_batchnorm,snapshot,state_changes
from smoke_ocr_training_native_v1 import state_digest
from train_native_recognition_v1 import process_peak_ram


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--config-sha256',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert digest(a.config)==a.config_sha256;c=json.loads(a.config.read_text('utf-8'))
    for q in c['pins']:assert digest(ROOT/q['path'])==q['sha256'],q['path']
    assert c['protocol']['train_passes_per_arm']==1 and c['protocol']['test_model_calls']==0
    runtime=json.loads((ROOT/c['runtime_config']['path']).read_text('utf-8'));native_environment(ROOT,runtime['source_root'])
    import paddle
    from ppocr.losses import build_loss
    assert paddle.__version__=='3.2.2';paddle.set_device('cpu')
    def seed():paddle.seed(20261108);np.random.seed(20261108);random.seed(20261108)
    seed();model,chars,upstream=native_model(ROOT,runtime);buffers=batchnorm_buffer_keys(model,paddle)
    initial=snapshot(model.state_dict());initial_digest=state_digest(initial)
    target=a.output.resolve();assert not target.exists() and target.is_relative_to(ROOT/'artifacts');target.mkdir()
    (target/'tensors').mkdir();train=[];val=[]
    for samples,values,save in [(c['train_samples'],train,False),(c['validation_samples'],val,True)]:
        for sample in samples:
            q=sample['image'];assert digest(ROOT/q['path'])==q['sha256']
            with Image.open(ROOT/q['path']) as im:x,meta=tensor_rgb(np.array(im.convert('RGB')))
            values.append(x)
            if save:np.save(target/'tensors'/f"{sample['id']}.npy",x,allow_pickle=False)
    arms={};timings={};started=time.perf_counter()
    def evaluate(name):
        tick=time.perf_counter();model.eval();values=[];(target/name).mkdir()
        with paddle.no_grad():
            for sample,x in zip(c['validation_samples'],val,strict=True):
                prob=model(paddle.to_tensor(x[None])).numpy()[0];text,confidence=decode_ctc(prob,chars)
                values.append({'id':sample['id'],'prediction':text,'confidence':confidence,'status':'ok' if text else 'empty'})
                np.save(target/name/f"{sample['id']}.npy",prob,allow_pickle=False)
        arms[name]=values;timings[name]=time.perf_counter()-tick
    batches=epoch_batches(121,np.random.default_rng(20261108),8)
    def train_pass(frozen):
        seed();tick=time.perf_counter()
        if frozen:train_with_frozen_batchnorm(model,paddle)
        else:model.train()
        consumed=[]
        with paddle.no_grad():
            for indices in batches:
                consumed.extend(indices);texts=[c['train_samples'][i]['text'] for i in indices]
                ctc,nrtr,lengths=encode_targets(texts,chars)
                batch=[paddle.to_tensor(np.stack([train[i] for i in indices])),paddle.to_tensor(ctc),paddle.to_tensor(nrtr),
                       paddle.to_tensor(lengths),paddle.ones([len(indices)],dtype='float32')]
                model(batch[0],data=batch[1:])
        assert sorted(consumed)==list(range(121))
        return time.perf_counter()-tick
    evaluate('baseline');assert state_digest(model.state_dict())==initial_digest
    timings['bn_live_train_pass']=train_pass(False);live=snapshot(model.state_dict());live_changes=state_changes(initial,live,buffers)
    assert live_changes['batchnorm_changed'] and not live_changes['other_changed']
    np.savez(target/'live-bn-buffers.npz',**{k:live[k] for k in buffers});evaluate('bn_live')
    for key in buffers:model.state_dict()[key].set_value(initial[key])
    assert state_digest(model.state_dict())==initial_digest;evaluate('restored')
    assert arms['restored']==arms['baseline']
    timings['bn_frozen_train_pass']=train_pass(True);frozen=snapshot(model.state_dict());frozen_changes=state_changes(initial,frozen,buffers)
    assert not frozen_changes['batchnorm_changed'] and not frozen_changes['other_changed'];evaluate('bn_frozen')
    assert arms['bn_frozen']==arms['baseline']
    # Synthetic backward qualifies affine gradients and the future mode; no customer gradient step.
    seed();train_with_frozen_batchnorm(model,paddle)
    synthetic=np.random.default_rng(20261108).uniform(-1,1,(2,3,48,320)).astype('float32')
    ctc,nrtr,lengths=encode_targets(['10','AB'],chars)
    batch=[paddle.to_tensor(synthetic),paddle.to_tensor(ctc),paddle.to_tensor(nrtr),paddle.to_tensor(lengths),paddle.ones([2],dtype='float32')]
    criterion=build_loss(upstream['Loss']);optimizer=paddle.optimizer.Adam(learning_rate=1e-5,beta1=0.9,beta2=0.999,weight_decay=3e-5,parameters=model.parameters())
    loss=criterion(model(batch[0],data=batch[1:]),batch)['loss'];scalar=float(loss.numpy().item());assert np.isfinite(scalar);loss.backward()
    affine=[];finite_gradients=0
    affine_keys={layer+'.'+suffix for layer in batchnorm_layers(model,paddle) for suffix in ('weight','bias')}
    for name,param in model.named_parameters():
        if param.grad is not None:
            assert bool(paddle.all(paddle.isfinite(param.grad)).item());finite_gradients+=1
            if name in affine_keys:affine.append(name)
    assert affine;optimizer.step();optimizer.clear_grad();updated=snapshot(model.state_dict())
    synthetic_changes=state_changes(initial,updated,buffers)
    assert not synthetic_changes['batchnorm_changed'] and synthetic_changes['other_changed']
    paddle.save(model.state_dict(),str(target/'frozen-bn-synthetic-step.pdparams'))
    for key in initial:model.state_dict()[key].set_value(initial[key])
    assert state_digest(model.state_dict())==initial_digest
    for q in c['pins']:assert digest(ROOT/q['path'])==q['sha256'],q['path']
    report={'schema_version':'ocr-batchnorm-probe-result/1.0','config_sha256':a.config_sha256,'arms':arms,'batchnorm_layers':len(batchnorm_layers(model,paddle)),
        'batchnorm_buffers':buffers,'live_changes':live_changes,'frozen_changes':frozen_changes,
        'initial_state_sha256':initial_digest,'final_customer_state_restored':True,'train_rows_per_arm':121,'customer_gradient_steps':0,
        'synthetic':{'steps':1,'loss':scalar,'finite_gradient_parameters':finite_gradients,'bn_affine_gradient_parameters':len(affine),
            'changes':synthetic_changes,'checkpoint_sha256':digest(target/'frozen-bn-synthetic-step.pdparams')},
        'test_model_calls':0,'timings':timings,'elapsed_seconds':time.perf_counter()-started,'memory':process_peak_ram()}
    (target/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'bn_layers':report['batchnorm_layers'],'customer_gradient_steps':0,'synthetic_steps':1,'test_model_calls':0}))


if __name__=='__main__':main()
