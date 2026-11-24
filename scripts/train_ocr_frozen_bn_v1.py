"""One sealed BN-mode variant; all customer inference remains train/validation."""
import argparse,hashlib,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from cad_recovery.ocr_frozen_bn_experiment_v1 import PARAMETERS,validate_scope
from cad_recovery.ocr_batchnorm_policy_v1 import batchnorm_buffer_keys,train_with_frozen_batchnorm,snapshot,state_changes
from cad_recovery.ocr_recognition_core_v1 import native_environment,native_model,tensor_rgb,encode_targets,decode_ctc,epoch_batches,selection_score,improves
from smoke_ocr_training_native_v1 import state_digest
from train_native_recognition_v1 import process_peak_ram


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--experiment-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if digest(a.experiment)!=a.experiment_sha256:raise ValueError('Experiment changed')
    c=json.loads(a.experiment.read_text('utf-8'));validate_scope(c)
    for q in c['inputs']:
        if digest(ROOT/q['path'])!=q['sha256']:raise ValueError('Input changed: '+q['path'])
    runtime=json.loads((ROOT/c['runtime_config']['path']).read_text('utf-8'));native_environment(ROOT,runtime['source_root'])
    import paddle
    from ppocr.losses import build_loss
    if paddle.__version__!='3.2.2':raise ValueError('Original CPU runtime required')
    paddle.set_device('cpu');paddle.seed(PARAMETERS['seed']);np.random.seed(PARAMETERS['seed']);random.seed(PARAMETERS['seed'])
    model,chars,upstream=native_model(ROOT,runtime);buffers=batchnorm_buffer_keys(model,paddle);initial=snapshot(model.state_dict())
    def prepare(rows):
        values=[]
        for r in rows:
            q=r['image_pin']
            if digest(ROOT/q['path'])!=q['sha256']:raise ValueError('Crop changed')
            with Image.open(ROOT/q['path']) as im:x,_=tensor_rgb(np.array(im.convert('RGB')))
            values.append(x)
        return values
    train=prepare(c['train_rows']);val=prepare(c['validation_rows']);encode_targets([r['text'] for r in c['train_rows']],chars)
    def evaluate():
        model.eval();predictions=[]
        with paddle.no_grad():
            for r,x in zip(c['validation_rows'],val,strict=True):
                text,confidence=decode_ctc(model(paddle.to_tensor(x[None])).numpy()[0],chars)
                predictions.append({'id':r['id'],'prediction':text,'confidence':confidence,'status':'ok' if text else 'empty'})
        return predictions,selection_score(c['validation_rows'],predictions)
    zero,baseline=evaluate()
    if zero!=c['epoch_zero_predictions'] or baseline!=c['epoch_zero_selection_score']:raise ValueError('Epoch-zero baseline changed')
    target=a.output.resolve()
    if target.parent!=ROOT/'artifacts' or target.exists():raise ValueError('Choose a new training output')
    target.mkdir();started=time.perf_counter();paddle.save(model.state_dict(),str(target/'epoch-00.pdparams'))
    criterion=build_loss(upstream['Loss']);optimizer=paddle.optimizer.Adam(learning_rate=PARAMETERS['learning_rate'],
        beta1=PARAMETERS['beta1'],beta2=PARAMETERS['beta2'],weight_decay=PARAMETERS['weight_decay'],parameters=model.parameters())
    best=baseline;best_epoch=0;best_path='epoch-00.pdparams';rng=np.random.default_rng(PARAMETERS['seed'])
    for epoch in range(1,PARAMETERS['epochs']+1):
        train_with_frozen_batchnorm(model,paddle);batches=epoch_batches(121,rng,8);consumed=[];losses=[]
        for indices in batches:
            consumed.extend(indices);ctc,nrtr,lengths=encode_targets([c['train_rows'][i]['text'] for i in indices],chars)
            batch=[paddle.to_tensor(np.stack([train[i] for i in indices])),paddle.to_tensor(ctc),paddle.to_tensor(nrtr),
                paddle.to_tensor(lengths),paddle.ones([len(indices)],dtype='float32')]
            loss=criterion(model(batch[0],data=batch[1:]),batch)['loss'];value=float(loss.numpy().item())
            if not np.isfinite(value):raise ValueError('Non-finite loss')
            loss.backward()
            if any(p.grad is not None and not bool(paddle.all(paddle.isfinite(p.grad)).item()) for p in model.parameters()):raise ValueError('Non-finite gradient')
            optimizer.step();optimizer.clear_grad();losses.append(value)
        if sorted(consumed)!=list(range(121)):raise ValueError('Loader skipped/repeated rows')
        changes=state_changes(initial,snapshot(model.state_dict()),buffers)
        if changes['batchnorm_changed']:raise ValueError('Frozen BN statistics changed')
        predictions,score=evaluate();accepted=improves(score,best,baseline)
        if accepted:
            best=score;best_epoch=epoch;best_path=f'epoch-{epoch:02d}.pdparams'
            paddle.save(model.state_dict(),str(target/best_path));paddle.save(optimizer.state_dict(),str(target/f'epoch-{epoch:02d}.pdopt'))
        record={'epoch':epoch,'mean_batch_loss':float(np.mean(losses)),'consumed_rows':121,'batch_sizes':[len(b) for b in batches],
            'selection':score,'new_best':accepted,'predictions':predictions,'bn_stats_unchanged':True,
            'changed_non_buffer_state_entries':len(changes['other_changed']),'elapsed_seconds':time.perf_counter()-started}
        (target/f'epoch-{epoch:02d}.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print(json.dumps({'epoch':epoch,'new_best':accepted,'macro_raw_cer':score['macro_raw_cer']}),flush=True)
    paddle.save(model.state_dict(),str(target/'final-epoch.pdparams'));paddle.save(optimizer.state_dict(),str(target/'final-epoch.pdopt'))
    for q in c['inputs']:
        if digest(ROOT/q['path'])!=q['sha256']:raise ValueError('Input changed during training')
    final={'schema_version':'ocr-frozen-bn-training-result/1.0','experiment_sha256':a.experiment_sha256,'epochs_completed':PARAMETERS['epochs'],
        'train_rows_consumed_total':2420,'selected_epoch':best_epoch,'selected_checkpoint':best_path,'selected_checkpoint_sha256':digest(target/best_path),
        'selection':best,'baseline_selection':baseline,'initial_state_sha256':state_digest(initial),'final_state_sha256':state_digest(model.state_dict()),
        'bn_stats_unchanged':True,'final_checkpoint_sha256':digest(target/'final-epoch.pdparams'),'test_evaluated':False,'test_inference_allowed':False,
        'production_model_updated':False,'training_and_validation_seconds':time.perf_counter()-started,'memory':process_peak_ram(),
        'remaining':['offline verification of all epochs and frozen buffers','selected export qualification','new reviewed holdout protocol before any final test']}
    (target/'selected.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n',encoding='utf-8')


if __name__=='__main__':main()
