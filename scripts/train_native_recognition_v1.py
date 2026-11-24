"""Sealed first CPU OCR experiment. Invoked separately in P16.2b, not by baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from cad_recovery.ocr_recognition_core_v1 import (TRAINING,native_environment,native_model,tensor_rgb,
    decode_ctc,encode_targets,epoch_batches,selection_score,improves)


def process_peak_ram():
    if os.name!='nt':return {'peak_bytes':None,'scope':'platform measurement unavailable'}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ('peak_working_set','working_set','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
    value=Counters();value.cb=ctypes.sizeof(value)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb):raise ctypes.WinError(ctypes.get_last_error())
    return {'peak_bytes':int(value.peak_working_set),'scope':'native process-lifetime PeakWorkingSetSize including model/data/authentication; CPU, not isolated model RAM'}


def digest(path):
    with path.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def validate_training_scope(config):
    if config['schema_version']!='ocr-first-training-experiment/1.0' or config['training_ready'] is not True or config['parameters']!=TRAINING:
        raise ValueError('Unsealed or altered first training experiment')
    train=config['train_rows'];val=config['validation_rows']
    if len(train)!=121 or len(val)!=64 or any(r['split']!='train' for r in train) or any(r['split']!='validation' for r in val):
        raise ValueError('Wrong train/validation routing')
    if len({r['id'] for r in train+val})!=185:raise ValueError('Duplicate training/evaluation IDs')
    for key in ('image_sha256','source_sha256','group_id'):
        if {r[key] for r in train}&{r[key] for r in val}:raise ValueError('Train/validation leakage: '+key)
    if config['test_used_for_selection'] is not False or config['automatic_production_update'] is not False:
        raise ValueError('Test/production scope changed')


def main():
    from smoke_ocr_training_native_v1 import state_digest
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--experiment-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if digest(a.experiment)!=a.experiment_sha256:raise ValueError('Experiment changed')
    c=json.loads(a.experiment.read_text('utf-8'));validate_training_scope(c)
    for q in c['inputs']:
        if digest(ROOT/q['path'])!=q['sha256']:raise ValueError('Pinned training input changed: '+q['path'])
    runtime=json.loads((ROOT/c['runtime_config']['path']).read_text('utf-8'));native_environment(ROOT,runtime['source_root'])
    import paddle
    from ppocr.losses import build_loss
    if paddle.__version__!='3.2.2':raise ValueError('Training must use the original Paddle 3.2.2 CPU environment')
    paddle.set_device('cpu');paddle.seed(TRAINING['seed']);np.random.seed(TRAINING['seed']);random.seed(TRAINING['seed'])
    model,chars,upstream=native_model(ROOT,runtime)
    encode_targets([r['text'] for r in c['train_rows']],chars)
    def prepare(rows):
        tensors=[]
        for row in rows:
            q=row['image_pin']
            if digest(ROOT/q['path'])!=q['sha256']:raise ValueError('Training crop changed')
            with Image.open(ROOT/q['path']) as im:x,_=tensor_rgb(np.array(im.convert('RGB')))
            tensors.append(x)
        return tensors
    train_x=prepare(c['train_rows']);val_x=prepare(c['validation_rows'])
    def evaluate():
        model.eval();predictions=[]
        for row,x in zip(c['validation_rows'],val_x,strict=True):
            with paddle.no_grad():prob=model(paddle.to_tensor(x[None])).numpy()[0]
            text,confidence=decode_ctc(prob,chars)
            predictions.append({'id':row['id'],'prediction':text,'confidence':confidence,'status':'ok' if text else 'empty'})
        return predictions,selection_score(c['validation_rows'],predictions)
    zero,baseline=evaluate()
    if zero!=c['epoch_zero_predictions'] or baseline!=c['epoch_zero_selection_score']:
        raise ValueError('Epoch-zero model differs from the sealed baseline')
    target=a.output.resolve()
    if target.parent!=ROOT/'artifacts' or target.exists():raise ValueError('Choose a new artifacts experiment output')
    target.mkdir();started=time.perf_counter();initial_state=state_digest(model.state_dict())
    paddle.save(model.state_dict(),str(target/'epoch-00.pdparams'))
    best=baseline;best_epoch=0;best_path='epoch-00.pdparams';history=[]
    criterion=build_loss(upstream['Loss'])
    optimizer=paddle.optimizer.Adam(learning_rate=TRAINING['learning_rate'],beta1=TRAINING['beta1'],beta2=TRAINING['beta2'],
         weight_decay=TRAINING['weight_decay'],parameters=model.parameters())
    rng=np.random.default_rng(TRAINING['seed'])
    for epoch in range(1,TRAINING['epochs']+1):
        model.train();batches=epoch_batches(len(train_x),rng,TRAINING['batch_size']);consumed=[];losses=[]
        for indices in batches:
            consumed.extend(indices);texts=[c['train_rows'][i]['text'] for i in indices]
            ctc,nrtr,lengths=encode_targets(texts,chars)
            images=paddle.to_tensor(np.stack([train_x[i] for i in indices]))
            batch=[images,paddle.to_tensor(ctc),paddle.to_tensor(nrtr),paddle.to_tensor(lengths),paddle.ones([len(indices)],dtype='float32')]
            values=criterion(model(images,data=batch[1:]),batch);loss=values['loss']
            scalar=float(loss.numpy().item())
            if not np.isfinite(scalar):raise ValueError('Non-finite loss; experiment stopped')
            loss.backward()
            if any(p.grad is not None and not bool(paddle.all(paddle.isfinite(p.grad)).item()) for p in model.parameters()):
                raise ValueError('Non-finite gradient; experiment stopped')
            optimizer.step();optimizer.clear_grad();losses.append(scalar)
        if sorted(consumed)!=list(range(121)):raise ValueError('Loader skipped or repeated train rows')
        predictions,score=evaluate();accepted=improves(score,best,baseline)
        if accepted:
            best=score;best_epoch=epoch;best_path=f'epoch-{epoch:02d}.pdparams'
            paddle.save(model.state_dict(),str(target/best_path));paddle.save(optimizer.state_dict(),str(target/f'epoch-{epoch:02d}.pdopt'))
        record={'epoch':epoch,'mean_batch_loss':float(np.mean(losses)),'consumed_rows':len(consumed),'batch_sizes':[len(b) for b in batches],
                'selection':score,'new_best':accepted,'predictions':predictions,'elapsed_seconds':time.perf_counter()-started}
        history.append(record);(target/f'epoch-{epoch:02d}.json').write_text(json.dumps(record,ensure_ascii=True,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print(json.dumps({'epoch':epoch,'new_best':accepted,'macro_raw_cer':score['macro_raw_cer']}),flush=True)
    for q in c['inputs']:
        if digest(ROOT/q['path'])!=q['sha256']:raise ValueError('Pinned training input changed during training')
    final={'schema_version':'ocr-first-training-result/1.0','experiment_sha256':a.experiment_sha256,'epochs_completed':len(history),
       'train_rows_consumed_total':121*len(history),'selected_epoch':best_epoch,'selected_checkpoint':best_path,
       'selected_checkpoint_sha256':digest(target/best_path),'selection':best,'baseline_selection':baseline,
       'initial_state_sha256':initial_state,'dataset_training_started':True,'test_evaluated':False,'production_model_updated':False,
       'training_and_validation_seconds':time.perf_counter()-started,'memory':process_peak_ram(),'gpu_memory_measured':False,
       'remaining':['selected-checkpoint ONNX export and equivalence','one final test evaluation','full training experiment verification']}
    (target/'selected.json').write_text(json.dumps(final,ensure_ascii=True,indent=2,sort_keys=True)+'\n',encoding='utf-8')


if __name__=='__main__':main()
