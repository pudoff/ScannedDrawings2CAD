"""Pinned CPU validation inference, or pixel-only preparation of final test."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from cad_recovery.ocr_recognition_core_v1 import native_environment,native_model,tensor_rgb,decode_ctc
from smoke_ocr_training_native_v1 import state_digest
from train_native_recognition_v1 import process_peak_ram


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--config-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if digest(a.config)!=a.config_sha256:raise ValueError('Changed inference configuration')
    c=json.loads(a.config.read_text('utf-8'));mode=c['mode']
    if mode not in ('validation','prepare_test'):raise ValueError('Invalid inference scope')
    for q in c['pins']:
        if digest(ROOT/q['path'])!=q['sha256']:raise ValueError('Changed input: '+q['path'])
    runtime=json.loads((ROOT/c['runtime_config']['path']).read_text('utf-8'));native_environment(ROOT,runtime['source_root'])
    target=a.output.resolve()
    if target.exists() or not target.is_relative_to(ROOT/'artifacts'):raise ValueError('Choose a new output')
    target.mkdir();(target/'tensors').mkdir();results=[];timings=[];before=None
    if mode=='validation':
        import paddle
        if paddle.__version__!='3.2.2':raise ValueError('Original training runtime required')
        paddle.set_device('cpu');paddle.seed(20261108)
        model,chars,_=native_model(ROOT,{**runtime,'checkpoint':c['checkpoint']['path']})
        before=state_digest(model.state_dict());(target/'probabilities').mkdir()
    for sample in c['samples']:
        q=sample['image']
        if digest(ROOT/q['path'])!=q['sha256']:raise ValueError('Changed crop')
        tick=time.perf_counter()
        with Image.open(ROOT/q['path']) as im:x,meta=tensor_rgb(np.array(im.convert('RGB')))
        elapsed=time.perf_counter()-tick
        np.save(target/'tensors'/f"{sample['id']}.npy",x,allow_pickle=False)
        if mode=='validation':
            tick=time.perf_counter()
            with paddle.no_grad():prob=model(paddle.to_tensor(x[None])).numpy()[0]
            timings.append({'id':sample['id'],'preprocess_seconds':elapsed,'recognize_seconds':time.perf_counter()-tick})
            text,confidence=decode_ctc(prob,chars)
            results.append({'id':sample['id'],'prediction':text,'confidence':confidence,'status':'ok' if text else 'empty','preprocessing':meta})
            np.save(target/'probabilities'/f"{sample['id']}.npy",prob,allow_pickle=False)
        else:results.append({'id':sample['id'],'preprocessing':meta})
    if mode=='validation' and state_digest(model.state_dict())!=before:raise ValueError('Inference changed model state')
    for q in c['pins']:
        if digest(ROOT/q['path'])!=q['sha256']:raise ValueError('Input changed during inference')
    report={'schema_version':'selected-native-recognition/1.0','mode':mode,'config_sha256':a.config_sha256,
        'results':results,'state_sha256':before,'model_state_unchanged':True,'timings':timings,
        'test_model_calls':0,'memory':process_peak_ram(),'gradient_steps':0}
    (target/'result.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'mode':mode,'samples':len(results),'test_model_calls':0}))


if __name__=='__main__':main()
