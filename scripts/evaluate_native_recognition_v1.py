"""Original Paddle 3.2.2 recognition only; no labels enter the model."""
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


def digest(p):
    with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--config-sha256',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert digest(a.config)==a.config_sha256;c=json.loads(a.config.read_text('utf-8'))
    runtime=json.loads((ROOT/c['runtime_config']['path']).read_text('utf-8'))
    native_environment(ROOT,runtime['source_root'])
    import paddle
    assert paddle.__version__=='3.2.2';paddle.set_device('cpu');paddle.seed(20261108)
    for q in c['pins']:
        assert digest(ROOT/q['path'])==q['sha256'],q['path']
    target=a.output.resolve();assert target.is_relative_to(ROOT/'artifacts') and not target.exists();target.mkdir()
    started=time.perf_counter();model,chars,_=native_model(ROOT,runtime);initialization=time.perf_counter()-started
    before=state_digest(model.state_dict());results=[];timings=[]
    (target/'tensors').mkdir();(target/'probabilities').mkdir()
    for sample in c['samples']:
        q=sample['image'];assert digest(ROOT/q['path'])==q['sha256'];tick=time.perf_counter()
        with Image.open(ROOT/q['path']) as im:rgb=np.array(im.convert('RGB'))
        x,pre=tensor_rgb(rgb);pre_time=time.perf_counter()-tick
        tick=time.perf_counter()
        with paddle.no_grad():prob=model(paddle.to_tensor(x[None])).numpy()[0]
        elapsed=time.perf_counter()-tick
        text,confidence=decode_ctc(prob,chars)
        np.save(target/'tensors'/f"{sample['id']}.npy",x,allow_pickle=False)
        np.save(target/'probabilities'/f"{sample['id']}.npy",prob,allow_pickle=False)
        results.append({'id':sample['id'],'prediction':text,'confidence':confidence,'status':'ok' if text else 'empty','preprocessing':pre})
        timings.append({'id':sample['id'],'decode_preprocess_seconds':pre_time,'recognize_seconds':elapsed})
    assert state_digest(model.state_dict())==before
    updated_runtime={**runtime,'checkpoint':c['export_cases']['updated_smoke']['path']}
    updated,_,_=native_model(ROOT,updated_runtime)
    for count in (1,2):
        synthetic=np.random.default_rng(20261108).uniform(-1,1,(count,3,48,320)).astype('float32')
        with paddle.no_grad():prob=updated(paddle.to_tensor(synthetic)).numpy()
        np.save(target/f'updated-batch-{count}.npy',prob,allow_pickle=False)
    for q in c['pins']:assert digest(ROOT/q['path'])==q['sha256'],q['path']
    report={'schema_version':'native-recognition-result/1.0','config_sha256':a.config_sha256,'results':results,
        'runtime':{'paddle':paddle.__version__,'executable':str(Path(sys.executable).resolve()),'initialization_seconds':initialization,
                   'rows':timings,'recognize_total_seconds':sum(r['recognize_seconds'] for r in timings)},
        'pretrained_state_sha256':before,'updated_checkpoint_state_sha256':state_digest(updated.state_dict()),
        'model_state_unchanged':True,'training_started':False,'test_used':False}
    (target/'result.json').write_text(json.dumps(report,ensure_ascii=True,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'rows':len(results),'model_state_unchanged':True}))


if __name__=='__main__':main()
