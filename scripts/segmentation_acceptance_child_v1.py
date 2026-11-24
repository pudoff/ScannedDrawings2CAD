"""Read-only CUDA inference on authenticated RGB inputs, without reading targets."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from cad_recovery.dataset_layout_v1 import checked,encoded,read,pin
from cad_recovery.family_catalog_v2 import require
from cad_recovery.gpu_training_probe_v1 import build_model,state_digest
from cad_recovery.gpu_inference_v1 import deny_python_network
from cad_recovery import segmentation_experiment_v1 as tiles


def main(a):
    import numpy as np
    from PIL import Image
    import torch
    config=read(checked(ROOT,{'path':a.config,'sha256':a.sha256}))
    for r in config['code']:checked(ROOT,r)
    inputs=read(checked(ROOT,{'path':a.inputs,'sha256':a.inputs_sha256}))
    output=ROOT/a.output;require(output.parent.exists() and not output.exists(),'Choose fresh inference output')
    variant=next(v for v in config['models'] if v['variant']==a.variant)
    torch.set_num_threads(4);torch.manual_seed(20261110);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.conv.fp32_precision='ieee'
    require(torch.__version__=='2.14.1+cu130' and torch.cuda.is_available(),'Unqualified CUDA inference runtime')
    model=build_model();model.final[-1]=torch.nn.Conv2d(16,10,1)
    state=torch.load(checked(ROOT,variant['weights']),map_location='cpu',weights_only=True)
    before=state_digest(state);require(before==variant['tensor_digest'],'Changed frozen model tensors')
    model.load_state_dict(state);del state;model.cuda().eval();torch.cuda.reset_peak_memory_stats()
    output.mkdir();results=[]
    with torch.inference_mode():
        for item in inputs:
            if item['kind']=='full_view' and a.variant not in ('pretrained-gray','pretrained-gray-repeat'):continue
            rgb=np.asarray(Image.open(checked(ROOT,item['image'])).convert('RGB'))
            h,w=rgb.shape[:2];require([w,h]==item['size_px'] and h*w<=config['policy']['maximum_full_view_pixels'],'Invalid native input')
            sums=np.zeros((10,h,w),np.float32);counts=np.zeros((h,w),np.uint16)
            boxes=tiles.boxes(w,h,stride=config['policy']['stride'])
            torch.cuda.synchronize();start=time.perf_counter()
            for box in boxes:
                x,y,right,bottom=box
                image=np.full((256,256,3),255,np.uint8);image[:bottom-y,:right-x]=rgb[y:bottom,x:right]
                values=model(torch.from_numpy(tiles.preprocess(image,variant['mode'])[None]).cuda())[0].cpu().numpy()
                require(np.isfinite(values).all(),'Nonfinite CUDA logits')
                sums[:,y:bottom,x:right]+=values[:,:bottom-y,:right-x];counts[y:bottom,x:right]+=1
            require(bool((counts>0).all()),'Uncovered native pixels')
            np.divide(sums,counts[None],out=sums)
            prediction=sums.argmax(0).astype(np.uint8)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            name=item['id']+'.png';Image.fromarray(prediction).save(output/name)
            results.append({'id':item['id'],'kind':item['kind'],'size_px':[w,h],'tiles':len(boxes),
                'coverage_min':int(counts.min()),'coverage_max':int(counts.max()),'inference_seconds':seconds,
                'prediction':pin(ROOT,(output/name).relative_to(ROOT).as_posix())})
            print(a.variant,item['id'],len(boxes),'tiles',round(seconds,3),'s',flush=True)
            del rgb,sums,counts,prediction
    after=state_digest(model.state_dict());require(before==after,'Inference mutated model state')
    result={'variant':a.variant,'input_mode':variant['mode'],'tensor_digest_before':before,'tensor_digest_after':after,
        'protocol':{'path':a.config,'sha256':a.sha256},'inputs_sha256':a.inputs_sha256,
        'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
        'allocated_peak_MiB':torch.cuda.max_memory_allocated()/1024**2,
        'reserved_peak_MiB':torch.cuda.max_memory_reserved()/1024**2,'items':results,
        'model_eval':True,'gradients_enabled':False,'targets_read':False,'training_executed':False,
        'network_disabled':True,'AMP':False,'TF32':False,'production_admitted':False}
    require(result['allocated_peak_MiB']<=config['policy']['max_allocated_MiB'] and result['reserved_peak_MiB']<=config['policy']['max_reserved_MiB'],'CUDA memory limit')
    (output/'result.json').write_bytes(encoded(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('config','sha256','inputs','inputs-sha256','variant','output'):p.add_argument('--'+n,required=True)
    with deny_python_network():main(p.parse_args())
