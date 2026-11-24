"""Blinded image-only predictions; raw tensors remain available for CPU replay."""
import time
import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked,encoded,pin,read
from .family_catalog_v2 import require
from .gpu_training_probe_v1 import state_digest
from .yolox_model_v1 import build
from .yolox_runtime_v1 import BUDGETS,preprocess
from .yolox_training_v1 import worker_config,configure_torch
from .semantic_regions_v1 import legacy,candidate


def infer(root,config_pin,output):
    config=worker_config(root,config_pin)
    request=read(checked(root,config['inference']))
    require(set(request)=={'cases'} and len(request['cases']) in (4,6),'Invalid inference envelope')
    require(len({r['id'] for r in request['cases']})==len(request['cases']),'Repeated inference case')
    for row in request['cases']:require(set(row)=={'id','image','size_px'},'Targets leaked into inference')
    require(sum(r['size_px'][0]*r['size_px'][1] for r in request['cases'])<=BUDGETS['maximum_inference_batch_pixels'],'Pixel budget exceeded')
    require(not output.exists(),'Choose a fresh inference output');output.mkdir(parents=True)
    method=config['method']; model=None;initialization=time.perf_counter()
    if method=='yolox':
        torch=configure_torch(root);model,transfer=build(root)
        checkpoint=torch.load(checked(root,config['checkpoint']),map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['model'],strict=True)
        require(state_digest(model.state_dict())==config['trained_final'],'Trained model rebound')
        model=model.to(config['device']).eval()
        if config['device']=='cuda':torch.cuda.reset_peak_memory_stats()
    else:require(method in ('legacy_grid','context_cv_rejected_candidate'),'Unknown CV method')
    initialization=time.perf_counter()-initialization;rows=[]
    for case in request['cases']:
        begin=time.perf_counter()
        with Image.open(checked(root,case['image'])) as image:
            require(image.mode=='RGB' and list(image.size)==case['size_px'],'Inference RGB/grid changed')
            rgb=np.asarray(image).copy()
        if method=='yolox':
            x,geometry=preprocess(rgb,config['arm']);tensor=torch.from_numpy(x[None]).to(config['device'])
            if config['device']=='cuda':torch.cuda.synchronize()
            compute=time.perf_counter()
            with torch.inference_mode():raw=model(tensor)
            if config['device']=='cuda':torch.cuda.synchronize()
            compute=time.perf_counter()-compute
            require(tuple(raw.shape)==(1,8400,10) and bool(torch.isfinite(raw).all()),'Invalid raw detections')
            values=raw[0].cpu().numpy();np.save(output/f"{case['id']}.npy",values,allow_pickle=False)
            rows.append({'id':case['id'],'image':case['image'],'size_px':case['size_px'],'geometry':geometry,
                'raw':pin(root,(output/f"{case['id']}.npy").relative_to(root).as_posix()),
                'compute_seconds':compute,'page_seconds':time.perf_counter()-begin})
        else:
            predictions=(legacy if method=='legacy_grid' else candidate)(rgb)
            rows.append({'id':case['id'],'image':case['image'],'size_px':case['size_px'],
                'predictions':predictions,'page_seconds':time.perf_counter()-begin})
        print(method,case['id'],round(rows[-1]['page_seconds'],4),flush=True)
    report={'schema_version':'yolox-inference-run/1.0','config':config_pin,'method':method,'rows':rows,
        'initialization_seconds':initialization,'device':config.get('device','cpu'),
        'ground_truth_read':False,'production_admitted':False}
    if method=='yolox' and config['device']=='cuda':
        report.update(allocated_peak_MiB=torch.cuda.max_memory_allocated()/1024**2,
                      reserved_peak_MiB=torch.cuda.max_memory_reserved()/1024**2)
    (output/'report.json').write_bytes(encoded(report));return report
