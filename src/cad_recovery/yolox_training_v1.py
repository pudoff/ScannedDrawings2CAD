"""Train the registered detector on admitted train inputs, never validation."""
import sys
import time

import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked,encoded,pin,read
from .family_catalog_v2 import require
from .gpu_training_probe_v1 import state_digest
from .yolox_model_v1 import build,DEPENDENCIES
from .yolox_runtime_v1 import POLICY,BUDGETS,preprocess,learning_rate,schedule


def worker_config(root,config_pin):
    config=read(checked(root,config_pin))
    require(config['policy']==POLICY and config['budgets']==BUDGETS,'Registered experiment changed')
    for item in config['code']:checked(root,item)
    return config


def configure_torch(root):
    sys.path.insert(0,str(root/DEPENDENCIES))
    import torch
    require(torch.__version__=='2.14.1+cu130','Changed torch backend')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.conv.fp32_precision='ieee'
    return torch


def train(root,config_pin,output):
    config=worker_config(root,config_pin);torch=configure_torch(root)
    require(torch.cuda.is_available() and config['arm'] in ('rgb','gray') and not output.exists(),'Invalid training environment/output')
    request=read(checked(root,config['training']))
    require(set(request)=={'cases','schedule'} and request['schedule']==schedule([c['id'] for c in request['cases']]),'Training schedule changed')
    arrays={};labels={};geometries={}
    for row in request['cases']:
        require(set(row)=={'id','image','size_px','labels_cxcywh_model_pixels'},'Unregistered training fields')
        with Image.open(checked(root,row['image'])) as image:
            require(list(image.size)==row['size_px'] and image.mode=='RGB','Training RGB/grid changed')
            arrays[row['id']],geometries[row['id']]=preprocess(np.asarray(image),config['arm'])
        values=np.asarray(row['labels_cxcywh_model_pixels'],dtype=np.float32)
        require(values.ndim==2 and values.shape[1]==5 and 0<len(values)<=16 and np.isfinite(values).all(),'Invalid training targets')
        require(np.all(values[:,0]==values[:,0].astype(int)) and np.all((values[:,0]>=0)&(values[:,0]<5))
                and np.all(values[:,3:]>0),'Invalid class or training box')
        labels[row['id']]=values
    model,transfer=build(root);initial=state_digest(model.state_dict())
    require(initial==config['qualified_initial'],'Initialization differs from qualified model')
    model=model.cuda().train(); optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01)
    output.mkdir(parents=True);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    history=[]
    for step,identifiers in enumerate(request['schedule']):
        x=np.stack([arrays[i] for i in identifiers]);y=np.zeros((2,16,5),np.float32)
        for index,identifier in enumerate(identifiers):y[index,:len(labels[identifier])]=labels[identifier]
        rate=learning_rate(step)
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.zero_grad(set_to_none=True)
        values=model(torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda());loss=values['total_loss']
        require(bool(torch.isfinite(loss)),'Nonfinite training loss');loss.backward()
        require(all(p.grad is not None for p in model.parameters()),'Missing trained parameter gradient')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),POLICY['clip_grad_norm'],error_if_nonfinite=True)
        optimizer.step()
        row={'update':step+1,'case_ids':identifiers,'learning_rate':rate,'gradient_norm':float(norm),
             'losses':{k:float(v.detach()) if hasattr(v,'detach') else float(v) for k,v in values.items()}}
        history.append(row)
        if (step+1)%30==0:
            (output/f'progress-{step+1:03d}.json').write_bytes(encoded(row))
            print(config['arm'],'update',step+1,row['losses']['total_loss'],flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    final=state_digest(model.state_dict());require(final!=initial,'Model did not train')
    torch.save({'model':model.state_dict()},output/'final.pt')
    report={'schema_version':'yolox-training-run/1.0','config':config_pin,'arm':config['arm'],'repeat':config['repeat'],
        'initial':initial,'final':final,'transfer':transfer,'updates':len(history),'history':history,'geometries':geometries,
        'checkpoint':pin(root,(output/'final.pt').relative_to(root).as_posix()),'training_seconds':seconds,
        'allocated_peak_MiB':torch.cuda.max_memory_allocated()/1024**2,'reserved_peak_MiB':torch.cuda.max_memory_reserved()/1024**2,
        'device':torch.cuda.get_device_name(0),'validation_training_calls':0,'inference_calls':0,
        'checkpoint_selection':POLICY['checkpoint_selection'],'production_admitted':False}
    (output/'report.json').write_bytes(encoded(report));return report
