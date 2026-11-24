"""GPU runtime without importing the evaluator or its CAD/data-review dependencies.

The frozen preparation module cannot be refactored. These small geometry/LR
operations are checked for exact equivalence to it before customer execution.
"""
import math
from pathlib import Path
import cv2
import numpy as np
from .dataset_layout_v1 import checked,read
from .family_catalog_v2 import require

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL={'path':'configs/reviews/yolox-experiment-v1.json',
          'sha256':'67e98548fafeb76c50d29046f2b4a7f2761341e28413903d5fb3a7bec5262c6f'}
_SPEC=read(checked(ROOT,PROTOCOL));POLICY=_SPEC['policy'];BUDGETS=_SPEC['budgets']

def geometry(size,side=640):
    require(len(size)==2 and all(type(v) is int and v>0 for v in size),'Invalid image grid')
    w,h=size;scale=min(side/w,side/h);rw=max(1,int(w*scale));rh=max(1,int(h*scale));sx,sy=rw/w,rh/h
    return {'size_px':size,'side':side,'resized_size_px':[rw,rh],'pad_left_top':[0,0],
        'edges_to_model':[[sx,0,0],[0,sy,0],[0,0,1]],'model_to_edges':[[1/sx,0,0],[0,1/sy,0],[0,0,1]],
        'centres_to_model':[[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1]]}

def preprocess(rgb,arm):
    require(arm in ('rgb','gray'),'Unknown image arm')
    require(isinstance(rgb,np.ndarray) and rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3
            and 0<rgb.shape[0]*rgb.shape[1]<=BUDGETS['maximum_input_page_pixels'],'Bounded RGB overview required')
    spec=geometry(list(rgb.shape[:2][::-1]));rw,rh=spec['resized_size_px']
    image=rgb if arm=='rgb' else cv2.cvtColor(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
    canvas=np.full((640,640,3),255,dtype=np.uint8);canvas[:rh,:rw]=cv2.resize(image,(rw,rh),interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1),dtype=np.float32),spec

def learning_rate(update):
    require(type(update) is int and 0<=update<POLICY['updates'],'Invalid optimizer update')
    if update<POLICY['warmup_updates']:return POLICY['learning_rate']*(update+1)/POLICY['warmup_updates']
    progress=(update-POLICY['warmup_updates'])/(POLICY['updates']-POLICY['warmup_updates']-1)
    return POLICY['final_learning_rate']+(.5+.5*math.cos(math.pi*progress))*(POLICY['learning_rate']-POLICY['final_learning_rate'])

def schedule(train_ids):
    require(len(train_ids)==6 and len(set(train_ids))==6,'Exactly six admitted train pages required')
    return [[train_ids[(2*i+j)%6] for j in range(2)] for i in range(POLICY['updates'])]
