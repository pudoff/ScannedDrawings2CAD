"""Local upstream YOLOX-Tiny with a new five-class head; no implicit downloads."""
import sys
from pathlib import Path

from .dataset_layout_v1 import checked, read
from .family_catalog_v2 import require

ASSETS={'path':'artifacts/yolox-assets-v1/assets.json',
        'sha256':'f1692ca6f1f932531471f5ed67732c5615bb56aa36f8038f78162ac83e061a1f'}
DEPENDENCIES='artifacts/environments/yolox-v1'
UPSTREAM='artifacts/yolox-assets-v1/upstream'
SEED=20261112

def build(root,*,seed=SEED):
    assets=read(checked(root,ASSETS))
    for p in assets['files'].values(): checked(root,p)
    sys.path[:0]=[str(root/DEPENDENCIES),str(root/UPSTREAM)]
    import torch
    from yolox.models import YOLOX,YOLOPAFPN,YOLOXHead
    torch.manual_seed(seed)
    model=YOLOX(YOLOPAFPN(depth=.33,width=.375),YOLOXHead(5,width=.375))
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm2d): module.eps=.001; module.momentum=.03
    model.head.initialize_biases(.01)
    checkpoint=torch.load(checked(root,assets['files']['yolox_tiny.pth']),map_location='cpu',weights_only=True)
    # Load feature, box and objectness weights. Three COCO class projections are replaced.
    dropped={f'head.cls_preds.{i}.{name}' for i in range(3) for name in ('weight','bias')}
    saved=checkpoint['model']; target=model.state_dict()
    require(set(saved)==set(target),'Upstream model inventory changed')
    for name,value in saved.items():
        if name not in dropped: require(value.shape==target[name].shape,'Unexpected pretrained shape')
        else: require(value.shape[0]==80 and target[name].shape[0]==5,'Wrong class-head transfer')
    state={k:v for k,v in saved.items() if k not in dropped}
    result=model.load_state_dict(state,strict=False)
    require(set(result.missing_keys)==dropped and not result.unexpected_keys,'Partial pretrained transfer changed')
    return model,{'parameters':sum(p.numel() for p in model.parameters()),
        'transferred_tensors':len(state),'new_class_tensors':sorted(dropped),
        'pretrained_optimizer_loaded':False,'pretraining_classes':80,'classes':5,'seed':seed}
