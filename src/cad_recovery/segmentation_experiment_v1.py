"""Fixed partial-supervision experiment: tiling, masked loss and bounded metrics."""
import numpy as np
from PIL import Image

from .family_catalog_v2 import require

POLICY = {'version':'partial-segmentation-experiment/1.0', 'seed':20261110,
    'tile_side':256, 'inference_stride':192, 'batch':4, 'updates':120,
    'classes':10, 'ignore_index':255, 'minimum_tile_foreground_pixels':24,
    'optimizer':'AdamW', 'learning_rate':0.0003, 'weight_decay':0.01,
    'class_weights':'inverse square-root train pixel frequency, mean-normalized, max=10',
    'primary':'pretrained-gray', 'variants':['pretrained-gray','random-gray','pretrained-rgb','pretrained-gray-repeat'],
    'checkpoint_step':116, 'checkpoint_selection':'fixed final update, no validation-based selection',
    'AMP':False, 'TF32':False, 'deterministic_algorithms':True,
    'max_child_seconds':900, 'max_host_MiB':6144, 'max_allocated_MiB':4096,
    'max_reserved_MiB':6144, 'max_device_MiB':8192,
    'independent_evaluation':False, 'production_admitted':False,
    'physical_geometry_approved':False, 'test_used':False}


def starts(length, side=256, stride=192):
    require(type(length) is int and length > 0 and 0 < stride <= side, 'Invalid tile extent')
    return sorted(set(range(0,max(1,length-side+1),stride)) | {max(0,length-side)})


def boxes(width,height,*,side=256,stride=192):
    return [[x,y,min(x+side,width),min(y+side,height)] for y in starts(height,side,stride) for x in starts(width,side,stride)]


def preprocess(rgb, mode):
    require(mode in ('gray','rgb'), 'Unknown experiment input mode')
    image=Image.fromarray(rgb.astype(np.uint8))
    if mode=='gray': image=image.convert('L').convert('RGB')
    array=np.asarray(image,dtype=np.float32)/255
    return ((array-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1).copy()


def padded_tile(rgb, labels, box, side=256):
    x,y,right,bottom=box
    require(0 <= x < right <= rgb.shape[1] and 0 <= y < bottom <= rgb.shape[0]
            and right-x <= side and bottom-y <= side and labels.shape == rgb.shape[:2], 'Invalid tile box')
    image=np.full((side,side,3),255,np.uint8)
    target=np.full((side,side),255,np.uint8)
    image[:bottom-y,:right-x]=rgb[y:bottom,x:right]
    target[:bottom-y,:right-x]=labels[y:bottom,x:right]
    return image,target


def class_weights(counts):
    counts=np.asarray(counts,dtype=float)
    require(counts.shape==(10,) and np.isfinite(counts).all() and (counts>0).all(), 'Each class needs train support')
    result=1/np.sqrt(counts)
    return np.minimum(result/result.mean(),10).astype(np.float32)


def masked_loss(logits, targets, weights):
    """Weighted CE with ignore, deterministic CUDA path; no CPU/NumPy conversion."""
    import torch
    from torch.nn import functional as F
    valid=targets!=255
    if not bool(valid.any()): raise ValueError('Batch has no supervised pixels')
    if not bool(((targets[valid]>=0)&(targets[valid]<logits.shape[1])).all()): raise ValueError('Target outside taxonomy')
    safe=targets.masked_fill(~valid,0)
    one_hot=F.one_hot(safe,num_classes=logits.shape[1]).movedim(-1,1).to(logits.dtype)
    losses=-(torch.log_softmax(logits,dim=1)*one_hot).sum(dim=1)
    pixel_weights=weights[safe]*valid
    return (losses*pixel_weights).sum()/pixel_weights.sum()


def confusion(target,prediction,classes=10):
    target=np.asarray(target);prediction=np.asarray(prediction)
    require(target.shape==prediction.shape, 'Prediction/target shape mismatch')
    require(set(np.unique(target)) <= set(range(classes))|{255}
            and set(np.unique(prediction)) <= set(range(classes)), 'Class outside taxonomy')
    valid=target!=255
    return np.bincount((target[valid].astype(np.int64)*classes+prediction[valid]),minlength=classes*classes).reshape(classes,classes)


def metrics(matrix):
    matrix=np.asarray(matrix,dtype=np.int64)
    require(matrix.shape==(10,10) and (matrix>=0).all(), 'Invalid confusion matrix')
    rows=[]
    for c in range(10):
        tp=int(matrix[c,c]);support=int(matrix[c].sum());pred=int(matrix[:,c].sum());union=support+pred-tp
        rows.append({'class_id':c,'target_pixels':support,'predicted_pixels_within_review':pred,
            'precision_within_review':tp/pred if pred else None, 'recall':tp/support if support else None,
            'IoU_within_review':tp/union if support and union else None})
    fg=[r['IoU_within_review'] for r in rows[1:] if r['IoU_within_review'] is not None]
    return {'confusion':matrix.tolist(),'classes':rows,'reviewed_pixels':int(matrix.sum()),
        'foreground_macro_IoU_within_review':float(np.mean(fg)) if fg else None,
        'pixel_accuracy_within_review':float(np.trace(matrix)/matrix.sum()) if matrix.sum() else None,
        'full_page_precision':None, 'independent_evaluation':False}


def heuristic(rgb):
    """Explicit weak floor, not a substitute for the unclassified production CV."""
    gray=np.asarray(Image.fromarray(rgb.astype(np.uint8)).convert('L'))
    return np.where(gray<240,1,0).astype(np.uint8)
