"""Explicit BatchNorm mode and exact state checks; Paddle is passed by caller."""
import numpy as np


def batchnorm_layers(model,paddle):
    types=tuple(getattr(paddle.nn,name) for name in ('BatchNorm','BatchNorm1D','BatchNorm2D','BatchNorm3D','SyncBatchNorm') if hasattr(paddle.nn,name))
    return {name:layer for name,layer in model.named_sublayers() if isinstance(layer,types)}


def batchnorm_buffer_keys(model,paddle):
    keys=[]
    for name,layer in batchnorm_layers(model,paddle).items():
        state=layer.state_dict()
        if not {'_mean','_variance'}.issubset(state):raise ValueError('Unexpected BatchNorm state')
        keys.extend(name+'.'+key for key in ('_mean','_variance'))
    if not keys or not set(keys).issubset(model.state_dict()):raise ValueError('Missing BatchNorm buffers')
    return sorted(keys)


def train_with_frozen_batchnorm(model,paddle):
    before={name:p.stop_gradient for name,p in model.named_parameters()}
    model.train();layers=batchnorm_layers(model,paddle)
    if not layers:raise ValueError('Model has no BatchNorm')
    for layer in layers.values():layer.eval()
    if before!={name:p.stop_gradient for name,p in model.named_parameters()}:
        raise ValueError('BatchNorm mode changed trainable parameter flags')
    return sorted(layers)


def snapshot(state):return {key:value.numpy().copy() for key,value in state.items()}


def state_changes(before,after,buffers):
    if set(before)!=set(after) or not set(buffers).issubset(before):raise ValueError('Changed state keys')
    changed=[]
    for key,a in before.items():
        b=after[key]
        if a.shape!=b.shape or a.dtype!=b.dtype or not np.isfinite(b).all():raise ValueError('Invalid model state')
        if not np.array_equal(a,b):changed.append(key)
    bn=set(buffers)
    return {'batchnorm_changed':sorted(k for k in changed if k in bn),'other_changed':sorted(k for k in changed if k not in bn)}
