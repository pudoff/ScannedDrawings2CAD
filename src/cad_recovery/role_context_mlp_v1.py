"""Small deterministic tanh MLP; fit-only statistics, fixed full-batch Adam."""
from collections import Counter
import numpy as np
from .role_context_v1 import FEATURES
from .role_interval_softmax_v1 import sampling_weights

SOLVER=dict(hidden_units=64,epochs=400,learning_rate=.003,L2=.001,
    beta1=.9,beta2=.999,epsilon=1e-8,seed=0,standardized_clip=8.,full_batch=True)
FIELDS={'mean','scale','classes','W1','b1','W2','b2','clip'}


def features(x):
    x=np.asarray(x,np.float64)
    if x.ndim!=2 or x.shape[1]!=len(FEATURES) or not np.isfinite(x).all():
        raise ValueError('Finite context feature matrix required')
    return x


def loss_gradient(z,targets,weights,params,regularization):
    hidden=np.tanh(z@params['W1']+params['b1'])
    logits=hidden@params['W2']+params['b2'];logits-=logits.max(axis=1,keepdims=True)
    exp=np.exp(logits);sums=exp.sum(axis=1,keepdims=True);p=exp/sums
    loss=float(-weights@(logits[np.arange(len(z)),targets]-np.log(sums[:,0]))
        +.5*regularization*(np.sum(params['W1']**2)+np.sum(params['W2']**2)))
    delta=p.copy();delta[np.arange(len(z)),targets]-=1;delta*=weights[:,None]
    back=(delta@params['W2'].T)*(1-hidden**2)
    gradients=dict(W2=hidden.T@delta+regularization*params['W2'],b2=delta.sum(0),
        W1=z.T@back+regularization*params['W1'],b1=back.sum(0))
    return loss,gradients


def fit(x,y,families,solver):
    x=features(x);y=np.asarray(y)
    if y.shape!=(len(x),) or not len(y) or not np.isin(y,range(1,10)).all():
        raise ValueError('Only reviewed fit labels 1..9 permitted')
    if (set(solver)!=set(SOLVER) or not solver['full_batch'] or
            any(type(solver[k])is not int or solver[k]<=0 for k in ('epochs','hidden_units'))
            or any(not np.isfinite(solver[k]) or solver[k]<=0 for k in ('learning_rate','epsilon','standardized_clip'))
            or not np.isfinite(solver['L2']) or solver['L2']<0
            or not 0<=solver['beta1']<1 or not 0<=solver['beta2']<1):
        raise ValueError('Fixed finite MLP solver required')
    classes=np.unique(y).astype(np.uint8)
    if len(classes)<2:raise ValueError('At least two observed classes required')
    mean=x.mean(0);scale=x.std(0);scale[scale==0]=1
    z=np.clip((x-mean)/scale,-solver['standardized_clip'],solver['standardized_clip'])
    target=np.searchsorted(classes,y);weights,weighting=sampling_weights(y,families)
    rng=np.random.Generator(np.random.PCG64(solver['seed']));h=solver['hidden_units']
    params=dict(W1=rng.normal(0,1/np.sqrt(x.shape[1]),(x.shape[1],h)),b1=np.zeros(h),
        W2=rng.normal(0,1/np.sqrt(h),(h,len(classes))),b2=np.zeros(len(classes)))
    first={k:np.zeros_like(v) for k,v in params.items()};second={k:v.copy() for k,v in first.items()}
    history=[]
    for step in range(1,solver['epochs']+1):
        loss,grad=loss_gradient(z,target,weights,params,solver['L2'])
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grad.values()):
            raise ValueError('Nonfinite MLP optimization')
        history.append(dict(epoch=step-1,fit_loss=loss))
        for k in params:
            first[k]=solver['beta1']*first[k]+(1-solver['beta1'])*grad[k]
            second[k]=solver['beta2']*second[k]+(1-solver['beta2'])*grad[k]**2
            params[k]-=solver['learning_rate']*(first[k]/(1-solver['beta1']**step))/(
                np.sqrt(second[k]/(1-solver['beta2']**step))+solver['epsilon'])
    history.append(dict(epoch=solver['epochs'],fit_loss=loss_gradient(z,target,weights,params,solver['L2'])[0]))
    model=dict(**params,mean=mean,scale=scale,classes=classes,clip=np.array(solver['standardized_clip']))
    audit=dict(observations=len(y),class_counts={str(int(k)):int(v) for k,v in Counter(y).items()},
        weighting=weighting,epochs=solver['epochs'],seed=solver['seed'],
        normalization='fit-only population mean/std; constants scale=1; standardized clipping fixed before fit',
        held_data_seen=False,diagnostic_data_seen=False,random_sampling=False,history=history)
    return model,audit


def probabilities(model,x):
    x=features(x)
    if set(model)!=FIELDS:raise ValueError('Unexpected context checkpoint fields')
    k=len(model['classes']);h=len(model['b1']);d=len(FEATURES)
    if (k<2 or model['classes'].shape!=(k,) or not np.isin(model['classes'],range(1,10)).all()
            or len(set(model['classes']))!=k or np.any(np.diff(model['classes'].astype(int))<=0)
            or model['mean'].shape!=(d,) or model['scale'].shape!=(d,) or np.any(model['scale']<=0)
            or model['W1'].shape!=(d,h) or not h or model['W2'].shape!=(h,k)
            or model['b2'].shape!=(k,) or model['clip'].shape!=() or model['clip']<=0
            or any(not np.isfinite(v).all() for v in model.values())):
        raise ValueError('Invalid context checkpoint')
    result=np.zeros((len(x),9),np.float64)
    if len(x):
        z=np.clip((x-model['mean'])/model['scale'],-model['clip'],model['clip'])
        logits=np.tanh(z@model['W1']+model['b1'])@model['W2']+model['b2']
        logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
        result[:,model['classes'].astype(int)-1]=p
    return result


def save_model(path,model):
    probabilities(model,np.zeros((0,len(FEATURES))))
    with path.open('xb') as stream:np.savez(stream,**model)


def load_model(path):
    with np.load(path,allow_pickle=False) as data:model={name:data[name].copy() for name in data.files}
    probabilities(model,np.zeros((0,len(FEATURES))))
    return model
