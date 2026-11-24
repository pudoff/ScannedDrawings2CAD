"""CPU typed message passing; forward inputs never include semantic labels."""
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

POLICY = dict(input_features=21, hidden_units=64, steps=2, edge_types=8, roles=9, relations=3,
              pair_features=16, activation='tanh', typed_messages='row mean with learned diagonal channel gates',
              self_messages='learned dense matrix', pair_head='sum(h_i,h_j), abs(h_i-h_j), source geometry',
              epochs=400, learning_rate=.003, L2=.001, seed=0, edge_loss_weight=.25,
              optimizer='Adam', beta1=.9, beta2=.999, epsilon=1e-8, gradient_norm_cap=10.,
              standardized_clip=8., score=.90, margin=.20, minimum_macro_f1=.90,
              checkpoint='last fixed epoch, no held selection', dtype='float32', backend='numpy/scipy CPU, one thread')


def balanced_weights(targets, units, groups):
    """One base vote per source unit, then fit-only family and class balance."""
    y = np.asarray(targets, int)
    units, groups = np.asarray(units, str), np.asarray(groups, str)
    if y.ndim != 1 or not len(y) or units.shape != y.shape or groups.shape != y.shape:
        raise ValueError('Nonempty aligned FIT supervision required')
    if (y < 0).any() or not all(units) or not all(groups):
        raise ValueError('Unknown targets/units are not loss observations')
    count = Counter(units)
    owners = {}
    for unit, group in zip(units, groups, strict=True):
        if unit in owners and owners[unit] != group:
            raise ValueError('One source unit cannot cross conservative groups')
        owners[unit] = group
    base = np.array([1 / count[u] for u in units], np.float64)
    weights = base.copy()
    for group in sorted(set(groups)):
        mask = groups == group
        weights[mask] /= weights[mask].sum()
    for role in sorted(set(y)):
        mask = y == role
        weights[mask] /= weights[mask].sum()
    weights /= weights.sum()
    return base, weights


def normalizer(x, base_weights):
    x, w = np.asarray(x, np.float64), np.asarray(base_weights, np.float64)
    if x.ndim != 2 or len(x) != len(w) or not len(w) or not np.isfinite(x).all() or not np.all(w > 0):
        raise ValueError('Finite FIT rows with positive source-unit weights required')
    w = w / w.sum()
    mean = w @ x
    scale = np.sqrt(w @ ((x-mean)**2))
    scale[scale < 1e-6] = 1.
    return dict(mean=mean, scale=scale)


def standardize(x, stats, dtype=np.float32):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite source input')
    return np.clip((x-stats['mean'])/stats['scale'], -POLICY['standardized_clip'], POLICY['standardized_clip']).astype(dtype)


def operators(n, edges, dtype=np.float32):
    e = np.asarray(edges)
    if e.ndim != 2 or e.shape[1] != 3 or not np.issubdtype(e.dtype, np.integer):
        raise ValueError('Typed integer source edges required')
    if len(e) and ((e[:,:2] < 0).any() or (e[:,:2] >= n).any() or (e[:,0] == e[:,1]).any()
                   or (e[:,2] < 0).any() or (e[:,2] >= POLICY['edge_types']).any()):
        raise ValueError('Invalid source edges')
    if len(np.unique(e, axis=0)) != len(e):
        raise ValueError('Duplicate typed edges bias message means')
    result = []
    for kind in range(POLICY['edge_types']):
        a,b = e[e[:,2] == kind,:2].T
        degree = np.bincount(b, minlength=n)
        weights = (1/np.maximum(1, degree[b])).astype(dtype)
        result.append(csr_matrix((weights,(b,a)), shape=(n,n), dtype=dtype))
    return result


def initialize(seed=0, hidden=64, dtype=np.float32):
    rng = np.random.default_rng(seed)
    def matrix(a,b):
        return rng.normal(0, np.sqrt(2/(a+b)), (a,b)).astype(dtype)
    p = dict(in_w=matrix(21,hidden), in_b=np.zeros(hidden,dtype),
             role_w=matrix(hidden,9), role_b=np.zeros(9,dtype),
             pair_w=matrix(2*hidden+16,3), pair_b=np.zeros(3,dtype))
    for step in range(2):
        p[f's{step}_w'] = matrix(hidden,hidden)
        p[f's{step}_g'] = rng.normal(0, .1, (8,hidden)).astype(dtype)
        p[f's{step}_b'] = np.zeros(hidden,dtype)
    return p


def encode(x, adjacency, params):
    """Pure source features/edges plus model parameters; no labels or groups."""
    if x.ndim != 2 or x.shape[1] != 21 or not np.isfinite(x).all() or len(adjacency) != 8:
        raise ValueError('Finite source features and eight typed operators required')
    h = np.tanh(x @ params['in_w'] + params['in_b'])
    states = [h]
    messages = []
    for step in range(2):
        m = np.stack([a @ h for a in adjacency])
        h = np.tanh(h @ params[f's{step}_w'] + np.sum(m * params[f's{step}_g'][:,None,:], axis=0)
                    + params[f's{step}_b'])
        messages.append(m)
        states.append(h)
    return h, (states,messages)


def pair_input(hidden, pairs, geometry):
    a,b = pairs.T
    return np.c_[hidden[a]+hidden[b], np.abs(hidden[a]-hidden[b]), geometry]


def logits(hidden, params, pairs, geometry):
    return hidden @ params['role_w'] + params['role_b'], pair_input(hidden,pairs,geometry) @ params['pair_w'] + params['pair_b']


def probabilities(scores):
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError('Finite multi-class logits required')
    shifted = scores-scores.max(axis=1,keepdims=True)
    exp = np.exp(shifted)
    return exp/exp.sum(axis=1,keepdims=True)


def cross_entropy(scores, targets, weights):
    if not len(targets):
        return 0., np.zeros_like(scores)
    shifted = scores-scores.max(axis=1,keepdims=True)
    p = probabilities(scores)
    loss = float(-np.asarray(weights,np.float64) @ (shifted[np.arange(len(targets)),targets]-np.log(np.exp(shifted).sum(axis=1))))
    p[np.arange(len(targets)),targets] -= 1
    p *= np.asarray(weights,dtype=p.dtype)[:,None]
    return loss,p


def loss_gradient(x, adjacency, params, node_indices, node_targets, node_weights,
                  pairs, geometry, pair_targets, pair_weights, regularization=.001, edge_weight=.25):
    """The training interface accepts FIT arrays only; held labels have no slot."""
    if (len(node_indices) != len(node_targets) or len(node_targets) != len(node_weights)
            or not np.isin(node_targets,range(9)).all() or not np.isin(pair_targets,range(3)).all()
            or len(pairs) != len(pair_targets) or len(pairs) != len(pair_weights)
            or geometry.shape != (len(pairs),16) or not np.isfinite(geometry).all()
            or (np.asarray(node_weights) <= 0).any() or (np.asarray(pair_weights) <= 0).any()):
        raise ValueError('Aligned known FIT targets, geometry and positive loss weights required')
    h, (states,messages) = encode(x,adjacency,params)
    role_scores, pair_scores = logits(h,params,pairs,geometry)
    node_loss, delta = cross_entropy(role_scores[node_indices],node_targets,node_weights)
    pair_loss, pair_delta = cross_entropy(pair_scores,pair_targets,pair_weights)
    pair_delta *= edge_weight
    grad = {k:np.zeros_like(v) for k,v in params.items()}
    grad['role_w'] = h[node_indices].T @ delta
    grad['role_b'] = delta.sum(axis=0)
    back = np.zeros_like(h)
    np.add.at(back,node_indices,delta @ params['role_w'].T)
    descriptor = pair_input(h,pairs,geometry)
    grad['pair_w'] = descriptor.T @ pair_delta
    grad['pair_b'] = pair_delta.sum(axis=0)
    pair_back = pair_delta @ params['pair_w'].T
    size = h.shape[1]
    a,b = pairs.T
    sign = np.sign(h[a]-h[b])
    np.add.at(back,a,pair_back[:,:size]+pair_back[:,size:2*size]*sign)
    np.add.at(back,b,pair_back[:,:size]-pair_back[:,size:2*size]*sign)
    for step in reversed(range(2)):
        d = back*(1-states[step+1]**2)
        grad[f's{step}_w'] = states[step].T @ d
        grad[f's{step}_b'] = d.sum(axis=0)
        grad[f's{step}_g'] = np.sum(messages[step]*d[None,:,:],axis=1)
        back = d @ params[f's{step}_w'].T
        for kind, a in enumerate(adjacency):
            back += a.T @ (d * params[f's{step}_g'][kind])
    d = back*(1-states[0]**2)
    grad['in_w'] = x.T @ d
    grad['in_b'] = d.sum(axis=0)
    penalty = 0.
    for key,value in params.items():
        if not key.endswith('_b'):
            penalty += .5*regularization*float(np.sum(value**2))
            grad[key] += regularization*value
    loss = node_loss+edge_weight*pair_loss+penalty
    if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in grad.values()):
        raise ValueError('Nonfinite graph optimization; no retry')
    return loss,grad


def adam_step(params, grad, moments, epoch, policy=POLICY):
    norm = np.sqrt(sum(float(np.sum(g.astype(np.float64)**2)) for g in grad.values()))
    factor = min(1.,policy['gradient_norm_cap']/max(norm,1e-12))
    for key in params:
        g = grad[key]*factor
        m,v = moments[key]
        m *= policy['beta1']; m += (1-policy['beta1'])*g
        v *= policy['beta2']; v += (1-policy['beta2'])*g*g
        params[key] -= policy['learning_rate']*(m/(1-policy['beta1']**epoch))/(np.sqrt(v/(1-policy['beta2']**epoch))+policy['epsilon'])


def fit(x, adjacency, node_indices, node_targets, node_weights, pairs, geometry, pair_targets, pair_weights, policy=POLICY):
    if policy != POLICY:
        raise ValueError('Use the frozen single solver policy')
    params = initialize(policy['seed'],policy['hidden_units'])
    moments = {k:(np.zeros_like(v),np.zeros_like(v)) for k,v in params.items()}
    history = []
    for epoch in range(1,policy['epochs']+1):
        loss,grad = loss_gradient(x,adjacency,params,node_indices,node_targets,node_weights,
                                 pairs,geometry,pair_targets,pair_weights,policy['L2'],policy['edge_loss_weight'])
        adam_step(params,grad,moments,epoch,policy)
        history.append(dict(epoch=epoch,fit_loss=loss))
    return params,history


def propose(scores, supported):
    p = probabilities(scores)
    winners = np.argmax(p,axis=1)
    order = np.sort(p,axis=1)
    accepted = (order[:,-1] >= POLICY['score']) & (order[:,-1]-order[:,-2] >= POLICY['margin']) & np.asarray(supported,bool)[winners]
    return np.where(accepted,winners,-1).astype(np.int16)


def metrics(targets, predictions, classes):
    y,p = np.asarray(targets,int),np.asarray(predictions,int)
    if y.shape != p.shape or not np.isin(y,range(classes)).all() or not np.isin(p,[-1,*range(classes)]).all():
        raise ValueError('Every primary target requires a prediction or explicit refusal')
    rows = []
    for k in range(classes):
        tp = int(np.count_nonzero((y==k)&(p==k)))
        fp = int(np.count_nonzero((y!=k)&(p==k)))
        fn = int(np.count_nonzero((y==k)&(p!=k)))
        precision = tp/(tp+fp) if tp+fp else 0.
        recall = tp/(tp+fn) if tp+fn else 0.
        rows.append(dict(class_id=k,targets=int(np.count_nonzero(y==k)),TP=tp,FP=fp,FN=fn,
                         precision=precision,recall=recall,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.))
    return dict(targets=len(y),refusals=int(np.count_nonzero(p==-1)),classes=rows,macro_f1=float(np.mean([r['f1'] for r in rows])))
