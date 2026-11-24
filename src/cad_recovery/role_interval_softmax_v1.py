"""Fixed CPU multinomial logistic baseline for partial source-interval roles.

Training sees fit arrays and conservative family IDs for weights only. There
is no held/diagnostic argument, checkpoint selection or hyperparameter search.
"""
from collections import Counter

import numpy as np

from .role_interval_dataset_v1 import FEATURES


def _features(X):
    X = np.asarray(X, np.float64)
    if X.ndim != 2 or X.shape[1] != len(FEATURES) or not np.isfinite(X).all():
        raise ValueError('Finite aligned source-only features required')
    return X


def sampling_weights(y, families):
    """Inverse class frequency capped at 10, normalized equally per family."""
    y = np.asarray(y); groups = np.asarray(families, dtype=str)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or not len(y) or not np.isin(y, range(1, 10)).all():
        raise ValueError('Reviewed fit roles and aligned family IDs required')
    classes, counts = np.unique(y, return_counts=True)
    class_weights = {int(role): min(10., len(y)/(len(classes)*int(count))) for role, count in zip(classes, counts)}
    weights = np.asarray([class_weights[int(role)] for role in y], np.float64)
    unique_groups = sorted(set(groups))
    if any(not group for group in unique_groups):
        raise ValueError('Nonempty family IDs required')
    for group in unique_groups:
        selected = groups == group
        weights[selected] /= weights[selected].sum()*len(unique_groups)
    return weights, dict(class_weights={str(role): value for role, value in class_weights.items()},
        family_weight_sums={group: float(weights[groups == group].sum()) for group in unique_groups},
        policy='class inverse frequency cap 10, then equal total weight per fit family')


def loss_gradient(Z, targets, weights, coefficients, intercept, regularization):
    logits = Z@coefficients+intercept
    maximum = logits.max(axis=1, keepdims=True)
    shifted = logits-maximum
    exponential = np.exp(shifted); sums = exponential.sum(axis=1, keepdims=True)
    probabilities = exponential/sums
    log_probability = shifted[np.arange(len(Z)), targets]-np.log(sums[:, 0])
    loss = float(-weights@log_probability+.5*regularization*np.sum(coefficients**2))
    delta = probabilities.copy(); delta[np.arange(len(Z)), targets] -= 1
    delta *= weights[:, None]
    return loss, Z.T@delta+regularization*coefficients, delta.sum(axis=0)


def fit(X, y, families, solver):
    X = _features(X); y = np.asarray(y)
    if len(X) != len(y) or y.ndim != 1 or not len(y) or not np.isin(y, range(1, 10)).all():
        raise ValueError('Only reviewed fit labels 1..9 are permitted')
    if (type(solver['epochs']) is not int or solver['epochs'] <= 0
            or not np.isfinite([solver['learning_rate'], solver['L2']]).all()
            or solver['learning_rate'] <= 0 or solver['L2'] < 0 or not solver['full_batch']):
        raise ValueError('Fixed finite full-batch solver required')
    classes = np.unique(y).astype(np.uint8)
    if len(classes) < 2:
        raise ValueError('At least two observed fit classes required')
    mean = X.mean(axis=0); scale = X.std(axis=0); scale[scale == 0] = 1.
    Z = (X-mean)/scale
    targets = np.searchsorted(classes, y)
    weights, weighting = sampling_weights(y, families)
    coefficients = np.zeros((len(FEATURES), len(classes)), np.float64)
    intercept = np.zeros(len(classes), np.float64)
    history = []
    for epoch in range(solver['epochs']):
        loss, grad, bias_grad = loss_gradient(Z, targets, weights, coefficients, intercept, solver['L2'])
        if not np.isfinite(loss) or not np.isfinite(grad).all() or not np.isfinite(bias_grad).all():
            raise ValueError('Non-finite logistic optimization')
        history.append(dict(epoch=epoch, fit_loss=loss))
        coefficients -= solver['learning_rate']*grad
        intercept -= solver['learning_rate']*bias_grad
    final_loss = loss_gradient(Z, targets, weights, coefficients, intercept, solver['L2'])[0]
    history.append(dict(epoch=solver['epochs'], fit_loss=final_loss))
    model = dict(mean=mean, scale=scale, classes=classes, coefficients=coefficients, intercept=intercept)
    audit = dict(observations=len(y), fit_class_counts={str(int(k)): int(v) for k, v in Counter(y).items()},
        weighting=weighting, epochs=solver['epochs'], initialization='deterministic zero coefficients and intercept',
        seed=solver['seed'], random_sampling=False, normalization='unweighted population mean/std of fit rows only',
        held_data_seen=False, diagnostic_data_seen=False, training_history=history)
    return model, audit


def probabilities(model, X):
    X = _features(X)
    classes = np.asarray(model['classes'])
    if (classes.ndim != 1 or len(classes) < 2
            or model['mean'].shape != (len(FEATURES),) or model['scale'].shape != (len(FEATURES),)
            or model['coefficients'].shape != (len(FEATURES), len(classes))
            or model['intercept'].shape != (len(classes),) or len(set(classes)) != len(classes)
            or not np.isin(classes, range(1, 10)).all() or np.any(model['scale'] <= 0)
            or any(not np.isfinite(value).all() for value in model.values())):
        raise ValueError('Invalid logistic checkpoint')
    result = np.zeros((len(X), 9), np.float64)
    if len(X):
        logits = ((X-model['mean'])/model['scale'])@model['coefficients']+model['intercept']
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
        result[:, classes.astype(int)-1] = p
    return result


def proposals(p, emission_roles, gate):
    p = np.asarray(p, np.float64)
    if (p.ndim != 2 or p.shape[1] != 9 or not np.isfinite(p).all()
            or (p < 0).any() or not np.allclose(p.sum(axis=1), 1., rtol=0, atol=1e-10)):
        raise ValueError('Valid nine-role score vectors required')
    if not set(emission_roles) <= set(range(1, 10)):
        raise ValueError('Invalid emission role')
    order = np.argsort(-p, axis=1, kind='stable')
    first, second = order[:, 0], order[:, 1]
    top = p[np.arange(len(p)), first]; margin = top-p[np.arange(len(p)), second]
    raw = (first+1).astype(np.uint8)
    emit = (np.isin(raw, emission_roles) & (top >= gate['minimum_probability'])
            & (margin >= gate['minimum_top_two_margin']))
    return dict(prediction=np.where(emit, raw, 255).astype(np.uint8), raw_argmax=raw,
        top_score=top, top_two_margin=margin, calibration_claimed=False)


def metrics(truth, prediction, supported_roles):
    truth = np.asarray(truth); prediction = np.asarray(prediction)
    if (truth.ndim != 1 or prediction.shape != truth.shape or not len(truth)
            or not np.isin(truth, range(1, 10)).all() or not np.isin(prediction, [*range(1, 10), 255]).all()):
        raise ValueError('All reviewed held roles must remain in the metric denominator')
    column = np.where(prediction == 255, 9, prediction.astype(int)-1)
    matrix = np.bincount((truth.astype(int)-1)*10+column, minlength=90).reshape(9, 10)
    rows = []
    for index in range(9):
        support = int(matrix[index].sum()); tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()-tp); fn = support-tp
        rows.append(dict(role=index+1, support=support, tp=tp, fp=fp, fn=fn,
            precision=tp/(tp+fp) if tp+fp else None, recall=tp/support if support else None,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
    emitted = int(np.count_nonzero(prediction != 255)); correct = int(np.trace(matrix[:, :9]))
    full = [row['f1'] for row in rows if row['f1'] is not None]
    selected = [row['f1'] for row in rows if row['role'] in supported_roles and row['f1'] is not None]
    supported_targets = np.isin(truth, supported_roles)
    return dict(confusion=matrix.tolist(), role_rows=rows, labelled_intervals=len(truth),
        correct=correct, emitted=emitted, abstained=len(truth)-emitted,
        coverage=emitted/len(truth), selective_accuracy=correct/emitted if emitted else None,
        accuracy_with_abstentions=correct/len(truth), macro_f1=float(np.mean(full)) if full else None,
        supported_roles_macro_f1=float(np.mean(selected)) if selected else None,
        supported_target_coverage=float(np.mean(prediction[supported_targets] != 255)) if supported_targets.any() else None,
        emissions_on_unsupported_targets=int(np.count_nonzero(~supported_targets & (prediction != 255))),
        unsupported_targets_kept=True, independent_samples_claimed=False)


def save_model(path, model):
    with path.open('xb') as stream:
        np.savez(stream, **model)


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'mean', 'scale', 'classes', 'coefficients', 'intercept'}:
            raise ValueError('Unexpected checkpoint fields')
        model = {name: data[name].copy() for name in data.files}
    probabilities(model, np.zeros((0, len(FEATURES))))
    return model
