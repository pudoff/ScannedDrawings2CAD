"""All-family train-only cross-validation, with per-role readiness and abstention."""
from collections import Counter

import numpy as np


def folds(metadata, targets, gate):
    if len(metadata) != len(targets) or any(row['declared_split'] != 'train' for row in metadata):
        raise ValueError('Only original train observations may define selection folds')
    y = np.asarray(targets)
    if y.dtype != np.uint8 or not np.isin(y, [*range(1, 10), 255]).all():
        raise ValueError('Reviewed roles or explicit ignore required')
    families = sorted({row['canonical_family'] for row in metadata})
    if len(families) < 3:
        raise ValueError('At least three disjoint conservative train families required')
    group = np.array([row['canonical_family'] for row in metadata])
    results = []; valid = {role: [] for role in range(1, 10)}
    for held in families:
        train = np.flatnonzero((group != held) & (y != 255))
        validation = np.flatnonzero((group == held) & (y != 255))
        unknown = np.flatnonzero((group == held) & (y == 255))
        counts = {}
        for role in range(1, 10):
            selected = train[y[train] == role]
            fit_families = sorted(set(group[selected]))
            count = int(np.count_nonzero(y[validation] == role))
            allowed = len(selected) >= gate['minimum_fit_intervals'] and len(fit_families) >= gate['minimum_fit_families']
            counts[str(role)] = dict(fit_intervals=len(selected), fit_families=fit_families,
                held_intervals=count, fit_support_sufficient=allowed)
            if allowed and count:
                valid[role].append(dict(held_family=held, held_intervals=count))
        results.append(dict(id='held-'+held, held_family=held, fit_indices=train.tolist(),
            selection_indices=validation.tolist(), unlabelled_selection_indices=unknown.tolist(), roles=counts))
    coverage = {}; supported = []
    for role in range(1, 10):
        count = sum(row['held_intervals'] for row in valid[role])
        reasons = []
        if len(valid[role]) < gate['minimum_selection_families']: reasons.append('too_few_evaluable_held_families')
        if count < gate['minimum_internal_validation_intervals']: reasons.append('too_few_evaluable_held_intervals')
        coverage[str(role)] = dict(all_train_intervals=int(np.count_nonzero(y == role)),
            all_train_families=sorted(set(group[y == role])), evaluable_held_families=valid[role],
            evaluable_held_intervals=count, publication_ready=not reasons, reasons=reasons)
        if not reasons: supported.append(role)
    actual = Counter(index for fold in results for index in fold['selection_indices'])
    if set(actual) != set(np.flatnonzero(y != 255)) or any(count != 1 for count in actual.values()):
        raise ValueError('Every supervised observation must be held exactly once')
    return results, coverage, supported
