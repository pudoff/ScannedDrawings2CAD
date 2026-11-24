"""Train-only fold consumer; metadata and diagnostic validation are not features."""
import json
from pathlib import Path

import numpy as np

from .dataset_layout_v1 import checked, read
from .role_interval_dataset_v1 import FEATURES
from .role_interval_folds_v1 import folds


def load(workspace, manifest_ref, *, fold_id, purpose):
    if purpose not in ('fit', 'model_selection'):
        raise ValueError('Only original-train fit or held-family model selection is allowed')
    root = Path(workspace).resolve(); path = checked(root, manifest_ref); manifest = read(path)
    if manifest['schema_version'] != 'role-interval-family-fold-manifest/1.0':
        raise ValueError('Unknown role fold manifest')
    folder = path.parent; refs = {ref['path']: ref for ref in manifest['files']}
    report = read(checked(folder, refs['report.json']))
    protocol = read(checked(folder, refs['candidate-protocol.json']))
    if not report['prototype_ready']:
        raise ValueError('Role prototype is not ready')
    X = np.load(checked(folder, refs['train-features.npy']), allow_pickle=False)
    y = np.load(checked(folder, refs['train-targets.npy']), allow_pickle=False)
    rows = [json.loads(line) for line in checked(folder, refs['train-metadata.jsonl']).read_text('utf-8').splitlines()]
    if X.shape != (len(rows), len(FEATURES)) or len(y) != len(rows) or not np.isfinite(X).all():
        raise ValueError('Role fold arrays are misaligned')
    generated, coverage, supported = folds(rows, y, protocol['candidate']['coverage_gate'])
    if generated != read(checked(folder, refs['folds.json'])) or supported != report['supported_roles'] or coverage != report['coverage']:
        raise ValueError('Stored folds differ from original-train family membership')
    matches = [row for row in generated if row['id'] == fold_id]
    if len(matches) != 1:
        raise ValueError('Unknown fold ID')
    fold = matches[0]
    indices = fold['fit_indices'] if purpose == 'fit' else fold['selection_indices']
    emission = [role for role in supported if fold['roles'][str(role)]['fit_support_sufficient']]
    return dict(features=X[indices], targets=y[indices],
        metadata=[rows[index] for index in indices], feature_names=list(FEATURES),
        observed_fit_classes=sorted({int(y[index]) for index in fold['fit_indices']}),
        emission_roles=emission, fold_id=fold_id, purpose=purpose,
        all_reviewed_held_classes_kept=True, diagnostic_validation_used=False,
        independent_acceptance=False)
