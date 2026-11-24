"""Purpose-bound consumers for the prepared experimental role dataset."""
from pathlib import Path

import numpy as np

from .dataset_layout_v1 import checked, read
from .role_interval_dataset_v1 import FEATURES


def load(workspace, manifest_ref, *, usage, purpose):
    allowed = {'fit': 'fit', 'internal_validation': 'model_selection',
               'diagnostic_validation': 'diagnostic_only'}
    if usage not in allowed or purpose != allowed[usage]:
        raise ValueError('Dataset usage does not permit this purpose')
    root = Path(workspace).resolve()
    manifest_path = checked(root, manifest_ref); manifest = read(manifest_path)
    if manifest['schema_version'] != 'role-interval-preparation-manifest/1.0':
        raise ValueError('Unknown role preparation manifest')
    folder = manifest_path.parent
    refs = {ref['path']: ref for ref in manifest['files']}
    report = read(checked(folder, refs['report.json']))
    if purpose != 'diagnostic_only' and not report['candidate']['prototype_fit_ready']:
        raise ValueError('Prepared roles do not meet the frozen prototype coverage gate')
    dataset = report['datasets'][usage]
    arrays = {name: np.load(checked(root, dataset[name]), allow_pickle=False) for name in ('features', 'targets')}
    X, y = arrays['features'], arrays['targets']
    if (X.ndim != 2 or X.shape != (len(y), len(FEATURES)) or not np.isfinite(X).all()
            or y.dtype != np.uint8 or not np.isin(y, [*range(1, 10), 255]).all()):
        raise ValueError('Invalid prepared role arrays')
    import json
    metadata = [json.loads(line) for line in checked(root, dataset['metadata']).read_text('utf-8').splitlines()]
    if len(metadata) != len(y) or any(row['usage'] != usage for row in metadata):
        raise ValueError('Metadata differs from the requested role usage')
    if usage in ('fit', 'internal_validation') and any(row['declared_split'] != 'train' for row in metadata):
        raise ValueError('Only original train sources may fit or select a model')
    selected = (y != 255) if purpose == 'diagnostic_only' else np.isin(y, report['candidate']['supported_roles'])
    return dict(features=X[selected], targets=y[selected],
        metadata=[row for row, keep in zip(metadata, selected) if keep], usage=usage, purpose=purpose,
        feature_names=list(FEATURES), independent_acceptance=False, dense_ground_truth=False)
