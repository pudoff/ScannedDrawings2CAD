"""Fixed reading order for explicitly upright, single physical line crops."""
from copy import deepcopy
import numpy as np

from .engineering_symbols_v1 import ARMS
from .family_catalog_v2 import require

RULE = {'id': 'given-single-line-left-to-right/2.0',
    'scope': 'Only the diagnostic supplied upright single-line crops; never a full page or arbitrary OCR regions.',
    'order': 'centroid x ascending, then centroid y; stable ties', 'join': 'one ASCII space',
    'raw_region_text_changed': False, 'OCR_reexecuted': False, 'target_text_used': False}


def assemble(predictions):
    require(set(predictions) == set(ARMS), 'Missing OCR arm')
    output = deepcopy(predictions); changes = []
    for arm in ('full_native', 'full_2x'):
        for row in output[arm]:
            observations = row['observations']
            for item in observations:
                points = np.asarray(item['polygon_px'])
                require(points.shape == (4,2) and np.isfinite(points).all() and isinstance(item['raw'],str), 'Invalid OCR region')
            observations.sort(key=lambda o: (float(np.asarray(o['polygon_px'])[:,0].mean()),
                                            float(np.asarray(o['polygon_px'])[:,1].mean())))
            text = ' '.join(o['raw'] for o in observations)
            if text != row['prediction']:
                changes.append({'id': row['id'], 'arm': arm, 'before': row['prediction'], 'after': text})
            row.update(prediction=text, status='ok' if text else 'empty')
    return output, changes
