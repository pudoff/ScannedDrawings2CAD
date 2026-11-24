"""Keep coarse role guards; localize duplicate own-head guards at source scale."""
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v4 as previous
from . import native_dimension_transfer_v2 as measured
from .native_dimension_source_partition_v1 import observe as partition_source
from .native_head_protection_v1 import protection_ledger, refine, _Budget
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import source_windows


def propose(original_source, working_image, *, geometry, ocr, roles, source_heads,
            working_to_source, arrows, authority=None, authority_root=None, protected=None):
    result = previous.propose(original_source, working_image, geometry=geometry, ocr=ocr,
        roles=roles, source_heads=source_heads, working_to_source=working_to_source, arrows=arrows,
        authority=authority, authority_root=authority_root, protected=protected)
    with Image.open(working_image) as image:
        gray = np.asarray(image.convert('L'))
    chain = result['chain_result']; contexts = {r['id']: r for r in chain['admission']['context']['contexts']}
    all_arrows = {'candidates': [*arrows.get('candidates', []), *chain['admission']['arrows']['candidates']]}
    result['export_protection'] = {}
    try:
        ledger = protection_ledger(gray.shape, roles, ocr, all_arrows, source_heads,
                                   np.asarray(working_to_source), protected)
    except _Budget as exc:
        result['report']['head_protection_ledger'] = {'complete': False, 'limits': [str(exc)],
                                                     'coarse_protection_preserved': True}
        result['report']['source_head_protection_refined'] = False
        return result
    if not np.array_equal(ledger['mask'], result['protection']):
        raise ValueError('Head protection ledger differs from existing coarse guard')
    ledger['context_sha256'] = object_sha(chain['admission']['context'])
    report = result['report']; report['head_protection_ledger'] = ledger['report']
    rows = {r['id']: r for r in report['links']}
    eligible = [r for r in chain['report']['links'] if r['id'] in rows
                and rows[r['id']]['status'] == 'source_dimension_bodies_admitted']
    boxes = list(dict.fromkeys(tuple(r['source_box_px']) for r in eligible))
    windows = source_windows(original_source, report['source_sha256'], source_heads['processed_size_px'], boxes) if boxes else []
    by_box = dict(zip(boxes, windows))
    for row in eligible:
        record = rows[row['id']]; transfer = record['transfer']
        source = by_box[tuple(row['source_box_px'])]
        work = {'source_mask_pixels': 0, 'width_samples': 0}
        masks, failure = measured._source_target_masks(source, row, contexts[row['id']], work)
        if failure:
            record['head_protection_refinement'] = {'complete': False, 'reason': failure,
                                                   'coarse_protection_preserved': True}
            continue
        targets, ink, blocked = masks
        refined = refine(source, gray, geometry, roles, row, contexts[row['id']], working_to_source,
            protection=result['protection'], ledger=ledger, targets=targets, ink=ink,
            source_protection=blocked)
        record['head_protection_refinement'] = refined['report']
        if not refined['complete']:
            continue
        native = partition_source(source, gray, geometry, roles, row, working_to_source,
            protection=refined['hard_protection'], targets=targets, ink=ink,
            source_protection=refined['native_protection'])
        if not native['complete']:
            record['head_protection_refinement'].update(partition_complete=False, partition_limits=native['report']['limits'])
            continue
        record['head_protection_refinement'].update(partition_complete=True,
            previous_partition_sha256=object_sha(transfer['source_partition']),
            previous_class_pixels=transfer['source_partition']['class_pixels'])
        transfer['source_partition'] = native['report']
        result['export_protection'][row['id']] = refined['hard_protection']
    report['source_head_protection_refined'] = True
    return result


def joint_propose(*args, **kwargs):
    with patch.object(joint, 'native_propose', propose):
        result = joint.propose(*args, **kwargs)
    result['report']['schema_version'] = 'joint-dimension-roles/8.0'
    return result
