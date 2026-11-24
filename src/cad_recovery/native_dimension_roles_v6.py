"""Attach possible source-head memberships without changing admitted CAD roles."""
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v5 as previous
from .native_head_membership_v1 import observe
from .source_arrow_observation_v1 import source_windows


def propose(original_source, working_image, *, geometry, ocr, roles, source_heads,
            working_to_source, arrows, authority=None, authority_root=None, protected=None):
    result = previous.propose(original_source, working_image, geometry=geometry, ocr=ocr,
        roles=roles, source_heads=source_heads, working_to_source=working_to_source,
        arrows=arrows, authority=authority, authority_root=authority_root, protected=protected)
    chain = result['chain_result']
    contexts = {row['id']: row for row in chain['admission']['context']['contexts']}
    records = {row['id']: row for row in result['report']['links']}
    eligible = [row for row in chain['report']['links'] if row['id'] in records
        and records[row['id']]['status'] == 'source_dimension_bodies_admitted'
        and records[row['id']].get('head_protection_refinement', {}).get('complete')
        and row['id'] in result['export_protection']]
    boxes = list(dict.fromkeys(tuple(row['source_box_px']) for row in eligible))
    windows = source_windows(original_source, result['report']['source_sha256'],
                             source_heads['processed_size_px'], boxes) if boxes else []
    by_box = dict(zip(boxes, windows))
    result['head_membership'] = {}
    if eligible:
        with Image.open(working_image) as image:
            gray = np.asarray(image.convert('L'))
        for row in eligible:
            record = records[row['id']]
            membership = observe(by_box[tuple(row['source_box_px'])], gray, geometry,
                roles, row, contexts[row['id']], working_to_source,
                head_refinement=record['head_protection_refinement'],
                hard_protection=result['export_protection'][row['id']],
                source_heads=source_heads, admitted_heads=chain['admission']['arrows'])
            record['head_membership'] = membership['report']
            if membership['complete']:
                result['head_membership'][row['id']] = membership['report']
    result['report']['source_head_membership_checked'] = True
    result['report']['source_head_membership_observed'] = bool(result['head_membership'])
    result['report']['source_head_membership_complete_count'] = len(result['head_membership'])
    return result


def joint_propose(*args, **kwargs):
    with patch.object(joint, 'native_propose', propose):
        result = joint.propose(*args, **kwargs)
    result['report']['schema_version'] = 'joint-dimension-roles/9.0'
    return result
