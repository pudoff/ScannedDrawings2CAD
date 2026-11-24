"""Measure source head flanks/base while preserving all V21 role authority."""
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v6 as previous
from .native_head_shape_v1 import observe


def propose(original_source, working_image, *, geometry, ocr, roles, source_heads,
            working_to_source, arrows, authority=None, authority_root=None, protected=None):
    # Reuse only source windows decoded and SHA-checked during this same call.
    # No persisted crop or caller-provided shape report authorizes a revision.
    windows = {}
    decode = previous.source_windows

    def current_windows(path, sha, size, boxes):
        result = decode(path, sha, size, boxes)
        windows.update({tuple(box): gray for box, gray in zip(boxes, result, strict=True)})
        return result

    with patch.object(previous, 'source_windows', current_windows):
        result = previous.propose(original_source, working_image, geometry=geometry, ocr=ocr,
            roles=roles, source_heads=source_heads, working_to_source=working_to_source,
            arrows=arrows, authority=authority, authority_root=authority_root, protected=protected)
    chain = result['chain_result']
    contexts = {row['id']: row for row in chain['admission']['context']['contexts']}
    records = {row['id']: row for row in result['report']['links']}
    eligible = [row for row in chain['report']['links'] if row['id'] in result['head_membership']]
    result['head_shapes'] = {}
    if eligible:
        with Image.open(working_image) as image:
            gray = np.asarray(image.convert('L'))
        for row in eligible:
            record = records[row['id']]
            shape = observe(windows[tuple(row['source_box_px'])], gray, geometry, roles,
                row, contexts[row['id']], working_to_source,
                head_refinement=record['head_protection_refinement'],
                hard_protection=result['export_protection'][row['id']],
                source_heads=source_heads, admitted_heads=chain['admission']['arrows'],
                membership=record['head_membership'])
            record['head_shape_observation'] = shape['report']
            # Additive metadata: the old RLE and all possible claimant bits stay
            # intact. The parent hash inside shape refers to the unextended RLE.
            record['head_membership']['head_shape_observation'] = shape['report']
            result['head_shapes'][row['id']] = shape['report']
    count = sum(bool(head['shape_pattern_observed']) for report in result['head_shapes'].values()
                for head in report['heads'])
    result['report'].update(source_head_shapes_checked=True,
        source_head_shape_observed_count=count, source_head_shapes_observed=count > 0)
    return result


def joint_propose(*args, **kwargs):
    with patch.object(joint, 'native_propose', propose):
        result = joint.propose(*args, **kwargs)
    result['report']['schema_version'] = 'joint-dimension-roles/10.0'
    return result
