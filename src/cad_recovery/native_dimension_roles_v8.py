"""Compare head ink models without changing V22 semantic or CAD authority."""
from unittest.mock import patch

from . import joint_dimension_roles_v6 as joint
from . import native_dimension_roles_v7 as previous
from .native_head_ink_v1 import observe


def propose(original_source, working_image, *, geometry, ocr, roles, source_heads,
            working_to_source, arrows, authority=None, authority_root=None, protected=None):
    windows = {}
    source_observe = previous.observe

    def current_shapes(source_gray, working_gray, source_geometry, input_roles,
                       chain, context, matrix, **kwargs):
        result = source_observe(source_gray, working_gray, source_geometry, input_roles,
                                chain, context, matrix, **kwargs)
        windows[chain['id']] = source_gray
        return result

    with patch.object(previous, 'observe', current_shapes):
        result = previous.propose(original_source, working_image, geometry=geometry, ocr=ocr,
            roles=roles, source_heads=source_heads, working_to_source=working_to_source,
            arrows=arrows, authority=authority, authority_root=authority_root, protected=protected)
    records = {row['id']: row for row in result['report']['links']}
    result['head_inks'] = {}
    for identity, shape in result['head_shapes'].items():
        record = records[identity]
        ink = observe(windows[identity], shape_report=shape, membership=record['head_membership'])
        record['head_ink_observation'] = ink['report']
        record['head_membership']['head_ink_observation'] = ink['report']
        result['head_inks'][identity] = ink['report']
    heads = [head for report in result['head_inks'].values() for head in report['heads']]
    result['report'].update(source_head_ink_checked=True,
        source_head_ink_support_head_count=sum(head['model_consensus_pixels'] > 0 for head in heads),
        source_head_ink_consensus_pixels=sum(head['model_consensus_pixels'] for head in heads))
    return result


def joint_propose(*args, **kwargs):
    with patch.object(joint, 'native_propose', propose):
        result = joint.propose(*args, **kwargs)
    result['report']['schema_version'] = 'joint-dimension-roles/11.0'
    return result
