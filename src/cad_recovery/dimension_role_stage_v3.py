"""Observe original dimension context after fresh native heads, preserving roles."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import dimension_role_stage_v2 as previous
from .native_dimension_context_v1 import inspect
from .part_admission_v1 import object_sha


def _summary(context):
    rows = context['contexts']
    options = [option for row in rows for group in row.get('head_extension_bindings', [])
               for option in group['extension_options']]
    return {
        'object_sha256': object_sha(context),
        'artifact': 'native-dimension-context.json',
        'complete': context['complete'],
        'limits': deepcopy(context['limits']),
        'context_status_counts': dict(Counter(row['status'] for row in rows)),
        'source_window_count': len({tuple(row['source_box_px']) for row in rows if 'source_box_px' in row}),
        'head_extension_option_count': len(options),
        'shape_rechecked_option_count': sum(option['source_shape_rechecked'] for option in options),
        'tip_revisions_beyond_prior_bound': sum(not option['old_tip_consistent'] for option in options),
        'arc_anchor_observation_count': sum(len(option.get('arc_anchors', [])) for option in options),
        'role_revision_count': 0,
        'input_role_frame': 'joint_input_before_role_proposals',
        'roles_sha256': context['roles_sha256'],
        'source_sha256': context['source_sha256'],
        'working_source_sha256': context['working_source_sha256'],
        'physical_geometry_approved': False,
        'complete_dimension_acceptance': False,
        'source_context_is_diagnostic_evidence': True}


def analyze(folder, packet, gray, roles, *, original_source, working_to_source,
            frame_rules=(), ocr=None):
    folder = Path(folder)
    propose = previous.propose
    observed = {}

    def with_native_context(image, **kwargs):
        result = propose(image, **kwargs)
        published_roles = result['roles'].copy()
        context = inspect(kwargs['original_source'], image,
            geometry=kwargs['geometry'], ocr=kwargs['ocr'], roles=kwargs['roles'],
            source_heads=kwargs['source_heads'], working_to_source=kwargs['working_to_source'])
        if context['roles_changed'] != 0 or not np.array_equal(result['roles'], published_roles):
            raise ValueError('Native context observation must not revise published role proposals')
        observed['context'] = context
        result['report']['native_context'] = _summary(context)
        return result

    with patch.object(previous, 'propose', with_native_context):
        result = previous.analyze(folder, packet, gray, roles, original_source=original_source,
            working_to_source=working_to_source, frame_rules=frame_rules, ocr=ocr)
    previous.previous.write(folder / 'native-dimension-context.json', observed['context'])
    return result
