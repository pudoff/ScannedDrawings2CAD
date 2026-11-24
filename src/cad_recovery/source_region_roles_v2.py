"""Separate semantic completeness from a checked, unchanged fallback map.

The frozen producer already rolls back all optional claims on a budget failure.
Only a local glyph-count failure above a complete baseline may continue. Neither
the old result.complete flag nor any resource threshold is changed.
"""
from copy import deepcopy
from unittest.mock import patch

import numpy as np

from . import source_region_roles_v1 as previous
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha

LOCAL_FALLBACK_LIMITS = {'glyph_count_budget'}


def validate_continuation(gray, graph, raw_roles, result):
    """Check the certificate against actual arrays, not just a delivery boolean."""
    report = result['semantic_coverage']
    if report['input_sha256'] != sha(gray) or report['raw_roles_sha256'] != sha(raw_roles) or report['graph_sha256'] != object_sha(graph):
        raise ValueError('Partial role evidence does not match these inputs')
    if result['complete']:
        return True
    limits = set(result['audit']['limits'])
    base = result['previous']
    return bool(limits and limits <= LOCAL_FALLBACK_LIMITS
        and report['baseline_complete'] and not report['baseline_limits']
        and report['failed_contexts']
        and report['baseline_roles_sha256'] == sha(base)
        and not result['audit'].get('source_ownership', {}).get('incomplete', True)
        and all(np.array_equal(result[arm], base) and not result['owners'][arm].any()
                for arm in previous.ARMS[1:]))


def analyze(gray, graph, roles, *, authority=None):
    nested = {}; failed = []
    grid_analyze = previous.grids.analyze
    glyph_context = previous.glyph_context
    before = (sha(gray), object_sha(graph), sha(roles))

    def checked_baseline(*args, **kwargs):
        result = grid_analyze(*args, **kwargs)
        nested.update(complete=result['complete'], limits=deepcopy(result['audit']['limits']),
                      roles_sha256=sha(result['guarded']))
        return result

    def checked_context(grid, ink, labels, stats, distance, items, audit):
        start = len(audit['limits'])
        result = glyph_context(grid, ink, labels, stats, distance, items, audit)
        limits = audit['limits'][start:]
        if limits:
            failed.append(dict(limits=list(limits), interval_ids=list(grid['interval_ids']),
                u=deepcopy(grid['u']), n=deepcopy(grid['n']), local_box=deepcopy(grid['local_box']),
                coordinate_space='producer_working_pixels',
                context_rectangle_is_not_a_semantic_mask=True))
        return result

    with (patch.object(previous.grids, 'analyze', checked_baseline),
          patch.object(previous, 'glyph_context', checked_context)):
        result = previous.analyze(gray, graph, roles, authority=authority)
    if before != (sha(gray), object_sha(graph), sha(roles)):
        raise ValueError('Role producer mutated source inputs')
    report = dict(schema_version='semantic-role-coverage/1.0', semantic_complete=bool(result['complete']),
        input_sha256=before[0], graph_sha256=before[1], raw_roles_sha256=before[2],
        baseline_complete=bool(nested['complete']), baseline_limits=nested['limits'],
        baseline_roles_sha256=nested['roles_sha256'], limits=list(result['audit']['limits']),
        failed_contexts=failed, fallback='none' if result['complete'] else 'unchanged_complete_baseline_only',
        optional_context_promotions=0 if not result['complete'] else int(np.count_nonzero(result['guarded'] != result['previous'])),
        baseline_unknown_pixels=int(np.count_nonzero(result['previous'] == 255)),
        fallback_unknown_pixels=int(np.count_nonzero(result['guarded'] == 255)),
        semantic_recall_measured=False, thresholds_changed=False)
    result['semantic_coverage'] = report
    result['continuation_safe'] = validate_continuation(gray, graph, roles, result)
    report['continuation_safe'] = result['continuation_safe']
    return result
