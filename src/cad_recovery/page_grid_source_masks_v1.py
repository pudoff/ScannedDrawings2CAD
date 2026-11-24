"""Exact observed-stroke masks per grounded rule; not a standalone PART veto."""
from copy import deepcopy
import cv2
import numpy as np

from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .native_frame_v1 import side_masks
from .stroke_roles_v1 import checked_gray

POLICY = {'version': 'page-grid-source-masks/1.0', 'maximum_pixels': 4194304,
          'maximum_rules': 512, 'maximum_local_mask_pixels': 8000000,
          'requires_owner_direction_binding': True, 'semantic_roles_assigned': False,
          'physical_geometry_approved': False}
BUDGET_REASONS = {'column_budget', 'cross_section_budget', 'source_sample_budget'}


def build(gray, observed):
    checked_gray(gray)
    if gray.size > POLICY['maximum_pixels'] or len(observed['rules']) > POLICY['maximum_rules']:
        raise ValueError('Bounded grounded-rule source masks required')
    if (observed['source_sha256'] != sha(gray) or observed['size_px'] != list(gray.shape[::-1])
            or observed.get('semantic_GRID_or_PART_assigned') is not False):
        raise ValueError('Original pixels and unconfirmed source-rule observations required')
    identifiers = {r['id'] for r in observed['rules']}
    active = {i for group in observed['groups'] for i in group['rule_ids']}
    if len(identifiers) != len(observed['rules']) or not active <= identifiers:
        raise ValueError('Unique bound grounded-rule IDs required')
    if any(r['grounded_group_member'] != (r['id'] in active) for r in observed['rules']):
        raise ValueError('Grounded membership differs')
    before, pixels_before = object_sha(observed), sha(gray)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
    union = np.zeros_like(gray); masks = []; failures = []
    audit = {'source_samples': 0, 'local_mask_pixels': 0, 'discarded_fitted_rules': 0, 'limits': []}
    if not observed['complete']: audit['limits'].append('incomplete_source_attachment')
    for rule in observed['rules'] if not audit['limits'] else []:
        if not rule['grounded_group_member']: continue
        points = np.asarray(rule['points_source_px'], float)
        width, error = float(rule['width_px']), float(rule['uncertainty_px'])
        if (points.shape != (2, 2) or not np.isfinite(points).all() or not np.isfinite(width+error)
                or width <= 0 or error < 0 or np.linalg.norm(points[1]-points[0]) <= 0):
            raise ValueError('Finite observed rule and measured uncertainty required')
        fitted, reason = side_masks(ink, points, width, error, audit)
        if fitted is None:
            failures.append({'rule_id': rule['id'], 'reason': reason})
            if reason in BUDGET_REASONS: audit['limits'].append(reason); break
            continue
        y, x = np.nonzero(fitted['safe'])
        if not len(x): failures.append({'rule_id': rule['id'], 'reason': 'empty_safe_band'}); continue
        x0, y0, x1, y1 = int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1
        count = (x1-x0)*(y1-y0); audit['local_mask_pixels'] += count
        if audit['local_mask_pixels'] > POLICY['maximum_local_mask_pixels']:
            audit['limits'].append('local_mask_pixel_budget'); break
        safe = fitted['safe'][y0:y1, x0:x1].astype(np.uint8)*255
        union[y0:y1, x0:x1] |= safe
        masks.append({'rule_id': rule['id'], 'bbox_source_px': [x0, y0, x1, y1],
                      'safe': safe, 'safe_sha256': sha(safe), 'source_sections': fitted['sections'],
                      'edge_model': fitted['edge_model'], 'source_ink_pixels': int(np.count_nonzero(safe))})
    complete = not audit['limits']
    if not complete:
        audit['discarded_fitted_rules'] = len(masks); masks = []; union[:] = 0
    if np.any((union > 0) & ~ink): raise ValueError('Source mask painted unobserved paper')
    if sha(gray) != pixels_before or object_sha(observed) != before: raise ValueError('Source-mask builder mutated observations')
    return {'source_sha256': pixels_before, 'observations_sha256': before, 'safe_mask': union,
            'safe_mask_sha256': sha(union), 'rule_masks': masks, 'fitted_rule_count': len(masks),
            'unfitted_rules': failures, 'audit': audit, 'complete': complete, 'source_immutable': True,
            'policy': deepcopy(POLICY), 'requires_owner_direction_binding': True,
            'semantic_GRID_or_PART_assigned': False, 'source_pixels_erased': 0, 'physical_geometry_approved': False}
