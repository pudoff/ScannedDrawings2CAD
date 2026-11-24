"""Keep the joint role rules while adding explicitly rechecked native heads."""
from copy import deepcopy
from unittest.mock import patch

import cv2
import numpy as np

from . import joint_dimension_roles_v1 as previous
from .arrow_shaft_support_v1 import observe as legacy_support
from .native_head_bridge_v1 import build


def propose(source_image, *, geometry, ocr, arrows, roles, original_source,
            working_to_source, source_heads):
    bridge = build(original_source, source_image, geometry=geometry, ocr=ocr,
                   source_heads=source_heads, working_to_source=working_to_source)
    native = bridge['arrows']['candidates']
    combined = deepcopy(arrows)
    duplicate_ids = []
    retained = []
    for old in combined['candidates']:
        polygon = np.asarray(old['polygon_px'], np.float32)
        duplicate = False
        for head in native:
            if (np.dot(old['direction'], head['direction']) < .98
                    or set(old.get('evidence', {}).get('geometry_ids', [])) != set(head['evidence']['geometry_ids'])):
                continue
            other = np.asarray(head['polygon_px'], np.float32)
            intersection, _ = cv2.intersectConvexConvex(polygon, other)
            union = abs(cv2.contourArea(polygon))+abs(cv2.contourArea(other))-intersection
            if union > 0 and intersection/union >= .5:
                duplicate = True
                duplicate_ids.append({'working_head_id': old['id'], 'native_head_id': head['id'],
                                      'reason': 'same_carrier_direction_and_overlapping_head_footprint'})
                break
        if not duplicate:
            retained.append(old)
    if {h['id'] for h in retained}.intersection(h['id'] for h in native):
        raise ValueError('Native and legacy head IDs collide')
    combined['candidates'] = retained + deepcopy(native)
    retained_ids = {h['id'] for h in retained}

    def support(image, *, geometry, ocr, arrows):
        old = {**arrows, 'candidates': [h for h in arrows['candidates'] if h['id'] in retained_ids]}
        baseline = legacy_support(image, geometry=geometry, ocr=ocr, arrows=old)
        result = {**baseline, 'spans': baseline['spans']+deepcopy(bridge['support']['spans']),
                  'events': baseline['events']+deepcopy(bridge['support']['events']),
                  'complete': baseline['complete'] and bridge['support']['complete'],
                  'native_bridge': bridge['report'], 'confidence_recalibrated': False}
        if not result['complete']:
            result['spans'] = []
        return result

    with patch.object(previous, 'observe', support):
        result = previous.propose(source_image, geometry=geometry, ocr=ocr, arrows=combined, roles=roles)
    result['report'].update(schema_version='joint-dimension-roles/2.0', native_bridge=bridge['report'],
        merged_head_duplicates=duplicate_ids, confidence_recalibrated=False,
        chain_verification_frame='working_source_pixels', head_verification_frames=['working_legacy', 'original_source_windows'],
        original_head_evidence_is_not_whole_native_dimension_proof=True)
    result['projected_arrows'] = combined
    result['native_bridge'] = bridge
    return result
