"""Conditional angular re-reading: immutable observations, separate proposals.

Only signed incomplete literals with native small-ink evidence and exactly one
compatible observed angular anchor can request a new reading. No truth enters.
"""
from copy import deepcopy
import math
import re

import numpy as np

from . import angular_text_components_v1 as layout
from .family_catalog_v2 import require

POLICY = {'id': 'conditional-ocr/1.0', 'max_observations': 32, 'max_retries_per_context': 8,
          'max_anchor_checks': 1024, 'alternative_confidence_min': layout.POLICY['confidence'],
          'frame_height_fraction': .15, 'layout': layout.POLICY,
          'eligible_literal': 'signed incomplete number only; small upper-right ink and one spatial nominal anchor',
          'selection': 'complete same semantic stem; own alternative confidence; no original confidence rewrite',
          'minus_semantics': 'U+002D and U+2212 have same polarity; raw/codepoint exact scored separately',
          'production_admitted': False, 'physical_geometry_approved': False, 'graph_acceptance': False}
LITERAL = re.compile(r'([+−-]?)([0-9]{1,6}(?:[.,][0-9]{1,4})?)(°?)\Z')


def parse(raw):
    if not isinstance(raw, str):
        return None
    match = LITERAL.fullmatch(raw)
    if not match:
        return None
    sign, number, degree = match.groups()
    return {'sign': sign, 'number': number, 'complete': bool(degree),
            'semantic_stem': ('-' if sign in ('-', '−') else sign)+number}


def validate_observation(row):
    require(isinstance(row, dict) and isinstance(row.get('raw'), str) and type(row.get('confidence')) in (float, int),
            'Expected literal and numeric confidence')
    layout._row(row)


def compatible_anchor(candidate, anchor):
    p, n = layout._row(candidate), layout._row(anchor)
    if n['reading_class'] != 'nominal_candidate':
        return False
    if float(p['u'] @ n['u']) < math.cos(math.radians(layout.POLICY['axis_tolerance_deg'])):
        return False
    scale = n['height']; ratio = p['height']/scale
    gap = float(np.min(p['p'] @ n['u'])-np.max(n['p'] @ n['u']))/scale
    offset = float((p['center']-n['center']) @ n['v'])/scale
    return (layout.POLICY['height_ratio_min'] <= ratio <= layout.POLICY['height_ratio_max']
            and -layout.POLICY['left_intrusion_height_max'] <= gap <= layout.POLICY['right_gap_height_max']
            and layout.POLICY['vertical_offset_min'] <= abs(offset) <= layout.POLICY['vertical_offset_max'])


def plan(observations, pixel_evidence, *, audit_complete, upstream_incomplete=False):
    require(isinstance(observations, list) and len(observations) <= POLICY['max_observations']
            and type(audit_complete) is bool and type(upstream_incomplete) is bool, 'Invalid bounded context')
    require(len({r['id'] for r in observations}) == len(observations), 'Repeated OCR observation')
    for row in observations:
        validate_observation(row)
    require(isinstance(pixel_evidence, dict) and set(pixel_evidence) <= {r['id'] for r in observations}, 'Unassigned pixel evidence')
    assignments = []; checks = 0; count = 0
    for row in observations:
        literal = parse(row['raw']); anchors = []; evidence = pixel_evidence.get(row['id'])
        reason = 'outside_signed_angular_domain'
        if literal and literal['complete']:
            reason = 'original_complete_no_call'
        elif literal and literal['sign']:
            if not audit_complete or upstream_incomplete or evidence is None or row.get('segmentation_requires_review'):
                reason = 'incomplete_or_ambiguous_pixel_evidence'
            elif not evidence['small_upper_right']:
                reason = 'no_small_upper_right_ink'
            else:
                for anchor in observations:
                    if anchor['id'] == row['id']:
                        continue
                    checks += 1
                    if checks > POLICY['max_anchor_checks']:
                        break
                    if compatible_anchor(row, anchor):
                        anchors.append(anchor['id'])
                reason = ('anchor_check_budget' if checks > POLICY['max_anchor_checks'] else
                          'no_unique_spatial_anchor' if len(anchors) != 1 else
                          'retry_budget' if count >= POLICY['max_retries_per_context'] else 'signed_incomplete_with_pixel_support')
        call = reason == 'signed_incomplete_with_pixel_support'
        count += int(call)
        assignments.append({'original_id': row['id'], 'call': call, 'reason': reason, 'anchor_ids': anchors})
    return {'policy': POLICY, 'assignments': assignments, 'anchor_checks': checks,
            'retries_planned': count, 'targets_received': False}


def resolve(original, alternative, assignment):
    validate_observation(original)
    require(assignment.get('original_id') == original['id'] and type(assignment.get('call')) is bool
            and isinstance(assignment.get('reason'), str), 'Invalid original assignment')
    require(assignment['call'] or alternative is None, 'Unrequested alternate reading')
    selected = original; reason = assignment['reason']; proposed = False
    if assignment['call']:
        reason = 'missing_alternative'
        if alternative is not None:
            validate_observation(alternative)
            require(alternative['id'] != original['id'] and alternative['polygon_px'] == original['polygon_px'],
                    'Alternate must keep native geometry and have its own ID')
            before, after = parse(original['raw']), parse(alternative['raw'])
            require(before and before['sign'] and not before['complete'], 'Ineligible original cannot be replaced')
            reason = ('alternative_incomplete' if not after or not after['complete'] else
                      'semantic_stem_conflict' if after['semantic_stem'] != before['semantic_stem'] else
                      'alternative_low_confidence' if alternative['confidence'] < POLICY['alternative_confidence_min'] else
                      'complete_same_stem_alternative')
            if reason == 'complete_same_stem_alternative':
                selected = alternative; proposed = True
    return {'schema_version': 'conditional-ocr-proposal/1.0',
            'original': deepcopy(original), 'alternative': deepcopy(alternative),
            'selected': deepcopy(selected), 'selection': 'alternative' if proposed else 'original',
            'reason': reason, 'requires_review': True, 'graph_acceptance': False,
            'physical_geometry_approved': False, 'production_admitted': False}
