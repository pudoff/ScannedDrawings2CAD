"""Native-ink accounting and review evidence around unchanged pixel-v3 crops.

Shape/position are evidence of unaccounted ink, never identities of glyphs.
No target, nominal, OCR string or physical geometry enters analyze().
"""
import hashlib
import math

import cv2
import numpy as np

from . import angular_component_pixels_v3 as baseline
from .angular_text_components_v1 import reading_class
from .family_catalog_v2 import require

POLICY = {'id': 'small-sign-evidence/1.0', 'ink_threshold': 200,
          'maximum_audit_pixels': 1048576, 'maximum_crop_evidence_pixels': 1048576,
          'maximum_components_per_window': 256, 'maximum_relation_checks': 8192,
          'small_height_fraction': .5, 'small_aspect_min': .4, 'small_aspect_max': 2.5,
          'right_gap_min_fraction': -.1, 'right_gap_max_fraction': .6,
          'left_neighbor_gap_fraction': .35, 'left_neighbor_center_fraction': .25,
          'white_frame_height_fraction': .15,
          'native_pixels_unscaled': True, 'glyph_identity_inferred': False,
          'training_eligible': False, 'physical_geometry_approved': False,
          'production_admitted': False, 'graph_acceptance': False}


def ink(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY) < POLICY['ink_threshold']


def mask_digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def framed(rgb):
    """A single frozen white margin, with the native crop left byte-identical."""
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3 and min(rgb.shape[:2]) > 0,
            'Expected native RGB crop')
    padding = math.ceil(rgb.shape[0] * POLICY['white_frame_height_fraction'])
    result = np.full((rgb.shape[0]+2*padding, rgb.shape[1]+2*padding, 3), 255, np.uint8)
    result[padding:padding+rgb.shape[0], padding:padding+rgb.shape[1]] = rgb
    require(result.shape[0]*result.shape[1] <= 100000, 'Framed crop exceeds OCR budget')
    return result, padding


def crop_evidence(rgb):
    binary = ink(rgb)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    parts = [{'box_px': [int(x), int(y), int(x+w), int(y+h)], 'pixels': int(area)}
             for x, y, w, h, area in stats[1:]]
    if not parts:
        return {'small_upper_right': [], 'parts': [], 'ink_pixels': 0}
    maximum = max(p['box_px'][3]-p['box_px'][1] for p in parts)
    bodies = [p for p in parts if p['box_px'][3]-p['box_px'][1] >= .45*maximum]
    left = min(p['box_px'][0] for p in bodies); top = min(p['box_px'][1] for p in bodies)
    right = max(p['box_px'][2] for p in bodies); bottom = max(p['box_px'][3] for p in bodies)
    scale = bottom-top; small = []
    for part in parts:
        a, b, c, d = part['box_px']; h = d-b; w = c-a
        if (h <= POLICY['small_height_fraction']*scale and POLICY['small_aspect_min'] <= w/h <= POLICY['small_aspect_max']
                and POLICY['right_gap_min_fraction']*scale <= a-right <= POLICY['right_gap_max_fraction']*scale
                and (b+d)/2 < (top+bottom)/2):
            small.append(part)
    return {'small_upper_right': small, 'parts': parts, 'ink_pixels': int(binary.sum()),
            'body_box_px': [left, top, right, bottom], 'glyph_identity_inferred': False}


def residual_mask(rgb, window, crops):
    x0, y0, x1, y1 = window['box_px']; binary = ink(rgb[y0:y1, x0:x1])
    covered = np.zeros_like(binary)
    for crop in crops:
        a, b, c, d = crop['box_px']
        a, b, c, d = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
        if a < c and b < d:
            covered[b-y0:d-y0, a-x0:c-x0] = True
    return binary & ~covered, binary & covered


def analyze(rgb, regions):
    original = baseline.propose(rgb, regions)
    costs = {'audit_pixels': 0, 'crop_evidence_pixels': 0, 'relation_checks': 0}
    windows = []; crop_rows = []; issues = []
    for window in original['windows']:
        if window.get('replaced_by_expansion'):
            continue
        a, b, c, d = window['box_px']; area = (c-a)*(d-b)
        if costs['audit_pixels']+area > POLICY['maximum_audit_pixels']:
            issues.append('audit_pixel_limit'); break
        costs['audit_pixels'] += area
        residual, assigned = residual_mask(rgb, window, original['crops'])
        n, labels, stats, _ = cv2.connectedComponentsWithStats(residual.astype(np.uint8), connectivity=8)
        row = {'id': window['id'], 'box_px': window['box_px'], 'quarter_turn': window['quarter_turn'],
               'assigned_ink_pixels': int(assigned.sum()), 'residual_ink_pixels': int(residual.sum()),
               'total_ink_pixels': int(assigned.sum()+residual.sum()), 'residual_sha256': mask_digest(residual),
               'residual_components': [], 'component_listing_complete': n-1 <= POLICY['maximum_components_per_window']}
        if not row['component_listing_complete']:
            issues.append('audit_component_limit')
        else:
            for x, y, w, h, count in stats[1:]:
                row['residual_components'].append({'box_px': [int(a+x), int(b+y), int(a+x+w), int(b+y+h)],
                                                   'ink_pixels': int(count), 'glyph_identity': 'unknown'})
        windows.append(row)
    for crop in original['crops']:
        a, b, c, d = crop['box_px']; area = (c-a)*(d-b)
        if costs['crop_evidence_pixels']+area > POLICY['maximum_crop_evidence_pixels']:
            issues.append('crop_evidence_pixel_limit'); break
        costs['crop_evidence_pixels'] += area
        pixels = np.ascontiguousarray(np.rot90(rgb[b:d, a:c], crop['quarter_turn']))
        evidence = crop_evidence(pixels)
        neighbors = []
        # Nearby residual ink is never attached or called a minus. Long bars stay
        # independent and can cause review; that cost is measured in the controls.
        for window in windows:
            for component in window['residual_components']:
                costs['relation_checks'] += 1
                if costs['relation_checks'] > POLICY['maximum_relation_checks']:
                    issues.append('audit_relation_limit'); break
                x, y, u, v = component['box_px']
                if crop['quarter_turn'] == 0:
                    gap = a-u; scale = d-b
                    if 0 <= gap <= POLICY['left_neighbor_gap_fraction']*scale and abs((y+v-b-d)/2) <= POLICY['left_neighbor_center_fraction']*scale:
                        neighbors.append(component)
                else:
                    # In canonical orientation, project native pixel boxes exactly.
                    q = crop['quarter_turn']; points = np.array([[x-a, y-b], [u-a, y-b], [u-a, v-b], [x-a, v-b]])
                    rotation_width, rotation_height = c-a, d-b
                    for _ in range(q):
                        points = np.column_stack((points[:, 1], rotation_width-points[:, 0]))
                        rotation_width, rotation_height = rotation_height, rotation_width
                    lo, hi = points.min(axis=0), points.max(axis=0); scale = pixels.shape[0]
                    if 0 <= -hi[0] <= POLICY['left_neighbor_gap_fraction']*scale and abs((lo[1]+hi[1])/2-scale/2) <= POLICY['left_neighbor_center_fraction']*scale:
                        neighbors.append(component)
            if 'audit_relation_limit' in issues:
                break
        crop_rows.append({'id': crop['id'], **evidence, 'nearby_unassigned_ink': neighbors,
                          'requires_review': True, 'graph_acceptance': False})
        if 'audit_relation_limit' in issues:
            break
    return {'schema_version': 'small-sign-evidence/1.0', 'policy': POLICY, 'baseline': original,
            'windows': windows, 'crop_evidence': crop_rows, 'costs': costs,
            'audit_complete': not issues, 'issues': sorted(set(issues)),
            'raw_strings_received': False, 'production_admitted': False, 'physical_geometry_approved': False}


def guard(raw, evidence, *, incomplete=False):
    require(isinstance(raw, str), 'Expected literal OCR output')
    reasons = []
    if incomplete or evidence is None:
        reasons.append('incomplete_pixel_evidence')
    if evidence:
        if evidence['small_upper_right'] and not raw.endswith('°'):
            reasons.append('small_upper_right_ink_not_accounted_for')
        if evidence['nearby_unassigned_ink']:
            reasons.append('nearby_unassigned_ink')
    syntax = reading_class(raw)
    if raw.startswith('−') and reading_class('-'+raw[1:]) == 'signed_deviation_candidate':
        syntax = 'signed_deviation_candidate'  # syntax only; the returned raw stays byte-identical
    if syntax not in ('nominal_candidate', 'signed_deviation_candidate', 'zero_angle_candidate'):
        reasons.append('unresolved_angular_literal')
    return {'raw': raw, 'status': 'unknown' if reasons else 'literal_observed_unverified',
            'reasons': reasons, 'requires_review': True, 'graph_acceptance': False,
            'physical_geometry_approved': False, 'production_admitted': False}
