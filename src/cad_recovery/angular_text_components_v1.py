"""Diagnostic layout proposals for angular nominals and separate deviations.

This module does not modify OCR, association ownership or accepted graph fields.
Whole-string syntax classifies observations; numerical values do not rank layout.
"""
from itertools import combinations
import math
import re

import cv2
import numpy as np

from .angular_link_metrics_v1 import overlap
from .family_catalog_v2 import require

POLICY = {'id': 'angular-text-components/1.0', 'max_rows': 128, 'max_pairs': 8192,
          'max_edges': 64, 'confidence': .65, 'spatial_iou': .35,
          'duplicate_iou': .65, 'merge_target_coverage': .5,
          'height_ratio_min': .35, 'height_ratio_max': .85,
          'right_gap_height_max': 1.5, 'left_intrusion_height_max': .1,
          'vertical_offset_min': .12, 'vertical_offset_max': .8,
          'axis_tolerance_deg': 12, 'nominal_used_for_selection': False,
          'physical_geometry_approved': False, 'production_admitted': False,
          'training_eligible': False, 'graph_acceptance': False}
NUMBER = r'(?:[0-9]{1,6}(?:[.,][0-9]{1,4})?)'
ROLES = ('nominal', 'upper_deviation', 'lower_deviation', 'unresolved')


def reading_class(raw):
    if raw is None:
        return 'unresolved'
    if re.fullmatch(rf'[+-]{NUMBER}\s*\u00b0', raw):
        return 'signed_deviation_candidate'
    if re.fullmatch(r'0(?:[.,]0+)?\s*\u00b0', raw):
        return 'zero_angle_candidate'
    if re.fullmatch(rf'{NUMBER}\s*\u00b0', raw):
        return 'nominal_candidate'
    if re.fullmatch(rf'{NUMBER}\s*\u00b0?\s*[+-]{NUMBER}\s*\u00b0?', raw):
        return 'merged_reading_needs_pixel_components'
    return 'unresolved'


def _row(row):
    require(isinstance(row['id'], str) and row['id'], 'Missing component observation ID')
    p = np.asarray(row['polygon_px'], float)
    require(p.shape == (4, 2) and np.isfinite(p).all() and np.abs(p).max() <= 1e7
            and cv2.isContourConvex(p.astype(np.float32)) and cv2.contourArea(p.astype(np.float32)) > 0,
            'Invalid component polygon')
    raw = row['raw']; score = row['confidence']
    require((raw is None or isinstance(raw, str) and len(raw) <= 512)
            and isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1, 'Invalid component reading')
    # TL/TR/BR/BL supplied by OCR or a given crop; preserve its directed first edge.
    u = p[1]-p[0]; width = float(np.linalg.norm(u))
    require(width > 0, 'Zero-width component')
    u /= width; v = np.array([-u[1], u[0]])
    height = float(np.ptp(p@v))
    require(height > 0, 'Zero-height component')
    return dict(row, p=p, center=p.mean(axis=0), u=u, v=v, height=height, reading_class=reading_class(raw))


def group(rows, *, upstream_incomplete=False):
    """Suggest component roles from explicit sign/zero and relative pixel layout.

    A signed upper deviation alone does not imply a missing lower zero. Returned
    groups are diagnostic proposals; none suppress an original OCR observation.
    """
    require(type(upstream_incomplete) is bool and isinstance(rows, list), 'Invalid component request')
    ids = [r['id'] for r in rows]
    require(len(set(ids)) == len(ids), 'Duplicate observation IDs')
    if len(rows) > POLICY['max_rows']:
        return {'groups': [], 'observations': [], 'search_incomplete': True, 'limit': 'row_limit',
                'checks': 0, 'policy': POLICY}
    normalized = [_row(r) for r in rows]; duplicate = set(); checks = 0; limited = None
    for a, b in combinations(normalized, 2):
        checks += 1
        if checks > POLICY['max_pairs']:
            limited = 'pair_limit'; break
        if overlap(a['polygon_px'], b['polygon_px']) >= POLICY['duplicate_iou']:
            duplicate.update((a['id'], b['id']))
    nominals = [r for r in normalized if r['reading_class'] == 'nominal_candidate']
    deviations = [r for r in normalized if r['reading_class'] in ('signed_deviation_candidate', 'zero_angle_candidate')]
    edges = []; parents = {}
    if limited is None:
        for nominal in nominals:
            for deviation in deviations:
                checks += 1
                if checks > POLICY['max_pairs']:
                    limited = 'pair_limit'; break
                if float(nominal['u']@deviation['u']) < math.cos(math.radians(POLICY['axis_tolerance_deg'])):
                    continue
                height = nominal['height']; ratio = deviation['height']/height
                if not POLICY['height_ratio_min'] <= ratio <= POLICY['height_ratio_max']:
                    continue
                gap = float(np.min(deviation['p']@nominal['u'])-np.max(nominal['p']@nominal['u']))/height
                offset = float((deviation['center']-nominal['center'])@nominal['v'])/height
                if not -POLICY['left_intrusion_height_max'] <= gap <= POLICY['right_gap_height_max']:
                    continue
                if not POLICY['vertical_offset_min'] <= abs(offset) <= POLICY['vertical_offset_max']:
                    continue
                role = 'upper_deviation' if offset < 0 else 'lower_deviation'
                edge = {'nominal_observation_id': nominal['id'], 'component_observation_id': deviation['id'],
                        'role': role, 'evidence': {'height_ratio': ratio, 'right_gap_in_nominal_heights': gap,
                        'vertical_offset_in_nominal_heights': offset, 'nominal_used_for_selection': False}}
                edges.append(edge); parents.setdefault(deviation['id'], set()).add(nominal['id'])
                if len(edges) >= POLICY['max_edges']:
                    limited = 'edge_limit'; break
            if limited: break
    incomplete = bool(limited or upstream_incomplete)
    by = {r['id']: r for r in normalized}; groups = []
    for nominal in nominals:
        children = [e for e in edges if e['nominal_observation_id'] == nominal['id']]
        if not children: continue
        reasons = []
        members = {nominal['id'], *(e['component_observation_id'] for e in children)}
        if incomplete: reasons.append('search_incomplete')
        if members & duplicate: reasons.append('overlapping_OCR_observations_not_independent')
        if any(by[i]['confidence'] < POLICY['confidence'] or by[i].get('conflicting_line_ids') for i in members):
            reasons.append('reading_requires_review')
        if any(len(parents[e['component_observation_id']]) > 1 for e in children):
            reasons.append('competing_nominal_parents')
        if any(sum(e['role'] == role for e in children) > 1 for role in ROLES[1:3]):
            reasons.append('competing_components_in_same_slot')
        groups.append({'nominal_observation_id': nominal['id'], 'components': children,
                       'status': 'ambiguous' if reasons else 'proposed_diagnostic', 'reasons': reasons,
                       'missing_upper': not any(e['role'] == 'upper_deviation' for e in children),
                       'missing_lower': not any(e['role'] == 'lower_deviation' for e in children),
                       'missing_values_inferred': False, 'graph_accepted': False})
    return {'groups': groups, 'observations': [{'id': r['id'], 'raw': r['raw'],
            'reading_class': r['reading_class'], 'overlap_requires_review': r['id'] in duplicate} for r in normalized],
            'search_incomplete': incomplete, 'limit': limited, 'checks': checks, 'policy': POLICY}


def validate_case(case):
    require(case['split'] in ('train', 'development', 'procedural') and case['group_id'], 'Unknown component data scope')
    require(all(case[k] is False for k in ('human_verified', 'training_eligible', 'physical_geometry_approved',
                                          'production_admitted', 'independent_evaluation')), 'Diagnostic components acquired authority')
    width, height = case['size_px']; ids = []
    for component in case['components']:
        ids.append(component['id']); box = component['box_px']
        require(len(box) == 4 and all(type(v) is int for v in box) and 0 <= box[0] < box[2] <= width
                and 0 <= box[1] < box[3] <= height and component['role'] in ROLES, 'Invalid component target')
        known = component['reading_status'] == 'accepted_diagnostic'
        require(known or component['reading_status'] == 'unresolved', 'Unknown reading status')
        require((known and isinstance(component['reference'], str) and component['reference']) or
                (not known and component['reference'] is None and component['unresolved_reason']), 'Invented unresolved reading')
    require(len(set(ids)) == len(ids), 'Repeated component targets')
    by = {r['id']: r for r in case['components']}
    for edge in case['expected_edges']:
        require(edge['nominal_id'] in by and edge['component_id'] in by
                and by[edge['nominal_id']]['role'] == 'nominal'
                and by[edge['component_id']]['role'] == edge['role'] and edge['role'] in ROLES[1:3], 'Invalid role relation')


def polygon(box):
    a, b, c, d = box
    return [[a, b], [c, b], [c, d], [a, d]]


def evaluate(case, rows, report):
    """Independent spatial assignment; strings/IDs never determine matching."""
    validate_case(case); targets = case['components']
    links = sorted((-overlap(polygon(t['box_px']), r['polygon_px']), i, j)
                   for i, t in enumerate(targets) for j, r in enumerate(rows)
                   if overlap(polygon(t['box_px']), r['polygon_px']) >= POLICY['spatial_iou'])
    assigned = {}; used = set()
    for score, i, j in links:
        if i not in assigned and j not in used:
            assigned[i] = j; used.add(j)
    observation_target = {rows[j]['id']: targets[i]['id'] for i, j in assigned.items()}
    known = [i for i, t in enumerate(targets) if t['reading_status'] == 'accepted_diagnostic']
    truth = {(e['nominal_id'], e['component_id'], e['role']) for e in case['expected_edges']}
    proposed = []; candidates = []
    for g in report['groups']:
        for e in g['components']:
            edge = (observation_target.get(e['nominal_observation_id']), observation_target.get(e['component_observation_id']), e['role'])
            candidates.append(edge)
            if g['status'] == 'proposed_diagnostic': proposed.append(edge)
    merged = 0
    for row in rows:
        covered = 0
        for t in targets:
            if t['reading_status'] != 'accepted_diagnostic': continue
            p = np.asarray(polygon(t['box_px']), np.float32)
            intersection, _ = cv2.intersectConvexConvex(p, np.asarray(row['polygon_px'], np.float32))
            covered += intersection/cv2.contourArea(p) >= POLICY['merge_target_coverage']
        merged += covered >= 2
    return {'known_component_targets': len(known), 'unresolved_component_targets': len(targets)-len(known),
            'localized_known_components': sum(i in assigned for i in known),
            'literal_exact_known_components': sum(i in assigned and rows[assigned[i]]['raw'] == targets[i]['reference'] for i in known),
            'localized_unresolved_components': sum(i in assigned for i in range(len(targets)) if i not in known),
            'expected_role_edges': len(truth), 'correct_role_edges': len(set(proposed)&truth),
            'false_role_edges': sum(e not in truth for e in proposed),
            'correct_role_edges_among_alternatives': len(set(candidates)&truth),
            'proposed_groups': sum(g['status'] == 'proposed_diagnostic' for g in report['groups']),
            'ambiguous_groups': sum(g['status'] == 'ambiguous' for g in report['groups']),
            'observations_covering_multiple_targets': int(merged),
            'unmatched_observations': len(rows)-len(used), 'search_incomplete': int(report['search_incomplete']),
            'matches': [{'target_id': targets[i]['id'], 'observation_id': rows[j]['id'], 'raw': rows[j]['raw'],
                         'reference': targets[i]['reference'], 'reading_status': targets[i]['reading_status'],
                         'iou': overlap(polygon(targets[i]['box_px']), rows[j]['polygon_px'])} for i, j in sorted(assigned.items())]}


def totals(records):
    keys = ('known_component_targets', 'unresolved_component_targets', 'localized_known_components',
            'literal_exact_known_components', 'localized_unresolved_components', 'expected_role_edges',
            'correct_role_edges', 'false_role_edges', 'correct_role_edges_among_alternatives', 'proposed_groups',
            'ambiguous_groups', 'observations_covering_multiple_targets', 'unmatched_observations', 'search_incomplete')
    return {k: sum(r[k] for r in records) for k in keys}
