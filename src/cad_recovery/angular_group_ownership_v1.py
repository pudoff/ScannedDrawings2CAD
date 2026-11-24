"""Shared angular ownership proved by component layout and native leader paths.

No OCR is repaired or deleted. Parent representations, unresolved alternatives
and the original ungrouped associations remain available for review.
"""
from copy import deepcopy
import math
import re

import cv2
import numpy as np

from . import angular_association_v1 as core
from . import angular_leader_search_v2 as search
from . import angular_text_components_v1 as text
from . import dimension_association_v3 as linear
from .family_catalog_v2 import require
from .notation import parse_notation

POLICY = {'id': 'angular-group-ownership/1.0', 'max_groups': 32, 'max_parent_regions': 64,
          'max_parent_roi_pixels': 262144, 'max_total_parent_pixels': 1048576,
          'max_pixel_cells': 2000000, 'max_checks': 50000, 'geometry_corridor_px': 4,
          'maximum_unexplained_ink_fraction': .05, 'minimum_parent_ink_pixels': 3,
          'nominal_used_for_selection': False, 'physical_geometry_approved': False,
          'production_admitted': False, 'graph_acceptance': False}


def signature(choice):
    path = choice.get('evidence', {}).get('leader_path')
    if not path or not path.get('junction_proof', {}).get('unique_ink_support'): return None
    return (choice['dimension_arc_candidate_id'], path['shelf_candidate_id'], tuple(path['leader_candidate_ids']))


def clear(item, reason):
    item.update(status='ambiguous', anchors_px=[], measured_candidate_ids=[], dimension_line_candidate_ids=[],
                arrow_candidate_ids=[], evidence={})
    item.pop('observed_sector_deg', None)
    if reason not in item['reasons']: item['reasons'].append(reason)


def _parent_coverage(parent, children, verified_geometry, raster, budget, cost):
    p = np.asarray(parent['polygon_px'], float); h, w = raster.gray.shape
    low = np.maximum([0, 0], np.floor(p.min(axis=0))).astype(int)
    high = np.minimum([w, h], np.ceil(p.max(axis=0))+1).astype(int)
    x0, y0 = low; x1, y1 = high; area = int((x1-x0)*(y1-y0))
    if area > POLICY['max_parent_roi_pixels'] or cost['parent_roi_pixels']+area > POLICY['max_total_parent_pixels']:
        raise core.SearchLimit('parent_region_pixel_limit')
    cost['parent_roi_pixels'] += area; budget.cells += 8*area
    if budget.cells > POLICY['max_pixel_cells']: raise core.SearchLimit('ownership_pixel_limit')
    inside = np.zeros((y1-y0, x1-x0), np.uint8); assigned = inside.copy()
    cv2.fillPoly(inside, [(p-low).round().astype(np.int32)], 1)
    for child in children:
        cv2.fillPoly(assigned, [(np.asarray(child['polygon_px'])-low).round().astype(np.int32)], 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]; points = np.stack((xx, yy), axis=-1)
    explained_geometry = np.zeros(inside.shape, bool)
    # Only the finite, pixel-verified shelf/leader members of a group may explain
    # non-text ink inside a broad parent OCR polygon. Other geometry is not exempt.
    for line in verified_geometry:
        budget.check()
        a, b = line['points']; delta = b-a; length2 = float(delta@delta)
        t = np.clip(((points-a)@delta)/length2, 0, 1)
        distance = np.linalg.norm(points-(a+t[..., None]*delta), axis=-1)
        explained_geometry |= distance <= POLICY['geometry_corridor_px']
    ink = (raster.gray[y0:y1, x0:x1] < 200) & (inside != 0) & ~explained_geometry
    total = int(ink.sum()); residual = int((ink & (assigned == 0)).sum())
    fraction = residual/total if total else None
    return {'parent_line_id': parent['id'], 'roi_pixels': area, 'text_ink_pixels': total,
            'unexplained_ink_pixels': residual, 'unexplained_fraction': fraction,
            'geometry_exemption_ids': sorted(r['id'] for r in verified_geometry),
            'complete_representation': total >= POLICY['minimum_parent_ink_pixels'] and fraction <= POLICY['maximum_unexplained_ink_fraction']}


def resolve(source_image, report, *, geometry, ocr, arrows, leader_geometry=None,
            parent_ocr=None, parent_report=None, component_search_incomplete=False):
    """Pure report transformation; all support search is bounded and nominal-free."""
    result = deepcopy(report); rows = {r['id']: r for r in ocr['lines']}
    angular_rows = [r for r in ocr['lines'] if text.reading_class(r['raw']) != 'unresolved']
    groups = text.group(angular_rows, upstream_incomplete=component_search_incomplete)
    result['component_layout'] = groups
    associations = {a['source_line_id']: a for a in result['associations']}
    result['ungrouped_associations'] = deepcopy(result['associations'])
    budget = search.Budget(); cost = {'parent_roi_pixels': 0}; reasons = []; candidates = []; parents = {}; coverage = []
    if parent_ocr is not None:
        linear._validate(source_image, geometry, parent_ocr, arrows)
        parents = {r['id']: r for r in parent_ocr['lines']}
        if len(parents) > POLICY['max_parent_regions']: reasons.append('parent_region_limit')
        for row in rows.values():
            ids = row.get('source_region_ids', [])
            require(ids and len(set(ids)) == len(ids) and set(ids) <= set(parents), 'Missing or invalid component provenance')
    if not groups['groups'] and not groups['search_incomplete']:
        if parent_ocr is not None:
            for item in result['associations']:
                if item['status'] == 'proposed' and item['kind'] in ('angular', 'unknown'):
                    clear(item, 'no_verified_component_group_for_parent_representation')
        result['group_ownership'] = {'policy': POLICY, 'groups': [], 'parent_coverage': [], 'incomplete': reasons, 'checks': 0,
                                     'pixel_cells': 0, 'parent_roi_pixels': 0, 'parent_observations': deepcopy(parent_ocr)}
        return result
    _, _, _, curves, _ = linear._validate(source_image, geometry, ocr, arrows)
    _, _, lines, _, _ = linear._validate(source_image, leader_geometry or geometry, ocr, arrows)
    by_curve = {c['id']: c for c in curves}; by_line = {l['id']: l for l in lines}
    raster = core.RasterEvidence(source_image, ocr, arrows, budget)
    try:
        if reasons: raise core.SearchLimit(reasons[0])
        if len(groups['groups']) > POLICY['max_groups']: raise core.SearchLimit('ownership_group_limit')
        for group in groups['groups']:
            budget.notation = 0; budget.junction_notation = 0; budget.check()
            nominal = associations.get(group['nominal_observation_id'])
            members = [group['nominal_observation_id']]+[e['component_observation_id'] for e in group['components']]
            record = {'nominal_line_id': group['nominal_observation_id'], 'component_line_ids': members,
                      'layout': group, 'status': 'ambiguous', 'reasons': [], 'shared_support': None}
            candidates.append(record)
            if nominal is None or group['status'] != 'proposed_diagnostic':
                record['reasons'].append('layout_or_reading_requires_review'); continue
            choices = nominal.get('alternatives', [])
            if len(choices) != 1 or not choices[0].get('complete_support') or signature(choices[0]) is None:
                record['reasons'].append('nominal_support_missing_or_not_unique'); continue
            choice = choices[0]; key = signature(choice); record['shared_support'] = list(key[:2])+[list(key[2])]
            for identifier in members[1:]:
                child = associations.get(identifier)
                if child is None:
                    record['reasons'].append('component_not_localized'); continue
                if any(signature(c) != key or not c.get('complete_support') for c in child.get('alternatives', [])):
                    record['reasons'].append('component_has_other_or_incomplete_support'); continue
                # A near-square zero can have an uncertain long-edge OCR orientation.
                # Its axis comes from the independently proposed text group, not a nominal value.
                axis = text._row(rows[members[0]])['u']; poly = np.asarray(rows[identifier]['polygon_px'])
                proxy = linear._text_geometry(rows[identifier]); proxy.update(u=axis,
                    width=float(np.ptp(poly@axis)), height=max(1., float(np.ptp(poly@np.array([-axis[1], axis[0]])))))
                paths = search.paths(proxy, lines, by_curve[key[0]], raster, budget)
                matched = [p for p in paths if (key[0], p['shelf_candidate_id'], tuple(p['leader_candidate_ids'])) == key
                           and p['junction_proof']['unique_ink_support']]
                if len(matched) != 1:
                    record['reasons'].append('component_lacks_unique_common_pixel_path')
                else:
                    record.setdefault('component_pixel_paths', []).append({'line_id': identifier, 'path': matched[0],
                        'orientation_basis': 'observed_component_group_nominal_axis'})
            # Values are checked only AFTER roles and supports have been selected.
            deviation = {}
            for edge in group['components']:
                raw = rows[edge['component_observation_id']]['raw']
                deviation[edge['role']] = float(raw.replace('\u00b0', '').replace(',', '.'))
            if ('upper_deviation' in deviation and 'lower_deviation' in deviation and
                    deviation['upper_deviation'] < deviation['lower_deviation']):
                record['reasons'].append('deviation_order_requires_review')
            record['deviation_degree_values_proposed'] = deviation
            if parent_ocr is not None:
                related = sorted({p for identifier in members for p in rows[identifier]['source_region_ids']})
                verified = [by_line[key[1]], *(by_line[i] for i in key[2])]
                for identifier in related:
                    children = [r for r in rows.values() if identifier in r['source_region_ids']]
                    proof = _parent_coverage(parents[identifier], children, verified, raster, budget, cost)
                    coverage.append(proof)
                    if not proof['complete_representation']: record['reasons'].append('unresolved_raw_parent_representation')
                    parsed = parse_notation(parents[identifier]['raw'])
                    if parsed['kind'] == 'angle' and any(text.reading_class(r['raw']) == 'nominal_candidate'
                        and parse_notation(r['raw']).get('values', {}).get('angle_deg') != parsed['values']['angle_deg'] for r in children):
                        record['reasons'].append('raw_parent_angle_reading_conflicts')
            if budget.page > POLICY['max_checks']: raise core.SearchLimit('ownership_check_limit')
            if budget.cells > POLICY['max_pixel_cells']: raise core.SearchLimit('ownership_pixel_limit')
    except core.SearchLimit as error:
        if str(error) not in reasons: reasons.append(str(error))
    if groups['search_incomplete']: reasons.append('component_layout_search_incomplete')
    if report['leader_search']['upstream_incomplete'] or report['leader_search']['incomplete'] or report['angular_search']['incomplete']:
        reasons.append('upstream_or_angular_search_incomplete')
    # Project only completely proved memberships. Remaining texts retain their own votes.
    eligible = [r for r in candidates if not r['reasons'] and not reasons]
    projection = {}
    for record in eligible:
        for identifier in record['component_line_ids']: projection[identifier] = record['nominal_line_id']
    owners = {}
    for item in result['associations']:
        for choice in item.get('alternatives', []):
            if 'dimension_arc_candidate_id' in choice:
                owners.setdefault(choice['dimension_arc_candidate_id'], set()).add(projection.get(item['source_line_id'], item['source_line_id']))
    covered_parent_ids = {p['parent_line_id'] for p in coverage if p['complete_representation']}
    if parent_report is not None:
        for item in parent_report['associations']:
            if item['source_line_id'] in covered_parent_ids: continue
            for choice in item.get('alternatives', []):
                if 'dimension_arc_candidate_id' in choice:
                    owners.setdefault(choice['dimension_arc_candidate_id'], set()).add('raw-parent:'+item['source_line_id'])
    for record in eligible:
        nominal = associations[record['nominal_line_id']]; choice = nominal['alternatives'][0]
        if owners.get(choice['dimension_arc_candidate_id'], set()) != {record['nominal_line_id']}:
            record['reasons'].append('competing_arc_owners'); continue
        blockers = [r for r in nominal['reasons'] if r != 'competing_texts_for_dimension_arc']
        syntax = parse_notation(nominal['raw'])
        if nominal['kind'] != 'angular' or syntax['kind'] != 'angle' or syntax['status'] != 'parsed' or blockers:
            record['reasons'].append('nominal_reading_or_support_requires_review'); continue
        value = syntax['values']['angle_deg']
        if not 0 < value < 360 or abs(choice['observed_sector_deg']-value) > 3:
            record['reasons'].append('observed_sector_disagrees_with_text'); continue
        record['status'] = 'proposed_diagnostic'
        nominal.update(status='proposed', reasons=[], anchors_px=choice['points_px'],
            measured_candidate_ids=choice['measured_candidate_ids'], dimension_line_candidate_ids=[choice['dimension_arc_candidate_id']],
            arrow_candidate_ids=choice['arrow_candidate_ids'], evidence=deepcopy(choice['evidence']), observed_sector_deg=choice['observed_sector_deg'])
        nominal.pop('competing_notation_ids', None)
        nominal['text_components'] = [{'source_line_id': identifier, 'raw': rows[identifier]['raw'],
            'polygon_px': rows[identifier]['polygon_px'], 'role': 'nominal' if identifier == record['nominal_line_id'] else
                next(e['role'] for e in record['layout']['components'] if e['component_observation_id'] == identifier)} for identifier in record['component_line_ids']]
        nominal['evidence']['group_ownership'] = deepcopy(record)
        for edge in record['layout']['components']:
            child = associations[edge['component_observation_id']]
            child.update(kind='angular_component', status='associated_component', semantic_role=edge['role'],
                type_hypotheses=['angular_deviation_component'],
                component_of_association_id=nominal['id'], anchors_px=[], measured_candidate_ids=[], dimension_line_candidate_ids=[],
                arrow_candidate_ids=[], nominal_mm=None, nominal_unit='degree', reasons=['observed_component_group_owns_shared_dimension_arc'],
                evidence={'shared_owner_association_id': nominal['id'], 'independent_dimension': False})
            child.pop('graph_contract', None); child.pop('observed_sector_deg', None); child.pop('competing_notation_ids', None)
            child['component_contract'] = {'unit': 'degree', 'exportable': False, 'graph_accepted': False, 'independent_dimension': False}
    for record in candidates:
        nominal = associations.get(record['nominal_line_id'])
        if record['status'] != 'proposed_diagnostic' and nominal and nominal['status'] == 'proposed':
            clear(nominal, 'component_group_ownership_requires_review')
    if reasons:
        for item in result['associations']:
            if item['status'] == 'proposed': clear(item, 'group_ownership_search_incomplete')
    result['group_ownership'] = {'policy': POLICY, 'groups': candidates, 'parent_coverage': coverage, 'incomplete': reasons,
        'checks': budget.page, 'pixel_cells': budget.cells, 'parent_roi_pixels': cost['parent_roi_pixels'],
        'parent_observations': deepcopy(parent_ocr), 'parent_representation_is_not_independent_vote': True}
    return result
