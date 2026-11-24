"""Versioned partial directed references; never infer labels from predictions."""
from copy import deepcopy
import math

import numpy as np

from .family_catalog_v2 import require

POLICY = {'version': 'native-directed-controls/1.0', 'changed_ids': ['train-extra-1-2-arrow-1'],
    'visible_heads': 21, 'negative_regions': 14, 'maximum_patch_side': 1024,
    'maximum_DXF_corroboration_distance_px': 3, 'maximum_DXF_direction_error_deg': 5,
    'evaluation_tip_tolerance_px': 8, 'evaluation_direction_tolerance_deg': 20,
    'reviewer': 'assistant_visual_review', 'human_verified': False, 'blind_review': False,
    'purpose': 'diagnostic_arrow_evaluation', 'exhaustive_page_labels': False,
    'training_eligible': False, 'physical_geometry_approved': False,
    'production_admitted': False, 'independent_evaluation': False, 'closed_test_used': False}


def validate_control(control, size):
    width, height = size
    require(isinstance(control.get('id'), str) and control['id'], 'Missing control identity')
    if control['state'] == 'visible_head':
        point = np.asarray(control['tip_px'], float); direction = np.asarray(control['direction'], float)
        require(point.shape == (2,) and np.isfinite(point).all() and 0 <= point[0] < width and
            0 <= point[1] < height, 'Invalid native directed point')
        require(direction.shape == (2,) and np.isfinite(direction).all() and
            abs(float(np.linalg.norm(direction))-1) <= 1e-6, 'Direction must be a finite unit vector')
    elif control['state'] == 'negative_region':
        box = control['box']
        require(len(box) == 4 and all(type(v) is int for v in box) and
            0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height,
            'Invalid assigned negative region')
    else: raise ValueError('Unsupported directed control state')


def materialize_references(parents, reading, evidence):
    require(reading['reviewer'] == POLICY['reviewer'] and reading['human_verified'] is False
        and reading['blind_review'] is False and reading['training_eligible'] is False
        and reading['physical_geometry_approved'] is False and reading['independent_evaluation'] is False
        and reading['closed_test_used'] is False, 'Changed reading authority or scope')
    by_id = {r['patch_id']: r for r in reading['patches']}
    require(len(by_id) == len(reading['patches']) and set(by_id) == {r['id'] for r in parents},
            'Lost/repeated/unexpected reviewed patch')
    records, changes, identities = [], [], set()
    for row in parents:
        original = row['review']['arrow_controls']; decisions = by_id[row['id']]['controls']
        require([r['id'] for r in decisions] == [r['id'] for r in original], 'Changed directed control inventory')
        controls = []
        for old, decision in zip(original, decisions):
            require(old['id'] not in identities, 'Repeated directed control identity'); identities.add(old['id'])
            require(decision['state'] == old['state'] and len(decision['observation']) > 30, 'Invalid review evidence')
            result = deepcopy(old)
            if decision['decision'] == 'retain':
                require(not {'tip_px', 'direction', 'manual_triangle_px', 'box'} & decision.keys(),
                        'Retained reference contains a hidden replacement')
            else:
                require(decision['decision'] == 'replace_directed_reference' and old['id'] in POLICY['changed_ids']
                    and old['state'] == 'visible_head', 'Unexpected reference replacement')
                require(decision['previous_tip_px'] == old['tip_px'] and decision['previous_direction'] == old['direction'],
                        'Replacement does not bind the historical reference')
                triangle = np.asarray(decision['manual_triangle_px'], float)
                require(triangle.shape == (3, 2) and np.isfinite(triangle).all() and
                    (triangle >= 0).all() and (triangle[:, 0] < row['size_px'][0]).all() and
                    (triangle[:, 1] < row['size_px'][1]).all(), 'Invalid manual raster triangle')
                require(abs(float(np.linalg.det(np.column_stack((triangle[1]-triangle[0], triangle[2]-triangle[0]))))) > 1,
                        'Degenerate manual triangle')
                vector = triangle[0]-triangle[1:].mean(0); vector /= np.linalg.norm(vector)
                require(np.array_equal(triangle[0], decision['tip_px']) and
                    np.allclose(vector, decision['direction'], atol=1e-8, rtol=0), 'Manual direction does not follow triangle')
                corroborated = np.asarray(evidence['registered_SOLID_triangle_px'], float)
                require(corroborated.shape == (3, 2) and np.isfinite(corroborated).all(), 'Invalid corroborating triangle')
                corroborated_direction = corroborated[0]-corroborated[1:].mean(0)
                require(np.linalg.norm(corroborated_direction) > 0, 'Degenerate corroborating head')
                corroborated_direction /= np.linalg.norm(corroborated_direction)
                distance = float(np.linalg.norm(triangle[0]-corroborated[0]))
                angle = math.degrees(math.acos(float(np.clip(vector@corroborated_direction, -1, 1))))
                require(distance <= POLICY['maximum_DXF_corroboration_distance_px'] and
                    angle <= POLICY['maximum_DXF_direction_error_deg'], 'Raster reading not corroborated by DXF')
                result.update(tip_px=deepcopy(decision['tip_px']), direction=deepcopy(decision['direction']),
                              observation=decision['observation'])
                changes.append({'patch_id': row['id'], 'id': old['id'], 'before': deepcopy(old), 'after': deepcopy(result),
                    'manual_triangle_px': triangle.tolist(), 'DXF_tip_distance_px': distance,
                    'DXF_direction_error_deg': angle, 'reason': 'Original point marked the crossing below the actual apex'})
            validate_control(result, row['size_px']); controls.append(result)
        require(row['split'] in ('train', 'validation'), 'Closed/unknown split in directed release')
        matrix = np.asarray(row['patch_to_source'], float)
        require(matrix.shape == (3, 3) and np.isfinite(matrix).all() and
            np.array_equal(matrix[:2, :2], np.eye(2)) and np.array_equal(matrix[2], [0, 0, 1]), 'Invalid native crop transform')
        records.append({'id': row['id'], 'split': row['split'], 'group_id': row['group_id'],
            'source_id': row['source_id'], 'source': row['source'], 'size_px': row['size_px'],
            'patch_to_source': row['patch_to_source'], 'image': 'images/'+row['id']+'.png',
            'annotation_version': POLICY['version'], 'target_scope': 'partial_directed_arrow_controls',
            'arrow_controls': controls, 'training_eligible': False, 'physical_geometry_approved': False})
    require([v['id'] for v in changes] == POLICY['changed_ids'], 'Lost/unexpected directed correction')
    all_controls = [c for r in records for c in r['arrow_controls']]
    require(sum(c['state'] == 'visible_head' for c in all_controls) == POLICY['visible_heads'] and
        sum(c['state'] == 'negative_region' for c in all_controls) == POLICY['negative_regions'], 'Changed control counts')
    return records, changes


def require_purpose(purpose):
    require(purpose == POLICY['purpose'], 'This partial release is admitted only for diagnostic arrow evaluation')
