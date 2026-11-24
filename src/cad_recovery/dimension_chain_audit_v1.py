"""Read-only evidence chains for contour-association-v3 pixel proposals.

IDs in the producer are SETS, not positional tuples. Reconstruct every feasible
anchor/arrow/extension/object path; never select a path by a nominal or ID order.
Consistency is not correctness, physical calibration, or semantic acceptance.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import math

import numpy as np

from .dimension_association_v3 import (
    _validate, _line, _point, _segment_distance, _intersection, _text_geometry, _cross,
)

MAX_ASSOCIATIONS = 1024
MAX_CHOICES = 4096
MAX_PATH_CHECKS = 1_000_000
COS = math.cos(math.radians(12))
SIN = math.sin(math.radians(12))
REFERENCE_FIELDS = ('measured_candidate_ids', 'dimension_line_candidate_ids',
                    'extension_line_candidate_ids', 'arrow_candidate_ids')


def _unit(value):
    result = _point(value, 'Direction')
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError('Direction must be nonzero')
    return result / norm


def map_observation(matrix, point, direction=None):
    """Map a pixel CENTER and local tangent using a homography Jacobian.

    A direction is not a homogeneous point. Translation must not affect it;
    for perspective transforms the Jacobian depends on the arrow tip location.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Mapping must be a finite 3x3 matrix')
    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        raise ValueError('Mapping must be invertible')
    matrix = matrix / scale
    if abs(float(np.linalg.det(matrix))) < 1e-15:
        raise ValueError('Mapping must be invertible and well-conditioned')
    point = _point(point, 'Mapping point')
    value = matrix @ np.r_[point, 1.]
    if abs(value[2]) < 1e-12:
        raise ValueError('Mapping crosses the projective horizon')
    mapped = value[:2] / value[2]
    result = {'point': mapped.tolist()}
    if direction is not None:
        jacobian = (matrix[:2, :2] - np.outer(mapped, matrix[2, :2])) / value[2]
        result['direction'] = _unit(jacobian @ _unit(direction)).tolist()
    if not np.isfinite(mapped).all():
        raise ValueError('Nonfinite mapped coordinates')
    return result


def _index(items, label):
    result = {}
    for row in items:
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in result:
            raise ValueError('Nonunique or absent '+label+' ID')
        result[identifier] = row
    return result


def _mapping_record(mapping, source_sha):
    if mapping is None:
        return None
    if (mapping.get('processed_sha256') != source_sha or
            mapping.get('coordinate_convention') != 'pixel_centers' or
            not isinstance(mapping.get('native_sha256'), str) or
            len(mapping['native_sha256']) != 64 or
            any(c not in '0123456789abcdef' for c in mapping['native_sha256'])):
        raise ValueError('Native mapping requires bound source hashes and pixel centers')
    map_observation(mapping['processed_to_native'], [0, 0])
    return deepcopy(mapping)


def _arrow_shape(arrow):
    """The pinned filled-arrow producer stores tip, base corner, base corner.

    Do not invent a base for a different polygon layout or an open arrow.
    """
    polygon = np.asarray(arrow['polygon_px'], dtype=float)
    tip = np.asarray(arrow['tip_px'], dtype=float)
    if polygon.shape != (3, 2) or not np.allclose(polygon[0], tip, atol=1e-6, rtol=0):
        return {'status': 'not_inferable_from_polygon'}
    base = polygon[1:].mean(axis=0)
    if float(np.linalg.norm(tip-base)) < 1e-8:
        return {'status': 'not_inferable_from_polygon'}
    dot = float(_unit(tip-base) @ _unit(arrow['direction']))
    return {'status': 'forward' if dot > 0 else 'contradictory',
            'base_px': base.tolist(), 'base_to_tip_dot': dot}


def audit_dimension_chains(source_image, *, geometry, ocr, arrows, associations,
                           native_mapping=None):
    """Audit existing outputs without consuming truth or modifying inputs.

    This deliberately does not run any detector, OCR, association search or CAD
    solver. Missing evidence is explicit; a consistent chain stays needs_review.
    """
    from pathlib import Path
    digest, size, _, _, _ = _validate(Path(source_image), geometry, ocr, arrows)
    if (associations.get('source_sha256') != digest or associations.get('processed_size_px') != size or
            associations.get('algorithm_version') != 'contour-association-v3'):
        raise ValueError('Association provenance/version mismatch')
    records = associations.get('associations', [])
    if len(records) > MAX_ASSOCIATIONS:
        raise ValueError('Association audit limit exceeded')
    _index(records, 'association')
    geo = _index(geometry['candidates'], 'geometry')
    lines = {key: _line(row) for key, row in geo.items() if row['kind'] == 'line'}
    arrow_index = _index(arrows['candidates'], 'arrow')
    texts = _index(ocr.get('lines', []), 'text')
    notations = _index(ocr.get('notations', []), 'notation')
    mapping = _mapping_record(native_mapping, digest)
    checks = 0
    choices_seen = 0
    output = []
    for association in records:
        row = {'association_id': association['id'], 'notation_id': association['notation_id'],
               'source_line_id': association.get('source_line_id'), 'kind': association['kind'],
               'producer_status': association['status'], 'producer_reasons': deepcopy(association['reasons']),
               'producer_diagnostics': deepcopy(association.get('evidence', {})),
               'source_ref': deepcopy(association.get('source_ref')),
               'view_id': None, 'scale_status': 'not_evaluated', 'choices': [], 'findings': []}
        notation = notations.get(row['notation_id'])
        text = texts.get(row['source_line_id'])
        if notation is None:
            row['findings'].append('notation_id_missing')
        elif notation.get('source_line_id') != row['source_line_id']:
            row['findings'].append('notation_text_reference_mismatch')
        if text is None:
            row['findings'].append('source_text_missing')
        else:
            row['text_observation'] = {key: deepcopy(text.get(key)) for key in
                ('id', 'raw', 'confidence', 'polygon_px', 'source_ref')}
            if association.get('text_polygon_px') != text['polygon_px']:
                row['findings'].append('association_text_polygon_differs')
        row['findings'].append('view_membership_not_recorded')
        if association['kind'] != 'linear':
            row['scope'] = 'nonlinear_relationship_not_audited'
            output.append(row)
            continue
        row['scope'] = 'linear_pixel_evidence_consistency'
        choices = ([association] if association['status'] == 'proposed'
                   else association.get('alternatives', []))
        if not choices:
            row['findings'].append('no_spatial_choice_to_trace')
        for choice in choices:
            choices_seen += 1
            if choices_seen > MAX_CHOICES:
                raise ValueError('Choice audit limit exceeded')
            trace = {'references': {key: deepcopy(choice.get(key, [])) for key in REFERENCE_FIELDS},
                     'evidence': deepcopy(choice.get('evidence', {})), 'anchors': [],
                     'findings': [], 'complete_two_anchor_pairing': False,
                     'pairings': [], 'path_assignment': 'missing', 'arrow_observations': []}
            for key, values in trace['references'].items():
                registry = arrow_index if key == 'arrow_candidate_ids' else geo
                if (not isinstance(values, list) or len(values) > 64 or
                        any(not isinstance(v, str) for v in values) or len(set(values)) != len(values)):
                    raise ValueError('Invalid bounded reference set')
                for identifier in values:
                    if identifier not in registry:
                        trace['findings'].append('missing_reference:'+key+':'+identifier)
            for identifier in trace['references']['arrow_candidate_ids']:
                if identifier in arrow_index:
                    arrow = arrow_index[identifier]
                    shape = _arrow_shape(arrow)
                    trace['arrow_observations'].append({'arrow_id': identifier,
                        'tip_px': deepcopy(arrow['tip_px']), 'direction': deepcopy(arrow['direction']),
                        'source_ref': deepcopy(arrow.get('source_ref')),
                        'detector_evidence': deepcopy(arrow.get('evidence', {})), 'shape_direction': shape})
                    if shape['status'] == 'contradictory':
                        trace['findings'].append('arrow_direction_disagrees_with_polygon:'+identifier)
            anchors = choice.get('anchors_px', [])
            if len(anchors) != 2:
                trace['findings'].append('two_anchors_required')
            if text is None or trace['findings']:
                row['choices'].append(trace)
                continue
            text_geometry = _text_geometry(text)
            tol = text_geometry['tol']
            endpoint_tol = max(6., text_geometry['height'] * .65)
            refs = trace['references']
            for anchor_index, anchor in enumerate(anchors):
                anchor = _point(anchor, 'Anchor')
                paths = []
                for aid, eid, mid, did in product(refs['arrow_candidate_ids'], refs['extension_line_candidate_ids'],
                                                 refs['measured_candidate_ids'], refs['dimension_line_candidate_ids']):
                    checks += 1
                    if checks > MAX_PATH_CHECKS:
                        raise ValueError('Path audit budget exhausted; no complete report')
                    if len({eid, mid, did}) != 3 or any(key not in lines for key in (eid, mid, did)):
                        continue
                    arrow = arrow_index[aid]
                    tip, direction = _point(arrow['tip_px'], 'Tip'), _unit(arrow['direction'])
                    extension, measured, dimension = (lines[key] for key in (eid, mid, did))
                    intersection = _intersection(extension, measured)
                    arrow_parallel = abs(float(direction @ dimension['u']))
                    extension_parallel = abs(float(extension['u'] @ dimension['u']))
                    distances = {'tip_to_extension_px': _segment_distance(tip, extension),
                                 'tip_to_dimension_px': _segment_distance(tip, dimension),
                                 'anchor_to_extension_px': _segment_distance(anchor, extension),
                                 'anchor_to_object_px': _segment_distance(anchor, measured),
                                 'anchor_to_intersection_px': float(np.linalg.norm(anchor-intersection)) if intersection is not None else None}
                    if (arrow['confidence'] < .65 or arrow_parallel < COS or extension_parallel > SIN or
                            distances['tip_to_extension_px'] > tol * 2 or distances['tip_to_dimension_px'] > tol * 3 or
                            distances['anchor_to_extension_px'] > endpoint_tol or distances['anchor_to_object_px'] > tol or
                            intersection is None or distances['anchor_to_intersection_px'] > tol or
                            float(np.linalg.norm(anchor-tip)) < max(12., text_geometry['height'])):
                        continue
                    path = {'arrow_id': aid, 'extension_id': eid, 'measured_id': mid, 'dimension_line_id': did,
                            'tip_px': tip.tolist(), 'direction': direction.tolist(),
                            'arrow_axis_abs_dot': arrow_parallel, 'extension_axis_abs_dot': extension_parallel,
                            'distances': distances}
                    if mapping:
                        path['native_arrow'] = map_observation(mapping['processed_to_native'], tip, direction)
                    paths.append(path)
                entry = {'anchor_index': anchor_index, 'anchor_px': anchor.tolist(), 'paths': paths}
                if mapping:
                    entry['anchor_native_px'] = map_observation(mapping['processed_to_native'], anchor)['point']
                trace['anchors'].append(entry)
            # Do not zip ID lists with anchors. Two different tips/extensions must
            # support the two anchors; a single arrow cannot corroborate both.
            pairings = []
            for i, first in enumerate(trace['anchors'][0]['paths']):
                for j, second in enumerate(trace['anchors'][1]['paths']):
                    checks += 1
                    if checks > MAX_PATH_CHECKS:
                        raise ValueError('Path audit budget exhausted; no complete report')
                    a, b = arrow_index[first['arrow_id']], arrow_index[second['arrow_id']]
                    opposed = float(_unit(a['direction']) @ _unit(b['direction']))
                    delta = np.array(b['tip_px'])-a['tip_px']
                    axis = _unit(delta) if float(np.linalg.norm(delta)) > 1e-8 else None
                    side_a = _cross(axis, np.array(anchors[0])-a['tip_px']) if axis is not None else 0.
                    side_b = _cross(axis, np.array(anchors[1])-b['tip_px']) if axis is not None else 0.
                    if (a['id'] != b['id'] and first['extension_id'] != second['extension_id'] and
                            opposed <= -COS and side_a * side_b > 0):
                        pairings.append([i, j])
            trace['pairings'] = pairings
            trace['complete_two_anchor_pairing'] = bool(pairings)
            if not pairings:
                trace['findings'].append('no_complete_geometric_pairing')
            trace['path_assignment'] = ('unique' if len(pairings) == 1 else 'ambiguous' if pairings else 'missing')
            row['choices'].append(trace)
        output.append(row)
    return {'schema_version': 'dimension-chain-audit/1.0', 'status': 'needs_review',
            'source_sha256': digest, 'processed_size_px': size, 'native_mapping': mapping,
            'native_mapping_status': 'supplied_pixel_mapping' if mapping else 'not_supplied',
            'rows': output, 'path_checks': checks, 'choice_count': choices_seen,
            'source_modified': False, 'physical_geometry_approved': False, 'production_admitted': False,
            'limitations': ['Consistency of existing pixel evidence only; not an independent accuracy estimate.',
                'No semantic view ownership, scale or accepted DrawingGraph entities inferred.',
                'Unresolved output retains producer diagnostics; no invented per-candidate rejection history.',
                'Mapped coordinates depend on the supplied bound matrix; metric paper mapping is not calibration.']}
