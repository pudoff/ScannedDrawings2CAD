"""Recheck original head pixels and bind their evidence to a working CAD frame."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from . import dimension_association_v3 as shared
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import (checked_matrix, digest, local_lines,
    source_windows, transform, verify)


POLICY = {'version': 'native-head-bridge/1.0', 'maximum_heads': 256,
          'maximum_windows': 256, 'maximum_roi_pixels': 67_108_864,
          'maximum_window_side': 1024, 'confidence_recalibrated': False,
          'working_head_pixels_rechecked': False, 'original_head_pixels_rechecked': True,
          'multiple_owners_resolved_by_nearest': False}


def build(original_source, working_image, *, geometry, ocr, source_heads, working_to_source):
    original_source, working_image = Path(original_source), Path(working_image)
    common = {k: geometry.get(k) for k in ('source_sha256', 'processed_size_px')}
    work_sha, work_size, _, _, _ = shared._validate(working_image, geometry, ocr, {**common, 'candidates': []})
    matrix = checked_matrix(working_to_source)
    inverse = np.linalg.inv(matrix)
    original_sha = digest(original_source)
    with Image.open(original_source) as image:
        original_size = list(image.size)
    if (source_heads.get('schema_version') != 'source-arrow-observation/1.0'
            or source_heads.get('source_sha256') != original_sha
            or source_heads.get('processed_size_px') != original_size
            or source_heads.get('geometry_source_sha256') != work_sha
            or not np.allclose(checked_matrix(source_heads.get('geometry_to_source')), matrix, rtol=0, atol=1e-10)):
        raise ValueError('Original head evidence, geometry frame or transform changed')
    if np.prod(original_size) > 256_000_000:
        raise ValueError('Original raster exceeds observation resource limit')
    heads = deepcopy(source_heads.get('verified_candidates', []))
    if (any(not isinstance(h.get('id'), str) or not h['id'] for h in heads)
            or len({h['id'] for h in heads}) != len(heads)):
        raise ValueError('Original heads require unique IDs')
    report = {'schema_version': POLICY['version'], 'working_source_sha256': work_sha,
        'original_source_sha256': original_sha, 'original_size_px': original_size,
        'working_size_px': work_size, 'working_to_source': matrix.tolist(),
        'source_to_working': inverse.tolist(), 'geometry_sha256': object_sha(geometry),
        'ocr_sha256': object_sha(ocr), 'source_heads_sha256': object_sha(source_heads),
        'events': [], 'limits': [], 'policy': deepcopy(POLICY),
        'registration_scope': 'caller_transform_checked_and_current_original_head_pixels_reverified',
        'full_view_pixel_correspondence_checked': False,
        'confidence_recalibrated': False, 'production_changed': False}
    if source_heads.get('search_complete') is not True or geometry.get('complete') is False:
        report['limits'].append('upstream_source_observation_incomplete')
    if len(heads) > POLICY['maximum_heads']:
        report['limits'].append('head_budget')
    scale = float(np.linalg.svd(matrix[:2, :2], compute_uv=False).max())
    native_lines = []
    for original in geometry['candidates']:
        if original['kind'] != 'line':
            continue
        row = deepcopy(original)
        uncertainty = row.get('uncertainty_px', 1.)
        if isinstance(uncertainty, bool) or not np.isfinite(uncertainty) or uncertainty < 0:
            raise ValueError('Finite nonnegative geometry uncertainty required')
        row['points_px'] = transform(row['points_px'], matrix).tolist()
        if 'support_points' in row:
            row['support_points'] = transform(row['support_points'], matrix).tolist()
        row['uncertainty_px'] = max(.5, uncertainty*scale)
        row['source_projection_quantization_px'] = np.sqrt(.5)*scale
        native_lines.append(row)
    polygons = [transform(line['polygon_px'], matrix) for line in ocr.get('lines', [])]
    boxes = []
    for head in heads if not report['limits'] else []:
        box = head.get('evidence', {}).get('source_window', {}).get('box_original_px')
        if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                or not 0 <= box[0] < box[2] <= original_size[0]
                or not 0 <= box[1] < box[3] <= original_size[1]
                or max(box[2]-box[0], box[3]-box[1]) > POLICY['maximum_window_side']):
            raise ValueError('Original head requires a bounded source window')
        if box not in boxes:
            boxes.append(box)
    cost = sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
    if len(boxes) > POLICY['maximum_windows'] or cost > POLICY['maximum_roi_pixels']:
        report['limits'].append('native_recheck_window_budget')
    projected, spans, events = [], [], []
    if not report['limits']:
        windows = source_windows(original_source, original_sha, original_size, boxes) if boxes else []
        by_box = {tuple(box): gray for box, gray in zip(boxes, windows)}
        for source_head in heads:
            box = source_head['evidence']['source_window']['box_original_px']
            gray = by_box[tuple(box)]
            event = {'source_arrow_id': source_head['id'], 'status': 'rejected', 'reasons': []}
            report['events'].append(event)
            reference = source_head.get('source_ref', {})
            if (reference.get('sha256') != original_sha
                    or reference.get('coordinate_space') != 'original_image_pixels'):
                raise ValueError('Native head source reference changed')
            if hashlib.sha256(gray.tobytes()).hexdigest() != source_head['evidence']['source_window'].get('gray_pixels_sha256'):
                raise ValueError('Original head window pixels changed')
            polygon = transform(source_head['polygon_px'], inverse)
            uncertainty = source_head.get('uncertainty_px')
            uncertainty_scale = float(np.linalg.svd(inverse[:2, :2], compute_uv=False).max())
            if (not isinstance(uncertainty, bool) and isinstance(uncertainty, (float, int))
                    and np.isfinite(uncertainty) and uncertainty > 0):
                margin = uncertainty*uncertainty_scale
                # A tip alone outside the crop does not exclude the head domain.
                if ((polygon.max(0)+margin < 0).any()
                        or (polygon.min(0)-margin >= work_size).any()):
                    event.update(status='outside_scope',
                                 reasons=['head_footprint_outside_working_frame'])
                    continue
            if source_head['evidence'].get('base_position_precisely_confirmed') is False:
                event['reasons'].append('head_base_position_unresolved')
                continue
            origin = np.array(box[:2])
            local = deepcopy(source_head)
            local['tip_px'] = (np.asarray(local['tip_px'])-origin).tolist()
            local['polygon_px'] = (np.asarray(local['polygon_px'])-origin).tolist()
            local_geometry = local_lines(native_lines, box)
            text = [(p-origin).tolist() for p in polygons
                    if (p.max(0) >= origin).all() and (p.min(0) < box[2:]).all()]
            pen = local.get('evidence', {}).get('shaft_width_px')
            if not isinstance(pen, (int, float)) or not np.isfinite(pen) or pen <= 0:
                raise ValueError('Positive measured native shaft width required')
            verified, reason = verify(local, gray, local_geometry, text, pen)
            if verified is None:
                event['reasons'].append(reason)
                continue
            uncertainty = source_head.get('uncertainty_px')
            uncertainty_bound = max(2*verified['evidence']['shaft_width_px'],
                                    verified['evidence']['base_width_px']/2)
            if (isinstance(uncertainty, bool) or not isinstance(uncertainty, (float, int))
                    or not np.isfinite(uncertainty) or uncertainty <= 0 or uncertainty > uncertainty_bound):
                event['reasons'].append('native_localization_uncertainty_unbounded')
                continue
            owners = verified['evidence']['geometry_ids']
            if set(owners) != set(source_head['evidence'].get('geometry_ids', [])):
                event['reasons'].append('current_geometry_changes_source_shaft_ownership')
                continue
            if len(owners) != 1:
                event.update(status='ambiguous', owner_ids=owners)
                event['reasons'].append('multiple_source_shaft_owners')
                continue
            tip = transform(source_head['tip_px'], inverse)
            if (tip < 0).any() or (tip >= work_size).any():
                event['reasons'].append('projected_tip_outside_working_frame')
                continue
            delta = polygon[0]-polygon[1:].mean(0)
            direction = delta/np.linalg.norm(delta)
            native_direction = np.asarray(source_head['direction'])
            normal = np.array([-native_direction[1], native_direction[0]])
            transverse_scale = float(np.linalg.norm(inverse[:2, :2]@normal))
            uncertainty_scale = float(np.linalg.svd(inverse[:2, :2], compute_uv=False).max())
            row = deepcopy(source_head)
            identity = 'native/'+source_head['id']
            row.update(id=identity, tip_px=tip.tolist(), polygon_px=polygon.tolist(), direction=direction.tolist(),
                       uncertainty_px=float(source_head['uncertainty_px'])*uncertainty_scale,
                       source_ref={'sha256': work_sha, 'coordinate_space': 'processed_image_pixels'})
            native_proof = deepcopy(source_head['evidence'])
            native_proof['current_recheck'] = deepcopy(verified['evidence'])
            row['evidence'] = dict(shaft_width_px=verified['evidence']['shaft_width_px']*transverse_scale,
                base_width_px=float(np.linalg.norm(polygon[1]-polygon[2])),
                arrow_length_px=float(np.linalg.norm(delta)), head_length_px=float(np.linalg.norm(delta)),
                geometry_ids=owners, measurement_coordinate_space='processed_image_pixels',
                original_head={'id': source_head['id'], 'tip_px': source_head['tip_px'],
                    'polygon_px': source_head['polygon_px'], 'source_sha256': original_sha,
                    'source_window': deepcopy(source_head['evidence']['source_window']), 'proof': native_proof},
                geometry_sha256=report['geometry_sha256'], source_heads_sha256=report['source_heads_sha256'],
                original_confidence=source_head['confidence'], confidence_recalibrated=False,
                projected_head_geometry_only=True)
            projected.append(row)
            evidence = {'arrow_id': identity, 'status': 'supported', 'reasons': [],
                'base_width_px': row['evidence']['base_width_px'], 'tip_px': tip.tolist(),
                'base_px': polygon[1:].mean(0).tolist(), 'verification_frame': 'original_source_window',
                'original_proof': deepcopy(row['evidence']['original_head']),
                'owner_checks': [{'shaft_id': owners[0]}], 'confidence_recalibrated': False}
            events.append(evidence)
            spans.append({'id': 'native-annotation-support/'+source_head['id'], 'arrow_id': identity,
                'shaft_id': owners[0], 'points_px': [tip.tolist(), polygon[1:].mean(0).tolist()],
                'role': 'annotation_support_only', 'uncertainty_px': row['uncertainty_px'],
                'source_ref': {'sha256': work_sha, 'coordinate_space': 'processed_image_pixels'},
                'accepted_as_part_geometry': False})
            event.update(status='supported', projected_arrow_id=identity, owner_ids=owners,
                         original_confidence=source_head['confidence'], projected_confidence=row['confidence'])
    withheld = [e['source_arrow_id'] for e in report['events']
                if e['status'] not in ('supported', 'outside_scope')]
    complete = not report['limits'] and not withheld
    # An unresolved middle head must not disappear from the joint-domain veto
    # and allow two outer heads to become one false dimension.
    if not complete:
        spans = []
    report.update(complete=complete, input_heads=len(heads), supported_heads=len(projected),
                  published_support_heads=len(spans), qualification_complete=not withheld,
                  qualification_withheld_source_head_ids=withheld, rechecked_roi_pixels=cost)
    return {'arrows': {**common, 'candidates': projected,
                       'analysis': {'search_limited': bool(report['limits']),
                                    'qualification_complete': not withheld}},
            'support': {**common, 'complete': complete, 'spans': spans, 'events': events,
                        'policy': deepcopy(POLICY), 'verification_frame': 'original_source_windows'},
            'report': report}
