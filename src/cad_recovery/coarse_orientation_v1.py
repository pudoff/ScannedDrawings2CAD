"""Bounded text-line orientation proposals with exact quarter-turn provenance.

Detector polygons, classifier scores and raw recognition are the only evidence.
No expected literal, template name, drawing geometry or physical scale is used.
"""
import hashlib
import math

import cv2
import numpy as np

from .family_catalog_v2 import require
from .ocr_label_schema_v1 import make_ocr_transform_v1

POLICY = {'id': 'coarse-orientation/1.0', 'max_image_pixels': 1048576,
          'max_side_px': 1536, 'max_detections': 64, 'max_lines': 12,
          'max_crop_pixels': 100000, 'line_aspect_min': 2., 'nms_iou': .5,
          'min_classifier_confidence': .9, 'min_recognition_confidence': .85,
          'min_characters': 4, 'min_letters': 2, 'min_line_votes': 2,
          'minimum_vote_fraction': .8, 'strong_conflict_votes': 2,
          'per_case_seconds': 10, 'training_executed': False,
          'production_admitted': False, 'physical_geometry_approved': False,
          'graph_acceptance': False}


def transforms(size_px, clockwise):
    require(isinstance(size_px, (list, tuple)) and len(size_px) == 2
            and all(type(v) is int and v > 0 for v in size_px), 'Invalid raster size')
    require(type(clockwise) is int and clockwise in (0, 90, 180, 270), 'Expected exact clockwise quarter-turn')
    w, h = size_px
    center = make_ocr_transform_v1([0, 0, w, h], clockwise)
    forward = np.asarray(center['source_to_image'], dtype=float)
    plus = np.array([[1., 0, .5], [0, 1., .5], [0, 0, 1.]])
    minus = np.array([[1., 0, -.5], [0, 1., -.5], [0, 0, 1.]])
    edge = plus @ forward @ minus
    return {'input_size_px': [w, h], 'output_size_px': center['size_px'], 'clockwise_deg': clockwise,
            'centers_input_to_output': center['source_to_image'],
            'centers_output_to_input': center['image_to_source'],
            'edges_input_to_output': edge.astype(int).tolist(),
            'edges_output_to_input': np.rint(np.linalg.inv(edge)).astype(int).tolist(),
            'convention': 'Raster samples/points: integer centers 0..W-1; boxes/polygons: edges 0..W, half-open boxes.'}


def map_points(points, matrix):
    p = np.asarray(points, dtype=float); m = np.asarray(matrix, dtype=float)
    require(p.ndim == 2 and p.shape[1] == 2 and len(p) > 0 and np.isfinite(p).all()
            and m.shape == (3, 3) and np.isfinite(m).all()
            and np.array_equal(m[2], [0, 0, 1]) and abs(np.linalg.det(m)) > 1e-12, 'Invalid affine pixel mapping')
    return (np.column_stack((p, np.ones(len(p)))) @ m.T)[:, :2].tolist()


def map_box(box, edge_matrix):
    require(isinstance(box, (list, tuple)) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in box)
            and box[0] < box[2] and box[1] < box[3], 'Expected nonempty half-open box')
    a, b, c, d = box
    p = np.asarray(map_points([[a, b], [c, b], [c, d], [a, d]], edge_matrix))
    return [*p.min(axis=0).tolist(), *p.max(axis=0).tolist()]


def pixel_hash(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


def validate_image(rgb):
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
            and 0 < rgb.shape[0]*rgb.shape[1] <= POLICY['max_image_pixels']
            and max(rgb.shape[:2]) <= POLICY['max_side_px'], 'Orientation image exceeds native RGB budget')


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union else 0.


def select_lines(polygons, size_px):
    """Deterministic pixel-only selection; all budget/conflict flags are retained."""
    require(isinstance(polygons, list), 'Expected detector polygon list')
    if len(polygons) > POLICY['max_detections']:
        return [], ['detection_limit']
    w, h = size_px; proposed = []
    for index, polygon in enumerate(polygons):
        p = np.asarray(polygon, dtype=float)
        require(p.shape == (4, 2) and np.isfinite(p).all() and (p >= 0).all()
                and (p <= [w, h]).all() and cv2.contourArea(p.astype(np.float32)) > 0, 'Invalid detector polygon')
        a, b = np.maximum(0, np.floor(p.min(axis=0))).astype(int)
        c, d = np.minimum([w, h], np.ceil(p.max(axis=0))).astype(int)
        if a >= c or b >= d:
            continue
        width, height = c-a, d-b
        if max(width, height)/min(width, height) < POLICY['line_aspect_min']:
            continue
        proposed.append({'detection_index': index, 'box_px': [int(a), int(b), int(c), int(d)],
                         'axis_clockwise': 0 if width >= height else 90, 'polygon_input_px': polygon})
    proposed.sort(key=lambda r: (-((r['box_px'][2]-r['box_px'][0])*(r['box_px'][3]-r['box_px'][1])), r['box_px'], r['detection_index']))
    selected = []; issues = []
    for row in proposed:
        if any(iou(row['box_px'], other['box_px']) >= POLICY['nms_iou'] for other in selected):
            continue
        # Two nonoverlapping words on the same physical baseline are one vote.
        def same_row(other):
            if other['axis_clockwise'] != row['axis_clockwise']:
                return False
            p, q = row['box_px'], other['box_px']; axis = 1 if row['axis_clockwise'] == 0 else 0
            overlap = min(p[axis+2], q[axis+2])-max(p[axis], q[axis])
            return overlap >= .5*min(p[axis+2]-p[axis], q[axis+2]-q[axis])
        if any(same_row(other) for other in selected):
            continue
        a, b, c, d = row['box_px']
        if (c-a)*(d-b) > POLICY['max_crop_pixels']:
            issues.append('line_crop_limit'); continue
        if len(selected) == POLICY['max_lines']:
            issues.append('line_limit'); break
        selected.append({**row, 'id': 'line-'+str(len(selected))})
    return selected, sorted(set(issues))


def line_vote(line, readings):
    require(len(readings) == 2 and {r['clockwise_deg'] for r in readings} ==
            {line['axis_clockwise'], (line['axis_clockwise']+180) % 360}, 'Missing opposite-direction evidence')
    for row in readings:
        require(row['classifier_label'] in ('0', '180') and isinstance(row['raw'], str)
                and all(type(row[k]) in (float, int) and math.isfinite(row[k]) and 0 <= row[k] <= 1
                        for k in ('classifier_confidence', 'recognition_confidence')), 'Invalid model score')
    forward = [r for r in readings if r['classifier_label'] == '0' and
               r['classifier_confidence'] >= POLICY['min_classifier_confidence']]
    inverse = [r for r in readings if r['classifier_label'] == '180' and
               r['classifier_confidence'] >= POLICY['min_classifier_confidence']]
    if len(forward) != 1 or len(inverse) != 1:
        return {'line_id': line['id'], 'clockwise_deg': None, 'reason': 'classifier_disagreement_or_weakness'}
    row = forward[0]
    if (row['recognition_confidence'] < POLICY['min_recognition_confidence'] or
            len(row['raw'].strip()) < POLICY['min_characters'] or
            sum(c.isalpha() for c in row['raw']) < POLICY['min_letters']):
        return {'line_id': line['id'], 'clockwise_deg': None, 'reason': 'insufficient_text_support'}
    return {'line_id': line['id'], 'clockwise_deg': row['clockwise_deg'], 'reason': 'paired_classifier_and_literal_support'}


def consensus(votes, issues=()):
    ids = [v['line_id'] for v in votes]; require(len(ids) == len(set(ids)), 'Repeated line cannot vote twice')
    require(all(v['clockwise_deg'] is None or (type(v['clockwise_deg']) is int and
                v['clockwise_deg'] in (0, 90, 180, 270)) for v in votes), 'Invalid orientation vote')
    counts = {str(k): sum(v['clockwise_deg'] == k for v in votes) for k in (0, 90, 180, 270)}
    ranked = sorted(((-n, int(k)) for k, n in counts.items())); winner = ranked[0][1]
    total = sum(counts.values()); reasons = list(issues)
    if total < POLICY['min_line_votes']:
        reasons.append('insufficient_distinct_line_votes')
    if -ranked[1][0] >= POLICY['strong_conflict_votes']:
        reasons.append('conflicting_text_orientations')
    if total and counts[str(winner)]/total < POLICY['minimum_vote_fraction']:
        reasons.append('insufficient_consensus')
    return {'status': 'abstain' if reasons else 'proposal', 'clockwise_deg': None if reasons else winner,
            'vote_counts': counts, 'eligible_votes': total, 'reasons': sorted(set(reasons)),
            'requires_review': True, 'production_admitted': False, 'graph_acceptance': False,
            'physical_geometry_approved': False}


def analyze(rgb, backend):
    validate_image(rgb); h, w = rgb.shape[:2]
    polygons = backend.detect(np.array(rgb, copy=True))
    lines, issues = select_lines(polygons, [w, h]); evidence = []; votes = []
    for line in lines:
        a, b, c, d = line['box_px']; crop = rgb[b:d, a:c]; readings = []
        for clockwise in (line['axis_clockwise'], (line['axis_clockwise']+180) % 360):
            pixels = np.array(np.rot90(crop, -(clockwise//90)), copy=True, order='C')
            observed = backend.read(pixels)
            require(set(observed) == {'classifier_label', 'classifier_confidence', 'raw', 'recognition_confidence'}, 'Unexpected backend fields')
            readings.append({'clockwise_deg': clockwise, **observed})
        votes.append(line_vote(line, readings)); evidence.append({'line': line, 'readings': readings})
    decision = consensus(votes, issues)
    return {'schema_version': 'coarse-orientation-evidence/1.0', 'policy': POLICY,
            'input_size_px': [w, h], 'input_pixel_sha256': pixel_hash(rgb),
            'detections': polygons, 'line_evidence': evidence, 'votes': votes, 'decision': decision,
            'calls': {'detector': 1, 'classifier': 2*len(lines), 'recognizer': 2*len(lines)},
            'raw_or_targets_received': False}


def replay(evidence):
    require(evidence['policy'] == POLICY, 'Changed orientation policy')
    lines, issues = select_lines(evidence['detections'], evidence['input_size_px'])
    require([r['line'] for r in evidence['line_evidence']] == lines, 'Changed detector selection')
    votes = [line_vote(r['line'], r['readings']) for r in evidence['line_evidence']]
    require(evidence['votes'] == votes and evidence['decision'] == consensus(votes, issues), 'Changed orientation decision')
    require(evidence['calls'] == {'detector': 1, 'classifier': 2*len(lines), 'recognizer': 2*len(lines)}, 'Model call accounting drift')
    return evidence['decision']


def apply(rgb, evidence):
    validate_image(rgb)
    require(evidence['input_size_px'] == [rgb.shape[1], rgb.shape[0]] and evidence['input_pixel_sha256'] == pixel_hash(rgb), 'Different source pixels')
    decision = replay(evidence)
    clockwise = decision['clockwise_deg'] if decision['status'] == 'proposal' else 0
    corrected = np.ascontiguousarray(np.rot90(rgb, -(clockwise//90))).copy()
    mapping = transforms(evidence['input_size_px'], clockwise)
    return corrected, {'applied_clockwise_deg': clockwise, 'proposal_requires_review': True,
                       'orientation_verified': False, 'model_consensus_reached': decision['status'] == 'proposal', 'transforms': mapping,
                       'input_pixel_sha256': pixel_hash(rgb), 'output_pixel_sha256': pixel_hash(corrected),
                       'production_admitted': False, 'physical_geometry_approved': False}
