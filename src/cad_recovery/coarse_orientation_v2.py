"""Same frozen orientation votes; crop ceiling matches the native page ceiling.

V1 is immutable. Its scoring/geometry utilities are reused unchanged. Selection
is versioned because v1's100k crop ceiling excluded every real-crop montage.
"""
import numpy as np

from . import coarse_orientation_v1 as prior
from .family_catalog_v2 import require

POLICY = {**prior.POLICY, 'id': 'coarse-orientation/2.0', 'max_crop_pixels': prior.POLICY['max_image_pixels']}
transforms = prior.transforms
map_points = prior.map_points
map_box = prior.map_box
pixel_hash = prior.pixel_hash
validate_image = prior.validate_image


def select_lines(polygons, size_px):
    """Preserve v1 selection and ordering; allow the already-bounded native area."""
    # Reuse v1 to validate geometry, NMS, physical-row uniqueness and ordering.
    # Cases with no over-ceiling crop retain the exact v1 result.
    selected, issues = prior.select_lines(polygons, size_px)
    if 'line_crop_limit' not in issues:
        return selected, issues
    # Obtain the same ranking without using text or changing global v1 policy.
    w, h = size_px; ranked = []
    for index, polygon in enumerate(polygons):
        p = np.asarray(polygon, dtype=float)
        a, b = np.maximum(0, np.floor(p.min(axis=0))).astype(int)
        c, d = np.minimum([w, h], np.ceil(p.max(axis=0))).astype(int)
        if a >= c or b >= d or max(c-a, d-b)/min(c-a, d-b) < POLICY['line_aspect_min']:
            continue
        ranked.append({'detection_index': index, 'box_px': [int(a), int(b), int(c), int(d)],
                       'axis_clockwise': 0 if c-a >= d-b else 90, 'polygon_input_px': polygon})
    ranked.sort(key=lambda r: (-((r['box_px'][2]-r['box_px'][0])*(r['box_px'][3]-r['box_px'][1])), r['box_px'], r['detection_index']))
    selected = []; issues = []
    for row in ranked:
        if any(prior.iou(row['box_px'], q['box_px']) >= POLICY['nms_iou'] for q in selected):
            continue
        def same_row(q):
            if q['axis_clockwise'] != row['axis_clockwise']:
                return False
            p, r = row['box_px'], q['box_px']; axis = 1 if row['axis_clockwise'] == 0 else 0
            return min(p[axis+2], r[axis+2])-max(p[axis], r[axis]) >= .5*min(p[axis+2]-p[axis], r[axis+2]-r[axis])
        if any(same_row(q) for q in selected):
            continue
        a, b, c, d = row['box_px']
        if (c-a)*(d-b) > POLICY['max_crop_pixels']:
            issues.append('line_crop_limit'); continue
        if len(selected) == POLICY['max_lines']:
            issues.append('line_limit'); break
        selected.append({**row, 'id': 'line-'+str(len(selected))})
    return selected, sorted(set(issues))


def analyze(rgb, backend):
    validate_image(rgb); h, w = rgb.shape[:2]
    polygons = backend.detect(np.array(rgb, copy=True))
    lines, issues = select_lines(polygons, [w, h]); evidence = []; votes = []
    for line in lines:
        a, b, c, d = line['box_px']; readings = []
        for clockwise in (line['axis_clockwise'], (line['axis_clockwise']+180) % 360):
            pixels = np.array(np.rot90(rgb[b:d, a:c], -(clockwise//90)), copy=True, order='C')
            observed = backend.read(pixels)
            require(set(observed) == {'classifier_label', 'classifier_confidence', 'raw', 'recognition_confidence'}, 'Unexpected backend fields')
            readings.append({'clockwise_deg': clockwise, **observed})
        votes.append(prior.line_vote(line, readings)); evidence.append({'line': line, 'readings': readings})
    return {'schema_version': 'coarse-orientation-evidence/2.0', 'policy': POLICY,
            'input_size_px': [w, h], 'input_pixel_sha256': pixel_hash(rgb), 'detections': polygons,
            'line_evidence': evidence, 'votes': votes, 'decision': prior.consensus(votes, issues),
            'calls': {'detector': 1, 'classifier': 2*len(lines), 'recognizer': 2*len(lines)},
            'raw_or_targets_received': False}


def replay(evidence):
    require(evidence['policy'] == POLICY, 'Changed v2 orientation policy')
    lines, issues = select_lines(evidence['detections'], evidence['input_size_px'])
    votes = [prior.line_vote(r['line'], r['readings']) for r in evidence['line_evidence']]
    require([r['line'] for r in evidence['line_evidence']] == lines and evidence['votes'] == votes
            and evidence['decision'] == prior.consensus(votes, issues), 'Changed orientation evidence')
    require(evidence['calls'] == {'detector': 1, 'classifier': 2*len(lines), 'recognizer': 2*len(lines)}, 'Hidden model calls')
    return evidence['decision']


def apply(rgb, evidence):
    validate_image(rgb)
    decision = replay(evidence)
    require(evidence['input_size_px'] == [rgb.shape[1], rgb.shape[0]] and evidence['input_pixel_sha256'] == pixel_hash(rgb), 'Different pixels')
    # Pixel/coordinate conventions are identical to v1; selection was checked above.
    clockwise = decision['clockwise_deg'] if decision['status'] == 'proposal' else 0
    pixels = np.array(np.rot90(rgb, -(clockwise//90)), copy=True, order='C')
    return pixels, {'applied_clockwise_deg': clockwise, 'proposal_requires_review': True,
                    'orientation_verified': False, 'model_consensus_reached': decision['status'] == 'proposal',
                    'transforms': transforms(evidence['input_size_px'], clockwise),
                    'input_pixel_sha256': pixel_hash(rgb), 'output_pixel_sha256': pixel_hash(pixels),
                    'production_admitted': False, 'physical_geometry_approved': False}
