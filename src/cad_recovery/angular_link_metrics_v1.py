"""Spatial angular control evaluation; no association implementation imported."""
import math
import numpy as np
import cv2


def overlap(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != (4, 2) or b.shape != (4, 2): return 0.
    intersection, _ = cv2.intersectConvexConvex(cv2.convexHull(a), cv2.convexHull(b))
    total = cv2.contourArea(a)+cv2.contourArea(b)-intersection
    return float(intersection/total) if total > 0 else 0.


def evaluate(report, truth):
    if report['source_sha256'] != truth['source_sha256']:
        raise ValueError('Angular evaluation source mismatch')
    proposals = report['associations']; targets = truth['targets']; edges = []
    for i, target in enumerate(targets):
        for j, row in enumerate(proposals):
            score = overlap(target['polygon_px'], row['text_polygon_px'])
            if score >= .35: edges.append((-score, i, j))
    matched_targets = set(); matched_proposals = set(); matches = []
    for negative_score, i, j in sorted(edges):
        if i in matched_targets or j in matched_proposals: continue
        matched_targets.add(i); matched_proposals.add(j)
        target, row = targets[i], proposals[j]; expected = target['expected']; correct = False
        if expected == 'angular':
            correct = (row['kind'] == 'angular' and row['status'] == 'proposed' and len(row['anchors_px']) == 3
                       and set(row['measured_candidate_ids']) == set(target['ray_ids'])
                       and math.dist(row['anchors_px'][0], target['vertex_px']) <= 6
                       and abs(row.get('observed_sector_deg', -1000)-target['sector_deg']) <= 3
                       and row.get('nominal_unit') == 'degree' and row.get('nominal_mm') is None)
        elif expected == 'linear':
            correct = row['kind'] == 'linear' and row['status'] == 'proposed' and len(row['anchors_px']) == 2
        elif expected == 'unknown':
            correct = row['kind'] == 'unknown' and row['status'] == 'ambiguous' and row.get('nominal_mm') is None
        elif expected == 'ambiguous':
            correct = row['status'] != 'proposed' and not row['anchors_px']
        matches.append({'target_id': target['id'], 'association_id': row['id'], 'expected': expected,
                        'correct': bool(correct), 'status': row['status'], 'kind': row['kind'], 'text_iou': -negative_score})
    correct_ids = {m['association_id'] for m in matches if m['correct'] and m['expected'] in ('angular', 'linear')}
    return {'targets': len(targets), 'localized_targets': len(matched_targets),
            'positive_targets': sum(t['expected'] in ('angular', 'linear') for t in targets),
            'correct_proposals': len(correct_ids),
            'false_proposals': sum(r['status'] == 'proposed' and r['id'] not in correct_ids for r in proposals),
            'unknown_targets': sum(t['expected'] == 'unknown' for t in targets),
            'correct_unknown_guards': sum(m['expected'] == 'unknown' and m['correct'] for m in matches),
            'review_targets': sum(t['expected'] == 'ambiguous' for t in targets),
            'correct_reviews': sum(m['expected'] == 'ambiguous' and m['correct'] for m in matches),
            'matches': matches, 'independent_evaluation': False,
            'matching_policy': 'greedy spatial IoU>=0.35, then stable input index; no nominal or ID matching'}
