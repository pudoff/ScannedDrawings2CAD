"""Bounded text-class experiment. Outputs are alternatives, never CAD authority."""
from collections import Counter
import re

import cv2
import numpy as np

from .engineering_glyphs_v1 import TEXT_CLASSES
from .family_catalog_v2 import require

CLASSES = ('degree', 'diameter_mark', 'digit_zero', 'letter_M', 'letter_R',
           'multiplication', 'phi_like_letter', 'plus_minus', 'round_letter')
PURPOSE = 'engineering_glyph_classifier_experiment_v1'
POLICY = {
    'purpose': PURPOSE, 'training': 'accepted real train only; one row per exact image SHA256',
    'validation': 'fixed parent assignment; occurrences and exact-image representatives separately',
    'feature': '32px aspect-preserving ink canvas, 4x4 unsigned 8-bin gradient cells + 8x8 ink + log aspect; L2',
    'ridge_lambda': 0.1, 'min_score': 0.35, 'min_margin': 0.15, 'max_distance': 0.85,
    'nearest_class_agreement': True, 'canvas': 32, 'max_proposals_per_context': 64,
    'max_total_proposals': 4096, 'localization_iou': 0.5,
    'tuning': 'none; train leave-group-out is diagnostic, not threshold selection',
    'patch': 'first component only; accepted diameter; x<=25% width; prefix [phi,$,0,O,diameter] immediately followed by digit',
    'synthetic_training': False, 'development_training': False, 'closed_test_used': False,
    'physical_geometry_approved': False, 'production_admitted': False,
    'independent_evaluation': False, 'human_verified': False,
    'decision': 'further review only if val automatic prefix alias-exact improves without regression, both unique diameters covered, no negative false diameter, no unresolved acceptance, oracle class correctness >= glyph OCR',
}


def feature(rgb):
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3
            and rgb.shape[2] == 3 and 0 < rgb.shape[0]*rgb.shape[1] <= 4_000_000,
            'Expected bounded RGB glyph')
    h, w = rgb.shape[:2]
    ink = (255-rgb.mean(axis=2)).astype(np.float64)/255
    scale = 28/max(h, w)
    sw, sh = max(1, round(w*scale)), max(1, round(h*scale))
    resized = cv2.resize(ink, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((32, 32), np.float64)
    x, y = (32-sw)//2, (32-sh)//2
    canvas[y:y+sh, x:x+sw] = resized
    gx = cv2.Sobel(canvas, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(canvas, cv2.CV_64F, 0, 1, ksize=3)
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    bins = np.minimum(7, (angle*8/np.pi).astype(int))
    magnitude = np.hypot(gx, gy)
    cells = []
    for cy in range(4):
        for cx in range(4):
            region = np.s_[cy*8:(cy+1)*8, cx*8:(cx+1)*8]
            cells.extend(np.bincount(bins[region].ravel(), weights=magnitude[region].ravel(), minlength=8))
    hog = np.array(cells); hog /= max(np.linalg.norm(hog), 1e-12)
    low = cv2.resize(canvas, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    low /= max(np.linalg.norm(low), 1e-12)
    vector = np.concatenate([hog, .5*low, [.25*np.clip(np.log(w/h), -3, 3)]])
    # Empty glyphs must be rejected, irrespective of aspect ratio.
    if ink.max() < 1/255:
        return np.zeros(193, np.float64)
    return vector/max(np.linalg.norm(vector), 1e-12)


def fit(rows):
    require(rows and len(rows) <= 512, 'Missing or excessive train rows')
    require(len({r['sha256'] for r in rows}) == len(rows), 'Deduplicate training pixels first')
    require(all(r['split'] == 'train' and r['purpose'] == PURPOSE and r['class'] in CLASSES
                and r['group_ids'] for r in rows), 'Only admitted real train text classes may be fitted')
    x = np.asarray([r['feature'] for r in rows], np.float64)
    require(x.shape == (len(rows), 193) and np.isfinite(x).all(), 'Invalid feature matrix')
    classes = sorted({r['class'] for r in rows})
    require(len(classes) >= 2, 'Classifier needs at least two train classes')
    y = np.eye(len(classes))[[classes.index(r['class']) for r in rows]]
    weights = x.T @ np.linalg.solve(x@x.T + POLICY['ridge_lambda']*np.eye(len(rows)), y)
    return {'schema_version': 'engineering-glyph-ridge/1.0', 'classes': classes,
            'weights': weights.tolist(), 'train_features': x.tolist(),
            'train_classes': [r['class'] for r in rows], 'train_ids': [r['id'] for r in rows],
            'train_image_sha256': [r['sha256'] for r in rows],
            'purpose': PURPOSE, 'production_admitted': False, 'physical_geometry_approved': False}


def predict(model, vector):
    x = np.asarray(vector, np.float64)
    require(x.shape == (193,) and np.isfinite(x).all(), 'Invalid inference feature')
    require(model['purpose'] == PURPOSE and model['production_admitted'] is False
            and model['physical_geometry_approved'] is False, 'Changed model authority')
    weights = np.asarray(model['weights'], np.float64)
    templates = np.asarray(model['train_features'], np.float64)
    classes = model['classes']
    require(weights.shape == (193, len(classes)) and templates.shape == (len(model['train_classes']), 193)
            and set(classes) <= set(CLASSES) and len(classes) >= 2
            and np.isfinite(weights).all() and np.isfinite(templates).all(), 'Invalid classifier model')
    scores = x@weights
    order = np.argsort(-scores, kind='stable')
    best, second = map(int, order[:2])
    distances = np.linalg.norm(templates-x, axis=1)
    nearest = int(np.argmin(distances)); distance = float(distances[nearest])
    score = float(scores[best]); margin = score-float(scores[second])
    reasons = []
    if np.linalg.norm(x) < .5: reasons.append('empty_ink')
    if score < POLICY['min_score']: reasons.append('low_ridge_score')
    if margin < POLICY['min_margin']: reasons.append('small_margin')
    if distance > POLICY['max_distance']: reasons.append('far_from_train')
    if model['train_classes'][nearest] != classes[best]: reasons.append('nearest_class_disagrees')
    return {'class': None if reasons else classes[best], 'top_class': classes[best],
            'score': score, 'margin': margin, 'nearest_distance': distance,
            'nearest_class': model['train_classes'][nearest], 'rejection_reasons': reasons,
            'scores': {c: float(scores[i]) for i, c in enumerate(classes)}}


def ocr_class(text):
    """Exactly one OCR character; no extraction of a convenient substring."""
    return TEXT_CLASSES.get(text.strip()) if len(text.strip()) == 1 else None


def alias(text):
    return text.translate(str.maketrans({'Ø': '⌀', 'ø': '⌀', 'x': '×'}))


def prefix_alternative(raw, proposals, width):
    """A deliberately narrow image-conditioned experiment, not a global repair."""
    if not proposals or not re.match(r'^[φΦфФ$0OО⌀Øø][0-9]', raw):
        return raw, False
    first = proposals[0]
    if first['prediction']['class'] == 'diameter_mark' and first['box'][0] <= .25*width:
        alternative = '⌀'+raw[1:]
        return alternative, alternative != raw
    return raw, False


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/area if area else 0.0


def class_summary(rows):
    known = [r for r in rows if r['target'] is not None]
    unknown = [r for r in rows if r['target'] is None]
    return {'targets': len(known), 'correct': sum(r['prediction'] == r['target'] for r in known),
            'rejected': sum(r['prediction'] is None for r in known),
            'wrong_accepted': sum(r['prediction'] is not None and r['prediction'] != r['target'] for r in known),
            'unresolved': len(unknown), 'unresolved_accepted': sum(r['prediction'] is not None for r in unknown),
            'false_diameter': sum(r['target'] != 'diameter_mark' and r['prediction'] == 'diameter_mark' for r in known),
            'by_class': {c: {'targets': sum(r['target'] == c for r in known),
                            'correct': sum(r['target'] == c and r['prediction'] == c for r in known),
                            'rejected': sum(r['target'] == c and r['prediction'] is None for r in known)}
                         for c in sorted({r['target'] for r in known})},
            'confusions': dict(sorted(Counter(str(r['target'])+' -> '+str(r['prediction']) for r in known
                                              if r['target'] != r['prediction']).items()))}


def representatives(rows):
    """Scoring representatives are selected by pixel SHA then ID, without predictions."""
    seen = set(); result = []
    for row in sorted(rows, key=lambda r: (r['sha256'], r['id'])):
        if row['sha256'] not in seen:
            result.append(row); seen.add(row['sha256'])
    return result
