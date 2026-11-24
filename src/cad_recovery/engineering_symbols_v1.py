"""Diagnostic glyph-family labels and metrics; never repairs OCR or approves CAD."""
import re

import cv2
import numpy as np

from .family_catalog_v2 import require
from .ocr_baseline_metrics_v1 import score_text_v1

FAMILIES = {'diameter_mark': '⌀Øø', 'letter_R': 'R', 'letter_M': 'M',
    'plus_minus': '±', 'degree': '°', 'multiplication': '×x'}
CONFUSABLES = '0OО'
ARMS = ('full_native', 'full_2x', 'rec_native', 'rec_2x')
POLICY = {'id': 'engineering-symbol-diagnostic/1.0',
    'selection': 'All glyph-positive train/validation lines; first20 glyph-free O/0 negatives per split by ID.',
    'comparison_aliases': {'Ø': '⌀', 'ø': '⌀', 'x': '×'},
    'alias_scope': 'Symmetric diagnostic comparison only; raw OCR remains unchanged. No O/О/0 mapping.',
    'glyph_metrics': 'One deterministic literal Levenshtein alignment; not independent character detection accuracy.',
    'localization_threshold': .8, 'upscale': 'exactly2x cv2.INTER_CUBIC; no threshold search or information gain claim',
    'training_eligible': False, 'physical_geometry_approved': False, 'production_admitted': False}


def family(char):
    if not isinstance(char, str) or len(char) != 1: return None
    return next((name for name, chars in FAMILIES.items() if char in chars), None)


def annotate(text):
    return [{'index': i, 'literal': c, 'family': family(c)} for i, c in enumerate(text) if family(c)]


def compare(reference, prediction):
    raw = score_text_v1(reference, prediction)
    glyphs = {name: {'assigned': 0, 'literal_correct': 0, 'family_correct': 0, 'deleted': 0} for name in FAMILIES}
    confusions = []; unexpected = 0
    confusable = {'assigned': 0, 'literal_correct': 0, 'false_diameter': 0}
    for item in raw['alignment']:
        a, b = item['reference'], item['prediction']; fa, fb = family(a), family(b)
        if fa:
            row = glyphs[fa]; row['assigned'] += 1; row['literal_correct'] += a == b
            row['family_correct'] += fa == fb; row['deleted'] += b is None
            if fa != fb: confusions.append({'family': fa, 'reference': a, 'prediction': b, 'operation': item['operation']})
        if fb and not fa: unexpected += 1
        if a is not None and a in CONFUSABLES:
            confusable['assigned'] += 1; confusable['literal_correct'] += a == b
            confusable['false_diameter'] += fb == 'diameter_mark'
    table = str.maketrans(POLICY['comparison_aliases'])
    return {'literal_exact': reference == prediction, 'comparison_alias_exact': reference.translate(table) == prediction.translate(table),
        'reference_characters': len(reference), 'edit_distance': raw['distance'], 'glyphs': glyphs,
        'confusions': confusions, 'confusables': confusable, 'unexpected_engineering_symbols': unexpected}


def validate_cases(cases):
    require(isinstance(cases, list) and 0 < len(cases) <= 128 and len({r['id'] for r in cases}) == len(cases), 'Invalid symbol case inventory')
    owners = {}; image_owners = {}
    for row in cases:
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', row['id']) and row['split'] in ('train', 'validation', 'development', 'synthetic'),
                'Invalid ID or closed-test scope')
        require(row['kind'] in ('text', 'nontext') and isinstance(row['reference'], str) and len(row['reference']) <= 128,
                'Invalid symbol transcription')
        require(row['annotations'] == annotate(row['reference']) and row['human_verified'] is False and
            row['training_eligible'] is False and row['physical_geometry_approved'] is False, 'Changed annotation authority')
        if row['kind'] == 'nontext': require(not row['reference'] and row['split'] == 'synthetic', 'Nontext case has a text target')
        if row['split'] in ('train', 'validation'):
            require(owners.setdefault(row['group_id'], row['split']) == row['split'] and
                image_owners.setdefault(row['image']['sha256'], row['split']) == row['split'], 'Cross-split group/image leakage')
        if 'symbol_box_in_source' in row:
            box, crop = row['symbol_box_in_source'], row['source_box']
            require(len(box) == len(crop) == 4 and all(type(v) is int for v in box+crop) and
                crop[0] <= box[0] < box[2] <= crop[2] and crop[1] <= box[1] < box[3] <= crop[3], 'Invalid symbol ROI')


def summarize(cases, predictions):
    validate_cases(cases)
    require(set(predictions) == set(ARMS), 'Incomplete OCR arms')
    output = {}
    for arm in ARMS:
        rows = predictions[arm]; by = {r['id']: r for r in rows}
        require(len(by) == len(rows) and set(by) == {r['id'] for r in cases}, 'Missing/duplicate OCR cases')
        details = []
        for case in cases:
            p = by[case['id']]
            require(p['status'] in ('ok', 'empty') and isinstance(p['prediction'], str) and
                (p['status'] == 'empty') == (not p['prediction']), 'Invalid prediction status')
            details.append({'id': case['id'], 'split': case['split'], 'kind': case['kind'],
                'image_sha256': case['image']['sha256'], 'reference': case['reference'],
                'prediction': p['prediction'], 'status': p['status'], **compare(case['reference'], p['prediction'])})
        groups = {}
        for split in ('train', 'validation', 'development', 'synthetic'):
            group = [r for r in details if r['split'] == split]; text = [r for r in group if r['kind'] == 'text']
            glyphs = {name: {k: sum(r['glyphs'][name][k] for r in text) for k in ('assigned', 'literal_correct', 'family_correct', 'deleted')}
                for name in FAMILIES}
            representatives = {}; [representatives.setdefault(r['image_sha256'], r) for r in text]
            groups[split] = {'cases': len(group), 'text_cases': len(text), 'literal_exact': sum(r['literal_exact'] for r in text),
                'comparison_alias_exact': sum(r['comparison_alias_exact'] for r in text),
                'empty_predictions': sum(r['status'] == 'empty' for r in group),
                'reference_characters': sum(r['reference_characters'] for r in text), 'edit_distance': sum(r['edit_distance'] for r in text),
                'glyphs': glyphs, 'unique_text_images': len(representatives),
                'unique_image_literal_exact': sum(r['literal_exact'] for r in representatives.values()),
                'confusables': {k: sum(r['confusables'][k] for r in group) for k in ('assigned', 'literal_correct', 'false_diameter')},
                'nontext_text_hallucinations': sum(r['kind'] == 'nontext' and bool(r['prediction']) for r in group),
                'false_symbol_cases': sum(r['unexpected_engineering_symbols'] > 0 for r in group),
                'unexpected_engineering_symbols': sum(r['unexpected_engineering_symbols'] for r in group)}
        output[arm] = {'by_split': groups, 'cases': details}
    return output


def glyph_coverage(polygon, box):
    contour = np.asarray(polygon, np.float32)
    require(contour.shape == (4, 2) and np.isfinite(contour).all(), 'Invalid detector polygon')
    x0, y0, x1, y1 = box
    reference = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    area, _ = cv2.intersectConvexConvex(cv2.convexHull(contour), reference)
    return float(max(0, min(1, area/((x1-x0)*(y1-y0)))))


def jpeg_localization(cases, observations):
    result = {}
    for arm in ('page_native', 'page_2x'):
        rows = []
        for case in cases:
            if 'symbol_box_in_source' not in case: continue
            scored = [(glyph_coverage(o['polygon_px'], case['symbol_box_in_source']), o) for o in observations[arm]]
            coverage, chosen = max(scored, key=lambda v: v[0], default=(0., None))
            raw = chosen['raw'] if chosen and coverage > 0 else ''
            rows.append({'id': case['id'], 'symbol_coverage': coverage,
                'located': coverage >= POLICY['localization_threshold'], 'overlapping_raw': raw,
                'reference': case['reference'], 'literal_exact': raw == case['reference'],
                'glyphs': compare(case['reference'], raw)['glyphs']})
        result[arm] = rows
    return result
