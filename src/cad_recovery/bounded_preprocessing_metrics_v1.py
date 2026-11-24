"""Fixed pixel transforms and paired decisions; no parameter search or CAD approval."""
from copy import deepcopy

import cv2
import numpy as np

from .family_catalog_v2 import require
from .hard_raster_metrics_v1 import transform_points
from .hard_raster_runs_v1 import text_scores, summarize
from .ocr_baseline_metrics_v1 import normalize_text_v1, score_text_v1

ARMS = ('original', 'upscale-2x', 'local-contrast')
PIXEL_BUDGET = 25_000_000


def variant_metadata(size, arm):
    require(arm in ARMS and len(size) == 2 and all(type(x) is int and x > 0 for x in size), 'Invalid variant')
    width, height = size; scale = 2 if arm == 'upscale-2x' else 1
    matrix = [[scale, 0, (scale-1)/2], [0, scale, (scale-1)/2], [0, 0, 1]]
    inverse = np.linalg.inv(np.array(matrix, dtype=float)).tolist()
    return {'arm': arm, 'source_size_px': size, 'derivative_size_px': [width*scale, height*scale],
            'status': 'pixel_budget_refused' if width*height*scale**2 > PIXEL_BUDGET else 'ready',
            'pixel_budget': PIXEL_BUDGET, 'source_to_derivative': matrix, 'derivative_to_source': inverse,
            'physical_scale_confirmed': False, 'silent_scale_reduction': False}


def derivative(rgb, arm):
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, 'Expected uint8 RGB')
    height, width = rgb.shape[:2]; meta = variant_metadata([width, height], arm)
    require(width*height <= PIXEL_BUDGET, 'Original exceeds frozen pixel budget')
    if meta['status'] != 'ready': return None, meta
    if arm == 'original': output = rgb.copy()
    elif arm == 'upscale-2x': output = cv2.resize(rgb, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
    else:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        output = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return output, meta


def derivative_box(box, arm):
    require(arm in ARMS and len(box) == 4 and all(type(x) is int for x in box)
            and 0 <= box[0] < box[2] and 0 <= box[1] < box[3], 'Invalid crop')
    # These are exclusive pixel edges; the center transform has +0.5 at scale 2.
    return [v*(2 if arm == 'upscale-2x' else 1) for v in box]


def original_frame(ocr, metadata):
    if ocr is None: return None
    result = deepcopy(ocr); inverse = metadata['derivative_to_source']
    for line in result['lines']:
        line['polygon_input_px'] = transform_points(line['polygon_input_px'], inverse).tolist()
    result['processed_to_input'] = (np.array(inverse)@np.array(ocr['processed_to_input'])).tolist()
    return result


def recognition_rows(cases, results, statuses):
    require(len(results) == len({r['id'] for r in results}), 'Repeated recognition output')
    expected = {c['id']+'--'+f['id'] for c in cases for f in c['texts'] if f['state'] == 'readable'}
    require({r['id'] for r in results}.issubset(expected), 'Foreign recognition output')
    require(all(statuses[c['id']] in ('ok', 'pixel_budget_refused', 'process_failed') for c in cases), 'Invalid recognition status')
    allowed = {c['id']+'--'+f['id'] for c in cases if statuses[c['id']] == 'ok' for f in c['texts'] if f['state'] == 'readable'}
    require({r['id'] for r in results} == allowed, 'Recognition crossed a refusal boundary or lost assigned lines')
    by_id = {r['id']: r for r in results}; rows = []
    for case in cases:
        for field in case['texts']:
            if field['state'] != 'readable': continue
            ident = case['id']+'--'+field['id']; observed = by_id.get(ident)
            status = statuses[case['id']]
            if status == 'ok': require(observed is not None, 'Recognition lost assigned line')
            prediction = observed['prediction'] if observed and status == 'ok' else ''
            raw = score_text_v1(field['transcription'], prediction)
            norm = score_text_v1(normalize_text_v1(field['transcription']), normalize_text_v1(prediction))
            rows.append({'case_id': case['id'], 'id': field['id'], 'reference': field['transcription'],
                'prediction': prediction, 'status': observed['status'] if observed and status == 'ok' else status,
                'raw_exact': raw['exact_match'], 'normalized_exact': norm['exact_match'],
                'distance': raw['distance'], 'characters': raw['reference_characters']})
    return rows


def aggregate(rows, recognition=False):
    result = summarize(rows)
    if recognition: result['detected'] = None
    return result


def compare_candidate(baseline, candidate, complete, budgets_pass):
    """Frozen validation decision on paired observations; per-field sign regressions."""
    base = {r['id']: r for r in baseline['rows']}; other = {r['id']: r for r in candidate['rows']}
    require(set(base) == set(other) and len(base) == len(baseline['rows']) == len(candidate['rows']), 'Unpaired validation fields')
    require(all(base[k]['reference'] == other[k]['reference'] for k in base), 'Validation references changed')
    a, b = aggregate(baseline['rows']), aggregate(candidate['rows'])
    signs_a = {(r['field_id'], r['glyph']) for r in baseline['symbols'] if r['recognized']}
    signs_b = {(r['field_id'], r['glyph']) for r in candidate['symbols'] if r['recognized']}
    regressions = sorted(signs_a-signs_b)
    reasons = []
    if not base: reasons.append('no_validation_fields')
    if b['raw_exact'] <= a['raw_exact']: reasons.append('literal_exact_not_strictly_better')
    if b['detected'] < a['detected']: reasons.append('more_missed_fields')
    if regressions: reasons.append('lost_previously_retained_signs')
    if not complete: reasons.append('assigned_validation_incomplete')
    if not budgets_pass: reasons.append('process_budget_or_measurement_failed')
    return {'eligible_for_followup': not reasons, 'reasons': reasons, 'baseline': a, 'candidate': b,
            'lost_signs': [list(x) for x in regressions],
            'repaired_fields': sorted(k for k in base if not base[k]['raw_exact'] and other[k]['raw_exact']),
            'regressed_fields': sorted(k for k in base if base[k]['raw_exact'] and not other[k]['raw_exact']),
            'deployment_changed': False}


def scoped_text(tasks, case_ids):
    rows = []; symbols = []
    for task in tasks:
        if task['case_id'] not in case_ids or task['text'] is None: continue
        for row in task['text']['rows']: rows.append({**row, 'id': task['case_id']+'--'+row['id']})
        for sign in task['text']['symbols']: symbols.append({**sign, 'field_id': task['case_id']+'--'+sign['field_id']})
    return {'rows': rows, 'symbols': symbols}
