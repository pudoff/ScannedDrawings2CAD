"""Pixel-only diagnostics on assigned development references, never CAD accuracy."""
from collections import Counter, defaultdict
import math
import re

import numpy as np

from .family_catalog_v2 import require
from .ocr_baseline_metrics_v1 import normalize_text_v1, score_text_v1

POLICY = {'text_reference_coverage_min': .5, 'text_prediction_area_ratio_max': 4.,
          'reference_box_policy': 'Erode the manual review crop by its declared pixel uncertainty for text-presence matching; area cap uses full crop.',
          'matching': 'greedy descending IoU, reference coverage, reference ID, prediction ID; no text/confidence oracle',
          'stroke_direction_tolerance_deg': 12., 'stroke_sample_spacing_px': .5,
          'stroke_coverage_min': .6, 'normalization': 'ocr-development-comparison/1.0',
          'global_precision': 'not identifiable outside assigned reference scope',
          'rec_only': 'not measurable from saved det+rec observations'}


def valid_box(box, size):
    return (isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
            and 0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1])


def points(value, size, minimum=1):
    require(isinstance(value, list) and len(value) >= minimum, 'Missing pixel points')
    require(all(isinstance(p, list) and len(p) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in p)
                and 0 <= p[0] < size[0] and 0 <= p[1] < size[1] for p in value), 'Pixel point outside original image')


def validate_references(ref):
    require(ref['schema_version'] == 'hard-raster-pixel-references/1.0' and ref['task'] == 'P15.2a'
            and ref['reviewer'] == 'assistant', 'Unknown pixel review')
    require(all(ref[f] is False for f in ('human_verified', 'training_eligible', 'independent_evaluation',
                'physical_geometry_approved', 'coordinate_regression_approved')), 'Pixel observations cannot approve training or physical geometry')
    require(ref['cohort'] == 'viewed_development_regression_only', 'Viewed example cannot become a closed test')
    size = ref['source_size_px']
    require(len(size) == 2 and all(type(v) is int and 0 < v <= 20000 for v in size), 'Invalid source size')
    for key in ('texts', 'symbols', 'strokes', 'arrows', 'links'):
        require(len({r['id'] for r in ref[key]}) == len(ref[key]), 'Repeated reference ID: '+key)
        require(all(isinstance(r['id'], str) and re.fullmatch(r'[a-z][a-z0-9-]{0,79}', r['id']) for r in ref[key]), 'Invalid reference ID')
    texts = {r['id']: r for r in ref['texts']}
    require(any(r['state'] == 'readable' for r in texts.values()), 'No readable evaluation fields')
    for row in texts.values():
        require(valid_box(row['box'], size) and row['state'] in ('readable', 'partial', 'unknown'), 'Invalid text reference')
        require(type(row['rotation_cw']) is int and row['rotation_cw'] in (0, 90, 180, 270), 'Invalid query rotation')
        require((row['state'] == 'readable' and isinstance(row['transcription'], str) and row['transcription'].strip())
                or (row['state'] != 'readable' and row['transcription'] is None), 'Unknown text cannot become an exact target')
        require(row['quantity_kind'] in (None, 'linear', 'diameter', 'angular'), 'Unsupported quantity kind')
        require(row['physical_value'] is None and row['physical_unit'] is None, 'Printed text is not a physical target')
        require(type(row['pixel_tolerance']) is int and 0 <= row['pixel_tolerance'] <= 3, 'Invalid text uncertainty')
    for row in ref['symbols']:
        require(row['text_id'] in texts and texts[row['text_id']]['state'] == 'readable'
                and valid_box(row['box'], size), 'Invalid symbol reference')
        b, parent = row['box'], texts[row['text_id']]['box']
        require(parent[0] <= b[0] < b[2] <= parent[2] and parent[1] <= b[1] < b[3] <= parent[3], 'Symbol outside parent text')
        require(row['glyph'] in texts[row['text_id']]['transcription'], 'Symbol not in literal transcription')
    strokes = {r['id']: r for r in ref['strokes']}
    for row in strokes.values():
        points(row['points'], size, 2)
        require(row['role'] in ('part_contour', 'axis_or_construction', 'sheet_frame', 'title_grid',
            'dimension_stroke', 'leader_stroke', 'dimension_arc_guide', 'fold_artifact'), 'Unknown stroke role')
        require(row['state'] in ('visible', 'partly_occluded') and type(row['band_px']) in (int, float)
                and 0 < row['band_px'] <= 5, 'Invalid stroke uncertainty')
    for row in ref['arrows']:
        points([row['tip']], size)
        require(type(row['radius_px']) in (int, float) and 0 < row['radius_px'] <= 10, 'Invalid arrow interval')
    require(len({r['text_id'] for r in ref['links']}) == len(ref['links']), 'Repeated display-link assignment')
    for row in ref['links']:
        require(row['text_id'] in texts and row['stroke_id'] in strokes, 'Display link references missing objects')
        require(all(row[k] is None for k in ('physical_geometry', 'physical_unit', 'physical_scale')), 'Display links cannot approve CAD anchors')
        if row['pixel_supports'] is not None:
            points(row['pixel_supports'], size, 2)


def bounds(polygon):
    a = np.asarray(polygon, dtype=float)
    require(a.ndim == 2 and a.shape[1] == 2 and len(a) >= 2 and np.isfinite(a).all(), 'Invalid prediction polygon')
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def overlap(reference, prediction):
    area = lambda b: max(0., b[2]-b[0])*max(0., b[3]-b[1])
    intersection = area([max(reference[0], prediction[0]), max(reference[1], prediction[1]),
                         min(reference[2], prediction[2]), min(reference[3], prediction[3])])
    ra, pa = area(reference), area(prediction)
    require(ra > 0 and pa > 0, 'Degenerate matching box')
    return intersection/ra, intersection/(ra+pa-intersection), pa/ra


def match_texts(texts, lines):
    require(len({r['id'] for r in lines}) == len(lines), 'Repeated OCR line ID')
    options = []
    for row in texts:
        if row['state'] != 'readable':
            continue
        for line in lines:
            b = row['box']; margin = row.get('pixel_tolerance', 0)
            dx, dy = min(margin, (b[2]-b[0]-1)/2), min(margin, (b[3]-b[1]-1)/2)
            core = [b[0]+dx, b[1]+dy, b[2]-dx, b[3]-dy]
            prediction_box = bounds(line['polygon_input_px'])
            coverage, _, _ = overlap(core, prediction_box)
            _, iou, ratio = overlap(b, prediction_box)
            if coverage >= POLICY['text_reference_coverage_min'] and ratio <= POLICY['text_prediction_area_ratio_max']:
                options.append((-iou, -coverage, row['id'], line['id']))
    assigned, used = {}, set()
    for _, _, reference, prediction in sorted(options):
        if reference not in assigned and prediction not in used:
            assigned[reference] = prediction
            used.add(prediction)
    duplicates = sorted({p for _, _, r, p in options if r in assigned and p not in used})
    return assigned, duplicates


def transform_points(values, matrix):
    m = np.asarray(matrix, dtype=float)
    require(m.shape == (3, 3) and np.isfinite(m).all() and np.array_equal(m[2], [0, 0, 1])
            and abs(np.linalg.det(m)) > 1e-12, 'Invalid fixed pixel transform')
    a = np.asarray(values, dtype=float)
    require(a.ndim == 2 and a.shape[1] == 2 and np.isfinite(a).all(), 'Invalid predicted points')
    return (np.column_stack((a, np.ones(len(a)))) @ m.T)[:, :2]


def stroke_coverage(guide, candidates, matrix):
    """Fraction of a sampled corridor supported by finite, similarly directed lines."""
    segments = [(c['id'], transform_points(c['points_px'], matrix)) for c in candidates if c['kind'] == 'line']
    supported = total = 0
    contributors = set()
    for left, right in zip(guide['points'], guide['points'][1:]):
        a, b = np.asarray(left, float), np.asarray(right, float)
        delta = b-a; length = float(np.linalg.norm(delta))
        require(length > 0, 'Degenerate manual stroke segment')
        count = max(2, math.ceil(length/POLICY['stroke_sample_spacing_px']))
        samples = a+(np.arange(count)+.5)[:, None]/count*delta
        covered = np.zeros(count, bool)
        for identifier, line in segments:
            if line.shape != (2, 2): continue
            direction = line[1]-line[0]; norm = float(np.linalg.norm(direction))
            if norm <= 1e-9 or abs(float(delta @ direction))/(length*norm) < math.cos(math.radians(POLICY['stroke_direction_tolerance_deg'])):
                continue
            t = (samples-line[0]) @ direction/(norm*norm)
            projected = line[0]+np.clip(t, 0, 1)[:, None]*direction
            matches = np.linalg.norm(samples-projected, axis=1) <= guide['band_px']
            if matches.any(): contributors.add(identifier)
            covered |= matches
        supported += int(covered.sum()); total += count
    return {'id': guide['id'], 'role': guide['role'], 'state': guide['state'],
            'coverage': supported/total, 'covered': supported/total >= POLICY['stroke_coverage_min'],
            'candidate_ids': sorted(contributors), 'primitive_accuracy': None}


def score_saved(ref, ocr, geometry, arrows, associations, graph, convert_counts):
    validate_references(ref)
    assigned, duplicates = match_texts(ref['texts'], ocr['lines'])
    by_id = {r['id']: r for r in ocr['lines']}
    notations = {r['source_line_id']: r['parsed']['kind'] for r in ocr['notations']}
    rows = []
    for field in ref['texts']:
        if field['state'] != 'readable': continue
        line = by_id.get(assigned.get(field['id']))
        prediction = line['raw'] if line else ''
        raw = score_text_v1(field['transcription'], prediction)
        normalized = score_text_v1(normalize_text_v1(field['transcription']), normalize_text_v1(prediction))
        kind = notations.get(line['id']) if line else None
        interpreted_kind = 'linear' if kind == 'nominal' else kind
        rows.append({'id': field['id'], 'line_id': line['id'] if line else None, 'detected': line is not None,
            'reference': field['transcription'], 'prediction': prediction, 'raw_exact': raw['exact_match'],
            'normalized_exact': normalized['exact_match'], 'edit_distance': raw['distance'],
            'reference_characters': raw['reference_characters'], 'expected_quantity_kind': field['quantity_kind'],
            'candidate_quantity_kind': interpreted_kind,
            'quantity_kind_correct': interpreted_kind == field['quantity_kind'] if field['quantity_kind'] else None})
    row_index = {r['id']: r for r in rows}
    symbols = [{**s, 'parent_detected': row_index[s['text_id']]['detected'],
                'recognized_in_associated_transcription': normalize_text_v1(s['glyph']) in normalize_text_v1(row_index[s['text_id']]['prediction'])}
               for s in ref['symbols']]
    strokes = [stroke_coverage(s, geometry['candidates'], ocr['processed_to_input']) for s in ref['strokes']]
    roles = {}
    for role in sorted({s['role'] for s in strokes}):
        subset = [s for s in strokes if s['role'] == role and s['state'] == 'visible']
        roles[role] = {'visible_guides': len(subset), 'covered_guides': sum(s['covered'] for s in subset),
                       'mean_corridor_coverage': sum(s['coverage'] for s in subset)/len(subset) if subset else None}
    arrow_rows = []
    for target in ref['arrows']:
        found = []
        for candidate in arrows['candidates']:
            point = transform_points([candidate['tip_px']], ocr['processed_to_input'])[0]
            if float(np.linalg.norm(point-np.asarray(target['tip']))) <= target['radius_px']:
                found.append(candidate['id'])
        arrow_rows.append({'id': target['id'], 'candidate_ids': found, 'covered': bool(found)})
    title = ocr['title_block']['fields']
    title_rows = []
    for key, reference_id in (('name', 'title-name'), ('document_number', 'title-designation')):
        field = title.get(key, {})
        title_rows.append({'field': key, 'reference': row_index[reference_id]['reference'],
            'value': field.get('value'), 'status': field.get('status', 'unknown'),
            'exact': field.get('value') == row_index[reference_id]['reference']})
    dimensions = [r for r in rows if r['expected_quantity_kind']]
    conditional = [r for r in rows if r['detected']]
    per_symbol = {kind: {'assigned': sum(s['kind'] == kind for s in symbols),
                        'recognized': sum(s['kind'] == kind and s['recognized_in_associated_transcription'] for s in symbols)}
                  for kind in sorted({s['kind'] for s in symbols})}
    summary = {'assigned_readable_fields': len(rows), 'partial_fields': sum(r['state'] == 'partial' for r in ref['texts']),
        'unknown_fields': sum(r['state'] == 'unknown' for r in ref['texts']), 'detected_fields': len(conditional),
        'missed_field_ids': [r['id'] for r in rows if not r['detected']],
        'raw_exact': sum(r['raw_exact'] for r in rows), 'normalized_exact': sum(r['normalized_exact'] for r in rows),
        'end_to_end_cer': sum(r['edit_distance'] for r in rows)/sum(r['reference_characters'] for r in rows),
        'conditional_detected_field_cer': sum(r['edit_distance'] for r in conditional)/sum(r['reference_characters'] for r in conditional) if conditional else None,
        'rec_only': None, 'duplicate_observation_ids': duplicates,
        'unscored_observation_ids': sorted(set(by_id)-set(assigned.values())-set(duplicates)), 'global_precision': None,
        'dimension_fields': len(dimensions), 'quantity_kind_correct': sum(r['quantity_kind_correct'] for r in dimensions),
        'wrong_quantity_candidate_ids': [r['id'] for r in dimensions if r['detected'] and r['candidate_quantity_kind'] is not None and not r['quantity_kind_correct']],
        'symbols_by_kind': per_symbol, 'title_fields': title_rows, 'stroke_roles': roles,
        'assigned_arrow_tips': len(arrow_rows), 'covered_arrow_tips': sum(r['covered'] for r in arrow_rows),
        'geometry_candidate_count': len(geometry['candidates']),
        'unclassified_geometry_candidates': sum(c['classification'] == 'unclassified' for c in geometry['candidates']),
        'assigned_display_links': len(ref['links']), 'proposed_associations': associations['summary']['proposed'],
        'accepted_graph_dimensions': len(graph['dimensions']), 'accepted_graph_primitives': len(graph['primitives']),
        'dxf_modelspace_counts': convert_counts, 'geometric_attachment_accuracy': None, 'physical_accuracy': None,
        'geometry_search_incomplete': bool(geometry['analysis']['hough_circle_limit_reached']),
        'manual_review_required': True, 'walltime_seconds': None, 'peak_rss_MiB': None}
    return {'schema_version': 'hard-raster-saved-observation-score/1.0', 'summary': summary, 'text_rows': rows,
        'symbols': symbols, 'strokes': strokes, 'arrow_rows': arrow_rows, 'policy': POLICY,
        'notes': ['Assigned partial page scope; unassigned predictions are not all false positives.',
                  'Conditional detected-field CER is not rec-only; missed fields are deletions only in end-to-end CER.',
                  'Corridor coverage is not primitive shape, contour closure, CAD anchor or physical accuracy.',
                  'Only frozen historical predictions are scored; no model was executed or tuned.']}
