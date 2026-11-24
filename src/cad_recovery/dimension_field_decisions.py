"""Separate reviewed drawing text, display contacts and measured CAD supports."""
from copy import deepcopy
from itertools import permutations
import math
import re

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_annotation(text):
    """Compare explicit visual glyph readings without deleting their semantics."""
    require(isinstance(text, str) and 0 < len(text) <= 4096, 'Expected bounded annotation text')
    value = ''.join(text.split()).replace('⌀', 'Ø').replace('∅', 'Ø').replace('ø', 'Ø')
    value = re.sub(r'(?<=\d),(?=\d)', '.', value)
    value = value.replace('–', '-').replace('−', '-')
    value = re.sub(r'^М(?=\d)', 'M', value)
    value = re.sub(r'^(M\d+(?:\.\d+)?-\d+)Н', r'\1H', value)
    value = re.sub(r'(?<=\d)[xхXХ](?=\d)', '×', value)
    # A semicolon marks the reviewer's separate count line, not a digit separator.
    if ';' in value:
        require(re.fullmatch(r'[^;]+;\d+(?:фаски|отв\.?|рад\.?)', value) is not None,
                'Unsupported annotation line separator')
    value = re.sub(r'(отв|рад)\.$', r'\1', value)
    return value


def join_native_display(pieces):
    require(isinstance(pieces, list) and 0 < len(pieces) <= 100
            and all(isinstance(p, str) and p for p in pieces), 'Invalid native display pieces')
    last = ''.join(pieces[-1].split())
    if len(pieces) > 1 and re.fullmatch(r'\d+(?:фаски|отв\.?|рад\.?)', last):
        return ''.join(pieces[:-1])+';'+pieces[-1]
    return ''.join(pieces)


def parse_reviewed_annotation(text):
    """Interpret the small declared review vocabulary; preserve modifiers."""
    value = canonical_annotation(text)
    number = r'(\d+(?:\.\d+)?)'
    result = None
    for pattern, kind in ((number+r'(\*)?', 'linear'),
                          ('Ø'+number+r'([A-Za-z]+\d+)?(?:;(\d+)отв)?', 'diameter'),
                          ('R'+number+r'(?:;(\d+)рад)?', 'radius'),
                          (number+'×'+number+r'°(?:;(\d+)фаски)?', 'chamfer'),
                          ('M'+number+r'-(\d+H)(?:;(\d+)отв)?', 'thread')):
        match = re.fullmatch(pattern, value)
        if match is None:
            continue
        nominal = float(match[1])
        require(math.isfinite(nominal) and nominal > 0, 'Annotation nominal must be finite and positive')
        result = {'kind': kind, 'nominal': nominal, 'normalized_text': value,
                  'nominal_units': 'printed_value_not_validated_physical_size'}
        if kind == 'linear':
            result['reference_only'] = bool(match[2])
        elif kind == 'diameter':
            result.update(fit=match[2], count=int(match[3]) if match[3] else None)
        elif kind == 'radius':
            result['count'] = int(match[2]) if match[2] else None
        elif kind == 'chamfer':
            angle = float(match[2])
            require(math.isfinite(angle) and 0 < angle < 180, 'Invalid chamfer angle')
            result.update(angle_deg=angle, count=int(match[3]) if match[3] else None)
        else:
            result.update(thread_tolerance_class=match[2], count=int(match[3]) if match[3] else None)
        if result.get('count') is not None:
            require(result['count'] > 0, 'Annotation count must be positive')
        break
    require(result is not None, 'Annotation is outside the explicit field-review vocabulary')
    return result


def compare_display_contacts(native, matrix, reading):
    """Compare arrow contacts, never promote them into geometry anchor truth."""
    matrix = np.asarray(matrix, dtype=float)
    require(matrix.shape == (3, 3) and np.isfinite(matrix).all()
            and np.array_equal(matrix[2], [0, 0, 1])
            and abs(np.linalg.det(matrix[:2, :2])) > 1e-12, 'Invalid fixed CAD-to-source transform')
    points, kind = native['definition_points'], native['native_kind']
    def xy(name):
        point = np.asarray(points[name], dtype=float)
        require(point.shape == (3,) and np.isfinite(point).all() and abs(point[2]) <= 1e-6,
                'Display contact requires a finite planar definition point')
        return point[:2]
    if kind in ('linear', 'aligned'):
        a, b, location = xy('defpoint2'), xy('defpoint3'), xy('defpoint')
        if kind == 'aligned':
            direction = b-a
            require(np.linalg.norm(direction) > 1e-12, 'Degenerate aligned display contact')
            direction /= np.linalg.norm(direction)
        else:
            angle = native['angle_deg']
            require(type(angle) in (int, float) and math.isfinite(angle), 'Invalid display angle')
            direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        contacts = [location+np.dot(point-location, direction)*direction for point in (a, b)]
        keys = ['defpoint2_projected_to_dimension_line', 'defpoint3_projected_to_dimension_line']
    elif kind == 'radius':
        contacts, keys = [xy('defpoint4')], ['radius_arrow_contact']
    elif kind == 'diameter':
        contacts, keys = [xy('defpoint'), xy('defpoint4')], ['diameter_contact_1', 'diameter_contact_2']
    else:
        raise ValueError('Unsupported display contact kind')
    support = reading['author_support']
    expected_roles = {'radius': {'leader_contact_window', 'leader_curve_contact_window'},
                      'diameter': {'diameter_arrow_curve_contact_windows'}}.get(
        kind, {'dimension_arrow_contact_windows'})
    require(support['role'] in expected_roles
            and support['status'] == 'author_visual_annotation_not_expert_ground_truth',
            'Arrow windows cannot be reinterpreted as measured geometry anchors')
    windows = support['source_windows_px']
    require(isinstance(windows, list) and len(windows) == len(contacts), 'Display contact window count differs')
    for window in windows:
        require(isinstance(window, list) and len(window) == 4
                and all(type(x) in (int, float) and math.isfinite(x) for x in window)
                and 0 <= window[0] < window[2] and 0 <= window[1] < window[3], 'Invalid source contact window')
    projected = [(matrix @ [*point, 1])[:2].tolist() for point in contacts]
    assignments = [list(order) for order in permutations(range(len(windows))) if all(
        windows[j][0] <= point[0] <= windows[j][2] and windows[j][1] <= point[1] <= windows[j][3]
        for point, j in zip(projected, order))]
    return {'status': 'inside_authored_contact_windows' if len(assignments) == 1 else
                     'ambiguous_contact_window_assignment' if assignments else 'outside_authored_contact_windows',
            'contact_keys': keys, 'native_contacts_xy': [point.tolist() for point in contacts],
            'projected_source_xy': projected, 'source_windows_px': deepcopy(windows), 'assignments': assignments,
            'meaning': 'Displayed arrow/leader contacts only; not measured support coordinates or an accuracy certificate.',
            'measured_anchor_accuracy_approved': False, 'transform_refitted': False,
            'annotation_approved': False, 'training_eligible': False}


def decide_display_fields(native, reading, contact_review, caption, *, broken_length_view=False):
    """Report separate text, syntax, factor and numeric observations."""
    require(type(broken_length_view) is bool, 'Break qualification must be explicit')
    interpreted = parse_reviewed_annotation(reading['text'])
    native_text = canonical_annotation(join_native_display(native['display_text_pieces']))
    text_matches = interpreted['normalized_text'] == native_text
    contact_supported = contact_review['status'] == 'inside_authored_contact_windows'
    override = native['normalized_override_matches_single_display']
    text_status = ('supported_observation' if text_matches and contact_supported and override is not False else
                   'qualified_observation' if text_matches and override is not False else 'conflict')
    match = re.fullmatch(r'(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)', caption)
    require(match is not None and float(match[1]) > 0 and float(match[2]) > 0, 'Invalid reviewed scale caption')
    expected_factor = float(match[2])/float(match[1])
    require(math.isfinite(expected_factor), 'Nonfinite caption ratio')
    raw, factor = native['measurement_raw'], native['dimlfac']
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (raw, factor)),
            'Invalid native measurement/factor')
    numeric = raw*factor
    require(math.isfinite(numeric), 'Nonfinite native display measurement')
    residual = abs(numeric-interpreted['nominal'])
    # Same numeric diagnostic tolerance as the retained semantic audit, never accuracy.
    numeric_status = 'agrees_numerically' if residual <= .005 else 'numeric_mismatch'
    if broken_length_view:
        numeric_status = 'broken_view_length_not_recoverable_from_drawn_span'
    return {'display_text': {'status': text_status, 'source_text': reading['text'],
                            'native_text_pieces': deepcopy(native['display_text_pieces']),
                            'normalized_source': interpreted['normalized_text'], 'normalized_native': native_text,
                            'source_matches_native': text_matches,
                            'reading_basis': 'explicit_independent_assistant_raster_transcription_not_OCR'},
            'notation': {'status': text_status, **interpreted},
            'view_scale': {'status': 'caption_and_native_display_factor_consistent' if abs(factor-expected_factor) <= 1e-12
                                    else 'caption_and_native_display_factor_differ',
                           'caption': caption, 'effective_dimlfac': factor, 'dimscale': native['dimscale'],
                           'physical_coordinate_factor': None, 'physical_scale_confirmed': False,
                           'factor_fitted_from_nominal': False, 'broken_length_view': broken_length_view},
            'numeric_consistency': {'status': numeric_status, 'native_display_measurement': numeric,
                'printed_nominal': interpreted['nominal'], 'absolute_residual': residual,
                'diagnostic_tolerance': .005, 'tolerance_is_physical_accuracy': False,
                'nominal_or_geometry_corrected': False},
            'physical_nominal': {'status': 'not_approved', 'value': None, 'unit': None},
            'physical_coordinate_accuracy': {'status': 'not_approved', 'accuracy_bound': None}}
