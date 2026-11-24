"""Read explicitly reviewed native facts without approving dimensions or labels.

Handles and coordinate snapshots are authored review inputs, never discovered
from a similarity score. Text values and selected contour topology are checked
again against the original document. No entity is changed.
"""
from __future__ import annotations

import math
import re


_FIELDS = frozenset(('designation', 'mass_g', 'strength_class', 'displayed_length', 'declared_scale'))
_NUMBER = r'\d+(?:[.,]\d+)?'
_EPS = 1e-7
_MAX_ENTITIES = 64


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    value = float(value.replace(',', '.'))
    _require(math.isfinite(value) and value > 0, 'Expected a positive finite native text value')
    return value


def _canonical_number(value):
    _number(value)
    integer, _, fraction = value.replace(',', '.').partition('.')
    integer, fraction = integer.lstrip('0') or '0', fraction.rstrip('0')
    return integer+('.'+fraction if fraction else '')


def parse_variant_text(field: str, plain: str) -> dict:
    """Parse a bounded reviewed transcription; never infer a missing value.

    Cyrillic М and Latin M are accepted as the visible thread-designation
    symbol. Decimal comma and point are accepted explicitly. All surrounding
    native text is retained by the caller as evidence.
    """
    _require(field in _FIELDS and isinstance(plain, str) and 0 < len(plain) <= 16384,
             'Unsupported field or native text size')
    value = plain.strip()
    if field == 'designation':
        match = re.fullmatch(r'\(([МM])('+_NUMBER+r')-('+_NUMBER+r')\)|([МM])('+_NUMBER+r')-('+_NUMBER+r')', value)
        _require(match is not None, 'Expected exactly one thread-length designation')
        diameter, length = (match[2], match[3]) if match[1] else (match[5], match[6])
        return {'designation': f'M{_canonical_number(diameter)}-{_canonical_number(length)}'}
    if field == 'mass_g':
        match = re.fullmatch('('+_NUMBER+r')\s*г', value)
        _require(match is not None, 'Expected a positive mass explicitly labelled in grams')
        return {'mass_g': _number(match[1])}
    if field == 'strength_class':
        matches = re.findall(r'класса\s+прочности\s+(\d+\.\d+)(?!\d|\.\d)', value, flags=re.IGNORECASE)
        _require(matches and len(set(matches)) == 1, 'Missing or contradictory strength-class statements')
        _require(_number(matches[0]) > 0, 'Invalid strength class')
        return {'strength_class': matches[0]}
    if field == 'displayed_length':
        match = re.fullmatch('('+_NUMBER+r')\s*±\s*('+_NUMBER+')', value)
        _require(match is not None, 'Expected a standalone nominal and explicit symmetric tolerance')
        return {'displayed_length': _number(match[1]), 'displayed_length_tolerance': _number(match[2])}
    match = re.fullmatch(r'([1-9]\d*)\s*:\s*([1-9]\d*)', value)
    _require(match is not None, 'Expected a sheet scale caption, not an inferred scale')
    _require(max(int(match[1]), int(match[2])) <= 10000, 'Sheet scale caption exceeds review limits')
    return {'declared_scale': f'{int(match[1])}:{int(match[2])}'}


def _xyz(value):
    values = [float(item) for item in value]
    _require(len(values) == 3 and all(math.isfinite(item) for item in values), 'Nonfinite native point')
    _require(abs(values[2]) <= _EPS, 'Only planar XY review geometry is supported')
    return values


def native_geometry_snapshot(entity) -> dict:
    """Capture native coordinates and spline parameters relevant to this review."""
    kind = entity.dxftype()
    result = {'native_handle': entity.dxf.handle, 'entity_type': kind, 'layer': entity.dxf.layer}
    if kind == 'LINE':
        result.update(start_xyz=_xyz(entity.dxf.start), end_xyz=_xyz(entity.dxf.end))
        result['length_drawing_units'] = float((entity.dxf.end-entity.dxf.start).magnitude)
        _require(math.isfinite(result['length_drawing_units']) and result['length_drawing_units'] > 0,
                 'Degenerate native contour line')
    elif kind == 'SPLINE':
        result.update(degree=int(entity.dxf.degree), flags=int(entity.dxf.flags),
                      control_points_xyz=[_xyz(point) for point in entity.control_points],
                      fit_points_xyz=[_xyz(point) for point in entity.fit_points],
                      knots=[float(value) for value in entity.knots], weights=[float(value) for value in entity.weights])
        _require(1 <= result['degree'] <= 5 and 2 <= len(result['control_points_xyz']) <= 1024
                 and len(result['fit_points_xyz']) <= 1024 and len(result['knots']) <= 2048,
                 'Spline geometry exceeds supported review scope')
        _require(all(math.isfinite(value) for key in ('knots', 'weights') for value in result[key]),
                 'Nonfinite spline parameters')
        for key in ('start_tangent', 'end_tangent'):
            result[key] = _xyz(entity.dxf.get(key)) if entity.dxf.hasattr(key) else None
    else:
        raise ValueError('Only selected LINE and SPLINE geometry can be reviewed')
    return result


def native_dimension_snapshot(entity) -> dict:
    """Record an explicitly named linear DIMENSION, without assigning its text."""
    _require(entity.dxftype() == 'DIMENSION' and int(entity.dxf.dimtype) & 15 in (0, 1),
             'Only linear DIMENSION observations are supported')
    measurement = float(entity.get_measurement())
    _require(math.isfinite(measurement) and measurement >= 0, 'Invalid DIMENSION measurement')
    return {'native_handle': entity.dxf.handle, 'entity_type': 'DIMENSION',
            'dimtype': int(entity.dxf.dimtype), 'layer': entity.dxf.layer,
            'definition_points': {key: _xyz(entity.dxf.get(key)) for key in
                ('defpoint', 'defpoint2', 'defpoint3', 'defpoint4', 'defpoint5') if entity.dxf.hasattr(key)},
            'text_override': entity.dxf.get('text', ''), 'geometry_block': entity.dxf.get('geometry'),
            'measurement_drawing_units': measurement,
            'standalone_text_attachment_verified': False}


def _native_entity(doc, handle, types):
    _require(isinstance(handle, str) and re.fullmatch('[0-9A-F]+', handle) is not None,
             'Expected a native uppercase DXF handle')
    entity = doc.entitydb.get(handle)
    _require(entity is not None and entity.is_alive and entity.dxftype() in types
             and entity.dxf.owner == doc.modelspace().block_record_handle
             and entity.origin_of_copy is None, 'Review entity must belong directly to native modelspace')
    _require(not entity.dxf.get('invisible', 0), 'Reviewed native entity is invisible')
    layer_name = entity.dxf.get('layer', '0')
    _require(layer_name in doc.layers, 'Reviewed entity layer is missing')
    layer = doc.layers.get(layer_name)
    _require(not layer.is_off() and not layer.is_frozen(), 'Reviewed entity is on a hidden layer')
    return entity


def _horizontal_groups(handles, snapshots):
    _require(isinstance(handles, list) and 1 <= len(handles) <= 8
             and len(set(handles)) == len(handles), 'Invalid selected contour chain')
    intervals, level = [], None
    for handle in handles:
        row = snapshots[handle]
        _require(row['entity_type'] == 'LINE', 'Contour edges must be native LINE entities')
        start, end = row['start_xyz'], row['end_xyz']
        _require(abs(start[1]-end[1]) <= _EPS and abs(start[0]-end[0]) > _EPS,
                 'Contour edge must be nondegenerate and horizontal')
        if level is None:
            level = start[1]
        _require(abs(start[1]-level) <= _EPS, 'One contour chain must share a native Y level')
        intervals.append([min(start[0], end[0]), max(start[0], end[0]), handle])
    intervals.sort()
    gaps = []
    for left, right in zip(intervals, intervals[1:]):
        _require(right[0] >= left[1]-_EPS, 'Contour intervals overlap ambiguously')
        if right[0]-left[1] > _EPS:
            gaps.append([left[1], right[0]])
    return {'native_y': level, 'intervals': intervals, 'gaps': gaps,
            'extent_x': [intervals[0][0], intervals[-1][1]]}


def _geometry(doc, spec):
    _require(isinstance(spec, dict) and spec.get('mode') in ('continuous', 'broken'),
             'Explicit reviewed contour mode is required')
    upper, lower, spline_handles = spec.get('upper_edges'), spec.get('lower_edges'), spec.get('break_splines')
    _require(all(isinstance(value, list) for value in (upper, lower, spline_handles)),
             'Explicit contour handles are required')
    handles = [*upper, *lower, *spline_handles]
    _require(2 <= len(handles) <= _MAX_ENTITIES and len(set(handles)) == len(handles),
             'Repeated or oversized contour selection')
    expected = spec.get('expected_entities')
    _require(isinstance(expected, dict) and set(expected) == set(handles), 'Missing or unused coordinate snapshots')
    snapshots = {handle: native_geometry_snapshot(_native_entity(doc, handle, ('LINE', 'SPLINE'))) for handle in handles}
    _require(snapshots == expected, 'Native geometry changed from its reviewed coordinate snapshot')
    top, bottom = _horizontal_groups(upper, snapshots), _horizontal_groups(lower, snapshots)
    _require(top['native_y'] > bottom['native_y']+_EPS, 'Upper and lower contour ordering differs')
    tolerance = spec.get('coordinate_join_tolerance', 0.0)
    _require(type(tolerance) in (int, float) and math.isfinite(tolerance) and 0 <= tolerance <= .01,
             'Native coordinate join tolerance must be explicitly bounded to 0.01 drawing unit')
    residuals = []
    if spec['mode'] == 'continuous':
        _require(not top['gaps'] and not bottom['gaps'] and not spline_handles,
                 'Continuous contour must have no selected break or open interval')
    else:
        _require(len(top['gaps']) == len(bottom['gaps']) == 1 and len(spline_handles) == 2,
                 'Broken contour requires exactly two edges of one gap and two splines')
        endpoints = []
        for handle in spline_handles:
            row = snapshots[handle]
            _require(row['entity_type'] == 'SPLINE' and row['flags'] & 3 == 0,
                     'Break boundaries must be open nonperiodic splines')
            points = row['fit_points_xyz'] or row['control_points_xyz']
            start, end = sorted((points[0], points[-1]), key=lambda point: point[1], reverse=True)
            endpoints.append((start[0]+end[0], handle, start, end))
        for side, (_, handle, start, end) in enumerate(sorted(endpoints)):
            target_top = [top['gaps'][0][side], top['native_y']]
            target_bottom = [bottom['gaps'][0][side], bottom['native_y']]
            distances = [math.dist(point[:2], target) for point, target in
                         ((start, target_top), (end, target_bottom))]
            _require(max(distances) <= tolerance+_EPS, 'Selected spline does not border the reviewed shaft gap')
            residuals.append({'handle': handle, 'endpoint_residuals_drawing_units': distances})
    return {'mode': spec['mode'], 'upper_chain': top, 'lower_chain': bottom,
            'entities': snapshots, 'spline_endpoint_residuals': residuals,
            'coordinate_join_tolerance_drawing_units': tolerance,
            'cross_edge_extent_differences_drawing_units': [abs(a-b) for a, b in zip(top['extent_x'], bottom['extent_x'])],
            'units': 'native drawing units', 'physical_length_approved': False}


def extract_variant_facts(doc, spec: dict) -> dict:
    """Validate an authored selection and return native facts plus qualifications."""
    _require(isinstance(spec, dict), 'Expected an authored native variant review specification')
    text_handles = spec.get('text_handles')
    _require(isinstance(text_handles, dict) and set(text_handles) == _FIELDS,
             'All five explicit text fields are required')
    _require(all(isinstance(record, dict) for record in text_handles.values()), 'Invalid text handle record')
    _require(len({record.get('handle') for record in text_handles.values()}) == len(_FIELDS),
             'Distinct native entities are required for different reviewed text fields')
    facts, evidence = {}, {}
    for field, record in text_handles.items():
        _require(isinstance(record, dict), 'Invalid text handle record')
        entity = _native_entity(doc, record.get('handle'), ('TEXT', 'MTEXT'))
        plain = entity.plain_text() if entity.dxftype() == 'MTEXT' else entity.dxf.text
        _require(plain == record.get('expected_plain'), 'Native text differs from its reviewed exact transcription')
        parsed = parse_variant_text(field, plain)
        facts.update(parsed)
        evidence[field] = {'native_handle': entity.dxf.handle, 'entity_type': entity.dxftype(),
            'owner_handle': entity.dxf.owner, 'plain_text': plain,
            'raw_text': entity.text if entity.dxftype() == 'MTEXT' else entity.dxf.text,
            'insert_xyz': _xyz(entity.dxf.insert), 'parsed': parsed}
    geometry = _geometry(doc, spec.get('geometry'))
    dimensions = spec.get('dimension_handles', {})
    _require(isinstance(dimensions, dict) and len(dimensions) <= 16, 'Invalid dimension observation selection')
    dimension_evidence = {}
    for handle, expected in dimensions.items():
        snapshot = native_dimension_snapshot(_native_entity(doc, handle, ('DIMENSION',)))
        _require(snapshot == expected, 'Native DIMENSION differs from its reviewed snapshot')
        dimension_evidence[handle] = snapshot
    geometry['dimension_observations'] = dimension_evidence
    facts['shaft_break'] = spec['geometry']['mode'] == 'broken'
    scale = facts.pop('declared_scale')
    nominal = float(facts['designation'].rsplit('-', 1)[1])
    conflicts = []
    if not math.isclose(nominal, facts['displayed_length'], rel_tol=0, abs_tol=1e-9):
        conflicts.append({'code': 'designation_length_disagrees_with_displayed_nominal',
            'designation_nominal': nominal, 'displayed_nominal': facts['displayed_length'],
            'native_handles': [text_handles['designation']['handle'], text_handles['displayed_length']['handle']],
            'correction_applied': False})
    qualifications = [
        {'code': 'native_units_not_physical_label_approval', 'insunits': int(doc.units)},
        {'code': 'declared_sheet_scale_not_applied', 'caption': scale},
        {'code': 'standalone_length_text_attachment_unverified', 'dimension_attachment_inferred': False},
        {'code': 'authored_selected_contour_not_global_geometry_approval',
         'cross_edge_extent_differences_drawing_units': geometry['cross_edge_extent_differences_drawing_units'],
         'spline_endpoint_residuals': geometry['spline_endpoint_residuals'],
         'coordinate_join_tolerance_drawing_units': geometry['coordinate_join_tolerance_drawing_units']},
    ]
    return {'facts': facts, 'text_evidence': evidence, 'geometry_evidence': geometry,
            'native_conflicts': conflicts, 'qualifications': qualifications,
            'source_modified': False, 'annotation_approved': False, 'training_eligible': False}
