"""Validate explicit per-object CAD observations without approving training labels.

The reviewer supplies view membership, caption readings and contour links. This
module checks those assertions against native modelspace entities. It does not
infer part contours from layer names, perform OCR, or certify pixel locations.
"""
from __future__ import annotations

from collections import Counter
import math
import re

import numpy as np

CAD_TOLERANCE = 1e-6
MAX_NUMERIC_TOLERANCE_MM = 0.0005
MAX_MODELSPACE_ENTITIES = 200_000
KINDS = {0: 'linear', 1: 'aligned', 3: 'diameter', 4: 'radius'}
DECISIONS = {'supported_observation', 'qualified_observation', 'quarantine'}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


def _point(value, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} must be a finite XYZ point') from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite XYZ point')
    return result


def _planar(entity):
    extrusion = _point(entity.dxf.get('extrusion', (0, 0, 1)), 'extrusion')
    if not np.allclose(extrusion, (0, 0, 1), rtol=0, atol=1e-12):
        raise ValueError('Only the default XY entity plane is supported')


def _xy(value, name):
    point = _point(value, name)
    if abs(point[2]) > CAD_TOLERANCE:
        raise ValueError(f'{name} is outside the XY plane')
    return point[:2]


def _box(value, size):
    if (not isinstance(value, list) or len(value) != 4
            or any(isinstance(v, bool) or not isinstance(v, int) for v in value)):
        raise ValueError('View source_box_px must contain four integer bounds')
    x0, y0, x1, y1 = value
    if not 0 <= x0 < x1 <= size[0] or not 0 <= y0 < y1 <= size[1]:
        raise ValueError('View box must lie within the source raster')
    return value


def _simple_nominal(raw):
    match = re.fullmatch(r'\s*(?:(%%[cC]|[Øø⌀Rr])\s*)?([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+))\s*', raw)
    if match is None:
        return None, None
    value = float(match[2].replace(',', '.'))
    return (value, match[1]) if math.isfinite(value) else (None, None)


def _native_measurement(entity, points):
    kind = entity.dimtype
    if kind in (0, 1):
        delta = points['defpoint3'] - points['defpoint2']
        if kind == 1:
            return float(np.linalg.norm(delta))
        angle = math.radians(_number(entity.dxf.get('angle', 0.), 'Dimension angle'))
        return abs(float(np.dot(delta[:2], [math.cos(angle), math.sin(angle)])))
    if kind in (3, 4):
        return float(np.linalg.norm(points['defpoint4'] - points['defpoint']))
    return None


def _binding(entity, point, relation, spec):
    _planar(entity)
    kind = entity.dxftype()
    p = _xy(point, 'Definition point')
    evidence = {'geometry_handle': entity.dxf.handle, 'geometry_type': kind,
                'geometry_layer': entity.dxf.layer, 'relation': relation,
                'contour_role_source': 'explicit_reviewer_assertion',
                'exact_finite_geometry_attachment': True}
    if relation in ('endpoint', 'segment', 'supporting_line'):
        if kind != 'LINE':
            raise ValueError(f'{relation} binding requires LINE')
        a, b = _xy(entity.dxf.start, 'Line start'), _xy(entity.dxf.end, 'Line end')
        delta = b-a
        length2 = float(np.dot(delta, delta))
        if length2 <= CAD_TOLERANCE**2:
            raise ValueError('Degenerate line cannot support a binding')
        t = float(np.dot(p-a, delta) / length2)
        finite = a + np.clip(t, 0., 1.)*delta
        support = a+t*delta
        endpoint_distance = float(min(np.linalg.norm(p-a), np.linalg.norm(p-b)))
        finite_distance = float(np.linalg.norm(p-finite))
        distance = (endpoint_distance if relation == 'endpoint' else
                    finite_distance if relation == 'segment' else float(np.linalg.norm(p-support)))
        evidence.update(distance_cad=distance, distance_to_finite_segment_cad=finite_distance,
                        distance_to_nearest_endpoint_cad=endpoint_distance,
                        unclamped_segment_parameter=t, closest_finite_xy=finite.tolist(),
                        on_finite_segment=finite_distance <= CAD_TOLERANCE)
        if relation == 'supporting_line':
            evidence['exact_finite_geometry_attachment'] = False
            evidence['qualification'] = 'Infinite supporting line; not an asserted finite endpoint.'
    elif relation == 'vertex':
        if kind != 'LWPOLYLINE':
            raise ValueError('vertex binding requires LWPOLYLINE')
        elevation = _number(entity.dxf.get('elevation', 0.), 'Polyline elevation')
        if abs(elevation) > CAD_TOLERANCE:
            raise ValueError('Polyline lies outside the XY plane')
        index = spec.get('vertex_index')
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(entity):
            raise ValueError('Invalid polyline vertex_index')
        vertex = np.asarray(entity[index][:2], dtype=float)
        if not np.isfinite(vertex).all():
            raise ValueError('Nonfinite polyline vertex')
        distance = float(np.linalg.norm(p-vertex))
        evidence.update(vertex_index=index, vertex_xy=vertex.tolist(), distance_cad=distance)
    elif relation in ('circle_point', 'arc_point', 'center'):
        expected = ('CIRCLE',) if relation == 'circle_point' else ('ARC',) if relation == 'arc_point' else ('ARC', 'CIRCLE')
        if kind not in expected:
            raise ValueError(f'{relation} binding has incompatible geometry')
        center = _xy(entity.dxf.center, 'Curve center')
        radius = _number(entity.dxf.radius, 'Curve radius')
        if radius <= CAD_TOLERANCE:
            raise ValueError('Degenerate curve cannot support a binding')
        distance = (float(np.linalg.norm(p-center)) if relation == 'center'
                    else abs(float(np.linalg.norm(p-center))-radius))
        evidence.update(center_xy=center.tolist(), radius=radius, distance_cad=distance)
        if relation == 'arc_point':
            start = _number(entity.dxf.start_angle, 'Arc start angle')
            end = _number(entity.dxf.end_angle, 'Arc end angle')
            angle = math.degrees(math.atan2(*(p-center)[::-1])) % 360.
            sweep = (end-start) % 360.
            relative = (angle-start) % 360.
            if sweep < 1e-9 or (relative > sweep + 1e-7 and relative < 360.-1e-7):
                raise ValueError('Definition point is outside the finite arc sweep')
            evidence.update(angle_deg=angle, start_angle_deg=start, end_angle_deg=end)
    else:
        raise ValueError('Unknown geometry binding relation')
    if not math.isfinite(distance) or distance > CAD_TOLERANCE:
        raise ValueError(f'Geometry link does not match the definition point: {relation}')
    return evidence


def validate_object_review(doc, case: dict, cad_to_source_pixel, source_size_px) -> dict:
    """Check a complete explicit modelspace DIMENSION review against native CAD.

    This deliberately narrow implementation accepts millimeter DXFs and planar
    native linear/aligned/radius/diameter definitions. Nested dimensions and
    compound notations need a separate review. No decision approves labels or
    makes global registration into validated pixel geometry.
    """
    if not isinstance(case, dict) or case.get('schema_version') != '1.0' or case.get('reviewer') != 'assistant':
        raise ValueError('Expected schema 1.0 and explicit assistant reviewer')
    if case.get('expected_insunits') != 4 or doc.units != 4:
        raise ValueError('This review requires declared millimeter INSUNITS=4')
    if (not isinstance(source_size_px, (list, tuple)) or len(source_size_px) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 32768 for v in source_size_px)):
        raise ValueError('Invalid source raster size')
    try:
        matrix = np.asarray(cad_to_source_pixel, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid CAD to source matrix') from error
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[2], [0., 0., 1.]) or abs(np.linalg.det(matrix[:2, :2])) < 1e-12):
        raise ValueError('CAD to source matrix must be a finite invertible XY affine transform')
    modelspace = {}
    for entity in doc.modelspace():
        if len(modelspace) >= MAX_MODELSPACE_ENTITIES:
            raise ValueError('Modelspace entity budget exceeded')
        modelspace[entity.dxf.handle] = entity
    # Explicitly reject instances until instance paths/transforms can be reviewed.
    if any(e.dxftype() == 'INSERT' for e in modelspace.values()):
        raise ValueError('INSERT instances are not supported by this modelspace review')
    if any(e.dxftype() in ('ARC_DIMENSION', 'LARGE_RADIAL_DIMENSION') for e in modelspace.values()):
        raise ValueError('Special dimension subtypes require a separate review')
    dimensions = {h: e for h, e in modelspace.items() if e.dxftype() == 'DIMENSION'}
    if not dimensions:
        raise ValueError('No modelspace DIMENSION objects to review')
    view_items = case.get('views')
    if not isinstance(view_items, list) or not 1 <= len(view_items) <= 1000:
        raise ValueError('Invalid view inventory')
    views = {}
    for view in view_items:
        if not isinstance(view, dict):
            raise ValueError('Invalid view record')
        name = _text(view.get('id'), 'View id')
        if name in views:
            raise ValueError('Duplicate view id')
        scale = _text(view.get('declared_scale'), 'Declared view scale')
        ratio = re.fullmatch(r'([1-9]\d*(?:\.\d+)?):([1-9]\d*(?:\.\d+)?)', scale)
        if ratio is None:
            raise ValueError('Declared view scale must be a positive ratio')
        factor = _number(view.get('drawing_to_physical_factor'), 'View factor')
        expected_factor = float(ratio[2])/float(ratio[1])
        if (factor <= 0 or not math.isfinite(expected_factor) or expected_factor <= 0
                or not math.isclose(factor, expected_factor, rel_tol=1e-12, abs_tol=0)):
            raise ValueError('View factor disagrees with the declared scale')
        evidence = view.get('scale_evidence')
        if not isinstance(evidence, dict):
            raise ValueError('Missing view scale evidence')
        caption = modelspace.get(evidence.get('handle'))
        expected_text = _text(evidence.get('expected_text'), 'Scale caption text')
        if caption is None or caption.dxftype() not in ('TEXT', 'MTEXT') or caption.plain_text() != expected_text:
            raise ValueError('Scale caption text does not match native CAD')
        if re.search(r'(?<![\d.:])'+re.escape(scale)+r'(?![\d.:])', expected_text) is None:
            raise ValueError('Scale caption does not contain the declared ratio')
        _text(evidence.get('region'), 'Scale caption raster region')
        views[name] = {'id': name, 'declared_scale': scale, 'drawing_to_physical_factor': factor,
                       'source_box_px': _box(view.get('source_box_px'), source_size_px),
                       'scale_evidence': dict(evidence), 'pixel_reading_verified_by_validator': False}
    items = case.get('objects')
    if not isinstance(items, list) or len(items) != len(dimensions):
        raise ValueError('Review must cover every modelspace DIMENSION exactly once')
    handles = [item.get('handle') if isinstance(item, dict) else None for item in items]
    if any(not isinstance(h, str) for h in handles) or len(set(handles)) != len(handles) or set(handles) != set(dimensions):
        raise ValueError('Review dimension inventory is incomplete or duplicated')
    results = []
    for item in items:
        entity = dimensions[item['handle']]
        view = views.get(item.get('view_id'))
        if view is None:
            raise ValueError('Dimension refers to an unknown view')
        decision = item.get('decision')
        if decision not in DECISIONS:
            raise ValueError('Unknown object review decision')
        reason = _text(item.get('reason'), 'Review reason')
        raw = entity.dxf.get('text', '')
        if item.get('expected_text') != raw:
            raise ValueError('Dimension override text changed from reviewed text')
        tolerance = _number(item.get('numeric_tolerance_mm', MAX_NUMERIC_TOLERANCE_MM), 'Numeric tolerance')
        if not 0 < tolerance <= MAX_NUMERIC_TOLERANCE_MM:
            raise ValueError('Numeric tolerance cannot exceed the strict review limit')
        nominal = item.get('nominal')
        parsed, prefix = _simple_nominal(raw)
        if nominal is not None:
            nominal = _number(nominal, 'Nominal value')
            if nominal <= 0 or parsed is None or abs(nominal-parsed) > 1e-9:
                raise ValueError('Nominal must agree with the explicit simple scalar text')
        kind = entity.dimtype
        required = ('defpoint2', 'defpoint3') if kind in (0, 1) else ('defpoint', 'defpoint4') if kind in (3, 4) else ()
        points = {name: _point(entity.dxf.get(name), name) for name in required if entity.dxf.hasattr(name)}
        if len(points) != len(required):
            raise ValueError('Dimension is missing required definition points')
        is_supported = (kind in KINDS and parsed is not None and nominal is not None
                        and not (prefix in ('R', 'r') and kind != 4))
        if decision != 'quarantine':
            if not is_supported:
                raise ValueError('Compound or unsupported dimension can only be quarantined')
            _planar(entity)
            for name, point in points.items():
                _xy(point, name)
        native = _native_measurement(entity, points)
        scaled = None if native is None else native*view['drawing_to_physical_factor']
        if scaled is not None and not math.isfinite(scaled):
            raise ValueError('Nonfinite scaled measurement')
        delta = None if nominal is None or scaled is None else scaled-nominal
        numeric_status = 'unresolved' if delta is None else 'consistent' if abs(delta) <= tolerance else 'conflict'
        if decision != 'quarantine' and numeric_status != 'consistent':
            raise ValueError('A nominal conflict cannot be accepted as a supported observation')
        specs = item.get('bindings')
        if not isinstance(specs, list) or len(specs) > 20:
            raise ValueError('Invalid binding inventory')
        links = {}
        for spec in specs:
            if not isinstance(spec, dict):
                raise ValueError('Invalid binding record')
            name = spec.get('point')
            if name not in points or name in links:
                raise ValueError('Unknown or duplicate definition point binding')
            geometry = modelspace.get(spec.get('geometry_handle'))
            if geometry is None:
                raise ValueError('Geometry handle must refer to a modelspace entity')
            link = _binding(geometry, points[name], spec.get('relation'), spec)
            link['point'] = name
            links[name] = link
        qualified = any(not link['exact_finite_geometry_attachment'] for link in links.values())
        if decision != 'quarantine':
            if set(links) != set(required):
                raise ValueError('Accepted observation requires all measurement point bindings')
            if decision == 'supported_observation' and qualified:
                raise ValueError('Supporting-line projection requires a qualified observation')
            if decision == 'qualified_observation' and not qualified:
                raise ValueError('Qualified observation needs an explicit supporting-line qualification')
            if kind in (3, 4):
                a, b = links['defpoint'], links['defpoint4']
                if a['geometry_handle'] != b['geometry_handle']:
                    raise ValueError('Radius or diameter points must refer to the same curve')
                if kind == 4 and (a['relation'] != 'center' or b['relation'] not in ('arc_point', 'circle_point')):
                    raise ValueError('Radius requires a center and a point on its curve')
                if kind == 3:
                    if a['relation'] != 'circle_point' or b['relation'] != 'circle_point':
                        raise ValueError('Diameter requires two circle-point bindings')
                    if np.linalg.norm((points['defpoint'][:2]+points['defpoint4'][:2])/2 - a['center_xy']) > CAD_TOLERANCE:
                        raise ValueError('Diameter endpoints are not antipodal')
        mapped = {}
        for name, point in points.items():
            source_xy = (matrix @ [point[0], point[1], 1.])[:2]
            if not np.isfinite(source_xy).all():
                raise ValueError('Nonfinite transformed definition point')
            x0, y0, x1, y1 = view['source_box_px']
            inside = x0 <= source_xy[0] < x1 and y0 <= source_xy[1] < y1
            if decision != 'quarantine' and not inside:
                raise ValueError('Accepted measurement anchor lies outside its reviewed view')
            mapped[name] = {'cad_xyz': point.tolist(), 'source_pixel_center_xy': source_xy.tolist(),
                            'inside_reviewed_view': bool(inside), 'coordinate_status': 'tentative_global_projection',
                            'approved': False}
        results.append({'handle': item['handle'], 'view_id': view['id'], 'kind': KINDS.get(kind, 'unsupported'),
                        'text_override': raw, 'nominal_mm': nominal, 'native_measurement': native,
                        'drawing_to_physical_factor': view['drawing_to_physical_factor'],
                        'scaled_measurement_mm': scaled, 'scaled_minus_nominal_mm': delta,
                        'numeric_status': numeric_status, 'numeric_tolerance_mm': tolerance,
                        'decision': decision, 'reason': reason, 'binding_evidence': list(links.values()),
                        'points': mapped, 'training_eligible': False, 'annotation_approved': False,
                        'raster_attachment_verified': False})
    return {'schema_version': '1.0', 'reviewer': 'assistant', 'human_verified': False,
            'scope': 'Explicit planar modelspace CAD observations; pixel evidence is reviewed separately.',
            'units': 'mm', 'insunits': 4, 'complete_modelspace_dimension_inventory': True,
            'view_membership_source': 'explicit_reviewer_assertion', 'views': list(views.values()),
            'objects': results, 'summary': {'dimension_count': len(results),
                                           'decisions': dict(Counter(r['decision'] for r in results)),
                                           'numeric_statuses': dict(Counter(r['numeric_status'] for r in results))},
            'training_eligible': False, 'annotation_approved': False,
            'physical_scale_source': 'reviewed drawing scale caption, not PNG DPI',
            'pixel_geometry_approved': False, 'dimension_style_factor_applied': False,
            'limitations': ['Caption pixel readings and contour roles are reviewer assertions, not OCR or automatic classification.',
                            'Native numeric agreement and attachment do not certify raster precision.',
                            'Modelspace INSERT instances and special dimension subtypes are rejected.']}
