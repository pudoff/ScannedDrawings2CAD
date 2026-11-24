"""Semantic DXF export of explicit v2 views, contours and reviewed dimensions.

No source geometry is fitted to a printed nominal. DXF dimensions are native,
editable and computed from their own defining points, but not CAD-associative
with a contour moved later in a CAD application.
"""
from collections import Counter
import math
from pathlib import Path

import ezdxf
from ezdxf import bbox
import numpy as np

from .graph_v2 import ArcV2, CircleV2, LineV2, PolylineV2, read_graph_v2
from .tolerances import dimension_tolerances
from .topology_v2 import (TOLERANCE, bulge_arc, geometry_segments, validate_contour,
                         validate_hatch_boundaries, polyline_boundary, TopologyBudget)

NUMERIC_TOLERANCE = 1e-5


def _xy(matrix, point):
    with np.errstate(over='ignore', invalid='ignore'):
        result = tuple((matrix @ [*point, 1.])[:2])
    if not all(math.isfinite(value) for value in result):
        raise ValueError('Transformed coordinate is nonfinite')
    return result


def _similarity(matrix):
    if not np.isfinite(matrix).all(): return False
    a, b = matrix[:2, 0], matrix[:2, 1]
    sx, sy = np.linalg.norm(a), np.linalg.norm(b)
    return math.isfinite(sx) and math.isfinite(sy) and sx >= 1e-12 and math.isclose(sx, sy, rel_tol=1e-9, abs_tol=1e-12) and abs(np.dot(a, b)) <= sx*sy*1e-9


def _angle(matrix, angle):
    direction = matrix[:2, :2] @ [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
    return math.degrees(math.atan2(direction[1], direction[0]))


def _curve_matrix(matrix):
    if not _similarity(matrix) or np.linalg.det(matrix[:2, :2]) <= 0:
        raise ValueError('Native circles, arcs and bulges need an orientation-preserving similarity transform')
    factor = float(np.linalg.norm(matrix[:2, 0]))
    if not math.isfinite(factor): raise ValueError('Nonfinite curve transform')
    return factor


def _scaled_positive(value, factor):
    result = value*factor
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Scaled size is not finite and positive')
    return result


def _on_segment(point, a, b, tolerance=TOLERANCE):
    delta = np.asarray(b)-a
    denominator = float(delta @ delta)
    if denominator <= tolerance*tolerance:
        return math.dist(point, a) <= tolerance
    parameter = min(1., max(0., float((np.asarray(point)-a) @ delta)/denominator))
    return math.dist(point, np.asarray(a)+parameter*delta) <= tolerance


def _on_geometry(point, primitive, *, allow_center=True):
    if isinstance(primitive, (CircleV2, ArcV2)):
        if allow_center and math.dist(point, primitive.center) <= TOLERANCE:
            return True
        if abs(math.dist(point, primitive.center)-primitive.radius) > TOLERANCE:
            return False
        if isinstance(primitive, CircleV2):
            return True
        angle = math.degrees(math.atan2(point[1]-primitive.center[1], point[0]-primitive.center[0]))
        delta = (angle-primitive.start_angle_deg) % 360
        sweep = (primitive.end_angle_deg-primitive.start_angle_deg) % 360
        return delta <= sweep+1e-7 or delta >= 360-1e-7
    for a, b, bulge in geometry_segments(primitive):
        if not bulge and _on_segment(point, a, b):
            return True
        if bulge:
            center, radius, start, sweep = bulge_arc(a, b, bulge)
            angle = math.atan2(point[1]-center[1], point[0]-center[0])
            delta = (angle-start) % (2*math.pi) if sweep > 0 else (start-angle) % (2*math.pi)
            if abs(math.dist(point, center)-radius) <= TOLERANCE and (delta <= abs(sweep)+1e-9 or delta >= 2*math.pi-1e-9):
                return True
    return False


def dimension_measurement(dimension, geometries, matrix):
    """Require explicit finite supports before measuring transformed coordinates."""
    references = [geometries[key] for key in dimension.measured_entity_ids]
    if any(g.status != 'accepted' or g.evidence.status != 'verified' for g in references):
        raise ValueError('Dimension references unverified geometry')
    points = dimension.points
    if dimension.kind in ('diameter', 'radius'):
        a, b = points
        if not all(isinstance(g, (CircleV2, ArcV2)) and math.dist(a, g.center) <= TOLERANCE
                   and _on_geometry(b, g, allow_center=False) for g in references):
            raise ValueError('Radius or diameter needs an explicit center and finite curve contact')
        _curve_matrix(matrix)
    elif dimension.kind == 'angular':
        center, first, second = points
        if len(references) != 2 or not all(isinstance(g, LineV2) for g in references):
            raise ValueError('Three-point angular dimensions require two explicit straight rays')
        if not all(_on_geometry(center, g) for g in references):
            raise ValueError('Angular vertex is not on both finite rays')
        if not any(_on_geometry(first, references[i]) and _on_geometry(second, references[1-i]) for i in (0, 1)):
            raise ValueError('Angular defining points are not on their finite rays')
    else:
        if not all(any(_on_geometry(point, g) for g in references) for point in points) or not all(
                any(_on_geometry(point, g) for point in points) for g in references):
            raise ValueError('Measured points do not attach to the explicit finite geometry')
        if dimension.kind == 'longitudinal_diameter' and not all(isinstance(g, (LineV2, PolylineV2)) for g in references):
            raise ValueError('Longitudinal diameter references the actual straight profile edges')
    transformed = [_xy(matrix, point) for point in points]
    if dimension.kind == 'angular':
        center, a, b = transformed
        if min(math.dist(center, a), math.dist(center, b)) <= NUMERIC_TOLERANCE:
            raise ValueError('Degenerate angular ray')
        first = math.atan2(a[1]-center[1], a[0]-center[0])
        second = math.atan2(b[1]-center[1], b[0]-center[0])
        measurement = math.degrees((second-first) % (2*math.pi))
    elif dimension.kind in ('radius', 'diameter', 'aligned'):
        measurement = math.dist(*transformed)*(2 if dimension.kind == 'diameter' else 1)
    else:
        angle = math.radians(_angle(matrix, dimension.angle_deg))
        delta = np.asarray(transformed[1])-transformed[0]
        measurement = abs(float(delta @ [math.cos(angle), math.sin(angle)]))
    if not math.isfinite(measurement) or measurement <= NUMERIC_TOLERANCE or not math.isclose(measurement, dimension.nominal, abs_tol=NUMERIC_TOLERANCE, rel_tol=1e-8):
        raise ValueError(f'Printed nominal {dimension.nominal:g} conflicts with measured geometry {measurement:g}')
    return measurement, transformed


def _number(value):
    return f'{value:.6f}'.rstrip('0').rstrip('.') or '0'


def _plain(value):
    return value.replace('%%c', 'Ø').replace('%%C', 'Ø').replace('%%d', '°').replace('%%p', '±')


def expected_dimension_text(dimension, measurement):
    suffix = '°' if dimension.kind == 'angular' else ''
    prefix = dimension.prefix
    if dimension.kind == 'longitudinal_diameter' and not prefix.startswith(('Ø', '⌀', '%%c')):
        prefix = 'Ø'+prefix
    value = _plain(prefix+_number(measurement)+suffix+dimension.suffix)
    automatic = {'radius': 'R', 'diameter': 'Ø'}.get(dimension.kind)
    if automatic and not value.startswith(automatic):
        value = automatic+value
    deviations = dimension_tolerances(dimension)
    if deviations is not None:
        value += deviations.display_suffix(_number, suffix)
    return value


def dimension_display_text(native):
    block = native.get_geometry_block()
    if block is None:
        raise ValueError('Native dimension has no display block')
    texts = [_plain(e.plain_text() if e.dxftype() == 'MTEXT' else e.dxf.text)
             for e in block if e.dxftype() in ('TEXT', 'MTEXT')]
    if len(texts) != 1:
        raise ValueError('Expected one computed dimension display text, not exploded or ambiguous labels')
    return texts[0]


def rerender_native_dimension(native):
    """Refresh a dimension's own display; this does not attach it to moved contours.

    ezdxf's renderer uses the angle attribute even for native aligned dimensions.
    Refresh that redundant direction from the defining endpoints before rendering.
    """
    if native.dimtype == 1:
        a, b = native.dxf.defpoint2, native.dxf.defpoint3
        if math.hypot(b.x-a.x, b.y-a.y) <= NUMERIC_TOLERANCE:
            raise ValueError('Aligned dimension endpoints became degenerate')
        native.dxf.angle = math.degrees(math.atan2(b.y-a.y, b.x-a.x))
    native.render()
    return dimension_display_text(native)


def _layer_attributes(layer):
    return {'layer': layer, 'color': 256, 'linetype': 'BYLAYER', 'lineweight': -1}


def semantic_entity_snapshot(entity):
    """Serializable geometric state, excluding volatile document metadata."""
    data = {'type': entity.dxftype(), 'handle': entity.dxf.handle,
            **{key: getattr(entity.dxf, key) for key in ('layer', 'color', 'linetype', 'lineweight')}}
    for name in ('start', 'end', 'center', 'insert', 'defpoint', 'defpoint2', 'defpoint3', 'defpoint4'):
        if entity.dxf.is_supported(name):
            value = getattr(entity.dxf, name, None)
            if value is not None: data[name] = list(value)
    for name in ('radius', 'start_angle', 'end_angle', 'text', 'height', 'rotation', 'angle', 'dimtype'):
        if entity.dxf.is_supported(name):
            value = getattr(entity.dxf, name, None)
            if value is not None: data[name] = value
    if entity.dxftype() == 'LWPOLYLINE':
        data.update(closed=entity.closed, vertices=[list(p) for p in entity.get_points('xyb')])
    elif entity.dxftype() == 'DIMENSION':
        data.update(measurement=float(entity.get_measurement()), display_text=dimension_display_text(entity))
    elif entity.dxftype() == 'HATCH':
        data.update(pattern_name=entity.dxf.pattern_name, pattern_angle=entity.dxf.pattern_angle,
            solid_fill=entity.dxf.solid_fill, hatch_style=entity.dxf.hatch_style, associative=entity.dxf.associative,
            paths=[{'closed': p.is_closed, 'flags': p.path_type_flags, 'vertices': [list(v) for v in p.vertices],
                    'boundary_handles': list(p.source_boundary_objects)} for p in entity.paths])
    return data


def _same_snapshot(left, right):
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(_same_snapshot(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return isinstance(right, (list, tuple)) and len(left) == len(right) and all(_same_snapshot(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, int)) and not isinstance(left, bool):
        return isinstance(right, (float, int)) and math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, abs_tol=1e-8, rel_tol=1e-10)
    return left == right


def export_dxf_v2(graph, output):
    graph = read_graph_v2(graph)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new DXF path; existing outputs are preserved')
    if output == Path(graph.source.path).resolve():
        raise ValueError('DXF output cannot overwrite the source')
    report = {'schema_version': '0.2', 'status': 'needs_review', 'dxf_version': 'AC1021',
              'source_sha256': graph.source.sha256, 'source_modified': False, 'graph_modified': False,
              'cad_application_checked': False, 'dimensions_associative': False,
              'training_eligible': False, 'views': [], 'entities': [], 'dimensions': [], 'hatches': [],
              'result_scope': 'semantic_export_of_explicit_graph_fields', 'reconstruction_approved': False,
              'contours': [], 'issues': [i.model_dump(mode='json') for i in graph.issues]}

    def issue(code, message, ids=()):
        report['issues'].append({'code': code, 'message': str(message), 'entity_ids': list(ids), 'severity': 'warning'})

    doc = ezdxf.new('R2007', setup=True)
    doc.units = ezdxf.units.MM
    doc.header['$MEASUREMENT'] = 1
    doc.header['$DIMASSOC'] = 1
    if 'CENTER' not in doc.linetypes:
        doc.linetypes.new('CENTER', dxfattribs={'pattern': [20., 12., -3., 2., -3.]})
    if 'HIDDEN' not in doc.linetypes:
        doc.linetypes.new('HIDDEN', dxfattribs={'pattern': [6., 3., -3.]})
    # View-specific layer extensions allow each paper viewport to hide other views.
    layer_names = {}
    for layer in graph.layers:
        attributes = {'linetype': layer.line_type, 'color': layer.color, 'lineweight': layer.lineweight}
        doc.layers.new(layer.name, dxfattribs=attributes)
        for index, view in enumerate(graph.views):
            name = f'{layer.name}__VIEW_{index:03d}'
            if name in {l.name for l in graph.layers}:
                raise ValueError('Declared layer collides with reserved view-layer extension')
            doc.layers.new(name, dxfattribs=attributes)
            layer_names[(view.id, layer.name)] = name
    matrices = {v.id: np.asarray(v.model_matrix() if v.physical_scale_confirmed else v.view_to_sheet, dtype=float)
                for v in graph.views}
    views = {v.id: v for v in graph.views}
    by_view = {v.id: [] for v in graph.views}
    model = doc.modelspace()
    geometries = {g.id: g for g in graph.primitives}
    emitted = {}
    for view in graph.views:
        if not view.physical_scale_confirmed:
            issue('view_physical_scale_unknown', 'Geometry remains a sheet-size observation; physical dimensions are omitted.', [view.id])
        report['views'].append({'id': view.id, 'coordinate_space': view.coordinate_space,
            'view_to_sheet': view.view_to_sheet, 'physical_scale_confirmed': view.physical_scale_confirmed,
            'model_scale_xy': view.model_scale_xy, 'export_matrix': matrices[view.id].tolist(),
            'export_units': 'physical_model_mm' if view.physical_scale_confirmed else 'paper_observation_mm'})
    for geometry in graph.primitives:
        if geometry.status == 'rejected':
            continue
        matrix = matrices[geometry.view_id]
        attributes = _layer_attributes(layer_names[(geometry.view_id, geometry.layer)])
        try:
            if isinstance(geometry, LineV2):
                entity = model.add_line(_xy(matrix, geometry.start), _xy(matrix, geometry.end), dxfattribs=attributes)
            elif isinstance(geometry, CircleV2):
                entity = model.add_circle(_xy(matrix, geometry.center), _scaled_positive(geometry.radius, _curve_matrix(matrix)), dxfattribs=attributes)
            elif isinstance(geometry, ArcV2):
                entity = model.add_arc(_xy(matrix, geometry.center), _scaled_positive(geometry.radius, _curve_matrix(matrix)),
                    _angle(matrix, geometry.start_angle_deg), _angle(matrix, geometry.end_angle_deg), dxfattribs=attributes)
            else:
                if any(v.bulge for v in geometry.vertices): _curve_matrix(matrix)
                vertices = [(*_xy(matrix, v.point), v.bulge) for v in geometry.vertices]
                entity = model.add_lwpolyline(vertices, format='xyb', close=geometry.closed, dxfattribs=attributes)
        except ValueError as error:
            issue('geometry_transform_unsupported', error, [geometry.id])
            continue
        emitted[geometry.id] = entity
        by_view[geometry.view_id].append(entity)
        report['entities'].append({'id': geometry.id, 'view_id': geometry.view_id, 'handle': entity.dxf.handle, 'type': entity.dxftype()})
        if geometry.status != 'accepted' or geometry.evidence.status != 'verified':
            issue('geometry_is_observation', 'Exported geometry is not a verified reconstruction.', [geometry.id])
    for text in graph.texts:
        if text.status == 'rejected': continue
        matrix = matrices[text.view_id]
        try:
            factor = _curve_matrix(matrix)
        except ValueError as error:
            issue('text_transform_unsupported', error, [text.id])
            continue
        entity = model.add_text(text.raw, dxfattribs={**_layer_attributes(layer_names[(text.view_id, text.layer)]),
            'insert': _xy(matrix, text.position), 'height': _scaled_positive(text.height, factor), 'rotation': _angle(matrix, text.rotation_deg)})
        by_view[text.view_id].append(entity)
        report['entities'].append({'id': text.id, 'view_id': text.view_id, 'handle': entity.dxf.handle, 'type': 'TEXT'})
        if text.status != 'accepted' or text.evidence.status != 'verified':
            issue('text_is_observation', 'Text is exported with its unverified status.', [text.id])
    contours = {c.id: c for c in graph.contours}
    valid_contours = {}
    topology_budget = TopologyBudget()
    for contour in graph.contours:
        try:
            checked = validate_contour(contour, geometries, budget=topology_budget)
            valid_contours[contour.id] = checked
            report['contours'].append({k: v for k, v in checked.items() if k not in ('segments', 'flat')})
        except ValueError as error:
            issue('contour_invalid', error, [contour.id])
    for hatch in graph.hatches:
        try:
            bounds = [contours[hatch.outer_contour_id], *[contours[key] for key in hatch.hole_contour_ids]]
            if hatch.status != 'accepted' or hatch.evidence.status != 'verified' or any(
                    c.status != 'accepted' or c.evidence.status != 'verified' or c.id not in valid_contours for c in bounds):
                raise ValueError('HATCH requires verified contour roles and valid boundaries')
            if any(e.primitive_id not in emitted or geometries[e.primitive_id].evidence.status != 'verified'
                   or geometries[e.primitive_id].status != 'accepted' for c in bounds for e in c.edges):
                raise ValueError('HATCH boundary geometry is missing or unverified')
            checked, net_area = validate_hatch_boundaries(bounds[0], bounds[1:], geometries, budget=topology_budget)
            matrix = matrices[hatch.view_id]
            if any(b for c in checked for _, _, b in c['segments']): _curve_matrix(matrix)
            factor = _curve_matrix(matrix) if hatch.pattern != 'SOLID' else None
            export_area = net_area*abs(float(np.linalg.det(matrix[:2, :2])))
            if not math.isfinite(export_area): raise ValueError('Transformed HATCH area is nonfinite')
            attributes = _layer_attributes(layer_names[(hatch.view_id, hatch.layer)])
            entity = model.add_hatch(dxfattribs=attributes)
            if hatch.pattern == 'SOLID': entity.set_solid_fill(color=256, style=0)
            else:
                # One exact line family; spacing is in local view units, angle follows the view.
                entity.set_pattern_fill('ANSI31', color=256, angle=_angle(matrix, hatch.angle_deg), scale=1,
                    style=0, definition=[[0., (0., 0.), (0., _scaled_positive(hatch.spacing, factor)), []]])
            for index, boundary in enumerate(checked):
                vertices = [(*_xy(matrix, (x, y)), bulge) for x, y, bulge in polyline_boundary(boundary, clockwise=index > 0)]
                path = entity.paths.add_polyline_path(vertices, is_closed=True, flags=1 if index == 0 else 0)
                path.source_boundary_objects = [emitted[e.primitive_id].dxf.handle for e in bounds[index].edges]
            # Boundary references record provenance; no untested CAD associative reactors are claimed.
            entity.dxf.associative = 0
            by_view[hatch.view_id].append(entity)
            report['hatches'].append({'id': hatch.id, 'handle': entity.dxf.handle, 'hole_count': len(checked)-1,
                'boundary_path_count': len(checked), 'net_area_local': net_area,
                'net_area_export': export_area, 'associative': False})
        except ValueError as error:
            issue('hatch_omitted', error, [hatch.id])
    for dimension in graph.dimensions:
        native = None
        try:
            view = views[dimension.view_id]
            if dimension.status != 'accepted' or dimension.evidence.status != 'verified' or not view.physical_scale_confirmed:
                raise ValueError('Physical DIMENSION requires verified fields and confirmed view scale')
            if any(key not in emitted for key in dimension.measured_entity_ids):
                raise ValueError('Referenced geometry was not exported')
            matrix = matrices[dimension.view_id]
            measurement, points = dimension_measurement(dimension, geometries, matrix)
            placement = _xy(matrix, dimension.placement)
            sheet_from_model = np.asarray(view.view_to_sheet) @ np.linalg.inv(matrix)
            if dimension.annotation_scale is None:
                if not _similarity(sheet_from_model):
                    raise ValueError('Unrepresentable per-view plotting scale requires explicit annotation size')
                annotation_scale = 1./float(np.linalg.norm(sheet_from_model[:2, 0]))
            else:
                annotation_scale = dimension.annotation_scale
            prefix = dimension.prefix
            if dimension.kind == 'longitudinal_diameter' and not prefix.startswith(('Ø', '⌀', '%%c')): prefix = 'Ø'+prefix
            style = {'dimlfac': 1., 'dimscale': annotation_scale, 'dimtxt': 2.5, 'dimasz': 2.5,
                     # ezdxf interprets loaded DIMRND=0 as rounding to an integer;
                     # an explicit 1e-6 increment matches the declared six decimal places.
                     'dimdec': 6, 'dimrnd': 1e-6, 'dimadec': 6, 'dimzin': 8, 'dimazin': 2, 'dimtzin': 8,
                     'dimclrd': 256, 'dimclre': 256, 'dimclrt': 256, 'dimpost': prefix+'<>'+dimension.suffix}
            deviations = dimension_tolerances(dimension)
            if deviations is not None:
                style.update(deviations.dxf_style())
            attributes = _layer_attributes(layer_names[(dimension.view_id, dimension.layer)])
            if dimension.kind in ('linear', 'longitudinal_diameter', 'aligned'):
                angle = (_angle(matrix, dimension.angle_deg) if dimension.kind != 'aligned' else
                    math.degrees(math.atan2(points[1][1]-points[0][1], points[1][0]-points[0][0])))
                native = model.add_linear_dim(base=placement, p1=points[0], p2=points[1], angle=angle, override=style, dxfattribs=attributes)
                if dimension.kind == 'aligned':
                    # ezdxf.add_aligned_dim also creates type 0; preserve true native type 1 semantics.
                    native.dimension.dxf.dimtype = (native.dimension.dxf.dimtype & ~7) | 1
            elif dimension.kind == 'angular':
                native = model.add_angular_dim_3p(base=placement, center=points[0], p1=points[1], p2=points[2],
                    override=style, dxfattribs=attributes)
            else:
                create = model.add_radius_dim if dimension.kind == 'radius' else model.add_diameter_dim
                native = create(center=points[0], mpoint=points[1], override=style, dxfattribs=attributes)
                native.user_location_override(placement)
            native.render()
            for child in native.dimension.get_geometry_block():
                child.update_dxf_attribs(attributes)
            text = dimension_display_text(native.dimension)
            expected = expected_dimension_text(dimension, measurement)
            if text != expected:
                model.delete_entity(native.dimension)
                raise ValueError(f'Cached dimension display differs from computed value: {text!r} != {expected!r}')
            by_view[dimension.view_id].append(native.dimension)
            report['dimensions'].append({'id': dimension.id, 'view_id': dimension.view_id, 'kind': dimension.kind,
                'handle': native.dimension.dxf.handle, 'nominal': dimension.nominal, 'measurement': measurement,
                'unit': 'degree' if dimension.kind == 'angular' else 'mm', 'display_text': text,
                'computed_text_override': native.dimension.dxf.text, 'annotation_scale': annotation_scale,
                'annotation_scale_basis': 'explicit_graph_value' if dimension.annotation_scale is not None else 'derived_per_view_plotting_ratio',
                'measured_entity_ids': dimension.measured_entity_ids})
        except (ValueError, ezdxf.DXFError) as error:
            if native is not None and native.dimension.is_alive:
                model.delete_entity(native.dimension)
            issue('dimension_omitted', error, [dimension.id])
    sheet = doc.layouts.get('Layout1')
    doc.layouts.rename('Layout1', 'Sheet')
    sheet.page_setup(size=(graph.page.width_mm, graph.page.height_mm), margins=(0, 0, 0, 0), units='mm', scale=(1, 1))
    all_layers = set(layer_names.values())
    for view in graph.views:
        entities = by_view[view.id]
        if not entities: continue
        matrix = matrices[view.id]
        sheet_from_model = np.asarray(view.view_to_sheet) @ np.linalg.inv(matrix)
        if not _similarity(sheet_from_model) or np.linalg.det(sheet_from_model[:2, :2]) <= 0:
            issue('sheet_viewport_transform_unrepresentable', 'Sheet transform cannot be represented by one undistorted viewport.', [view.id])
            continue
        extents = bbox.extents(entities)
        if not extents.has_data: continue
        minimum, maximum = extents.extmin, extents.extmax
        center = ((minimum.x+maximum.x)/2, (minimum.y+maximum.y)/2)
        paper_center = _xy(sheet_from_model, center)
        factor = float(np.linalg.norm(sheet_from_model[:2, 0]))
        # Rotation changes the viewport's paper bounding rectangle; use all four corners.
        corners = [_xy(sheet_from_model, p) for p in [(minimum.x, minimum.y), (maximum.x, minimum.y),
                                                    (maximum.x, maximum.y), (minimum.x, maximum.y)]]
        width = max(p[0] for p in corners)-min(p[0] for p in corners)+2
        height = max(p[1] for p in corners)-min(p[1] for p in corners)+2
        twist = _angle(sheet_from_model, 0)
        # DXF view_center_point is in the rotated display coordinate system.
        dcs_center = tuple((sheet_from_model[:2, :2]/factor) @ center)
        viewport = sheet.add_viewport(center=paper_center, size=(width, height), view_center_point=dcs_center, view_height=height/factor)
        viewport.dxf.view_twist_angle = twist
        actual_matrix = viewport.get_transformation_matrix()
        for probe in ((0., 0.), (1., 0.), (0., 1.)):
            actual_point = actual_matrix.transform((*probe, 0))
            if math.dist((actual_point.x, actual_point.y), _xy(sheet_from_model, probe)) > 1e-7:
                raise ValueError('Viewport does not reproduce the declared sheet transform')
        own_layers = {name for (vid, _), name in layer_names.items() if vid == view.id}
        viewport.frozen_layers = sorted(all_layers-own_layers)
        viewport.dxf.layer = 'THIN'
        next(row for row in report['views'] if row['id'] == view.id)['viewport'] = {
            'handle': viewport.dxf.handle, 'paper_center': paper_center, 'paper_size': (width, height),
            'model_center': center, 'dcs_center': dcs_center, 'paper_per_model': factor,
            'sheet_from_model': sheet_from_model.tolist(), 'frozen_other_view_layer_count': len(viewport.frozen_layers)}
    expected = Counter(e.dxftype() for e in model)
    report['topology_predicates_used'] = topology_budget.used
    if graph.constraints:
        issue('constraints_require_separate_verification', 'Export checks dimension consistency but does not solve the graph constraint system.',
              [c.id for c in graph.constraints])
    expected_geometry = [semantic_entity_snapshot(e) for e in model]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        doc.write(stream)
    saved = ezdxf.readfile(output)
    audit = saved.audit()
    actual = Counter(e.dxftype() for e in saved.modelspace())
    geometry_matches = _same_snapshot(expected_geometry, [semantic_entity_snapshot(e) for e in saved.modelspace()])
    report['counts'] = dict(actual)
    report['audit'] = {'errors': [str(e.message) for e in audit.errors], 'fixes': [str(e.message) for e in audit.fixes],
                       'readable': True, 'modelspace_types_match': actual == expected, 'native_geometry_matches': geometry_matches}
    failed = bool(audit.errors) or actual != expected or not geometry_matches
    if not geometry_matches:
        issue('geometry_roundtrip_mismatch', 'Native geometry, style or boundary state changed on saving.')
    for record in report['dimensions']:
        native = saved.entitydb[record['handle']]
        measurement = float(native.get_measurement())
        text = dimension_display_text(native)
        record['roundtrip_measurement'] = measurement
        record['roundtrip_display_text'] = text
        if not math.isclose(measurement, record['measurement'], abs_tol=NUMERIC_TOLERANCE, rel_tol=1e-8) or text != record['display_text'] or native.dxf.text != '<>':
            failed = True
            issue('dimension_roundtrip_mismatch', 'Computed value or cached display changed on saving.', [record['id']])
    for entity in saved.modelspace():
        if entity.dxf.color != 256 or entity.dxf.linetype != 'BYLAYER' or entity.dxf.lineweight != -1:
            failed = True
            issue('not_bylayer', 'Editable modelspace entity lost its ByLayer attributes.', [entity.dxf.handle])
    for view in report['views']:
        if 'viewport' not in view: continue
        record = view['viewport']
        native_viewport = saved.entitydb[record['handle']]
        transform = native_viewport.get_transformation_matrix()
        for probe in ((0., 0.), (1., 0.), (0., 1.)):
            point = transform.transform((*probe, 0))
            if math.dist((point.x, point.y), _xy(np.asarray(record['sheet_from_model']), probe)) > 1e-7:
                failed = True
                issue('viewport_roundtrip_mismatch', 'View no longer maps to its original sheet position.', [view['id']])
    if any(e.dxftype() in ('INSERT', 'IMAGE', 'OLE2FRAME') for e in saved.modelspace()):
        failed = True
        issue('unexpected_container', 'Main content must remain separate editable modelspace entities.')
    if audit.fixes: issue('dxf_audit_fixes', 'DXF audit needed repairs; review the saved file.')
    report['status'] = 'failed' if failed else 'needs_review' if report['issues'] else 'succeeded'
    return report
