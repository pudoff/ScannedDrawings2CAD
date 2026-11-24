"""Independent, bounded inspection of a DXF against DrawingGraph v2.

Exporter reports are optional untrusted handle locators, never expected values.
The validator does not import the exporter or its semantic snapshots.
"""
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path

import ezdxf

from .graph_v2 import read_graph_v2
from .tolerances import dimension_tolerances
from .cad_validation_geometry_v1 import (finite, transform, sheet_point, similarity_scale, angle,
    geometry_expected, raw_geometry, compare_geometry, segments, point_on_geometry,
    dimension_expectation, computed_text, angular_difference, validate_attachments, geometry_extrema)
from .cad_validation_topology_v1 import (PredicateBudget, oriented_segments, loop_data,
    check_simple, check_holes, loop_vertices, cyclic_equal)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                                    allow_nan=False).encode()).hexdigest()


def _planar(entity):
    if entity.dxf.hasattr('extrusion'):
        normal = tuple(entity.dxf.extrusion)
        if not all(finite(float(x)) for x in normal) or math.dist(normal, (0., 0., 1.)) > 1e-10:
            raise ValueError('Native OCS extrusion differs from the declared XY plane')
    if entity.dxf.hasattr('thickness') and (not finite(entity.dxf.thickness) or abs(entity.dxf.thickness) > 1e-10):
        raise ValueError('Unexpected native 3D thickness')


def _point(entity, name):
    p = getattr(entity.dxf, name)
    if not all(finite(float(x)) for x in p) or abs(p.z) > 1e-8:
        raise ValueError('Nonfinite or nonplanar native definition point')
    return (float(p.x), float(p.y))


def _cached_text(entity, annotation_scale, expected_main_text, has_tolerance):
    block = entity.get_geometry_block()
    if block is None or len(block) > 1000: raise ValueError('Missing or oversized dimension graphics block')
    text_entities = [e for e in block if e.dxftype() in ('TEXT', 'MTEXT')]
    texts = [e.plain_text() for e in text_entities]
    if len(texts) != 1: raise ValueError('Expected exactly one cached dimension text')
    if any(e.dxftype() not in ('LINE', 'ARC', 'CIRCLE', 'SOLID', 'TEXT', 'MTEXT', 'POINT', 'INSERT') for e in block):
        raise ValueError('Unexpected entity in dimension graphics block')
    for text in text_entities:
        height = text.dxf.char_height if text.dxftype() == 'MTEXT' else text.dxf.height
        if text.dxf.invisible or text.dxf.layer != entity.dxf.layer or not finite(height) or abs(height-2.5*annotation_scale) > 1e-8 or text.dxf.hasattr('transparency') and text.dxf.transparency != 0:
            raise ValueError('Cached dimension text is hidden or has an unexpected layer/font height')
        _point(text, 'insert')
        if math.dist(_point(text, 'insert'), _point(entity, 'text_midpoint')) > 3*annotation_scale:
            raise ValueError('Cached dimension text is detached from its native text position')
        _planar(text)
        if text.dxf.color != 256 or text.dxf.lineweight != -1 or text.dxf.linetype.upper() != 'BYLAYER':
            raise ValueError('Cached dimension text does not follow its layer')
        if text.dxftype() == 'MTEXT':
            controls = re.findall(r'\\([A-Za-z])([^;]*);', text.text)
            if any(code not in ('A', 'H', 'S') or code == 'A' and value != '0' or code == 'H' and value != '0.50x'
                   for code, value in controls):
                raise ValueError('Unsupported cached text formatting can alter visible measurement')
            if controls:
                match = re.fullmatch(r'\\A0;([^{}\\]*)\{\\H0\.50x;(?:\\S[^{}\\]*;|[^{}\\]*)\}', text.text)
                if not has_tolerance or match is None:
                    raise ValueError('Cached height/stack formatting must be confined to the declared tolerance')
                main_text = match.group(1)
                for code, symbol in (('c', 'Ø'), ('d', '°'), ('p', '±')):
                    main_text = main_text.replace('%%'+code, symbol).replace('%%'+code.upper(), symbol)
                if main_text != expected_main_text: raise ValueError('Cached formatting changes the height of the main measurement')
            if text.dxf.width not in (None, 0): raise ValueError('Unexpected cached text wrapping width')
    return texts[0]


def _eligible(entity):
    return entity.status == 'accepted' and entity.evidence.status == 'verified'


def validate_cad_v1(graph, dxf_path, *, binding_report=None, comparison_tolerance_mm=1e-5,
                    workspace_root=None, maximum_file_bytes=64*1024*1024,
                    maximum_entities=100000, maximum_topology_predicates=250000):
    """Read-only validation. No source/review/accuracy approval is manufactured.

    ``binding_report`` can be an exporter report or a mapping of graph IDs to
    handles. A supplied handle must actually identify the independently expected
    native object; duplicate, missing and unexpected objects are errors.
    """
    if not 1e-9 <= comparison_tolerance_mm <= 1e-3:
        raise ValueError('Comparison tolerance must be between 1e-9 and 1e-3 mm')
    if not 1 <= maximum_file_bytes <= 64*1024*1024 or not 1 <= maximum_entities <= 100000:
        raise ValueError('Requested resource limit exceeds supported maximum')
    if not 1 <= maximum_topology_predicates <= 250000:
        raise ValueError('Requested topology limit exceeds supported maximum')
    if isinstance(graph, (str, Path)) and Path(graph).stat().st_size > 16*1024*1024:
        raise ValueError('Graph JSON byte budget exceeded')
    graph = read_graph_v2(graph)
    if len(graph.views) > 64 or sum(len(getattr(graph, key)) for key in ('primitives', 'texts', 'dimensions', 'contours', 'hatches', 'constraints')) > 10000:
        raise ValueError('Graph object/view budget exceeded')
    path = Path(dxf_path).resolve()
    if not path.is_file() or path.stat().st_size > maximum_file_bytes: raise ValueError('Missing or oversized DXF')
    before = _digest(path)
    report = {'schema_version': '1.0', 'task': 'P15.1', 'status': 'failed',
        'graph_sha256': _json_hash(graph.model_dump(mode='json')), 'dxf_sha256': before,
        'comparison_tolerance_mm': comparison_tolerance_mm, 'source_accuracy_approved': False,
        'training_eligible': False, 'target_cad_gui_verified': False, 'cad_associativity_verified': False,
        'cached_dimension_graphics_verified': False, 'issues': [], 'entities': [], 'dimensions': [],
        'contours': [], 'hatches': [], 'viewports': [], 'source_hash_verified': False,
        'scope': 'independent_native_geometry_and_dimension_text_agreement_with_declared_graph'}
    def issue(code, message, entity_id=None, severity='error'):
        report['issues'].append({'code': code, 'message': str(message), 'entity_id': entity_id, 'severity': severity})
    root = Path(workspace_root).resolve() if workspace_root is not None else None
    if root:
        source = (root/graph.source.path).resolve()
        if source.is_relative_to(root) and source.is_file() and source.stat().st_size <= maximum_file_bytes:
            report['source_hash_verified'] = _digest(source) == graph.source.sha256
        if not report['source_hash_verified']: issue('source_hash_unverified', 'Source bytes are absent, outside workspace, oversized, or differ from graph.', severity='review')
    else: issue('source_hash_unverified', 'Workspace source verification was not requested.', severity='review')
    try:
        doc = ezdxf.readfile(path)
    except (OSError, ezdxf.DXFError, ValueError) as error:
        issue('unreadable_dxf', error)
        return report
    if len(doc.entitydb) > maximum_entities:
        issue('entity_budget', 'Total native entity budget exceeded')
        return report
    # Audit a separate in-memory read: repairs must never conceal differences
    # in the original object collection used for the independent comparisons.
    audited = ezdxf.readfile(path).audit()
    report['audit'] = {'errors': [str(e.message) for e in audited.errors],
                       'fixes': [str(e.message) for e in audited.fixes]}
    if audited.errors or audited.fixes: issue('native_audit', 'Native DXF audit reports errors or requires repairs.')
    model = list(doc.modelspace())
    report['native_counts'] = dict(sorted(Counter(e.dxftype() for e in model).items()))
    if doc.dxfversion != 'AC1021' or doc.header.get('$INSUNITS') != 4 or doc.header.get('$MEASUREMENT') != 1:
        issue('native_units_or_version', 'Expected AC1021 with millimetre units and metric measurement')
    external = [e for e in doc.entitydb.values() if e.dxftype() in ('IMAGE', 'IMAGEDEF', 'PDFUNDERLAY', 'DWFUNDERLAY', 'DGNUNDERLAY', 'OLE2FRAME')]
    if external or any(block.block.dxf.flags & 12 for block in doc.blocks):
        issue('external_resources', 'External or embedded resource objects are unsupported and will not be loaded')
    if any(e.dxftype() in ('INSERT', 'IMAGE', 'OLE2FRAME') for e in model):
        issue('unexpected_container', 'Editable drawing objects must be separate modelspace entities')
    views = {v.id: v for v in graph.views}
    layer_names = {(v.id, layer.name): f'{layer.name}__VIEW_{i:03d}'
                   for i, v in enumerate(graph.views) for layer in graph.layers}
    for layer in graph.layers:
        for name in [layer.name]+[layer_names[(v.id, layer.name)] for v in graph.views]:
            if name not in doc.layers: issue('missing_layer', name)
            else:
                actual = doc.layers.get(name)
                if actual.dxf.color != layer.color or actual.dxf.lineweight != layer.lineweight or actual.dxf.linetype.upper() != layer.line_type.upper():
                    issue('layer_style', 'Native layer properties differ: '+name)
                if actual.is_frozen() or actual.is_off() or not actual.dxf.plot:
                    issue('hidden_layer', 'Declared layer is frozen, off, or excluded from plotting: '+name)
    for view in graph.views:
        if not view.physical_scale_confirmed or view.evidence.status != 'verified':
            issue('unverified_view', 'View calibration/evidence is not verified; comparison cannot establish physical accuracy.', view.id, 'review')
    all_objects = [*graph.primitives, *graph.texts, *graph.dimensions, *graph.contours, *graph.hatches]
    for entity in all_objects:
        if entity.status != 'rejected' and not _eligible(entity):
            issue('unverified_graph_object', 'Object lacks accepted verified evidence.', entity.id, 'review')
    bindings, viewport_bindings = {}, {}
    if isinstance(binding_report, (str, Path)):
        binding_path = Path(binding_report)
        if binding_path.stat().st_size > 8*1024*1024: raise ValueError('Binding report budget exceeded')
        binding_report = json.loads(binding_path.read_text(encoding='utf-8'))
    if binding_report:
        if any(k in binding_report for k in ('entities', 'dimensions', 'hatches', 'views')):
            for key in ('entities', 'dimensions', 'hatches'):
                for row in binding_report.get(key, []):
                    if row['id'] in bindings: issue('duplicate_binding_id', row['id'])
                    bindings[row['id']] = str(row['handle'])
            viewport_bindings = {v['id']: str(v['viewport']['handle']) for v in binding_report.get('views', []) if 'viewport' in v}
        else: bindings = {str(k): str(v) for k, v in binding_report.items()}
    expected_ids = {e.id for e in [*graph.primitives, *graph.texts, *graph.dimensions, *graph.hatches] if e.status != 'rejected'}
    if set(bindings)-expected_ids: issue('unknown_binding_id', ','.join(sorted(set(bindings)-expected_ids)))
    if len(bindings.values()) != len(set(bindings.values())): issue('duplicate_binding_handle', 'A native object cannot fulfill several graph IDs')
    if not bindings and len(model) > 1000:
        issue('binding_budget', 'Explicit ID/handle locators are required above 1000 native objects')
        return report
    by_handle = {e.dxf.handle: e for e in model}
    used, natives, actual_geometry, expected_geometry = set(), {}, {}, {}
    bounds = {v.id: [] for v in graph.views}
    def locate(entity, native_type, expectation=None):
        handle = bindings.get(entity.id)
        native = by_handle.get(handle) if handle else None
        if not handle:
            candidates = [e for e in model if e.dxf.handle not in used and e.dxftype() == native_type
                          and e.dxf.layer == layer_names[(entity.view_id, entity.layer)]]
            if expectation:
                valid = []
                for candidate in candidates:
                    try:
                        _planar(candidate)
                        if not compare_geometry(expectation, raw_geometry(candidate), comparison_tolerance_mm)[0]: valid.append(candidate)
                    except (ValueError, AttributeError): pass
                candidates = valid
            if len(candidates) == 1: native = candidates[0]
            elif len(candidates) > 1:
                issue('ambiguous_entity_binding', 'Several native entities match; an explicit ID/handle locator is required.', entity.id)
                return None
        if native is None:
            issue('missing_native_entity', 'Expected native object is absent from modelspace.', entity.id)
            return None
        used.add(native.dxf.handle); natives[entity.id] = native
        if native.dxftype() != native_type:
            issue('wrong_native_type', f'Expected {native_type}, found {native.dxftype()}.', entity.id)
            return None
        if native.dxf.layer != layer_names[(entity.view_id, entity.layer)]: issue('wrong_entity_layer', native.dxf.layer, entity.id)
        if native.dxf.color != 256 or native.dxf.lineweight != -1 or native.dxf.linetype.upper() != 'BYLAYER' or native.dxf.hasattr('true_color'):
            issue('not_bylayer', 'Entity properties override its layer.', entity.id)
        if native.dxf.invisible or native.dxf.hasattr('transparency') and native.dxf.transparency != 0:
            issue('invisible_entity', 'Declared object is hidden or has explicit transparency.', entity.id)
        _planar(native)
        return native
    for entity in graph.primitives:
        if entity.status == 'rejected': continue
        try:
            expected = geometry_expected(entity, views[entity.view_id]); expected_geometry[entity.id] = expected
            native = locate(entity, expected['type'], expected)
            if native is None: continue
            actual = raw_geometry(native); actual_geometry[entity.id] = actual
            errors, point_error, radius_error = compare_geometry(expected, actual, comparison_tolerance_mm)
            if errors: issue('geometry_mismatch', ','.join(errors), entity.id)
            report['entities'].append({'id': entity.id, 'native_type': native.dxftype(), 'geometry_matches': not errors,
                'max_point_error_mm': point_error, 'radius_error_mm': radius_error,
                'graph_evidence_status': entity.evidence.status, 'graph_review_ref': entity.evidence.review_ref,
                'declared_uncertainty': entity.evidence.uncertainty.model_dump(mode='json') if entity.evidence.uncertainty else None})
            bounds[entity.view_id].extend(geometry_extrema(actual))
        except (ValueError, KeyError, AttributeError, OverflowError) as error: issue('invalid_native_geometry', error, entity.id)
    for entity in graph.texts:
        if entity.status == 'rejected': continue
        try:
            native = locate(entity, 'TEXT')
            if native is None: continue
            view = views[entity.view_id]
            if native.dxf.text != entity.raw: issue('text_content', 'Text differs from graph.', entity.id)
            if math.dist(_point(native, 'insert'), transform(view, entity.position)) > comparison_tolerance_mm:
                issue('text_position', 'Text insertion differs from graph.', entity.id)
            if not finite(native.dxf.height) or not finite(native.dxf.rotation) or abs(native.dxf.height-entity.height*similarity_scale(view)) > comparison_tolerance_mm or angular_difference(native.dxf.rotation, angle(view, entity.rotation_deg)) > 1e-7:
                issue('text_style', 'Text height or rotation differs from graph.', entity.id)
            bounds[entity.view_id].append(_point(native, 'insert'))
        except (ValueError, KeyError, AttributeError) as error: issue('invalid_native_text', error, entity.id)
    budget = PredicateBudget(maximum_topology_predicates)
    contours = {c.id: c for c in graph.contours}
    actual_contours, expected_contours = {}, {}
    for contour in graph.contours:
        if contour.status == 'rejected': continue
        try:
            parts = oriented_segments(contour, actual_geometry)
            expected_contours[contour.id] = oriented_segments(contour, expected_geometry)
            loop = loop_data(parts, comparison_tolerance_mm, closed=contour.closed)
            check_simple(loop, comparison_tolerance_mm, budget)
            actual_contours[contour.id] = parts
            report['contours'].append({'id': contour.id, 'closed': contour.closed, 'max_gap_mm': loop['max_gap'],
                'signed_area_mm2': loop['signed_area'], 'chord_error_bound_mm': loop['chord_error_bound'], 'simple_checked': True})
        except (ValueError, KeyError) as error: issue('contour_topology', error, contour.id)
    for hatch in graph.hatches:
        if hatch.status == 'rejected': continue
        try:
            contour_ids = [hatch.outer_contour_id]+list(hatch.hole_contour_ids)
            hatch_allowed = _eligible(hatch) and all(_eligible(contours[i]) for i in contour_ids)
            if not hatch_allowed:
                if hatch.id in bindings: issue('unverified_native_hatch', 'Native HATCH is based on unverified evidence.', hatch.id)
                continue
            native = locate(hatch, 'HATCH')
            if native is None: continue
            if not all(finite(v) for v in native.dxf.elevation) or math.hypot(*native.dxf.elevation) > 1e-8 or native.dxf.hatch_style != 0 or native.dxf.associative != 0 or native.gradient is not None:
                issue('hatch_style', 'Expected planar normal-style, explicitly non-associative HATCH.', hatch.id)
            if len(native.paths.paths) != len(contour_ids): issue('hatch_loop_count', 'Outer/hole loop inventory differs.', hatch.id)
            loops = []
            for index, path_item in enumerate(native.paths.paths):
                if index >= len(contour_ids): break
                if not hasattr(path_item, 'vertices') or not path_item.is_closed: raise ValueError('HATCH requires closed native polyline paths')
                vertices = [tuple(float(v) for v in p) for p in path_item.vertices]
                if len(vertices) > 512 or any(not all(finite(v) for v in p) or abs(p[2]) > 1e6 for p in vertices):
                    raise ValueError('Invalid or oversized HATCH boundary')
                if path_item.path_type_flags != (3 if index == 0 else 2): issue('hatch_boundary_role', 'Outer/hole native flag differs.', hatch.id)
                expected_vertices = loop_vertices(expected_contours[contour_ids[index]], counterclockwise=index == 0)
                if not cyclic_equal(expected_vertices, vertices, comparison_tolerance_mm): issue('hatch_boundary_geometry', 'Native HATCH boundary differs from graph contour.', hatch.id)
                wanted_handles = {natives[e.primitive_id].dxf.handle for e in contours[contour_ids[index]].edges}
                if set(path_item.source_boundary_objects) != wanted_handles: issue('hatch_boundary_references', 'Native boundary reference handles differ.', hatch.id)
                parts = [(a[:2], b[:2], a[2]) for a, b in zip(vertices, vertices[1:]+vertices[:1])]
                loop = loop_data(parts, comparison_tolerance_mm)
                if (loop['signed_area'] > 0) != (index == 0): issue('hatch_winding', 'Outer must be CCW, holes CW.', hatch.id)
                loops.append(loop)
            net = check_holes(loops, comparison_tolerance_mm, budget)
            view = views[hatch.view_id]
            if native.dxf.pattern_name != hatch.pattern or bool(native.dxf.solid_fill) != (hatch.pattern == 'SOLID'):
                issue('hatch_pattern', 'Native fill type differs from graph.', hatch.id)
            if hatch.pattern != 'SOLID':
                if native.pattern is None or len(native.pattern.lines) != 1: issue('hatch_pattern', 'Expected one ANSI31 pattern line family.', hatch.id)
                else:
                    line = native.pattern.lines[0]
                    pattern_angle = angle(view, hatch.angle_deg)
                    spacing = hatch.spacing*similarity_scale(view)
                    expected_offset = (-spacing*math.sin(math.radians(pattern_angle)), spacing*math.cos(math.radians(pattern_angle)))
                    if not all(finite(v) for v in (*line.offset, *line.base_point, line.angle)) or math.hypot(*line.base_point) > comparison_tolerance_mm or math.dist(tuple(line.offset), expected_offset) > comparison_tolerance_mm or angular_difference(line.angle, pattern_angle) > 1e-7 or line.dash_length_items or native.dxf.pattern_double or not finite(native.dxf.pattern_angle) or angular_difference(native.dxf.pattern_angle, pattern_angle) > 1e-7 or native.dxf.pattern_scale != 1:
                        issue('hatch_pattern', 'Hatch angle/spacing/dashes differ from graph.', hatch.id)
            report['hatches'].append({'id': hatch.id, 'hole_count': max(0, len(loops)-1), 'net_area_mm2': net,
                                      'native_boundaries_checked': True})
        except (ValueError, KeyError, AttributeError, OverflowError) as error: issue('invalid_hatch', error, hatch.id)
    primitives = {p.id: p for p in graph.primitives}
    for dimension in graph.dimensions:
        if dimension.status == 'rejected': continue
        try:
            view = views[dimension.view_id]
            allowed = _eligible(dimension) and view.physical_scale_confirmed and all(_eligible(primitives[i]) for i in dimension.measured_entity_ids)
            if not allowed:
                if dimension.id in bindings: issue('unauthorized_native_dimension', 'Dimension lacks verified physical premises.', dimension.id)
                continue
            expected = dimension_expectation(dimension, view)
            native = locate(dimension, 'DIMENSION')
            if native is None: continue
            if native.dimtype != expected['native_kind']: issue('dimension_native_kind', 'Native dimension type changes measurement semantics.', dimension.id)
            for name, point in expected['definitions'].items():
                if math.dist(_point(native, name), point) > comparison_tolerance_mm: issue('dimension_definition_point', name, dimension.id)
            if dimension.kind in ('linear', 'aligned', 'longitudinal_diameter'):
                a, b = expected['points']
                expected_angle = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0])) if dimension.kind == 'aligned' else angle(view, dimension.angle_deg)
                if not finite(native.dxf.angle) or angular_difference(native.dxf.angle, expected_angle) > 1e-7:
                    issue('dimension_direction', 'Native projection/render direction differs from graph.', dimension.id)
                base, placement = _point(native, 'defpoint'), transform(view, dimension.placement)
                theta = math.radians(expected_angle)
                offset = -(base[0]-placement[0])*math.sin(theta)+(base[1]-placement[1])*math.cos(theta)
                if abs(offset) > comparison_tolerance_mm:
                    issue('dimension_placement', 'Native dimension line offset differs from graph placement.', dimension.id)
            elif dimension.kind == 'angular':
                if math.dist(_point(native, 'defpoint'), transform(view, dimension.placement)) > comparison_tolerance_mm:
                    issue('dimension_placement', 'Native angular base differs from graph placement.', dimension.id)
            if native.dxf.text != '<>': issue('dimension_literal_override', 'Dimension must retain computed <> text.', dimension.id)
            measured = float(native.get_measurement())
            if not finite(measured) or abs(measured-expected['measurement']) > comparison_tolerance_mm:
                issue('dimension_measurement', 'Native measurement differs from independently transformed graph points.', dimension.id)
            if abs(expected['measurement']-dimension.nominal) > comparison_tolerance_mm:
                issue('dimension_graph_nominal', 'Graph nominal differs from its physical points.', dimension.id)
            try:
                validate_attachments(dimension, expected['points'], [actual_geometry[i] for i in dimension.measured_entity_ids], comparison_tolerance_mm)
            except (ValueError, KeyError) as error: issue('dimension_attachment', error, dimension.id)
            style = native.override()
            prefix = dimension.prefix
            if dimension.kind == 'longitudinal_diameter' and not prefix.startswith(('Ø', '⌀', '%%c')): prefix = 'Ø'+prefix
            expected_style = {'dimlfac': 1., 'dimpost': prefix+'<>'+dimension.suffix,
                              'dimdec': 6, 'dimadec': 6, 'dimrnd': 1e-6, 'dimtxt': 2.5,
                              'dimzin': 8, 'dimazin': 2, 'dimtzin': 8, 'dimclrt': 256}
            if dimension.annotation_scale is not None: annotation_scale = dimension.annotation_scale
            else:
                model_a, model_b = transform(view, (0, 0)), transform(view, (1, 0))
                paper_a, paper_b = sheet_point(view, (0, 0)), sheet_point(view, (1, 0))
                annotation_scale = math.dist(model_a, model_b)/math.dist(paper_a, paper_b)
            expected_style['dimscale'] = annotation_scale
            for key, value in expected_style.items():
                actual = style.get(key)
                if isinstance(value, str): okay = actual == value
                else: okay = isinstance(actual, (int, float)) and finite(actual) and math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-10)
                if not okay: issue('dimension_style', key, dimension.id)
            deviations = dimension_tolerances(dimension)
            tol = deviations is not None
            if bool(style.get('dimtol', 0)) != tol or (tol and (not finite(style.get('dimtp', 0)) or not finite(style.get('dimtm', 0)) or abs(style.get('dimtp', 0)-deviations.upper) > 1e-9 or abs(style.get('dimtm', 0)+deviations.lower) > 1e-9)):
                issue('dimension_tolerances', 'Native tolerance style differs from graph.', dimension.id)
            actual_text = _cached_text(native, annotation_scale, computed_text(dimension, expected['measurement'], include_tolerance=False), tol)
            wanted_text = computed_text(dimension, expected['measurement'])
            native_text = computed_text(dimension, measured) if finite(measured) else None
            if actual_text != wanted_text or actual_text != native_text: issue('stale_dimension_text', 'Actual cached text differs from graph or current native measurement/affixes/tolerances.', dimension.id)
            report['dimensions'].append({'id': dimension.id, 'expected_measurement': expected['measurement'],
                'native_measurement': measured if finite(measured) else None, 'expected_text': wanted_text,
                'actual_cached_text': actual_text, 'cached_text_matches': actual_text == wanted_text,
                'current_native_computed_text': native_text,
                'native_kind': native.dimtype, 'annotation_scale': annotation_scale})
            bounds[dimension.view_id].extend(expected['points']+[transform(view, dimension.placement), _point(native, 'text_midpoint')])
            bounds[dimension.view_id].extend(_point(t, 'insert') for t in native.get_geometry_block() if t.dxftype() in ('TEXT', 'MTEXT'))
        except (ValueError, KeyError, AttributeError, TypeError, OverflowError, ezdxf.DXFError) as error:
            issue('invalid_dimension', error, dimension.id)
    unexpected = sorted(set(by_handle)-used)
    if unexpected: issue('unexpected_native_entities', ','.join(unexpected))
    try:
        sheet = doc.layouts.get('Sheet')
        if set(doc.layouts.names()) != {'Model', 'Sheet'}:
            issue('unexpected_layout', 'Unexpected native layout inventory.')
        for other in doc.layouts:
            if other.name not in ('Model', 'Sheet') and any(e.dxftype() != 'VIEWPORT' or e.dxf.id != 1 for e in other):
                issue('unexpected_layout_content', 'Undeclared paper layout contains drawing content.')
        if abs(sheet.dxf.paper_width-graph.page.width_mm) > 1e-7 or abs(sheet.dxf.paper_height-graph.page.height_mm) > 1e-7:
            issue('sheet_size', 'Paper dimensions differ from graph.')
        viewports = [e for e in sheet if e.dxftype() == 'VIEWPORT' and e.dxf.id != 1]
        used_viewports = set()
        all_view_layers = set(layer_names.values())
        for view in graph.views:
            if not bounds[view.id]: continue
            own = {value for (vid, _), value in layer_names.items() if vid == view.id}
            wanted_frozen = all_view_layers-own
            candidates = [v for v in viewports if v.dxf.handle == viewport_bindings.get(view.id)] if view.id in viewport_bindings else [v for v in viewports if set(v.frozen_layers) == wanted_frozen]
            if len(candidates) != 1:
                issue('viewport_binding', 'Expected exactly one native viewport for the view.', view.id)
                continue
            viewport = candidates[0]; used_viewports.add(viewport.dxf.handle)
            if viewport.dxf.status <= 0 or viewport.dxf.flags & 0x20000:
                issue('inactive_viewport', 'Declared view is disabled.', view.id)
            if set(viewport.frozen_layers) != wanted_frozen: issue('viewport_layer_isolation', 'Frozen layers do not isolate the view.', view.id)
            if math.dist(tuple(viewport.dxf.view_direction_vector), (0, 0, 1)) > 1e-10 or abs(viewport.dxf.view_target_point.z) > 1e-10 or viewport.dxf.flags & 1:
                issue('viewport_projection', 'Expected an orthographic top view.', view.id)
            width, height, view_height = viewport.dxf.width, viewport.dxf.height, viewport.dxf.view_height
            if not all(finite(x) and x > 0 for x in (width, height, view_height)): raise ValueError('Invalid viewport extent')
            ratio = height/view_height
            if not finite(ratio): raise ValueError('Viewport scale is outside finite comparison range')
            theta = math.radians(viewport.dxf.view_twist_angle); c, s = math.cos(theta), math.sin(theta)
            center, target, dcs = viewport.dxf.center, viewport.dxf.view_target_point, viewport.dxf.view_center_point
            def project(p):
                x, y = p[0]-target.x, p[1]-target.y
                return (ratio*(c*x-s*y-dcs.x)+center.x, ratio*(s*x+c*y-dcs.y)+center.y)
            errors = [math.dist(project(transform(view, p)), sheet_point(view, p)) for p in ((0, 0), (1, 0), (0, 1))]
            if any(not finite(e) or e > comparison_tolerance_mm for e in errors): issue('viewport_transform', 'Native viewport does not reproduce graph sheet transform.', view.id)
            if any(abs(project(p)[0]-center.x) > width/2+comparison_tolerance_mm or abs(project(p)[1]-center.y) > height/2+comparison_tolerance_mm for p in bounds[view.id]):
                issue('viewport_clipping', 'Viewport clips required geometry or dimension positions.', view.id)
            report['viewports'].append({'id': view.id, 'paper_per_model': ratio, 'max_probe_error_mm': max(errors) if all(finite(e) for e in errors) else None, 'probe_count': 3})
        if len(used_viewports) != len(viewports): issue('unexpected_viewports', 'Unused or duplicate active sheet viewports exist.')
        if any(e.dxftype() != 'VIEWPORT' for e in sheet): issue('unexpected_sheet_entity', 'Sheet includes content outside declared viewports.')
    except (KeyError, ValueError, AttributeError, OverflowError) as error: issue('invalid_sheet', error)
    report['topology_predicates_used'] = budget.used
    report['input_dxf_unchanged'] = _digest(path) == before
    if not report['input_dxf_unchanged']: issue('input_changed', 'DXF bytes changed during read-only validation.')
    report['status'] = ('failed' if any(i['severity'] == 'error' for i in report['issues']) else
                        'needs_review' if report['issues'] else 'passed_against_graph')
    return report
