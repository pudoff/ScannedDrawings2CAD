"""Assemble pinned real observations and an explicitly synthetic schema control."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .clear_pairs import contained, digest
from .family_catalog_v2 import require, stable_digest
from .graph_v2 import read_graph_v2
from .tolerances import dimension_tolerances
from .label_schema_v1 import FIELDS, LabelPackage, transform_points_v1, read_json_v1 as read_json
from .label_release_v1 import (build_derivative_index_v1, checked_pin, screen_label_package_v1,
                               validate_derivative_index_v1)

DEPENDENCIES = {'catalog', 'split_policy', 'family_final', 'exclusions', 'field_review', 'field_final',
                'raster_evidence', 'raster_seal', 'point_report', 'point_references', 'point_seal',
                'control_graph', 'control_source', 'control_expectations', 'control_spec'}
MODULES = {'label_schema_v1.py', 'label_corpus_v1.py', 'label_release_v1.py', 'family_catalog_v2.py',
           'family_split_v2.py', 'training_target_policy.py', 'clear_pairs.py', 'graph_v2.py',
           'dimension_field_raster.py', 'raster_regions.py', 'raster_point_references.py', 'schema.py',
           'exporter_v2.py', 'topology_v2.py'}


def pin(workspace, path):
    path = Path(path).resolve()
    return {'path': path.relative_to(Path(workspace).resolve()).as_posix(), 'sha256': digest(path)}


def evidence(artifact, locator, method='review_record', domain='customer_observation', statement=None):
    return {'artifact': artifact, 'locator': locator, 'method': method, 'domain': domain,
            'statement': statement or 'Pinned observation only; neither physical ground truth nor customer approval.'}


def field(value, ev, *, unit='structured', frame=None, status=None, basis=None):
    return {'status': status or ('unknown' if value is None else 'observation'), 'value': deepcopy(value),
            'unit': unit, 'frame_id': frame, 'evidence': [ev],
            'quality': {'basis': basis or ('Not reviewed; no value inferred.' if value is None else ev['statement']),
                        'accuracy_bound': None, 'bound_unit': None, 'statistical_confidence': None}}


def obj(identifier, kind, fields, ev, view=None):
    return {'id': identifier, 'kind': kind, 'view_id': view,
            'fields': {key: fields.get(key, field(None, ev)) for key in sorted(FIELDS[kind])}}


def transform(identifier, start, end, matrix, ev, status='observation'):
    matrix = np.asarray(matrix, dtype=float)
    return {'id': identifier, 'from_frame': start, 'to_frame': end, 'matrix': matrix.tolist(),
            'inverse': np.linalg.inv(matrix).tolist(), 'status': status, 'evidence': ev}


def project(matrix, points):
    return (np.column_stack((np.asarray(points)[:, :2], np.ones(len(points)))) @ np.asarray(matrix).T)[:, :2].tolist()


def coverage(objects, domain):
    counts = Counter(o['kind'] for o in objects)
    return [{'kind': kind, 'object_count': counts[kind],
             'state': ('partial_observations' if domain == 'customer_observation' else 'analytical_control') if counts[kind] else 'unreviewed',
             'reason': ('Available field observations; missing values remain unknown.' if counts[kind]
                        else 'No completed field review exists; this is not a true negative label.')}
            for kind in sorted(FIELDS)]


def build_customer_labels_v1(pair, catalog, dependency_pins, points):
    """Convert existing review facts; never relabel their uncertainty as exact truth."""
    matches = [row for row in catalog['pairs'] if row['input_path'] == pair['input_path'] and row['target_path'] == pair['target_path']]
    require(len(matches) == 1, 'Reviewed pair must have one catalog record')
    match = matches[0]
    require(match['input_sha256'] == pair['input_sha256'] and match['target_sha256'] == pair['target_sha256']
            and match['target_status'] != 'excluded', 'Reviewed pair revision changed or target is excluded')
    ev = evidence(dependency_pins['field_review'], '/pairs/'+pair['source_id'])
    source = {'path': 'dataset/'+pair['input_path'], 'sha256': pair['input_sha256']}
    target = {'path': 'dataset/'+pair['target_path'], 'sha256': pair['target_sha256']}
    frames = [{'id': 'source', 'unit': 'source_pixel', 'axes': 'x_right_y_down', 'source_sha256': source['sha256'],
               'physical_scale_confirmed': False, 'meaning': 'Original PNG pixel centers; top-left origin; half-open region boxes.'},
              {'id': 'native', 'unit': 'native_unit', 'axes': 'x_right_y_up', 'source_sha256': source['sha256'],
               'physical_scale_confirmed': False, 'meaning': 'DXF WCS XY drawing units; header and DIMLFAC do not establish physical scale.'}]
    alignment = pair['fixed_alignment']['cad_to_source_pixel']
    transforms = [transform('native-to-source', 'native', 'source', alignment,
                            evidence(dependency_pins['field_review'], ev['locator']+'/fixed_alignment', 'derived_projection'))]
    objects = []
    for view in pair['views']:
        vid = 'view:'+view['id']
        box = view['source_box_px']
        frames.append({'id': vid, 'unit': 'view_pixel', 'axes': 'x_right_y_up', 'source_sha256': source['sha256'],
                       'physical_scale_confirmed': False, 'meaning': 'Local pixel-center axes from authored view crop; extent remains an observation.'})
        transforms.append(transform('source-to-'+vid, 'source', vid,
                                    [[1, 0, -box[0]], [0, -1, box[3]-1], [0, 0, 1]], ev))
        objects.append(obj(vid, 'view', {'extent': field(box, ev, unit='source_pixel', frame='source'),
            'printed_label': field(view['printed_label'], ev, unit='text'),
            'scale_caption': field(view['printed_scale_inherited_from_title'], ev, unit='text')}, ev))
    objects.append(obj('title', 'title_block', {'readings': field(pair['source_title_readings'], ev),
        'native_text': field(pair['native_title_instances'], ev),
        'identity': field({'filename_designation': pair['filename_designation'],
                          'filename_matches_native_title': pair['filename_designation_matches_native_title'],
                          'meaning': 'File/stamp discrepancy retained, no identity repair inferred.'}, ev)}, ev))
    for item in pair['objects']:
        handle = item['identity']['handle']
        base = 'dim:'+handle
        dims = item['fields']
        reading = item['source_reading']
        de = evidence(dependency_pins['field_review'], ev['locator']+'/objects/'+handle)
        re = evidence(dependency_pins['raster_evidence'], '/sources/'+pair['source_id']+'/dimension_readings/'+reading['id'], 'raster_transcription')
        vid = 'view:'+dims['view_membership']['view_id']
        objects.append(obj('text:'+handle, 'text', {'transcription': field(reading['text'], re, unit='text'),
            'box': field(reading['label_source_box_px'], re, unit='source_pixel', frame='source'),
            'native_text': field(item['native']['display_text_pieces'], de)}, de, vid))
        anchor_ids = []
        for name, attachment in dims['geometry_role_and_attachment'].items():
            aid = base+':'+name
            anchor_ids.append(aid)
            point = item['native']['definition_points'][name]
            interval_info = dims['independent_anchor_intervals']['points'].get(name)
            objects.append(obj(aid, 'anchor', {'native_point': field(point, de, unit='native_unit', frame='native'),
                'raster_projection': field(attachment['projected_definition_point_xy'], de, unit='source_pixel', frame='source',
                                            basis='Fixed native-to-raster projection; no measured raster accuracy.'),
                'binding': field({'role_observation': attachment['role_observation'],
                    'context_covers_support': attachment['definition_point_inside_context_region'],
                    'context_review_status': attachment['status'], 'independent_reference_links': interval_info,
                    'native_relation_object_ids': [aid+':relation:'+str(i) for i in range(len(attachment['native_relations']))]}, de)}, de, vid))
            for i, relation in enumerate(attachment['native_relations']):
                binding = relation['binding']
                finite = binding['exact_finite_geometry_attachment']
                geom = relation['native_geometry']
                objects.append(obj(aid+':relation:'+str(i), 'geometry', {
                    'shape': field(geom, de, unit='native_unit', frame='native'),
                    'role': field({'role': attachment['role_observation'], 'basis': attachment['role_basis'],
                                   'context_covers_support': attachment['definition_point_inside_context_region']}, de),
                    'binding': field(binding, de, status='observation' if finite else 'rejected',
                        basis='Finite native incidence only; physical attachment is unapproved.' if finite
                        else 'Infinite continuation is not attachment to a finite geometric entity.')}, de, vid))
        for i, interval in enumerate(reading['author_support']['source_windows_px']):
            ae = evidence(dependency_pins['raster_evidence'], re['locator']+'/author_support', 'author_interval')
            objects.append(obj('arrow:'+handle+':'+str(i), 'arrow', {
                'contact_interval': field({'box': interval, 'role': reading['author_support']['role']}, ae,
                    unit='source_pixel', frame='source', basis='Authored display arrow/leader window, never a measured geometry support.')}, ae, vid))
        objects.append(obj(base, 'dimension', {'notation': field(dims['notation'], re),
            'printed_nominal': field(dims['notation'].get('nominal'), re, unit='printed_value'),
            'supports': field(anchor_ids, de), 'view_scale': field(dims['view_scale'], de),
            'numeric_consistency': field(dims['numeric_consistency'], de)}, de, vid))
    for point in points:
        if point['source_sha256'] != source['sha256']:
            continue
        require(point['target_sha256'] == target['sha256'], 'Independent point reference native target mismatch')
        pe = evidence(dependency_pins['point_report'], '/points/'+point['point_id'], 'author_interval')
        objects.append(obj('reference:'+point['point_id'], 'anchor', {
            'native_point': field(point['cad_xyz'], pe, unit='native_unit', frame='native'),
            'raster_interval': field(point['interval_source_xyxy'], pe, unit='source_pixel', frame='source',
                basis='Independent authored interval; center is representative, not exact/stochastic ground truth.'),
            'raster_projection': field(point['projected_source_xy'], pe, unit='source_pixel', frame='source'),
            'binding': field({'native_features': point['cad_features'], 'comparison': point['comparison']}, pe)}, pe))
    for record in pair['curve_and_thin_line_observations']:
        ce = evidence(dependency_pins['raster_evidence'], '/sources/'+pair['source_id']+'/curve_and_thin_line_observations/'+record['id'], 'author_interval')
        objects.append(obj('curve-context:'+record['id'], 'geometry', {'role': field(record, ce)}, ce))
    return LabelPackage.model_validate({'id': 'customer-'+pair['source_id'], 'domain': 'customer_observation',
        'source': source, 'native_target': target, 'lineage': {'group_id': match['group_id'], 'family_id': match['family_id'],
            'train_only': match['train_only'], 'historical_reserved_split': match['group_reserved_split'], 'catalog': dependency_pins['catalog']},
        'frames': frames, 'transforms': transforms, 'objects': objects, 'coverage': coverage(objects, 'customer_observation'),
        'dependencies': list(dependency_pins.values()), 'assembly_inputs': dependency_pins, 'limitations': [
            'Observed source readings, native records and authored intervals are not approved physical labels.',
            'Native projections and role observations remain partial, including contexts that miss supports and rejected infinite incidences.',
            'Arrow contact windows do not locate measured geometry supports. No exact raster anchors or accuracy bounds are inferred.',
            'Physical scale, tolerances, contours, holes, HATCH boundaries and complete line geometry remain pending.',
            'Caption, DIMLFAC and filename similarity cannot grant scale, family identity or training eligibility.']})


def verify_analytical_construction_v1(graph_value, spec, dependency_pins, expectations):
    """Check raw JSON dictionaries before minting analytical fields.

    Raw input content is sealed independently of Pydantic defaults; a model or a
    migrated graph is not interchangeable with that pinned JSON definition.
    """
    from .exporter_v2 import dimension_measurement, expected_dimension_text
    from .topology_v2 import validate_contour, validate_hatch_boundaries
    require(isinstance(graph_value, dict) and isinstance(spec, dict) and isinstance(expectations, dict),
            'Analytical assembly requires raw pinned JSON objects')
    graph = read_graph_v2(graph_value)
    require(graph.source.path == dependency_pins['control_source']['path']
            and graph.source.sha256 == dependency_pins['control_source']['sha256'], 'Control graph source binding differs')
    require(spec['purpose'] == 'analytical_coordinate_plane_without_customer_raster' and spec['customer_approval'] is False,
            'Control spec must be explicitly analytical')
    require(spec['graph'] == dependency_pins['control_graph'] and spec['source'] == dependency_pins['control_source'],
            'Analytical definition must bind the exact graph/source revision')
    require(spec['expectations'] == dependency_pins['control_expectations']
            and stable_digest(graph_value) == spec['graph_content_sha256']
            and stable_digest(expectations) == spec['expectations_content_sha256'],
            'Analytical input content does not match the pinned construction definition')
    require(expectations['control_only'] is True and expectations['physical_scale_is_synthetic_premise'] is True,
            'Analytical expected values must be explicitly synthetic')
    entities = [*graph.primitives, *graph.texts, *graph.dimensions, *graph.contours, *graph.hatches]
    require(all(e.status == 'accepted' and e.evidence.status == 'verified'
                and e.evidence.source_ref == graph.source.path for e in entities),
            'Analytical entity facts require accepted construction evidence')
    geometries = {p.id: p for p in graph.primitives}
    views = {v.id: v for v in graph.views}
    contours = {c.id: c for c in graph.contours}
    require(set(expectations['nominals']) == {d.id for d in graph.dimensions}
            and set(expectations['display_texts']) == {d.id for d in graph.dimensions}, 'Control dimension inventory differs')
    measurements = {}
    for dimension in graph.dimensions:
        matrix = views[dimension.view_id].model_matrix()
        require(matrix is not None, 'Analytical physical measurement requires constructed scale')
        measured, _ = dimension_measurement(dimension, geometries, np.asarray(matrix))
        require(math.isclose(measured, expectations['nominals'][dimension.id], rel_tol=1e-10, abs_tol=1e-8)
                and expected_dimension_text(dimension, measured) == expectations['display_texts'][dimension.id],
                'Analytical measured nominal/display differs from independently pinned expectation')
        measurements[dimension.id] = measured
    for contour in graph.contours:
        validate_contour(contour, geometries)
    areas = []
    for hatch in graph.hatches:
        _, area = validate_hatch_boundaries(contours[hatch.outer_contour_id], [contours[h] for h in hatch.hole_contour_ids], geometries)
        require(len(hatch.hole_contour_ids) == expectations['hatch_holes']
                and math.isclose(area, expectations['hatch_net_area'], rel_tol=1e-10, abs_tol=1e-8),
                'Analytical HATCH topology/area differs from pinned expectations')
        areas.append(area)
    types = {'line': 'LINE', 'circle': 'CIRCLE', 'arc': 'ARC', 'polyline': 'LWPOLYLINE'}
    counts = Counter(types[p.kind] for p in graph.primitives)
    counts.update({'TEXT': len(graph.texts), 'HATCH': len(graph.hatches), 'DIMENSION': len(graph.dimensions)})
    require(dict(counts) == expectations['counts'], 'Analytical geometry inventory differs')
    require(set(expectations['view_paper_per_model']) == set(views), 'Analytical view inventory differs')
    for view in graph.views:
        factor = expectations['view_paper_per_model'][view.id]
        require(view.physical_scale_confirmed and view.model_scale_xy is not None
                and all(math.isclose(x, 1/factor, rel_tol=1e-12) for x in view.model_scale_xy),
                'Analytical view scale differs from construction expectation')
    return {'measurements': measurements, 'hatch_net_areas': areas, 'finite_supports_checked': True,
            'topology_checked': True, 'customer_approval': False}


def build_analytical_labels_v1(graph_value, spec, dependency_pins, expectations):
    construction = verify_analytical_construction_v1(graph_value, spec, dependency_pins, expectations)
    graph = read_graph_v2(graph_value)
    source = dependency_pins['control_spec']
    ev = evidence(source, '/', 'analytical_exact', 'analytical_control',
                  'Exact synthetic construction premise only; virtual pixel plane, no customer raster/approval.')
    frames = [{'id': 'source', 'unit': 'analytic_pixel', 'axes': 'x_right_y_down', 'source_sha256': source['sha256'],
               'physical_scale_confirmed': False, 'meaning': 'Virtual source pixel coordinate plane defined by the pinned analytical spec; no bitmap ground truth.'},
              {'id': 'sheet', 'unit': 'paper_mm', 'axes': 'x_right_y_up', 'source_sha256': source['sha256'],
               'physical_scale_confirmed': False, 'meaning': 'Analytical sheet coordinates.'}]
    transforms = [transform('source-to-sheet', 'source', 'sheet', spec['source_to_sheet'], ev, 'verified')]
    objects = []
    def vf(value, unit='structured', frame=None):
        return field(value, ev, unit=unit, frame=frame, status='verified' if value is not None else 'unknown')
    for view in graph.views:
        vid, mid = 'view:'+view.id, 'model:'+view.id
        require(view.physical_scale_confirmed and view.model_matrix() is not None, 'Control needs explicit synthetic scale')
        frames.extend([{'id': vid, 'unit': view.coordinate_space, 'axes': 'x_right_y_up', 'source_sha256': source['sha256'],
                        'physical_scale_confirmed': False, 'meaning': 'Analytical graph local view coordinates.'},
                       {'id': mid, 'unit': 'model_mm', 'axes': 'x_right_y_up', 'source_sha256': source['sha256'],
                        'physical_scale_confirmed': True, 'meaning': 'Analytical physical coordinates from explicitly constructed rulers.'}])
        transforms.append(transform('source-to-'+vid, 'source', vid,
            np.linalg.inv(np.asarray(view.view_to_sheet)) @ np.asarray(spec['source_to_sheet']), ev, 'verified'))
        transforms.append(transform(vid+'-to-model', vid, mid, view.model_matrix(), ev, 'verified'))
        objects.append(obj(vid, 'view', {'printed_label': vf(view.name, 'text'),
            'physical_scale': vf(list(view.model_scale_xy), 'dimensionless')}, ev))
    layers = {l.name: l.model_dump(mode='json') for l in graph.layers}
    for geom in graph.primitives:
        vid, mid = 'view:'+geom.view_id, 'model:'+geom.view_id
        shape = geom.model_dump(mode='json')
        objects.append(obj('geometry:'+geom.id, 'geometry', {'shape': vf(shape, 'paper_mm', vid),
            'role': vf(layers[geom.layer]), 'physical_geometry': vf({'view_geometry_ref': 'geometry:'+geom.id,
                'transform_id': vid+'-to-model'}, 'model_mm', mid)}, ev, vid))
    for text in graph.texts:
        objects.append(obj('text:'+text.id, 'text', {'transcription': vf(text.raw, 'text'),
            'native_text': vf(text.model_dump(mode='json'))}, ev, 'view:'+text.view_id))
    for dim in graph.dimensions:
        vid, mid = 'view:'+dim.view_id, 'model:'+dim.view_id
        anchor_ids = []
        view = next(v for v in graph.views if v.id == dim.view_id)
        source_to_view = np.linalg.inv(np.asarray(view.view_to_sheet)) @ np.asarray(spec['source_to_sheet'])
        for i, point in enumerate(dim.points):
            aid = 'anchor:'+dim.id+':'+str(i)
            anchor_ids.append(aid)
            objects.append(obj(aid, 'anchor', {
                'native_point': vf(list(point), 'paper_mm', vid),
                'raster_projection': vf(project(np.linalg.inv(source_to_view), [point])[0], 'source_pixel', 'source'),
                'physical_point': vf(project(view.model_matrix(), [point])[0], 'model_mm', mid),
                'binding': vf({'measured_entity_ids': dim.measured_entity_ids, 'definition_index': i})}, ev, vid))
        deviations = dimension_tolerances(dim)
        objects.append(obj('dimension:'+dim.id, 'dimension', {'notation': vf({'kind': dim.kind, 'raw_text': dim.raw_text,
            'prefix': dim.prefix, 'suffix': dim.suffix}), 'printed_nominal': vf(dim.nominal, 'printed_value'),
            'physical_nominal': vf(dim.nominal, 'degree' if dim.kind == 'angular' else 'model_mm'),
            'tolerances': vf({'plus': deviations.upper if deviations is not None else None,
                              'minus': deviations.lower if deviations is not None else None,
                              'convention': 'signed_deviations',
                              'meaning': 'Signed upper/lower deviations from nominal; null means no tolerance.'}),
            'supports': vf(anchor_ids), 'view_scale': vf(list(view.model_scale_xy)),
            'numeric_consistency': vf({'expected': expectations['nominals'][dim.id],
                'measured_from_finite_geometry': construction['measurements'][dim.id],
                'construction': 'pinned graph-v2-control-expectations'})}, ev, vid))
    for contour in graph.contours:
        vid = 'view:'+contour.view_id
        objects.append(obj('contour:'+contour.id, 'contour', {'edges': vf([e.model_dump(mode='json') for e in contour.edges]),
            'closure': vf(contour.closed), 'role': vf(contour.role),
            'physical_boundary': vf({'contour_ref': 'contour:'+contour.id, 'transform_id': vid+'-to-model'})}, ev, vid))
        if contour.role == 'hole':
            objects.append(obj('hole:'+contour.id, 'hole', {'boundary': vf('contour:'+contour.id),
                'parent': vf('contour:'+contour.parent_id),
                'physical_boundary': vf({'contour_ref': 'contour:'+contour.id, 'transform_id': vid+'-to-model'})}, ev, vid))
    for hatch in graph.hatches:
        objects.append(obj('hatch:'+hatch.id, 'hatch', {'boundary': vf({'outer': 'contour:'+hatch.outer_contour_id,
            'holes': ['contour:'+h for h in hatch.hole_contour_ids]}),
            'pattern': vf({'pattern': hatch.pattern, 'angle_deg': hatch.angle_deg, 'spacing': hatch.spacing})}, ev, 'view:'+hatch.view_id))
    return LabelPackage.model_validate({'id': 'analytical-graph-v2-control', 'domain': 'analytical_control',
        'source': source, 'lineage': {}, 'frames': frames, 'transforms': transforms, 'objects': objects,
        'coverage': coverage(objects, 'analytical_control'), 'dependencies': list(dependency_pins.values()), 'assembly_inputs': dependency_pins,
        'limitations': ['Virtual pixel plane tests coordinate recovery and schema conversion; it is not a rendered bitmap or customer source.',
            'Verified means the analytical construction only, never a customer approval or dataset release.',
            'Arrow glyphs, raster masks and absent title values have no exact labels in this control.']})


def coordinate_check_v1(package):
    package = LabelPackage.model_validate(package)
    cases = []
    probes = [[0., 0.], [1., 2.], [100., 200.], [-3., 7.]]
    for t in package.transforms:
        mapped = transform_points_v1(package, t.id, probes)
        reverse = transform_points_v1(package, t.id, mapped['points'], inverse=True)
        error = float(np.max(np.abs(np.asarray(reverse['points'])-probes)))
        require(error < 1e-8, 'Coordinate roundtrip failed')
        cases.append({'transform_id': t.id, 'max_roundtrip_error': error,
                      'status_preserved': mapped['status'] == reverse['status'] == t.status})
    anchors = []
    if package.domain == 'analytical_control':
        for item in package.objects:
            if item.kind != 'anchor':
                continue
            local = item.fields['native_point'].value
            source_xy = item.fields['raster_projection'].value
            recovered = transform_points_v1(package, 'source-to-'+item.view_id, [source_xy])['points'][0]
            physical = transform_points_v1(package, item.view_id+'-to-model', [recovered])['points'][0]
            error = max(float(np.max(np.abs(np.asarray(recovered)-local))),
                        float(np.max(np.abs(np.asarray(physical)-item.fields['physical_point'].value))))
            require(error < 1e-8, 'Analytical source/view/model point recovery failed')
            anchors.append({'anchor_id': item.id, 'max_error': error})
    return {'package_id': package.id, 'transforms': cases, 'analytical_anchor_recovery': anchors,
            'meaning': 'Arithmetic checks do not approve customer registration or physical scale.'}


def _write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def run_label_corpus_v1(workspace, config_path, output):
    workspace, config_path, output = (Path(p).resolve() for p in (workspace, config_path, output))
    require(config_path.is_relative_to(workspace), 'Config must remain inside the workspace')
    require(output.is_relative_to(workspace/'artifacts') and output != workspace/'artifacts', 'Use a new artifacts directory')
    if output.exists():
        raise FileExistsError('Preserve prior label corpus versions')
    config = read_json(config_path)
    require(config['schema_version'] == 'labels-run/1.0' and config['purpose'] == 'observation_corpus_no_training_release'
            and config['training_eligible'] is False and set(config['dependencies']) == DEPENDENCIES
            and set(config['implementation']) == MODULES, 'Invalid or incomplete label corpus config')
    pins = [pin(workspace, config_path)]
    values = {}
    for name, item in config['dependencies'].items():
        path = checked_pin(workspace, item)
        values[name] = read_json(path) if path.suffix == '.json' else path.read_text(encoding='utf-8')
        pins.append(item)
    for name, sha in config['implementation'].items():
        item = {'path': 'src/cad_recovery/'+name, 'sha256': sha}
        checked_pin(workspace, item)
        pins.append(item)
    for path, sha in config['entrypoints'].items():
        item = {'path': path, 'sha256': sha}
        checked_pin(workspace, item)
        pins.append(item)
    # Existing strict verifier reconstructs every one of the 52 raster ROIs.
    from .dimension_field_raster import verify_field_raster_evidence
    raster_check = verify_field_raster_evidence(workspace, config['dependencies']['raster_evidence'], config['dependencies']['raster_seal'])
    from .raster_point_references import validate_raster_references
    pp, ps = config['dependencies']['point_references'], config['dependencies']['point_seal']
    point_check = validate_raster_references(workspace, workspace/pp['path'], workspace/ps['path'],
                                           expected_reference_sha256=pp['sha256'], expected_seal_sha256=ps['sha256'])
    for point in values['point_report']['points']:
        e = point['reference_evidence']
        checked_pin(workspace, {'path': e['raw_crop_path'], 'sha256': e['raw_crop_sha256']})
    catalog = values['catalog']
    originals = [{'path': 'dataset/'+row['path'], 'sha256': row['sha256']} for row in catalog['files']]
    for item in originals:
        checked_pin(workspace, item)
    derivative_index = build_derivative_index_v1(workspace, catalog, values['family_final'])
    validate_derivative_index_v1(catalog, derivative_index)
    packages = [build_customer_labels_v1(pair, catalog, config['dependencies'], values['point_report']['points'])
                for pair in values['field_review']['pairs']]
    packages.append(build_analytical_labels_v1(values['control_graph'], values['control_spec'], config['dependencies'], values['control_expectations']))
    coordinates = [coordinate_check_v1(p) for p in packages]
    screens = [screen_label_package_v1(workspace, p, catalog, values['split_policy'], values['exclusions']) for p in packages]
    queue = {'schema_version': 'labels-queue/1.0', 'catalog': config['dependencies']['catalog'],
             'split_policy': config['dependencies']['split_policy'], 'training_eligible': False,
             'entries': [{**{k: p[k] for k in ['pair_id', 'input_path', 'input_sha256', 'target_path', 'target_sha256',
                                              'group_id', 'family_id', 'family_status', 'train_only', 'target_status']},
                         'status': 'excluded' if p['target_status'] == 'excluded' else 'pending_physical_field_review',
                         'observation_package_id': next((x.id for x in packages if x.source.path == 'dataset/'+p['input_path']
                                                         and x.native_target and x.native_target.path == 'dataset/'+p['target_path']), None),
                         'pending_reasons': p['pending_reasons'], 'split': None, 'annotation_approved': False,
                         'training_eligible': False} for p in catalog['pairs']]}
    counts = Counter(o.kind for p in packages if p.domain == 'customer_observation' for o in p.objects)
    statuses = Counter(f.status for p in packages if p.domain == 'customer_observation' for o in p.objects for f in o.fields.values())
    control_statuses = Counter(f.status for o in packages[-1].objects for f in o.fields.values())
    report = {'schema_version': 'labels-run/1.0', 'status': 'observation_schema_and_snapshot_complete_physical_release_pending',
        'config': pins[0], 'dependencies': config['dependencies'], 'summary': {
            'customer_packages': len(packages)-1, 'analytical_control_packages': 1,
            'customer_objects_by_kind': dict(sorted(counts.items())), 'customer_fields_by_status': dict(sorted(statuses.items())),
            'control_fields_by_status': dict(sorted(control_statuses.items())), 'queue_pairs': len(queue['entries']),
            'queue_excluded': sum(e['status'] == 'excluded' for e in queue['entries']),
            'source_files_rehashed': len(originals), 'prepared_sources_rehashed': derivative_index['source_count'],
            'derivative_files_rehashed': len(derivative_index['artifacts']),
            'cross_group_duplicate_hashes': len(derivative_index['cross_group_content']),
            'active_split_counts': {s: 0 for s in ['train', 'validation', 'test']}, 'training_released': 0,
            'augmentations_created': 0}, 'raster_verification': raster_check['verification'],
        'point_reference_verification': point_check['verification'],
        'coordinate_checks': coordinates, 'selection_gates': screens, 'training_eligible': False,
        'annotation_approved': False, 'source_modified': False, 'split_modified': False,
        'remaining': ['P14.1b: physical geometry/scale/nominal/tolerance/anchor/topology approvals and customer label release',
                      '373 new semantic families unresolved; 86 derivative hashes need deduplication or reviewed joint allocation',
                      'Split and complete label release must precede any augmentation or training']}
    # Recheck every pin and original before creating the immutable output.
    for item in [*pins, *originals]:
        checked_pin(workspace, item)
    output.mkdir(parents=True)
    outputs = {'report.json': report, 'queue.json': queue, 'derivatives.json': derivative_index,
               'schema.json': LabelPackage.model_json_schema()}
    outputs.update({p.id+'.json': p.model_dump(mode='json') for p in packages})
    for name, value in outputs.items():
        _write(output/name, value)
    _write(output/'manifest.json', {'schema_version': 'labels-manifest/1.0', 'files': [
        {'path': name, 'sha256': digest(output/name)} for name in sorted(outputs)],
        'training_eligible': False, 'source_modified': False})
    return report
