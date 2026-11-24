"""Local endpoint/junction evidence linked to an immutable object review."""
from __future__ import annotations

from collections import Counter
import html
import math
from pathlib import Path

import ezdxf
import numpy as np
from PIL import Image, ImageDraw

from .object_review_case import _digest, _identifier, _json, _path, _pinned, _write_json, _xy
from .raster_endpoints import localize_endpoint
from .raster_junctions import localize_junction
from .raster_regions import read_prepared_region


def _finite_number(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'Invalid {name}')
    return value


def _cad_ray(doc, spec):
    """Return a finite straight edge with explicit contour/axis/annotation origin."""
    entity = doc.entitydb.get(spec.get('geometry_handle'))
    role = spec.get('role', 'contour')
    if entity is None or role not in ('contour', 'axis', 'annotation'):
        raise ValueError('Invalid anchor ray entity or role')
    if role == 'annotation':
        dimension = doc.entitydb.get(spec.get('dimension_handle'))
        if (dimension is None or dimension.dxftype() != 'DIMENSION'
                or dimension.dxf.owner != doc.modelspace().block_record_handle):
            raise ValueError('Annotation ray requires an explicit modelspace DIMENSION')
        block = doc.blocks.get(dimension.dxf.get('geometry', ''))
        if (block is None or entity.dxf.owner != block.block_record_handle or entity.dxftype() != 'LINE'
                or not np.allclose(block.block.dxf.base_point, (0, 0, 0), rtol=0, atol=1e-9)
                or not np.allclose(dimension.dxf.get('insert', (0, 0, 0)), (0, 0, 0), rtol=0, atol=1e-9)
                or not np.allclose(dimension.dxf.get('extrusion', (0, 0, 1)), (0, 0, 1), rtol=0, atol=1e-9)):
            raise ValueError('Annotation line must belong to the untransformed dimension block')
    elif entity.dxf.owner != doc.modelspace().block_record_handle:
        raise ValueError('Contour/axis ray must reference a modelspace entity')
    if not np.allclose(entity.dxf.get('extrusion', (0, 0, 1)), [0, 0, 1], rtol=0, atol=1e-9):
        raise ValueError('Anchor rays require the default XY plane')
    kind = entity.dxftype()
    if kind == 'LINE':
        if 'origin_cad_xyz' in spec:
            a = np.asarray(spec['origin_cad_xyz'], dtype=float)
            toward = spec.get('towards_endpoint')
            if a.shape != (3,) or not np.isfinite(a).all() or toward not in ('start', 'end') or 'endpoint' in spec:
                raise ValueError('Invalid interior LINE ray origin/direction')
            start, end = np.array(entity.dxf.start), np.array(entity.dxf.end)
            delta = end-start
            length2 = float(delta @ delta)
            if length2 <= 1e-12:
                raise ValueError('Zero-length ray source')
            t = float((a-start) @ delta/length2)
            if np.linalg.norm(a-(start+np.clip(t, 0, 1)*delta)) > 1e-6:
                raise ValueError('Ray origin must lie on the finite LINE')
            b = start if toward == 'start' else end
        else:
            endpoint = spec.get('endpoint')
            if endpoint not in ('start', 'end'):
                raise ValueError('LINE ray requires an explicit endpoint')
            a = np.array(entity.dxf.get(endpoint))
            b = np.array(entity.dxf.get('end' if endpoint == 'start' else 'start'))
    elif kind == 'LWPOLYLINE':
        i, j = spec.get('vertex_index'), spec.get('neighbor_index')
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(entity) for v in (i, j)):
            raise ValueError('Invalid polyline vertex index')
        wrap = entity.closed and {i, j} == {0, len(entity)-1}
        if abs(i-j) != 1 and not wrap:
            raise ValueError('Polyline ray must follow an adjacent edge')
        edge = len(entity)-1 if wrap else min(i, j)
        if entity[edge][4] != 0:
            raise ValueError('Curved polyline edges require a different localizer')
        a = np.array([*entity[i][:2], entity.dxf.elevation])
        b = np.array([*entity[j][:2], entity.dxf.elevation])
    else:
        raise ValueError('Only straight LINE or LWPOLYLINE rays are supported')
    if not np.isfinite([a, b]).all() or abs(a[2]) > 1e-6 or abs(b[2]) > 1e-6 or np.linalg.norm(b-a) < 1e-6:
        raise ValueError('Nonfinite, nonplanar or zero-length anchor ray')
    return a, b


def _validate_inventory(case, parent, doc):
    if case.get('schema_version') != '1.0' or case.get('reviewer') != 'assistant':
        raise ValueError('Expected an explicit assistant anchor review, schema 1.0')
    half = case.get('patch_half_side_px')
    if isinstance(half, bool) or not isinstance(half, int) or not 48 <= half <= 512:
        raise ValueError('Invalid patch_half_side_px')
    anchors, unattempted = case.get('anchors'), case.get('not_attempted')
    if not isinstance(anchors, list) or not 1 <= len(anchors) <= 256 or not isinstance(unattempted, list):
        raise ValueError('Invalid anchor inventory')
    objects = {item['handle']: item for item in parent['objects']}
    expected = {(h, p) for h, o in objects.items() for p in o['points']}
    seen, ids, prepared = set(), set(), []
    for anchor in anchors:
        name = _identifier(anchor['id'])
        if name in ids:
            raise ValueError('Duplicate anchor id')
        ids.add(name)
        method, specs = anchor.get('method'), anchor.get('rays')
        if (method not in ('junction', 'endpoint') or not isinstance(specs, list)
                or (len(specs) != 1 if method == 'endpoint' else not 2 <= len(specs) <= 6)):
            raise ValueError('Pilot case requires one endpoint ray or 2..6 junction rays')
        if any(s.get('role') == 'annotation' and s.get('dimension_handle') not in objects for s in specs):
            raise ValueError('Annotation must belong to the reviewed dimension inventory')
        rays = [_cad_ray(doc, spec) for spec in specs]
        origin = rays[0][0]
        if any(np.linalg.norm(a-origin) > 1e-6 for a, _ in rays):
            raise ValueError('Junction rays do not meet at one native CAD vertex')
        refs = anchor.get('references')
        if not isinstance(refs, list) or not refs:
            raise ValueError('Anchor must reference reviewed definition points')
        ray_handles = {s['geometry_handle'] for s in specs if s.get('role', 'contour') == 'contour'}
        for ref in refs:
            key = (ref.get('handle'), ref.get('point'))
            if key not in expected or key in seen:
                raise ValueError('Unknown or repeated definition point reference')
            seen.add(key)
            obj = objects[key[0]]
            bindings = [b for b in obj['binding_evidence'] if b['point'] == key[1]]
            if len(bindings) != 1 or bindings[0]['geometry_handle'] not in ray_handles:
                raise ValueError('Ray is not linked to the reviewed definition point')
            binding = bindings[0]
            difference = np.linalg.norm(np.array(obj['points'][key[1]]['cad_xyz'])-origin)
            if difference > 1e-6 and not (method == 'endpoint' and binding['relation'] == 'supporting_line'):
                raise ValueError('Ray origin differs from the reviewed native anchor')
            if difference > 1e-6:
                closest = binding.get('closest_finite_xy')
                if closest is None or np.linalg.norm(np.array(closest)-origin[:2]) > 1e-6:
                    raise ValueError('Qualified anchor must use its recorded nearest finite endpoint')
        _finite_number(anchor.get('search_radius_px', 12), 2, 32, 'search radius')
        _finite_number(anchor.get('inner_radius_px', 20), 4, 64, 'inner radius')
        _finite_number(anchor.get('outer_radius_px', 96), 24, 256, 'outer radius')
        _finite_number(anchor.get('probe_length_px', 64), 24, 256, 'probe length')
        _finite_number(anchor.get('half_width_px', 12), 2, 32, 'half width')
        if anchor.get('cap_model', 'unknown') != 'unknown':
            raise ValueError('Real-source cap shape is unverified; keep cap_model unknown')
        prepared.append((anchor, rays))
    for item in unattempted:
        key = (item.get('handle'), item.get('point'))
        if key not in expected or key in seen or not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('Invalid unattempted definition point')
        seen.add(key)
    if seen != expected:
        raise ValueError('Every reviewed definition point must be attempted or explicitly excluded')
    return prepared, objects


def run_anchor_review_case(workspace: Path, case_path: Path, output: Path):
    root = Path(workspace).resolve()
    case_path, output = Path(case_path).resolve(), Path(output).resolve()
    if not case_path.is_relative_to(root) or not output.is_relative_to(root) or output == root:
        raise ValueError('Review paths must stay within the workspace')
    if output.exists():
        raise FileExistsError('Choose a new anchor review output directory')
    case = _json(case_path)
    if 'branch_inventory' in case:
        _pinned(_path(root, case['branch_inventory']), case.get('branch_inventory_sha256'))
    parent_path = _path(root, case['parent_review'])
    _pinned(parent_path, case['parent_review_sha256'])
    parent = _json(parent_path)
    for name, expected_hash in parent['code_sha256'].items():
        _pinned(_path(Path(__file__).parent.resolve(), name), expected_hash)
    paths = {k: _path(root, p) for k, p in parent['source_files'].items()}
    for kind, digest_key in [('png', 'source_sha256'), ('dxf', 'target_sha256'), ('case', 'case_sha256'), ('pair', 'pair_report_sha256')]:
        _pinned(paths[kind], parent[digest_key])
    prior_case = _json(paths['case'])
    prepared = _path(root, prior_case['prepared_dir'])
    _pinned(prepared / 'preparation.json', parent['preparation_report_sha256'])
    metadata = _json(prepared / 'preparation.json')
    if metadata['source_sha256'] != parent['source_sha256']:
        raise ValueError('Prepared source differs from parent object review')
    matrix = np.asarray(parent['cad_to_source_pixel'], dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.array_equal(matrix[2], [0, 0, 1])
            or abs(np.linalg.det(matrix[:2, :2])) < 1e-12):
        raise ValueError('Invalid parent coordinate transform')
    doc = ezdxf.readfile(paths['dxf'])
    ready, objects = _validate_inventory(case, parent, doc)
    width, height = metadata['source_size_px']
    half = case['patch_half_side_px']
    output.mkdir(parents=True, exist_ok=False)
    results, artifacts = [], {}
    for anchor, rays in ready:
        source_origin = _xy(matrix, rays[0][0])
        cx, cy = np.floor(source_origin).astype(int)
        box = [max(0, int(cx)-half), max(0, int(cy)-half), min(width, int(cx)+half+1), min(height, int(cy)+half+1)]
        pixels, provenance = read_prepared_region(prepared, box)
        expected = source_origin-np.array(box[:2])
        vectors = [_xy(matrix, b)-_xy(matrix, a) for a, b in rays]
        lengths = [float(np.linalg.norm(v)) for v in vectors]
        directions = [(v/length).tolist() for v, length in zip(vectors, lengths)]
        params = {'search_radius': anchor.get('search_radius_px', 12)}
        if anchor['method'] == 'junction':
            # Do not probe beyond the end of the actual finite adjacent edge.
            params.update(inner_radius=anchor.get('inner_radius_px', 20),
                          outer_radius=min(anchor.get('outer_radius_px', 96), .75*min(lengths)))
            if params['outer_radius'] < 24 or params['outer_radius']-params['inner_radius'] < 20:
                result = {'status': 'not_supported', 'observed_xy': None, 'reasons': ['adjacent_edge_too_short']}
            else:
                result = localize_junction(pixels, expected.tolist(), directions, **params)
        else:
            params.update(probe_length=min(anchor.get('probe_length_px', 64), .75*lengths[0]),
                          half_width=anchor.get('half_width_px', 12), cap_model='unknown')
            if params['probe_length'] < max(24, 2*params['search_radius']):
                result = {'status': 'not_supported', 'observed_xy': None, 'reasons': ['finite_edge_too_short']}
            else:
                result = localize_endpoint(pixels, expected.tolist(), directions[0], **params)
        result.update(annotation_approved=False, training_eligible=False)
        observed = result.get('observed_xy')
        observed_source = None if observed is None else (np.asarray(observed)+box[:2]).tolist()
        references = []
        for ref in anchor['references']:
            obj = objects[ref['handle']]
            predicted = obj['points'][ref['point']]['source_pixel_center_xy']
            references.append({**ref, 'parent_decision': obj['decision'], 'native_definition_source_xy': predicted,
                'definition_to_finite_geometry_px': float(np.linalg.norm(source_origin-predicted)),
                'observed_displacement_from_definition_px': None if observed_source is None else float(np.linalg.norm(np.asarray(observed_source)-predicted)),
                'annotation_approved': False, 'training_eligible': False})
        record = {'id': anchor['id'], 'method': anchor['method'], 'expected_geometry_source_xy': source_origin.tolist(),
                  'expected_roi_xy': expected.tolist(), 'directions_xy': directions, 'finite_ray_lengths_px': lengths,
                  'ray_specifications': anchor['rays'],
                  'effective_parameters': params, 'result': result, 'observed_source_xy': observed_source,
                  'displacement_from_geometry_px': None if observed_source is None else float(np.linalg.norm(np.asarray(observed_source)-source_origin)),
                  'references': references, 'provenance': provenance, 'pixel_ground_truth_available': False,
                  'displacement_is_error_estimate': False, 'annotation_approved': False, 'training_eligible': False}
        picture = Image.fromarray(pixels)
        raw_name, overlay_name = anchor['id']+'.png', anchor['id']+'-overlay.png'
        picture.save(output / raw_name)
        pen = ImageDraw.Draw(picture)
        x, y = expected
        pen.line((x-6, y, x+6, y), fill=(230, 120, 0), width=2)
        pen.line((x, y-6, x, y+6), fill=(230, 120, 0), width=2)
        for direction in directions:
            finish = expected+np.array(direction)*min(60, .6*min(lengths))
            pen.line((*expected, *finish), fill=(0, 90, 220), width=1)
        if observed is not None:
            x, y = observed
            pen.ellipse((x-4, y-4, x+4, y+4), outline=(0, 150, 0), width=2)
        picture.save(output / overlay_name)
        for filename in (raw_name, overlay_name):
            artifacts[filename] = _digest(output / filename)
        record.update(raw_image=raw_name, overlay_image=overlay_name)
        results.append(record)
    report = {'schema_version': '1.0', 'case_id': case['case_id'], 'date': case['date'],
              'parent_review_sha256': case['parent_review_sha256'], 'case_sha256': _digest(case_path),
              'branch_inventory_sha256': case.get('branch_inventory_sha256'),
              'source_sha256': parent['source_sha256'], 'target_sha256': parent['target_sha256'],
              'anchors': results, 'not_attempted': case['not_attempted'],
              'summary': {'unique_locations': len(results),
                          'statuses': dict(Counter(r['result']['status'] for r in results)),
                          'definition_point_statuses': dict(Counter([r['result']['status'] for r in results for _ in r['references']]
                                                                   + ['not_attempted']*len(case['not_attempted']))),
                          'attempted_definition_points': sum(len(r['references']) for r in results),
                          'not_attempted_definition_points': len(case['not_attempted'])},
              'code_sha256': {name: _digest(Path(__file__).with_name(name)) for name in
                             ['anchor_review_case.py', 'raster_endpoints.py', 'raster_junctions.py']},
              'source_modified': False, 'parent_decisions_modified': False, 'training_eligible': False,
              'annotation_approved': False, 'human_verified': False, 'global_transform_approved': False,
              'localization_accuracy_on_this_source_px': None,
              'limitations': ['Real-pixel ground truth is unavailable; displacement from CAD is not measured error.',
                             'A localized junction does not resolve a conflicting nominal or approve an annotation.',
                             'Curve anchors and unverified line caps require separate methods.']}
    _write_json(output/'report.json', report)
    _write_json(output/'case.json', case)
    rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in
                  [r['id'], r['result']['status'], r['displacement_from_geometry_px'], r['result'].get('reasons', [])])+'</tr>' for r in results)
    cards = ''.join(f'<section><h2>{html.escape(r["id"])}</h2><a href="{r["raw_image"]}">Исходные пиксели</a>'
                   f'<p><img src="{r["overlay_image"]}" alt="{html.escape(r["id"])}"></p></section>' for r in results)
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Опорные точки</title>'
            '<style>body{font:16px sans-serif;margin:2em}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.5em}section{display:inline-block;vertical-align:top;margin:1em}img{image-rendering:auto;max-width:100%}</style>'
            '<h1>Локальные опорные точки: Гайка накидная</h1>'
            '<p>Оранжевый крест — прогноз CAD, синие лучи — проверяемые ветви, зелёная точка — локализация. '
            'Смещение относительно CAD не является ошибкой: независимой растровой разметки этого листа нет. Обучающие метки не утверждены.</p>'
            '<p><a href="report.json">Расчёты и происхождение</a> · <a href="case.json">Параметры</a></p>'
            '<table><tr><th>Место</th><th>Статус</th><th>Смещение, px</th><th>Причины</th></tr>'+rows+'</table>'+cards+'</html>')
    (output/'index.html').write_text(page, encoding='utf-8')
    for filename in ('report.json', 'case.json', 'index.html'):
        artifacts[filename] = _digest(output/filename)
    _write_json(output/'manifest.json', {'schema_version': '1.0', 'status': 'complete', 'artifact_sha256': artifacts,
                                       'annotation_approved': False, 'training_eligible': False})
    return report
