"""Reproduce an explicitly authored object review without publishing labels."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import re

import ezdxf
import numpy as np
from PIL import Image, ImageDraw

from .object_review import validate_object_review
from .raster_anchor_evidence import probe_circle, probe_line
from .raster_regions import read_prepared_region


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _path(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts):
        raise ValueError('Expected a contained portable relative path')
    result = (root / name).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError('Path escapes its root')
    return result


def _json(path):
    if not 0 < path.stat().st_size <= 4*1024**2:
        raise ValueError('Review JSON exceeds its size limit')
    return json.loads(path.read_text(encoding='utf-8'))


def _pinned(path, expected):
    if (not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None
            or _digest(path) != expected):
        raise ValueError(f'Pinned SHA-256 mismatch: {path.name}')


def _identifier(value):
    if not isinstance(value, str) or re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,63}', value) is None:
        raise ValueError('Invalid review identifier')
    return value


def _xy(matrix, point):
    return (matrix @ np.r_[np.asarray(point)[:2], 1])[:2]


def _probe_geometry(doc, spec, pixels, matrix, box):
    """Project an explicit CAD entity; never search for the nearest CAD object."""
    entity = doc.entitydb.get(spec['geometry_handle'])
    if entity is None or entity.dxf.owner != doc.modelspace().block_record_handle:
        raise ValueError('Probe geometry must be a modelspace entity')
    if not np.allclose(entity.dxf.get('extrusion', (0, 0, 1)), (0, 0, 1), atol=1e-9, rtol=0):
        raise ValueError('Probe geometry requires planar WCS XY')
    origin = np.asarray(box[:2])
    transform = lambda point: _xy(matrix, point) - origin
    kind = entity.dxftype()
    radius = spec.get('search_radius_px', 12)
    if kind in {'LINE', 'LWPOLYLINE'}:
        if kind == 'LINE':
            start, end = np.array(entity.dxf.start), np.array(entity.dxf.end)
        else:
            index = spec.get('edge_index')
            points = list(entity.get_points('xyb'))
            if (isinstance(index, bool) or not isinstance(index, int) or index < 0
                    or index >= len(points) - (not entity.closed)):
                raise ValueError('Invalid polyline edge index')
            if points[index][2] != 0 or abs(entity.dxf.elevation) > 1e-9:
                raise ValueError('Probe requires a straight planar polyline edge')
            start, end = np.array(points[index][:2]), np.array(points[(index+1) % len(points)][:2])
        if len(start) == 3 and (abs(start[2]) > 1e-9 or abs(end[2]) > 1e-9):
            raise ValueError('Probe line requires Z=0')
        fractions = spec.get('line_fraction', [0, 1])
        if (not isinstance(fractions, list) or len(fractions) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in fractions)
                or not 0 <= fractions[0] < fractions[1] <= 1):
            raise ValueError('Invalid probe line fractions')
        delta = end-start
        a, b = transform(start+fractions[0]*delta), transform(start+fractions[1]*delta)
        result = probe_line(pixels, a, b, radius)
        result['line_fraction'] = fractions
    elif kind in {'CIRCLE', 'ARC'}:
        if abs(entity.dxf.center.z) > 1e-9:
            raise ValueError('Probe circle requires Z=0')
        singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
        projected_radius = float(entity.dxf.radius * np.mean(singular))
        approximation = float(entity.dxf.radius * (max(singular)-min(singular))/2)
        if approximation > .25:
            raise ValueError('Affine circle distortion exceeds the probe approximation budget')
        a, b = 0., 360.
        if kind == 'ARC':
            begin, finish = entity.dxf.start_angle, entity.dxf.end_angle
            span = (finish-begin) % 360
            angle = begin if np.linalg.det(matrix[:2, :2]) > 0 else finish
            vector = matrix[:2, :2] @ np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            a = math.degrees(math.atan2(vector[1], vector[0]))
            # The affine is close to a similarity. This approximation is recorded,
            # including radial error, rather than approving a global transform.
            b = a + span
        result = probe_circle(pixels, transform(entity.dxf.center), projected_radius, a, b, radius)
        result['circle_approximation_max_radial_error_px'] = approximation
    else:
        raise ValueError(f'Unsupported probe entity: {kind}')
    result.update(id=spec['id'], geometry_handle=entity.dxf.handle, geometry_type=kind,
                  geometry_layer=entity.dxf.layer, region=spec['region'],
                  object_handles=spec['object_handles'], region_to_source_offset=box[:2])
    if kind == 'LWPOLYLINE':
        result['edge_index'] = spec['edge_index']
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run_object_review_case(workspace: Path, case_path: Path, output: Path) -> dict:
    """Validate pinned sources, review decisions, native-resolution ROIs and probes.

    A review can document supported CAD/nominal observations while still lacking
    endpoint accuracy. No decision here creates a training label or changes DXF.
    """
    root = Path(workspace).resolve()
    case_path, output = Path(case_path).resolve(), Path(output).resolve()
    if not case_path.is_relative_to(root) or not output.is_relative_to(root) or output == root:
        raise ValueError('Review and output must be within the workspace')
    if output.exists():
        raise FileExistsError('Review output already exists; choose a new version')
    case = _json(case_path)
    pair_path = _path(root, case['pair_report'])
    _pinned(pair_path, case['pair_report_sha256'])
    pair = _json(pair_path)
    if pair.get('visual_status') != 'verified' or pair.get('alignment', {}).get('status') != 'visual_alignment_candidate':
        raise ValueError('Case requires a visually verified pair with an alignment candidate')
    dataset = _path(root, 'dataset')
    source, target = _path(dataset, pair['input_path']), _path(dataset, pair['target_path'])
    if pair['input_sha256'] != case['source_sha256'] or pair['target_sha256'] != case['target_sha256']:
        raise ValueError('Case and pair source hashes differ')
    _pinned(source, case['source_sha256'])
    _pinned(target, case['target_sha256'])
    prepared = _path(root, case['prepared_dir'])
    _pinned(prepared / 'preparation.json', case['preparation_report_sha256'])
    if case['preparation_report_sha256'] != pair['source_provenance']['preparation_report_sha256']:
        raise ValueError('Case and pair preparation hashes differ')
    preparation = _json(prepared / 'preparation.json')
    if preparation['source_sha256'] != case['source_sha256']:
        raise ValueError('Preparation belongs to another source')
    matrix = np.array(pair['alignment']['cad_to_source_pixel'], dtype=float)
    size = pair['source_provenance']['source_size_px']
    if preparation['source_size_px'] != size:
        raise ValueError('Preparation and pair dimensions differ')
    doc = ezdxf.readfile(target)
    review = validate_object_review(doc, case, matrix, size)
    regions, probes = case['regions'], case.get('probes', [])
    if not isinstance(regions, dict) or not 1 <= len(regions) <= 32 or not isinstance(probes, list) or len(probes) > 256:
        raise ValueError('Region or probe inventory exceeds review limits')
    for name in regions:
        _identifier(name)
    identifiers = [_identifier(p['id']) for p in probes]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate probe id')
    objects = {o['handle']: o for o in case['objects']}
    views = {v['id']: v for v in case['views']}
    for view in views.values():
        if regions.get(view['id'], {}).get('source_box_px') != view['source_box_px']:
            raise ValueError('View and region boxes differ')
        if view['scale_evidence']['region'] not in regions:
            raise ValueError('Scale caption region is missing')
    for obj in objects.values():
        if not obj.get('raster_reading') or not obj.get('raster_evidence_regions') or any(r not in regions for r in obj['raster_evidence_regions']):
            raise ValueError('Object requires an explicit raster reading and evidence regions')
    for probe in probes:
        if probe['region'] not in regions or not probe.get('object_handles'):
            raise ValueError('Probe requires region and object references')
        for handle in probe['object_handles']:
            if handle not in objects or probe['region'] != objects[handle]['view_id']:
                raise ValueError('Probe refers to another object or view')
            if probe['geometry_handle'] not in {b['geometry_handle'] for b in objects[handle]['bindings']}:
                raise ValueError('Probe is not linked to the reviewed object geometry')
            entity = doc.entitydb[probe['geometry_handle']]
            if entity.dxftype() == 'LWPOLYLINE':
                edge = probe.get('edge_index')
                if (isinstance(edge, bool) or not isinstance(edge, int) or edge < 0
                        or edge >= len(entity) - (not entity.closed)):
                    raise ValueError('Invalid polyline edge index')
                vertices = {b.get('vertex_index') for b in objects[handle]['bindings']
                            if b['geometry_handle'] == entity.dxf.handle and b['relation'] == 'vertex'}
                if not vertices.intersection({edge, (edge+1) % len(entity)}):
                    raise ValueError('Probe edge does not touch a reviewed vertex')

    output.mkdir(parents=True, exist_ok=False)
    artifacts, region_reports, results = {}, {}, []
    for name, region in regions.items():
        box = region['source_box_px']
        pixels, provenance = read_prepared_region(prepared, box)
        if provenance['source_sha256'] != case['source_sha256']:
            raise ValueError('Region belongs to another source')
        raw_name, overlay_name = f'{name}.png', f'{name}-anchors.png'
        picture = Image.fromarray(pixels)
        picture.save(output / raw_name)
        overlay = picture.copy()
        pen = ImageDraw.Draw(overlay)
        for obj in objects.values():
            if obj['view_id'] != name:
                continue
            dimension = doc.entitydb[obj['handle']]
            for binding in obj['bindings']:
                point = _xy(matrix, dimension.dxf.get(binding['point'])) - np.asarray(box[:2])
                x, y = point
                if 0 <= x < pixels.shape[1] and 0 <= y < pixels.shape[0]:
                    pen.ellipse((x-10, y-10, x+10, y+10), outline=(0, 140, 0), width=2)
                    pen.text((x+12, y+5), obj['handle']+':'+binding['point'], fill=(0, 130, 0))
        overlay.save(output / overlay_name)
        provenance.update(raw_image=raw_name, predicted_anchor_overlay=overlay_name,
                          source_original_rehashed=True, original_rehash_scope='Runner verifies original SHA-256 before extracting ROIs.')
        region_reports[name] = provenance
        for filename in (raw_name, overlay_name):
            artifacts[filename] = _digest(output / filename)
        for spec in probes:
            if spec['region'] == name:
                results.append(_probe_geometry(doc, spec, pixels, matrix, box))
        del pixels, picture, overlay

    review.update(schema_version='1.0', case_id=case['case_id'], date=case['date'], reviewer='assistant',
                  review_scope=case['review_scope'], training_eligible=False, annotation_approved=False,
                  human_verified=False, source_modified=False, source_sha256=case['source_sha256'],
                  target_sha256=case['target_sha256'], case_sha256=_digest(case_path),
                  pair_report_sha256=case['pair_report_sha256'], preparation_report_sha256=case['preparation_report_sha256'],
                  cad_to_source_pixel=matrix.tolist(), regions=region_reports, pixel_probes=results,
                  original_sources_rehashed=True, split_assignment_changed=False,
                  endpoint_accuracy_verified=False, global_transform_approved=False)
    review['source_files'] = {k: v.relative_to(root).as_posix() for k, v in
                              {'png': source, 'dxf': target, 'case': case_path, 'pair': pair_path}.items()}
    review['code_sha256'] = {name: _digest(Path(__file__).with_name(name)) for name in
                              ['object_review.py', 'object_review_case.py', 'raster_regions.py', 'raster_anchor_evidence.py']}
    review['pixel_probe_summary'] = {status: sum(p['status'] == status for p in results)
                                     for status in ['supported', 'not_supported']}
    for obj in review['objects']:
        dimension = doc.entitydb[obj['handle']]
        obj['dimension_graphics'] = {
            'layer': dimension.dxf.layer, 'block': dimension.dxf.get('geometry'),
            'color': dimension.dxf.get('color', 256), 'lineweight': dimension.dxf.get('lineweight', -1),
            'role': 'annotation_not_part_contour',
        }
        for binding in obj['binding_evidence']:
            if 'closest_finite_xy' in binding:
                actual = _xy(matrix, binding['closest_finite_xy'])
                proposed = np.asarray(obj['points'][binding['point']]['source_pixel_center_xy'])
                binding['distance_to_finite_segment_source_px'] = float(np.linalg.norm(proposed-actual))
        linked = [p for p in results if obj['handle'] in p['object_handles']]
        obj['local_stroke_evidence'] = {
            'probe_ids': [p['id'] for p in linked],
            'status': ('not_probed' if not linked else 'supported' if all(p['status'] == 'supported' for p in linked) else 'incomplete'),
            'endpoint_accuracy_verified': False,
            'interpretation': 'Fits of selected strokes only; CAD decision does not approve raster anchors.',
        }
    review['role_policy'] = {
        'part_geometry': 'Only explicitly linked modelspace entities; role asserted by reviewer.',
        'dimension_graphics': 'Annotation blocks, not part geometry.',
        'other_modelspace_entities': 'Unreviewed; shared layers do not imply a shared semantic role.',
        'layer_names_used_as_automatic_labels': False,
    }
    # Preserve authored text and decisions alongside independently recomputed CAD evidence.
    review['authored_review'] = {'views': case['views'], 'objects': case['objects']}
    _write_json(output / 'review.json', review)
    artifacts['review.json'] = _digest(output / 'review.json')
    _write_json(output / 'case.json', case)
    artifacts['case.json'] = _digest(output / 'case.json')
    computed = {o['handle']: o for o in review['objects']}
    rows = ''
    for obj in case['objects']:
        checked = computed[obj['handle']]
        values = [obj['handle'], obj['view_id'], obj['raster_reading'],
                  checked['native_measurement'], checked['scaled_measurement_mm'],
                  checked['scaled_minus_nominal_mm'], obj['decision'],
                  checked['local_stroke_evidence']['status'], obj['reason']]
        rows += '<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in values)+'</tr>'
    pictures = ''.join(f'<h2>{html.escape(name)}</h2><a href="{name}.png">Исходные пиксели</a> · '
                      f'<a href="{name}-anchors.png">Предварительные опорные точки</a>'
                      f'<p><img src="{name}.png" alt="{html.escape(name)}"></p>' for name in regions)
    probe_rows = ''.join(f'<tr><td>{html.escape(p["id"])}</td><td>{p["status"]}</td>'
                         f'<td>{p["coverage"]:.3f}</td><td>{html.escape(str(p.get("hypothesis_difference", {})))}</td></tr>' for p in results)
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Проверка размеров</title>'
            '<style>body{font:16px sans-serif;max-width:1400px;margin:2em auto;padding:1em}td,th{border:1px solid #bbb;padding:.5em}table{border-collapse:collapse}img{max-width:100%}</style>'
            f'<h1>{html.escape(str(case["case_id"]))}: журнал проверки объектов</h1>'
            '<p>Проверка ассистентом. Обучающие метки не утверждены. Масштабы заданы по подписям видов; '
            'опорные точки наложения предварительные. Пробы штрихов не подтверждают точность концов отрезков.</p>'
            '<p><a href="review.json">Расчёты и происхождение</a> · <a href="case.json">Решения и параметры</a></p>'
            '<table><tr><th>Handle</th><th>Вид</th><th>Прочитано</th><th>CAD</th><th>С учётом масштаба, мм</th><th>Δ, мм</th><th>Решение по CAD</th><th>Локальные штрихи</th><th>Основание</th></tr>'
            + rows + '</table><h2>Локальные пробы штрихов</h2><table><tr><th>Проба</th><th>Статус</th><th>Покрытие</th><th>Смещение</th></tr>'
            + probe_rows + '</table>' + pictures + '</html>')
    (output / 'index.html').write_text(page, encoding='utf-8')
    artifacts['index.html'] = _digest(output / 'index.html')
    _write_json(output / 'manifest.json', {'schema_version': '1.0', 'status': 'complete',
                'artifact_sha256': artifacts, 'training_eligible': False, 'annotation_approved': False})
    return review
