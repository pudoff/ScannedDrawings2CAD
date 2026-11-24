"""Compare explicit CAD junctions with previously sealed raster-only intervals."""
from __future__ import annotations

from collections import Counter
import html
from pathlib import Path

import ezdxf
import numpy as np
from PIL import Image, ImageDraw

from .clear_pairs import contained, digest, read_json
from .object_review_case import _pinned, _write_json
from .object_corpus_review import _project_points
from .raster_point_references import validate_raster_references


def _numbers(value, length):
    if (not isinstance(value, (list, tuple)) or len(value) != length
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value)):
        raise ValueError('Coordinates must be numeric arrays')
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('Coordinates must be finite')
    return result


def compare_point_interval(projected_xy, center_xy, interval_xyxy):
    """Inclusive rectangle agreement; distance to an interval is not GT error."""
    point, center, bounds = _numbers(projected_xy, 2), _numbers(center_xy, 2), _numbers(interval_xyxy, 4)
    if np.any(bounds[2:] <= bounds[:2]) or np.any(center < bounds[:2]) or np.any(center > bounds[2:]):
        raise ValueError('Invalid reference interval or representative center')
    gap = point - np.clip(point, bounds[:2], bounds[2:])
    return {'status': 'inside_reference_interval' if np.all(gap == 0) else 'outside_reference_interval',
            'delta_from_representative_center_px': (point-center).tolist(),
            'distance_from_representative_center_px': float(np.linalg.norm(point-center)),
            'distance_to_reference_interval_px': float(np.linalg.norm(gap)),
            'ground_truth_error_px': None, 'annotation_approved': False}


def _junction(doc, features):
    if not isinstance(features, list) or not 2 <= len(features) <= 6:
        raise ValueError('A junction requires two to six explicit native LINE endpoints')
    points, records, identities = [], [], set()
    for feature in features:
        if not isinstance(feature, dict):
            raise ValueError('Invalid CAD feature record')
        handle, end = feature.get('handle'), feature.get('endpoint')
        if not isinstance(handle, str) or end not in ('start', 'end') or handle in identities:
            raise ValueError('Invalid or repeated CAD feature')
        identities.add(handle)
        entity = doc.entitydb.get(handle)
        if (entity is None or entity.dxftype() != 'LINE'
                or entity.dxf.owner != doc.modelspace().block_record_handle):
            raise ValueError('Feature must be a native modelspace LINE')
        a, b = _numbers(list(entity.dxf.start), 3), _numbers(list(entity.dxf.end), 3)
        if (abs(a[2]) > 1e-6 or abs(b[2]) > 1e-6 or np.linalg.norm(b-a) <= 1e-6
                or not np.allclose(entity.dxf.get('extrusion', (0, 0, 1)), (0, 0, 1), rtol=0, atol=1e-12)):
            raise ValueError('CAD feature must be finite, planar and nondegenerate')
        point = a if end == 'start' else b
        points.append(point)
        records.append({'handle': handle, 'endpoint': end, 'cad_xyz': point.tolist(),
                        'line_start': a.tolist(), 'line_end': b.tolist(), 'layer': entity.dxf.layer,
                        'owner_block': '*Model_Space', 'role_basis': 'explicit_post_seal_review'})
    if any(np.linalg.norm(point-points[0]) > 1e-6 for point in points[1:]):
        raise ValueError('Specified CAD features do not share a native endpoint')
    # A true junction needs at least two nonparallel lines, not duplicate strokes.
    directions = [np.subtract(r['line_end'][:2], r['line_start'][:2]) for r in records]
    directions = [d/np.linalg.norm(d) for d in directions]
    if not any(abs(directions[0][0]*d[1]-directions[0][1]*d[0]) > 1e-6 for d in directions[1:]):
        raise ValueError('Collinear features do not define a junction')
    return points[0].tolist(), records


def review_raster_point_comparison(workspace: Path, config_path: Path, output: Path) -> dict:
    root, config_path, output = Path(workspace).resolve(), Path(config_path).resolve(), Path(output).resolve()
    if not config_path.is_relative_to(root) or not output.is_relative_to(root) or output == root:
        raise ValueError('Comparison paths must remain within the workspace')
    if output.exists():
        raise FileExistsError('Choose a new comparison output directory')
    config = read_json(config_path)
    if config.get('schema_version') != '1.0':
        raise ValueError('Unknown comparison schema')
    validated = validate_raster_references(root, contained(root, config['references']), contained(root, config['seal']),
                                          expected_reference_sha256=config['references_sha256'],
                                          expected_seal_sha256=config['seal_sha256'])
    references = validated['references']
    run, dataset = contained(root, config['parent_run']), contained(root, config['dataset'])
    _pinned(run/'report.json', config['parent_report_sha256'])
    parent = read_json(run/'report.json')
    if parent.get('complete') is not True:
        raise ValueError('Incomplete parent pair report')
    for name, expected in parent['implementation'].items():
        _pinned(contained(Path(__file__).parent.resolve(), name), expected)
    associations = config['associations']
    if not isinstance(associations, list) or not 1 <= len(associations) <= 1000:
        raise ValueError('Invalid association inventory')
    if any(not isinstance(item, dict) or not isinstance(item.get('point_id'), str) for item in associations):
        raise ValueError('Invalid association record')
    ids = [item['point_id'] for item in associations]
    if len(ids) != len(set(ids)) or set(ids) != {p['id'] for p in references['points']}:
        raise ValueError('Every frozen reference point must have one explicit association')
    by_id = {a['point_id']: a for a in associations}
    pairs = {}
    for source in references['sources']:
        matches = [p for p in parent['pairs'] if p['input_sha256'] == source['source_sha256'] and p['input_path'] == source['input_path']]
        if len(matches) != 1:
            raise ValueError('Reference source is not unique in parent report')
        pair = matches[0]
        if pair['visual_status'] != 'verified':
            raise ValueError('Independent comparison requires a visually reviewed pair')
        target = contained(dataset, pair['target_path'])
        _pinned(target, pair['target_sha256'])
        pairs[source['id']] = (pair, ezdxf.readfile(target))
    records = []
    for ref in references['points']:
        pair, doc = pairs[ref['source_id']]
        cad, features = _junction(doc, by_id[ref['id']]['features'])
        projection = _project_points({'definition_points_modelspace': {'junction': cad}}, pair)
        if projection['status'] != 'provisional_alignment':
            raise ValueError('Reference comparison has no candidate alignment')
        projected = projection['points']['junction']['source_pixel_center_xy']
        records.append({'point_id': ref['id'], 'source_id': ref['source_id'], 'description': ref['description'],
                        'source_sha256': ref['source_sha256'], 'target_sha256': pair['target_sha256'],
                        'cad_xyz': cad, 'cad_features': features, 'projected_source_xy': projected,
                        'center_source_xy': ref['center_source_xy'], 'interval_source_xyxy': ref['interval_source_xyxy'],
                        'comparison': compare_point_interval(projected, ref['center_source_xy'], ref['interval_source_xyxy']),
                        'reference_evidence': ref['evidence'], 'alignment_refitted': False,
                        'training_eligible': False, 'annotation_approved': False})
    # All inputs and links are checked before producing a review directory.
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for index, record in enumerate(records, 1):
        ev = record['reference_evidence']
        with Image.open(contained(root, ev['raw_crop_path'])) as image:
            overlay = image.convert('RGB')
        offset = np.asarray(ev['source_box_px'][:2])
        bounds = np.asarray(record['interval_source_xyxy']).reshape(2, 2)-offset
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(tuple(bounds.flatten()), outline=(0, 100, 255), width=1)
        px, py = np.subtract(record['projected_source_xy'], offset)
        draw.line((px-5, py-5, px+5, py+5), fill=(255, 90, 0), width=1)
        draw.line((px-5, py+5, px+5, py-5), fill=(255, 90, 0), width=1)
        name = f'point-{index:03d}.png'
        overlay.save(output/name)
        record['overlay_path'] = name
        artifacts[name] = digest(output/name)
    report = {'schema_version': '1.0', 'date': config['date'], 'status': 'complete',
              'config_sha256': digest(config_path), 'references_sha256': config['references_sha256'],
              'seal_sha256': config['seal_sha256'], 'parent_report_sha256': config['parent_report_sha256'],
              'summary': {'source_count': len(pairs), 'point_count': len(records),
                          'statuses': dict(Counter(p['comparison']['status'] for p in records)),
                          'approved_coordinates': 0},
              'verification': {'raster_reference_validation': validated['verification'],
                               'cad_sources_rehashed': len(pairs), 'parent_report_rehashed': True,
                               'parent_implementation_verified': True,
                               'cad_accessed_after_reference_validation': True}, 'points': records,
              'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                             ['raster_point_comparison.py', 'raster_point_references.py', 'object_corpus_review.py', 'raster_regions.py']},
              'alignment_refitted': False, 'references_modified': False,
              'human_verified': False, 'training_eligible': False, 'annotation_approved': False,
              'limitations': ['Assistant-authored intervals are development evidence, not human-verified ground truth.',
                              'Interval agreement does not bound geometric error or validate all points in either drawing.',
                              'CAD junction links are explicit post-seal observations, not automatic dimension associations.',
                              'Physical units, view scales, nominal values and CAD associativity are not approved.']}
    _write_json(output/'report.json', report)
    _write_json(output/'config.json', config)
    items = ''.join('<section><h2>'+html.escape(p['point_id'])+'</h2><p>'+html.escape(p['description'])+
                    '</p><p>'+html.escape(p['comparison']['status'])+f'</p><img src="{p["overlay_path"]}" alt="Интервал и CAD-точка"></section>' for p in records)
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Независимые растровые интервалы</title>'
            '<style>body{font:16px sans-serif;margin:2em}section{display:inline-block;vertical-align:top;width:23em;padding:1em}img{image-rendering:pixelated}</style>'
            f'<h1>Контроль реальных точек: {len(records)}</h1><p>Синий прямоугольник — интервал по растру, оранжевый крест — '
            'проекция явно связанного CAD-узла. Интервалы зафиксированы до сопоставления с CAD. '
            'Совпадение не подтверждает точность всех координат, физический номинал или размерную привязку.</p>'
            '<p><a href="report.json">Отчёт</a> · <a href="config.json">Параметры и связи</a></p>'+items+'</html>')
    (output/'index.html').write_text(page, encoding='utf-8')
    for filename in ('report.json', 'config.json', 'index.html'):
        artifacts[filename] = digest(output/filename)
    _write_json(output/'manifest.json', {'schema_version': '1.0', 'status': 'complete', 'artifact_sha256': artifacts})
    return report
