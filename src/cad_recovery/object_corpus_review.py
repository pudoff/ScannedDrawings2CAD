"""Complete field-by-field candidate review of a pinned PNG/DXF corpus."""
from __future__ import annotations

from collections import Counter
import html
from pathlib import Path

import ezdxf
import numpy as np

from .clear_pairs import contained, digest, read_json
from .object_review_case import _pinned, _write_json
from .dimension_observations import classify_dimension_observation
from .dimension_attachment_candidates import collect_attachment_candidates


def _preflight(root, config):
    if config.get('schema_version') != '1.0':
        raise ValueError('Unknown corpus review schema')
    run = contained(root, config['source_run'])
    dataset = contained(root, config['dataset'])
    _pinned(run/'report.json', config['source_report_sha256'])
    report = read_json(run/'report.json')
    if report.get('complete') is not True:
        raise ValueError('Parent corpus run is incomplete')
    intake = contained(root, config['intake'])
    _pinned(intake, config['intake_sha256'])
    if report['intake_sha256'] != config['intake_sha256']:
        raise ValueError('Parent report and intake differ')
    for dep in read_json(intake)['dependencies'].values():
        _pinned(contained(root, dep['path']), dep['sha256'])
    for name, expected in report['implementation'].items():
        _pinned(contained(Path(__file__).parent.resolve(), name), expected)
    pairs = report['pairs']
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= 1000 or len(pairs) != config['expected_pair_count']:
        raise ValueError('Unexpected pair inventory')
    sources, ids, prepared, total, artifact_count = set(), set(), [], 0, 0
    for pair in pairs:
        if pair['input_sha256'] in sources:
            raise ValueError('Duplicate raster in corpus review')
        sources.add(pair['input_sha256'])
        source = contained(dataset, pair['input_path'])
        target = contained(dataset, pair['target_path'])
        _pinned(source, pair['input_sha256'])
        _pinned(target, pair['target_sha256'])
        semantic_path = contained(run, pair['artifacts']['semantics'])
        pair_id = semantic_path.parent.name
        if pair_id in ids or not pair_id.startswith('pair-') or not pair_id[5:].isdigit():
            raise ValueError('Duplicate or invalid pair id')
        ids.add(pair_id)
        if read_json(semantic_path.parent/'pair.json') != pair:
            raise ValueError('Pair record disagrees with the pinned parent inventory')
        for key, expected in pair['artifact_sha256'].items():
            _pinned(contained(run, pair['artifacts'][key]), expected)
            artifact_count += 1
        if 'semantics' not in pair['artifact_sha256']:
            raise ValueError('CAD semantics must be pinned by the parent pair')
        semantics = read_json(semantic_path)
        if semantics.get('source_sha256') != pair['target_sha256'] or semantics.get('inventory_complete') is not True:
            raise ValueError('Incomplete or mismatched semantic inventory')
        dimensions = semantics['dimensions']
        if len(dimensions) != semantics['summary']['dimension_instances']:
            raise ValueError('Semantic dimension count differs from its inventory')
        total += len(dimensions)
        for dimension in dimensions:
            _project_points(dimension, pair)
        prepared.append((pair_id, pair, target, semantics))
    if total != config['expected_dimension_count']:
        raise ValueError('Unexpected dimension inventory')
    if not set(config.get('focus_pair_ids', [])).issubset(ids):
        raise ValueError('Unknown focus pair')
    return prepared, artifact_count


def _project_points(record, pair):
    alignment = pair.get('alignment')
    if alignment is None:
        return {'status': 'unavailable', 'points': {}, 'approved': False}
    if not isinstance(alignment, dict):
        raise ValueError('Invalid parent alignment')
    if alignment.get('cad_to_source_pixel') is None:
        return {'status': 'unavailable', 'points': {}, 'approved': False}
    try:
        matrix = np.asarray(alignment['cad_to_source_pixel'], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid parent alignment matrix') from error
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[2], [0, 0, 1]) or abs(np.linalg.det(matrix[:2, :2])) < 1e-12):
        raise ValueError('Invalid parent alignment matrix')
    result = {}
    points = record.get('definition_points_modelspace') or {}
    if not isinstance(points, dict):
        raise ValueError('Invalid definition point inventory')
    for name, point in points.items():
        if (not isinstance(point, (list, tuple)) or len(point) != 3
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in point)
                or not np.isfinite(point).all()):
            result[name] = {'status': 'unavailable', 'approved': False}
            continue
        with np.errstate(over='ignore', invalid='ignore'):
            projected = (matrix @ [point[0], point[1], 1])[:2]
        if not np.isfinite(projected).all():
            raise ValueError('Nonfinite projected coordinates')
        result[name] = {'source_pixel_center_xy': projected.tolist(), 'cad_z': point[2],
                        'planar_xy': abs(point[2]) <= 1e-6, 'approved': False}
    return {'status': 'provisional_alignment' if pair['visual_status'] == 'verified' else 'diagnostic_alignment_only',
            'points': result, 'approved': False, 'accuracy_bound_px': None}


def review_object_corpus(workspace: Path, config_path: Path, output: Path) -> dict:
    root = Path(workspace).resolve()
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if not config_path.is_relative_to(root) or not output.is_relative_to(root) or output == root:
        raise ValueError('Corpus paths must stay within the workspace')
    if output.exists():
        raise FileExistsError('Choose a new object corpus review directory')
    config = read_json(config_path)
    prepared, verified_artifacts = _preflight(root, config)
    output.mkdir(parents=True, exist_ok=False)
    records, artifacts = [], {}
    notation_counts, units_counts, field_counts, attachment_counts = Counter(), Counter(), {}, Counter()
    for pair_id, pair, target, semantics in prepared:
        document = ezdxf.readfile(target)
        if int(document.units) != semantics['units']:
            raise ValueError('CAD unit declaration differs from semantic audit')
        attachments = collect_attachment_candidates(document, semantics['dimensions'])
        attached = attachments['dimensions']
        if len(attached) != len(semantics['dimensions']):
            raise ValueError('Attachment candidate inventory lost dimensions')
        observations = []
        for dimension, attachment in zip(semantics['dimensions'], attached):
            if (attachment['handle'] != dimension['handle'] or attachment['insert_path'] != dimension['insert_path']):
                raise ValueError('Attachment instance identity differs from semantic record')
            observation = classify_dimension_observation(dimension, units_code=semantics['units'],
                                                        pair_visual_status=pair['visual_status'])
            observation['cad_attachment_candidates'] = attachment
            observation['raster_coordinates'] = _project_points(dimension, pair)
            observation['field_statuses']['raster_coordinates'] = observation['raster_coordinates']['status']
            observation['field_statuses']['view_scale'] = 'not_reviewed'
            observation['field_statuses']['independent_raster_reference'] = 'not_linked'
            observation['field_statuses']['geometry_role'] = 'not_reviewed'
            observation['field_statuses']['geometry_attachment'] = attachment['status']
            observation['corrections'] = []
            observation['annotation_approved'] = observation['training_eligible'] = False
            observations.append(observation)
            notation_counts[observation['notation']['kind']] += 1
            units_counts[observation['units']['status']] += 1
            for field, status in observation['field_statuses'].items():
                field_counts.setdefault(field, Counter())[status] += 1
            for point in attachment.get('points', {}).values():
                attachment_counts[point['status']] += 1
        data = {'schema_version': '1.0', 'pair_id': pair_id,
                'input_path': pair['input_path'], 'target_path': pair['target_path'],
                'source_sha256': pair['input_sha256'], 'target_sha256': pair['target_sha256'],
                'pair_visual_status': pair['visual_status'], 'units': semantics['units'],
                'semantic_inventory_complete': True, 'dimension_count': len(observations),
                'modelspace_insert_count': semantics['modelspace_entity_counts'].get('INSERT', 0),
                'geometry_inventory': attachments['geometry_inventory'], 'objects': observations,
                'training_eligible': False, 'annotation_approved': False, 'source_modified': False,
                'limitations': ['CAD incidence candidates include unreviewed frame/axis/annotation roles.',
                               'Modelspace INSERT geometry is not silently flattened into contour labels.',
                               'Numeric consistency does not confirm view scale or physical nominal.',
                               'Source pixel intervals are reviewed separately, never inferred from this alignment.']}
        folder = output/pair_id
        folder.mkdir()
        _write_json(folder/'objects.json', data)
        path = f'{pair_id}/objects.json'
        artifacts[path] = digest(output/path)
        records.append({k: data[k] for k in ['pair_id', 'input_path', 'pair_visual_status', 'units', 'dimension_count', 'modelspace_insert_count']}
                       | {'objects_path': path, 'objects_sha256': artifacts[path], 'focus': pair_id in config.get('focus_pair_ids', [])})
    summary = {'pair_count': len(records), 'dimension_count': sum(p['dimension_count'] for p in records),
               'visual_status': dict(Counter(p['pair_visual_status'] for p in records)),
               'pair_units': dict(Counter(str(p['units']) for p in records)),
               'notation_kinds': dict(notation_counts), 'dimension_units_statuses': dict(units_counts),
               'field_statuses': {k: dict(v) for k, v in field_counts.items()},
               'cad_point_candidate_statuses': dict(attachment_counts),
               'approved_fields': 0, 'training_eligible_objects': 0}
    report = {'schema_version': '1.0', 'date': config['date'], 'status': 'complete', 'summary': summary,
              'config_sha256': digest(config_path), 'parent_report_sha256': config['source_report_sha256'],
              'intake_sha256': config['intake_sha256'], 'pairs': records,
              'verification': {'source_files_rehashed': 2*len(records), 'parent_artifacts_rehashed': verified_artifacts,
                               'intake_dependencies_verified': True, 'parent_implementation_verified': True},
              'code_sha256': {name: digest(Path(__file__).with_name(name)) for name in
                             ['object_corpus_review.py', 'dimension_observations.py', 'dimension_attachment_candidates.py', 'notation.py']},
              'source_modified': False, 'split_modified': False, 'training_eligible': False,
              'annotation_approved': False, 'independent_coordinate_review': 'separate_pinned_reference_workflow'}
    _write_json(output/'report.json', report)
    _write_json(output/'config.json', config)
    rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in
                  [p['pair_id'], p['input_path'], p['pair_visual_status'], p['units'], p['dimension_count'], p['modelspace_insert_count'], p['focus']])+
                  f'<td><a href="{p["objects_path"]}">Поля объектов</a></td></tr>' for p in records)
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пообъектная проверка корпуса</title>'
            '<style>body{font:16px sans-serif;margin:2em}td,th{border:1px solid #bbb;padding:.5em}table{border-collapse:collapse}</style>'
            '<h1>Корпус PNG/DXF: состояния полей размерных объектов</h1>'
            '<p>Полная опись кандидатов. Совпадение чисел, геометрическое совпадение в CAD и растровые координаты '
            'проверяются отдельно. Неизвестные единицы и карантин сохранены. Обучающие метки не утверждены.</p>'
            '<p><a href="report.json">Сводный JSON</a> · <a href="config.json">Параметры</a></p>'
            '<table><tr><th>Пара</th><th>Источник</th><th>Визуальный статус</th><th>INSUNITS</th><th>Размеры</th><th>INSERT</th><th>Контрольный случай</th><th>Журнал</th></tr>'
            + rows + '</table></html>')
    (output/'index.html').write_text(page, encoding='utf-8')
    for filename in ('report.json', 'config.json', 'index.html'):
        artifacts[filename] = digest(output/filename)
    _write_json(output/'manifest.json', {'schema_version': '1.0', 'status': 'complete', 'artifact_sha256': artifacts,
                                       'annotation_approved': False, 'training_eligible': False})
    return report
