"""Independent, fixed synthetic controls for local raster anchor localization.

Truth is authored from continuous drawing coordinates before an algorithm runs.
This is a development benchmark, not an independent customer acceptance set.
Its negatives include missing evidence and declared out-of-scope topologies.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import html
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 384
SUPERSAMPLE = 8
TOLERANCE_PX = 2.0
BENCHMARK_VERSION = 'anchor-controls-v1'


def _direction(angle):
    return [math.cos(math.radians(angle)), math.sin(math.radians(angle))]


def _add(a, b, factor=1):
    return [a[0]+factor*b[0], a[1]+factor*b[1]]


def _segment(start, end, width, cap='round'):
    return {'type': 'segment', 'start_xy': list(start), 'end_xy': list(end),
            'width_px': width, 'cap': cap}


def _draw(commands, *, foreground=0, background=255, blur=0):
    """Draw explicit stroke polygons/caps, independently of the localizers.

    The +0.5 offset preserves source pixel-center coordinates when the enlarged
    canvas is downsampled. Polygon quantization is limited to 1/8 source pixel.
    """
    image = Image.new('L', (SIZE*SUPERSAMPLE, SIZE*SUPERSAMPLE), background)
    pen = ImageDraw.Draw(image)
    scale = lambda p: tuple((float(v)+.5)*SUPERSAMPLE-.5 for v in p)
    for command in commands:
        kind = command['type']
        if kind == 'segment':
            a, b = np.asarray(command['start_xy'], float), np.asarray(command['end_xy'], float)
            direction = b-a
            normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            radius = command['width_px']/2
            polygon = [scale(p) for p in (a+normal*radius, b+normal*radius,
                                          b-normal*radius, a-normal*radius)]
            pen.polygon(polygon, fill=foreground)
            if command['cap'] == 'round':
                for point in (a, b):
                    pen.ellipse((*scale(point-radius), *scale(point+radius)), fill=foreground)
        elif kind == 'ring':
            center, radius = np.asarray(command['center_xy']), command['radius_px']
            half_width = command['width_px']/2
            pen.ellipse((*scale(center-radius-half_width), *scale(center+radius+half_width)),
                        outline=foreground, width=round(command['width_px']*SUPERSAMPLE))
        elif kind == 'rectangle':
            pen.rectangle((*scale(command['box_xyxy'][:2]), *scale(command['box_xyxy'][2:])),
                          fill=foreground)
        else:
            raise ValueError('Unknown benchmark drawing primitive')
    image = image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return np.array(image.convert('RGB'))


def create_cases() -> list[dict]:
    """Return 32 fixed scenes: 16 positive and 16 required-refusal controls.

    Every scene retains its independent drawing specification and truth. A scene
    rejected by the localizer is never relabeled or removed from this inventory.
    """
    cases = []
    point = [191.25, 190.75]

    def add(name, kind, commands, parameters, localizable, reason, truth_point=None,
            foreground=0, background=255, blur=0):
        rendering = {'size_px': [SIZE, SIZE], 'supersample': SUPERSAMPLE,
                     'foreground': foreground, 'background': background, 'blur_sigma_px': blur,
                     'coordinate_convention': 'continuous source pixel centers', 'commands': commands}
        cases.append({'case_id': name, 'kind': kind, 'parameters': parameters,
                      'truth': {'localizable': localizable,
                                'point_xy': list(truth_point) if localizable else None,
                                'tolerance_px': TOLERANCE_PX, 'reason': reason},
                      'rendering': rendering,
                      'image': _draw(commands, foreground=foreground, background=background, blur=blur)})

    endpoint_variants = [
        ('butt_horizontal', 0, 3, 'butt', [-3, 2], 0, 0, 255),
        ('butt_oblique', 30, 5, 'butt', [2, -3], .4, 0, 255),
        ('butt_vertical', 90, 7, 'butt', [-2, 2], 0, 0, 255),
        ('butt_reverse_contrast', 155, 9, 'butt', [2, -2], .6, 70, 240),
        ('round_horizontal', 0, 5, 'round', [-3, 1], 0, 0, 255),
        ('round_oblique', -45, 7, 'round', [3, 2], .3, 0, 255),
        ('round_vertical', -90, 11, 'round', [-2, -2], 0, 0, 255),
        ('round_blur_contrast', 120, 9, 'round', [2, 3], .7, 90, 245),
    ]
    for name, angle, width, cap, shift, blur, fg, bg in endpoint_variants:
        direction = _direction(angle)
        add('endpoint_'+name, 'endpoint', [_segment(point, _add(point, direction, 140), width, cap)],
            {'expected_xy': _add(point, shift), 'inward_xy': direction, 'cap_model': cap},
            True, 'One isolated finite stroke; cap model is explicitly known from construction.', point,
            foreground=fg, background=bg, blur=blur)

    def endpoint_negative(name, commands, reason, expected=None, cap='butt'):
        add('endpoint_'+name, 'endpoint', commands,
            {'expected_xy': point if expected is None else expected, 'inward_xy': [1., 0.], 'cap_model': cap},
            False, reason)

    endpoint_negative('blank', [], 'No stroke evidence.')
    endpoint_negative('continued', [_segment([60, point[1]], [340, point[1]], 5)],
                      'Stroke continues through the whole endpoint search window.')
    endpoint_negative('double_parallel', [_segment(_add(point, [0, offset]), _add(point, [140, offset]), 3)
                                           for offset in (-4, 4)],
                      'Two distinct parallel endpoint candidates; identity is ambiguous.')
    endpoint_negative('covered_by_crossing', [_segment(point, _add(point, [140, 0]), 5),
                                               _segment(_add(point, [0, -80]), _add(point, [0, 80]), 9)],
                      'A crossing stroke obscures the tip; its center is not an exposed endpoint.')
    endpoint_negative('clipped_search', [_segment([4, 190], [160, 190], 5)],
                      'Pre-tip evidence extends outside the supplied ROI.', expected=[4, 190])
    endpoint_negative('merged_circle', [{'type': 'ring', 'center_xy': _add(point, [0, 24]),
                                         'radius_px': 24, 'width_px': 5},
                                        _segment(point, _add(point, [140, 0]), 5)],
                      'The line tip joins a circle; a unique isolated cap is not observed.')
    endpoint_negative('short_fragment', [_segment(point, _add(point, [12, 0]), 5)],
                      'The 12-pixel fragment cannot supply the required interior support.')
    endpoint_negative('unknown_cap', [_segment(point, _add(point, [140, 0]), 9, 'round')],
                      'Visible ink boundary alone does not identify the center without a cap model.', cap='unknown')

    junction_variants = [
        ('corner_axis', [0, 90], 4, [-3, 2], 0, 0),
        ('corner_rotated', [30, 120], 5, [2, -3], .3, 0),
        ('corner_acute', [10, 70], 3, [-2, -2], 0, 0),
        ('corner_obtuse', [-30, 90], 7, [3, 2], .5, 0),
        ('tee_axis', [0, 90, 180], 5, [-3, -2], 0, 0),
        ('tee_oblique_contrast', [25, 115, 205], 7, [2, 3], .6, 75),
        ('cross_rotated', [35, 125, 215, 305], 9, [-2, 2], .4, 0),
        ('three_way', [0, 120, 240], 5, [2, -2], 0, 0),
    ]
    for name, angles, width, shift, blur, fg in junction_variants:
        directions = [_direction(angle) for angle in angles]
        add('junction_'+name, 'junction', [_segment(point, _add(point, d, 140), width) for d in directions],
            {'expected_xy': _add(point, shift), 'directions_xy': directions}, True,
            'All specified finite branches meet at the authored centerline intersection.', point,
            foreground=fg, blur=blur)

    def junction_negative(name, commands, reason, directions=None, expected=None):
        add('junction_'+name, 'junction', commands,
            {'expected_xy': point if expected is None else expected,
             'directions_xy': [[1., 0.], [0., 1.]] if directions is None else directions}, False, reason)

    junction_negative('blank', [], 'No branch evidence.')
    junction_negative('missing_branch', [_segment(point, _add(point, [140, 0]), 5)],
                      'The second requested branch is absent.')
    junction_negative('gap_at_vertex', [_segment(_add(point, d, 12), _add(point, d, 140), 4, 'butt')
                                       for d in ([1, 0], [0, 1])],
                      'Extrapolated lines intersect, but finite branches leave a 12-pixel gap.')
    junction_negative('parallel', [_segment([50, point[1]], [335, point[1]], 5)],
                      'Collinear rays do not determine a unique two-dimensional junction.',
                      directions=[[1., 0.], [-1., 0.]])
    junction_negative('double_corners', [_segment(_add(point, [offset, offset]),
                                                   _add(_add(point, [offset, offset]), d, 140), 2)
                                            for offset in (-4, 4) for d in ([1, 0], [0, 1])],
                      'Two distinct nearby corners compete inside the search window.')
    junction_negative('curves_without_lines', [{'type': 'ring', 'center_xy': _add(point, [22, 22]),
                                              'radius_px': 31, 'width_px': 6}],
                      'Only a curved merged contour exists; two straight branches are unsupported.')
    junction_negative('clipped_branch', [_segment([4, 190], [140, 190], 5),
                                         _segment([4, 190], [4, 330], 5)],
                      'The vertical branch search strip is clipped by the ROI.', expected=[4, 190])
    directions = [_direction(-3), _direction(3)]
    junction_negative('near_parallel', [_segment(point, _add(point, d, 140), 3) for d in directions],
                      'A six-degree junction is ill-conditioned and branches merge in the search strip.',
                      directions=directions)
    return cases


def score_cases(cases: list[dict], results: list[dict]) -> dict:
    """Score all cases, including inaccurate positives and false localizations."""
    if len(cases) != len(results):
        raise ValueError('Case and result inventories differ')
    counts = Counter(positive_count=0, negative_count=0, localized_positive=0, correct_positive=0,
                     incorrect_positive=0, missed_positive=0, false_localizations_negative=0,
                     correct_refusals_negative=0, execution_errors=0)
    errors = []
    for case, result in zip(cases, results):
        positive = case['truth']['localizable']
        counts['positive_count' if positive else 'negative_count'] += 1
        status = result.get('status')
        if status == 'error':
            counts['execution_errors'] += 1
            continue
        if status not in {'localized', 'ambiguous', 'not_supported'}:
            raise ValueError('Invalid localizer status')
        if status == 'localized':
            point = np.asarray(result.get('observed_xy'), dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError('Localized result must contain a finite observed_xy')
            if positive:
                error = float(np.linalg.norm(point-np.asarray(case['truth']['point_xy'])))
                errors.append(error)
                counts['localized_positive'] += 1
                counts['correct_positive' if error <= case['truth']['tolerance_px'] else 'incorrect_positive'] += 1
            else:
                counts['false_localizations_negative'] += 1
        elif positive:
            counts['missed_positive'] += 1
        else:
            counts['correct_refusals_negative'] += 1
    return {**counts, 'positive_localization_coverage': counts['localized_positive']/counts['positive_count'] if counts['positive_count'] else None,
            'positive_correct_rate': counts['correct_positive']/counts['positive_count'] if counts['positive_count'] else None,
            'localized_positive_error_px': {'count': len(errors), 'mean': float(np.mean(errors)) if errors else None,
                                             'p95': float(np.percentile(errors, 95)) if errors else None,
                                             'maximum': max(errors) if errors else None},
            'all_controls_pass': (counts['correct_positive'] == counts['positive_count']
                                  and counts['correct_refusals_negative'] == counts['negative_count']
                                  and counts['execution_errors'] == 0)}


def _invoke(case):
    from .raster_endpoints import localize_endpoint
    from .raster_junctions import localize_junction
    function = localize_endpoint if case['kind'] == 'endpoint' else localize_junction
    try:
        return function(case['image'], **case['parameters'])
    except (ValueError, RuntimeError) as error:
        return {'status': 'error', 'error_type': type(error).__name__, 'message': str(error),
                'observed_xy': None, 'training_eligible': False, 'annotation_approved': False}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def evaluate(output: Path, *, run_date: str | None = None) -> dict:
    """Run the fixed controls into a new, reproducible artifact directory."""
    if run_date is not None and (not isinstance(run_date, str) or date.fromisoformat(run_date).isoformat() != run_date):
        raise ValueError('run_date must be an explicit ISO date, YYYY-MM-DD')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases, records, results, hashes = create_cases(), [], [], {}
    for case in cases:
        result = _invoke(case)
        results.append(result)
        source = f'{case["case_id"]}.png'
        overlay_name = f'{case["case_id"]}-overlay.png'
        picture = Image.fromarray(case['image'])
        picture.save(output/source)
        overlay = picture.copy()
        pen = ImageDraw.Draw(overlay)
        for xy, color, label in [(case['parameters']['expected_xy'], '#cc8800', 'hypothesis'),
                                 (case['truth']['point_xy'], '#00a060', 'truth'),
                                 (result.get('observed_xy'), '#d000c0', 'observed')]:
            if xy is None:
                continue
            x, y = xy
            pen.line((x-6, y, x+6, y), fill=color, width=1)
            pen.line((x, y-6, x, y+6), fill=color, width=1)
            pen.text((x+8, y+8), label, fill=color)
        overlay.save(output/overlay_name)
        record = {key: value for key, value in case.items() if key != 'image'}
        record.update(result=result, source_image=source, overlay_image=overlay_name)
        if case['truth']['localizable'] and result['status'] == 'localized':
            record['error_px'] = float(np.linalg.norm(np.asarray(result['observed_xy'])-case['truth']['point_xy']))
        records.append(record)
        hashes[source], hashes[overlay_name] = _digest(output/source), _digest(output/overlay_name)
    report = {'schema_version': '1.0', 'benchmark_version': BENCHMARK_VERSION,
              'benchmark_authored_date': '2026-10-27', 'run_date': run_date,
              'purpose': 'Fixed independent synthetic development controls, not customer acceptance.',
              'tolerance_px': TOLERANCE_PX, 'truth_source': 'Authored continuous drawing coordinates before localization.',
              'metric_scope': 'Error statistics include all localized positives, even outside tolerance; refusals remain in coverage denominator.',
              'summary': score_cases(cases, results),
              'by_kind': {kind: score_cases([c for c in cases if c['kind'] == kind],
                                            [r for c, r in zip(cases, results) if c['kind'] == kind])
                          for kind in ['endpoint', 'junction']},
              'cases': records, 'training_eligible': False, 'annotation_approved': False,
              'limitations': ['Synthetic scenes do not establish an accuracy bound for customer images.',
                              'Known cap models are supplied for positive endpoint scenes.',
                              'These controls were authored independently of detector outputs, but remain development data.'],
              'code_sha256': {name: _digest(Path(__file__).with_name(name)) for name in
                              ['anchor_localization_evaluation.py', 'raster_endpoints.py', 'raster_junctions.py']}}
    _write_json(output/'report.json', report)
    hashes['report.json'] = _digest(output/'report.json')
    rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in
                       [r['case_id'], r['truth']['localizable'], r['result']['status'],
                        round(r['error_px'], 4) if 'error_px' in r else '', r['truth']['reason']])+
                   f'<td><a href="{r["source_image"]}">исходник</a> · <a href="{r["overlay_image"]}">точки</a></td></tr>' for r in records)
    page = ('<!doctype html><html lang="ru"><meta charset="utf-8"><title>Контроль локализации опорных точек</title>'
            '<style>body{font:16px sans-serif;max-width:1400px;margin:2em auto}td,th{border:1px solid #bbb;padding:.5em}table{border-collapse:collapse}</style>'
            '<h1>Локализация концов и пересечений: контрольные сцены</h1>'
            '<p>32 синтетические сцены разработки. Истинные координаты заданы до запуска алгоритма. '
            'Порог ошибки — 2 пикселя; отказы не исключены из оценки полноты. Это не оценка точности на чертежах заказчика.</p>'
            f'<pre>{html.escape(json.dumps(report["summary"], ensure_ascii=False, indent=2))}</pre>'
            '<p><a href="report.json">Полный отчёт и параметры</a>. На наложениях: зелёный — истина, '
            'оранжевый — гипотеза, пурпурный — наблюдение.</p>'
            '<table><tr><th>Сцена</th><th>Локализуема</th><th>Результат</th><th>Ошибка, px</th><th>Основание</th><th>Просмотр</th></tr>'+rows+'</table></html>')
    (output/'index.html').write_text(page, encoding='utf-8')
    hashes['index.html'] = _digest(output/'index.html')
    _write_json(output/'manifest.json', {'schema_version': '1.0', 'status': 'complete',
                                        'artifact_sha256': hashes, 'training_eligible': False})
    return report
