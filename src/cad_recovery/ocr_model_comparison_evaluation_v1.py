"""Replayable OCR metrics, scoped component roles and predeclared decisions."""
from collections import Counter
import html
import statistics

from .dataset_layout_v1 import checked, encoded, read
from .ocr_baseline_metrics_v1 import aggregate_scores_v1, score_text_v1
from .ocr_model_comparison_protocol_v1 import ARMS, POLICY, require
from .ocr_model_comparison_runtime_v1 import canonical_observation
from . import angular_text_components_v1 as layout


def align_predictions(request_ids, rows):
    """All assignments survive missing output; duplicates/foreign IDs fail closed."""
    require(len({r['id'] for r in rows}) == len(rows) and
            {r['id'] for r in rows} <= set(request_ids), 'Duplicate/unassigned prediction')
    for row in rows:
        require(set(row) == {'id', 'status', 'raw', 'confidence', 'observations',
                            'warm_outputs_identical', 'case_budget_pass', 'error'}, 'Invalid prediction fields')
        canonical_observation({key: row[key] for key in ('raw', 'confidence', 'observations')})
        require(row['status'] in ('ok', 'empty', 'error') and
                (row['raw'] != '' if row['status'] == 'ok' else row['raw'] == '') and
                type(row['warm_outputs_identical']) is type(row['case_budget_pass']) is bool,
                'Invalid prediction status/completeness')
    lookup = {r['id']: r for r in rows}
    missing = {'status': 'error', 'raw': '', 'confidence': None, 'observations': [],
               'warm_outputs_identical': False, 'case_budget_pass': False, 'error': 'missing_worker_output'}
    return {identifier: lookup.get(identifier, {'id': identifier, **missing}) for identifier in request_ids}


def aggregate(targets, predictions, lineage, *, corpus, split, field=None, unique=False, by_group=False):
    selected = [t for t in targets if t['corpus'] == corpus and t['split'] == split and t['exact_target']
                and (field is None or t['field'] == field)]
    if unique:
        seen = set()
        distinct = []
        for target in selected:
            digest = lineage[target['id']]['image']['sha256']
            if digest not in seen:
                distinct.append(target)
                seen.add(digest)
        selected = distinct
    rows = []
    for target in selected:
        prediction = predictions[target['id']]
        provenance = lineage[target['id']]['provenance']
        source = provenance.get('source')
        source_id = target['group_id'] if by_group else (source['sha256'] if source else
                    provenance.get('source_sha256') or target['group_id'])
        rows.append({'id': target['id'], 'source_id': source_id, 'reference': target['literal'],
                     'prediction': prediction['raw'], 'status': prediction['status']})
    return aggregate_scores_v1(rows) if rows else None


def critical_errors(metrics):
    critical = set(POLICY['critical_codepoints'])
    events = Counter()
    for sample in metrics['samples']:
        for item in sample['raw']['alignment']:
            if item['operation'] != 'match' and (item['reference'] in critical or item['prediction'] in critical):
                events[item['operation']] += 1
    return dict(events)


def regression(root, config, predictions, arm):
    cases = read(checked(root, config['data']['regression-cases']))
    links = read(checked(root, config['data']['regression-links']))
    output = []
    for case in cases:
        rows = []
        if arm.startswith('R'):
            for component in case['components']:
                link = next(r for r in links if r['context_id'] == case['id'] and r['component_id'] == component['id'])
                result = predictions[link['id']]
                rows.append({'id': component['id'], 'raw': result['raw'],
                    'confidence': result['confidence'] if result['confidence'] is not None else 0.,
                    'polygon_px': layout.polygon(component['box_px']), 'conflicting_line_ids': []})
        else:
            link = next(r for r in links if r['context_id'] == case['id'] and r['component_id'] is None)
            for index, row in enumerate(predictions[link['id']]['observations']):
                rows.append({'id': f'ocr-{index:04d}', **row, 'conflicting_line_ids': []})
        grouped = layout.group(rows, upstream_incomplete=case['upstream_incomplete'])
        metrics = layout.evaluate(case, rows, grouped)
        output.append({'id': case['id'], 'split': case['split'], 'observations': rows,
                       'groups': grouped, 'metrics': metrics})
    return {'records': output, 'train_totals': layout.totals([r['metrics'] for r in output if r['split'] == 'train']),
            'development_totals': layout.totals([r['metrics'] for r in output if r['split'] == 'development']),
            'scope': 'four viewed contexts; R uses oracle component boxes, D actual full-context OCR; not a full CAD-link benchmark'}


def judge_pair(baseline, candidate, *, glyph_baseline, glyph_candidate,
               baseline_roles, candidate_roles, complete, repeat_equal, resource_pass, latency_ratio,
               negative_regressions):
    old, new = baseline['raw'], candidate['raw']
    old_samples = {r['id']: r for r in baseline['samples']}
    false_corrections = [r['id'] for r in candidate['samples']
                         if old_samples[r['id']]['raw']['exact_match'] and not r['raw']['exact_match']]
    source_regressions = [key for key, value in candidate['by_source'].items()
                          if value['raw']['distance'] > baseline['by_source'][key]['raw']['distance']]
    relative = (old['cer']-new['cer'])/old['cer'] if old['cer'] else None
    glyph_losses = []
    glyph_pass = None
    if glyph_baseline is not None and glyph_candidate is not None:
        previous = {r['id']: r for r in glyph_baseline['samples']}
        glyph_losses = [r['id'] for r in glyph_candidate['samples']
                         if previous[r['id']]['raw']['exact_match'] and not r['raw']['exact_match']]
        glyph_pass = not glyph_losses and glyph_candidate['raw']['exact_matches'] >= glyph_baseline['raw']['exact_matches']
    old_roles, new_roles = baseline_roles['train_totals'], candidate_roles['train_totals']
    lost_links = max(0, old_roles['correct_role_edges']-new_roles['correct_role_edges'])
    added_false = max(0, new_roles['false_role_edges']-old_roles['false_role_edges'])
    # Also retain case-level regressions: one recovered link cannot hide another lost link.
    old_case = {r['id']: r['metrics'] for r in baseline_roles['records']}
    role_regressions = [r['id'] for r in candidate_roles['records'] if
        r['metrics']['correct_role_edges'] < old_case[r['id']]['correct_role_edges'] or
        r['metrics']['false_role_edges'] > old_case[r['id']]['false_role_edges']]
    gates = {'primary_CER': relative >= POLICY['primary_relative_raw_CER_reduction_min'] if relative is not None else None,
             'primary_exact': new['exact_matches'] >= old['exact_matches'],
             'no_false_corrections': not false_corrections,
             'no_source_regressions': not source_regressions,
             'known_glyphs': glyph_pass,
             'no_new_negative_signs': not negative_regressions,
             'scoped_roles': not role_regressions,
             'complete': complete, 'repeatable': repeat_equal, 'resource_limits': resource_pass,
             'latency': latency_ratio is not None and latency_ratio <= POLICY['limits']['median_latency_ratio_max']}
    # D0/D1 do not change the recognizer, and glyphs are not assigned to text
    # detection. Their glyph gate is explicitly not evaluated, not a free pass.
    required = [value for key, value in gates.items() if key != 'known_glyphs' or glyph_pass is not None]
    failed = [key for key, value in gates.items() if value is False]
    return {'decision': 'reject_candidate' if failed else 'diagnostic_candidate_only' if all(required) else 'insufficient_evidence',
            'gates': gates, 'relative_raw_CER_reduction': relative,
            'false_corrections': false_corrections, 'source_regressions': source_regressions,
            'glyph_correct_to_wrong': glyph_losses, 'role_regressions': role_regressions,
            'lost_train_role_links': lost_links, 'new_false_train_role_links': added_false,
            'negative_regressions': negative_regressions, 'median_paired_latency_ratio': latency_ratio,
            'broad_engineering_admission': False, 'production_admitted': False,
            'reason': 'Bounded diagnostic data and scoped roles cannot establish independent full-service CAD acceptance.'}


def build_report(root, config, output):
    protocol = read(checked(root, config['protocol']))
    targets = read(checked(root, protocol['data']['targets']))
    lineage = {r['id']: r for r in read(checked(root, protocol['data']['lineage']))}
    runtimes = read(output/'runtimes.json')
    per_arm = {}
    for arm in ARMS:
        repeats = []
        for number in (1, 2):
            folder = output/f'repeat-{number}'/arm/'result'
            rows = read(folder/'results.json') if (folder/'results.json').exists() else []
            aligned = align_predictions(config['assignments'][arm], rows)
            timings = read(folder/'timings.json') if (folder/'timings.json').exists() else {'rows': [], 'init_seconds': None}
            repeats.append({'predictions': aligned, 'timings': timings})
        first = repeats[0]['predictions']
        scores = {}
        for cohort in POLICY['cohorts']:
            corpus, split = cohort.split('/')
            if arm.startswith('D') and corpus != 'ocr':
                continue
            scores[cohort] = aggregate(targets, first, lineage, corpus=corpus, split=split)
        primary = aggregate(targets, first, lineage, corpus='ocr', split='validation')
        fields = {field: aggregate(targets, first, lineage, corpus='ocr', split='validation', field=field)
                  for field in sorted({t['field'] for t in targets if t['corpus'] == 'ocr' and t['split'] == 'validation'})}
        unknown = [{'id': t['id'], 'raw': first[t['id']]['raw'], 'status': first[t['id']]['status'],
                    'split': t['split'], 'class': t['class']}
                   for t in targets if not t['exact_target'] and t['id'] in first]
        arm_runtimes = [r for r in runtimes if r['arm'] == arm]
        complete = all(len(rep['predictions']) == len(config['assignments'][arm]) and
            all(row['status'] != 'error' and row['case_budget_pass'] and row['warm_outputs_identical'] for row in rep['predictions'].values())
            for rep in repeats)
        per_arm[arm] = {'scores': scores, 'primary': primary, 'by_field': fields,
            'primary_unique_images': aggregate(targets, first, lineage, corpus='ocr', split='validation', unique=True),
            'primary_by_group': aggregate(targets, first, lineage, corpus='ocr', split='validation', by_group=True),
            'critical_raw_errors': critical_errors(primary), 'unknown': unknown,
            'critical_errors_by_cohort': {key: critical_errors(value) for key, value in scores.items() if value is not None},
            'roles': regression(root, config, first, arm), 'repeat_equal': repeats[0]['predictions'] == repeats[1]['predictions'],
            'complete': complete, 'runtime_pass': len(arm_runtimes) == 2 and all(r['budget_pass'] for r in arm_runtimes),
            'runtimes': arm_runtimes, 'timings': [rep['timings'] for rep in repeats]}
    comparisons = {}
    validation_ids = {t['id'] for t in targets if t['corpus'] == 'ocr' and t['split'] == 'validation'}
    for baseline, candidate in [('R0', 'R1'), ('D0', 'D1')]:
        first, second = per_arm[baseline], per_arm[candidate]
        ratios = []
        for repeat in range(2):
            before = {r['id']: statistics.median(r['warm_seconds']) for r in first['timings'][repeat]['rows'] if len(r['warm_seconds']) == 3}
            after = {r['id']: statistics.median(r['warm_seconds']) for r in second['timings'][repeat]['rows'] if len(r['warm_seconds']) == 3}
            ratios.extend(after[key]/before[key] for key in validation_ids if key in before and key in after and before[key] > 0)
        old_negative = {r['id']: r for r in first['unknown'] if r['class'] in POLICY['supported_negative_glyphs']}
        negative = [r['id'] for r in second['unknown'] if r['id'] in old_negative and
                    any(c in r['raw'] for c in POLICY['critical_codepoints']) and
                    not any(c in old_negative[r['id']]['raw'] for c in POLICY['critical_codepoints'])]
        comparisons[baseline+'/'+candidate] = judge_pair(first['primary'], second['primary'],
            glyph_baseline=first['scores'].get('glyph/validation'), glyph_candidate=second['scores'].get('glyph/validation'),
            baseline_roles=first['roles'], candidate_roles=second['roles'], complete=first['complete'] and second['complete'],
            repeat_equal=first['repeat_equal'] and second['repeat_equal'], resource_pass=first['runtime_pass'] and second['runtime_pass'],
            latency_ratio=statistics.median(ratios) if len(ratios) == 128 else None, negative_regressions=negative)
    return {'schema_version': 'ocr-model-comparison-result/1.0', 'task': 'P16.2i.2',
            'arms': per_arm, 'comparisons': comparisons, 'protocol': config['protocol'],
            'regression_scope': config['regression_scope'], 'closed_test_used': False,
            'training_updates': 0, 'independent_evaluation': False, 'production_admitted': False,
            'physical_geometry_approved': False}


def write_report(root, config, output):
    report = build_report(root, config, output)
    (output/'report.json').write_bytes(encoded(report))
    body = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>OCR comparison</title>',
        '<style>body{font:16px system-ui;margin:2rem}td,th{padding:.5rem;border:1px solid #aaa}</style>',
        '<h1>Сравнение OCR-моделей</h1><p>Просмотренный диагностический контроль. ',
        'Результаты не подтверждают физическую геометрию или готовность сервиса.</p>',
        '<p><a href="report.json">Полный отчёт: raw, normalized, срезы, связи, ресурсы</a></p>',
        '<table><tr><th>Вариант</th><th>Exact /64</th><th>Raw CER</th><th>Повторы</th></tr>']
    for arm, value in report['arms'].items():
        raw = value['primary']['raw']
        body.append(f'<tr><td>{arm}</td><td>{raw["exact_matches"]}/64</td><td>{raw["cer"]:.3%}</td><td>{value["repeat_equal"]}</td></tr>')
    body.append('</table><ul>')
    for pair, decision in report['comparisons'].items():
        body.append('<li>'+html.escape(pair+': '+decision['decision'])+'</li>')
    (output/'index.html').write_text(''.join(body)+'</ul></html>', encoding='utf-8')
    return report
