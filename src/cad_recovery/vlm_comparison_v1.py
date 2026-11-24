"""Frozen conditional VLM comparison; targets never enter the model request.

All assigned cases survive abstention, skips and worker failure. The fused arm
is a diagnostic counterfactual, never an edit to the OCR or accepted CAD graph.
"""
from collections import Counter
from copy import deepcopy
import math

from .dataset_layout_v1 import checked, encoded, pin, read
from .ocr_baseline_metrics_v1 import aggregate_scores_v1, normalize_text_v1
from .vlm_adapter_v1 import POLICY as ADAPTER_POLICY, envelope, require, trigger_allowed

PREVIOUS = {'path': 'artifacts/vlm-adapter-release-v1.json',
            'sha256': '93e65012b3037e548aa351a09c0a57ac875e6150c4e36e3d3fd4314c626f48b9'}
ADAPTER = {'path': 'configs/reviews/vlm-adapter-v2.json',
           'sha256': '66cfcb8f73955fa4b7ea452521513dd49a1506e5950166f232e3a3996967aa4b'}
ADMISSION = {'path': 'artifacts/vlm-qualification-v2/admission.json',
             'sha256': '7cbd3b80ff6eb1c56d567209015eb1ea12691a237d716e7161a52761aba1e7a7'}
INPUTS = {
    'ocr': 'artifacts/architecture-baseline-v1/ocr-baseline.json',
    'signs': 'artifacts/architecture-baseline-v1/sign-baseline.json',
    'glyph_confidence': 'artifacts/engineering-glyph-classifier-v1/result/raw-ocr.json',
    'component_cases': 'artifacts/angular-components-inputs-v1/cases.json',
}
CODE = ['src/cad_recovery/vlm_comparison_v1.py', 'scripts/run_vlm_comparison_v1.py',
        'src/cad_recovery/ocr_baseline_metrics_v1.py', 'src/cad_recovery/angular_text_components_v1.py',
        'docs/vlm-comparison-protocol-v1.md']
POLICY = {
    'task': 'P16.4b', 'repeats': 2,
    'selection': 'all 64 OCR validation, 50 glyph validation + 4 glyph development, 7 components, 4 angular',
    'trigger_order': ['no_existing_candidates', 'ambiguous_candidates', 'low_confidence', 'ocr_disagreement', 'not_triggered'],
    'low_confidence_threshold': .85,
    'primary': 'ocr/validation',
    'primary_relative_raw_CER_reduction_min': .10,
    'primary_exact_matches_must_not_decrease': True,
    'maximum_false_corrections': 0,
    'maximum_primary_source_distance_regressions': 0,
    'minimum_schema_valid_fraction': .95,
    'maximum_unknown_proposals': 0,
    'candidate_point_tolerance_px': 8,
    'resources': ADAPTER_POLICY,
    'failure': 'retain baseline, count missing reply as worker_error; no retry',
    'glyph_improved_arm': 'same cached glyph OCR; no admitted improved glyph model exists',
    'component_improved_arm': 'given component crops; oracle localization, not automatic recall',
    'links': 'reuse existing pixel observations and group rules; no new coordinates or acceptance',
    'unknown': 'unscored literal; proposal is manual-review risk, not proven incorrect',
    'dependent_cases_summed': False, 'independent_evaluation': False,
    'training_executed': False, 'graph_acceptance': False,
    'physical_geometry_approved': False, 'production_admitted': False,
}


def protected(root):
    old = read(checked(root, PREVIOUS))
    values = dict(old['prior_protected_pins'])
    for name, sha in old['files'].items():
        require(name not in values or values[name] == sha, 'Conflicting protected pin')
        values[name] = sha
    values[PREVIOUS['path']] = PREVIOUS['sha256']
    for name, sha in values.items():
        checked(root, {'path': name, 'sha256': sha})
    return values


def literal_trigger(ordinary, improved, confidence):
    """Uses predictions only. References/classes never determine a call."""
    if type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= .85:
        return {'reason': 'low_confidence', 'confidence': confidence}
    if ordinary != improved:
        return {'reason': 'ocr_disagreement', 'baseline': ordinary, 'alternative': improved}
    return {'reason': 'not_triggered'}


def matched(row, component_id):
    hits = [m for m in row['metrics']['matches'] if m['target_id'] == component_id]
    require(len(hits) <= 1, 'Duplicate spatial match')
    if not hits:
        return '', None, None
    hit = hits[0]
    observation = next(o for o in row['observations'] if o['id'] == hit['observation_id'])
    return observation['raw'], observation['confidence'], observation['id']


def assignments(root, config, cases):
    """Reconstruct cached baselines and trigger decisions without reading targets."""
    ocr = read(checked(root, config['inputs']['ocr']))
    predictions = read(checked(root, config['inputs']['ocr_predictions']))
    ocr_rows = {v['id']: v for v in ocr['rows']}
    ordinary = {v['id']: v for v in predictions['legacy_pipeline']}
    improved = {v['id']: v for v in predictions['legacy_rec']}
    signs = read(checked(root, config['inputs']['signs']))
    glyphs = {v['id']: v for v in signs['glyph_rows']}
    scores = {v['id']: v['score'] for v in read(checked(root, config['inputs']['glyph_confidence']))['glyphs']}
    components = {(v['id'], v['mode']): v for v in signs['real_component_results']}
    rows = []
    for case in cases:
        corpus, split, uid = case.get('corpus'), case['split'], case.get('upstream_id')
        selected = ((corpus == 'ocr' and split == 'validation') or
                    (corpus == 'glyph' and split in ('validation', 'development')) or
                    corpus in ('components', 'angular'))
        if not selected:
            continue
        checked(root, case['image'])
        row = {k: case[k] for k in ('id', 'corpus', 'split', 'group_id', 'upstream_id', 'image')}
        row.update(ordinary=None, improved=None, confidence=None, observation_id=None)
        if corpus == 'ocr':
            require(ocr_rows[uid]['image'] == case['image'], 'OCR fragment mismatch')
            row.update(ordinary=ordinary[uid]['prediction'], improved=improved[uid]['prediction'],
                       confidence=improved[uid]['confidence'])
        elif corpus == 'glyph':
            require(glyphs[uid]['image'] == case['image'], 'Glyph fragment mismatch')
            row.update(ordinary=glyphs[uid]['raw_OCR'], improved=glyphs[uid]['raw_OCR'], confidence=scores[uid])
        elif corpus == 'components':
            context_id, component_id = uid.split('/')
            a, _, _ = matched(components[context_id, 'full_ocr'], component_id)
            b, score, oid = matched(components[context_id, 'given_crops'], component_id)
            row.update(ordinary=a, improved=b, confidence=score, observation_id=oid)
        if corpus == 'angular':
            if not case['candidates']:
                trigger = {'reason': 'no_existing_candidates'}
            else:
                producers = {p['producer']['path']: read(checked(root, p['producer']))
                             for p in case['provenance']['candidate_mapping']}
                ambiguous = any(a['status'] == 'ambiguous' for p in producers.values() for a in p['associations'])
                trigger = {'reason': 'ambiguous_candidates', 'upstream_requires_review': ambiguous}
        else:
            trigger = literal_trigger(row['ordinary'], row['improved'], row['confidence'])
        row['trigger'] = trigger
        row['call_planned'] = trigger_allowed(case, trigger) and (case['task'] != 'candidate' or bool(case['candidates']))
        rows.append(row)
    rows.sort(key=lambda v: v['id'])
    require(Counter(r['corpus'] for r in rows) == {'ocr': 64, 'glyph': 54, 'components': 7, 'angular': 4},
            'Changed comparison population')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate assignment')
    return rows


def freeze(root, destination, folder):
    from .vlm_protocol_v1 import context
    from .architecture_data_readiness_v1 import Package
    require(not destination.exists() and not folder.exists(), 'Choose fresh comparison paths')
    previous = protected(root)
    adapter, cases = context(root, ADAPTER)
    admission = read(checked(root, ADMISSION))
    require(admission['qualified'] and admission['protocol'] == ADAPTER, 'Unqualified adapter')
    inputs = {k: {'path': p, 'sha256': previous[p]} for k, p in INPUTS.items() if p in previous}
    packages = []
    for key in ('glyph_confidence', 'component_cases'):
        name = INPUTS[key]; parent, member = name.rsplit('/', 1)
        manifest_name = parent + '/manifest.json'
        package_record = {'path': manifest_name, 'sha256': previous[manifest_name]}
        package = Package(root, package_record)
        package.path(member)
        inputs[key] = pin(root, name)
        packages.append(package_record)
    inputs['ocr_predictions'] = read(checked(root, inputs['ocr']))['cached_predictions']
    checked(root, inputs['ocr_predictions'])
    config = {'schema_version': 'vlm-comparison-protocol/1.0', 'policy': POLICY,
              'previous': PREVIOUS, 'adapter': ADAPTER, 'admission': ADMISSION, 'inputs': inputs, 'input_packages': packages,
              'targets': adapter['targets'], 'code': [pin(root, p) for p in CODE],
              'customer_calls_before_freeze': 0}
    rows = assignments(root, config, cases)
    folder.mkdir(parents=True)
    (folder / 'assignments.json').write_bytes(encoded(rows))
    request = {'schema_version': 'vlm-requests/1.0', 'scope': 'diagnostic_evaluation',
               'protocol': ADAPTER, 'admission': ADMISSION,
               'requests': [{'case_id': r['id'], 'trigger': r['trigger']} for r in rows if r['call_planned']]}
    require(0 < len(request['requests']) <= ADAPTER_POLICY['maximum_requests'], 'Invalid bounded request count')
    (folder / 'requests.json').write_bytes(encoded(request))
    config['assignments'] = pin(root, (folder / 'assignments.json').relative_to(root).as_posix())
    config['request'] = pin(root, (folder / 'requests.json').relative_to(root).as_posix())
    destination.write_bytes(encoded(config))
    return {'protocol': pin(root, destination.relative_to(root).as_posix()), 'assigned': len(rows),
            'requests_per_repeat': len(request['requests']), 'protected_pins': len(previous)}


def load(root, record):
    from .vlm_protocol_v1 import context
    value = read(checked(root, record))
    require(value['policy'] == POLICY and value['previous'] == PREVIOUS and value['adapter'] == ADAPTER
            and value['admission'] == ADMISSION and value['code'] == [pin(root, p) for p in CODE], 'Changed comparison protocol/code')
    adapter, cases = context(root, ADAPTER)
    for package in value['input_packages']:
        checked(root, package)
    require(adapter['targets'] == value['targets'], 'Changed targets')
    rows = read(checked(root, value['assignments']))
    require(rows == assignments(root, value, cases), 'Assignment/trigger drift')
    request = read(checked(root, value['request']))
    expected = [{'case_id': r['id'], 'trigger': r['trigger']} for r in rows if r['call_planned']]
    require(request == {'schema_version': 'vlm-requests/1.0', 'scope': 'diagnostic_evaluation',
                       'protocol': ADAPTER, 'admission': ADMISSION, 'requests': expected}, 'Changed requests')
    return value, {c['id']: c for c in cases}, rows


def combine(rows, cases, result):
    """Fail closed; score every assigned case, retain old OCR on every refusal."""
    returned = result.get('results', []) if result else []
    ids = [v['case_id'] for v in returned]
    expected = {v['id'] for v in rows if v['call_planned']}
    require(len(ids) == len(set(ids)) and set(ids) <= expected, 'Duplicate/unassigned response')
    by_id = {v['case_id']: v for v in returned}
    outcomes = []
    for row in rows:
        case = cases[row['id']]
        if not row['call_planned']:
            reply = envelope(case, skip=row['trigger']['reason'])
        elif row['id'] not in by_id:
            reply = envelope(case, skip='worker_error')
        else:
            reply = by_id[row['id']]
            reason = reply['reason']
            parsed = envelope(case, reply.get('raw'), skip=reason if reason not in
                              ('valid_proposal', 'model_abstention', 'invalid_model_output') else None)
            require({k: v for k, v in reply.items() if k != 'runtime'} == parsed, 'Altered reply/provenance')
        proposed = reply['status'] == 'propose'
        outcomes.append({**row, 'reply': reply,
                         'fused': reply['literal'] if proposed and case['task'] == 'literal' else row['improved'],
                         'baseline_preserved_on_disk': True, 'automatic_review_reduction': 0})
    return outcomes


def literal_scores(outcomes, targets, corpus, split=None):
    subset = [r for r in outcomes if r['corpus'] == corpus and (split is None or r['split'] == split)
              and targets[r['id']]['exact_target'] and targets[r['id']]['literal'] is not None]
    result = {}
    for arm in ('ordinary', 'improved', 'fused'):
        rows = [{'id': r['id'], 'source_id': r['group_id'], 'reference': targets[r['id']]['literal'],
                 'prediction': r[arm], 'status': 'ok' if r[arm] else 'empty'} for r in subset]
        result[arm] = aggregate_scores_v1(rows) if rows else None
    result['changes'] = {'corrected': [], 'false_corrections': [], 'changed_wrong_to_wrong': []}
    for r in subset:
        target = targets[r['id']]['literal']
        if r['fused'] == r['improved']:
            continue
        key = ('false_corrections' if r['improved'] == target else
               'corrected' if r['fused'] == target else 'changed_wrong_to_wrong')
        result['changes'][key].append(r['id'])
    return result


def candidate_scores(root, outcomes, cases, targets):
    records = []
    for row in outcomes:
        if row['corpus'] != 'angular':
            continue
        case = cases[row['id']]; chosen = row['reply']['candidate_id']
        points = None; pixel_match = None; geometry_aliases = []
        for mapping in case['provenance']['candidate_mapping']:
            producer = read(checked(root, mapping['producer']))
            association = next(a for a in producer['associations'] if a['id'] == mapping['association_id'])
            alternative = association['alternatives'][mapping['alternative_index']]
            geometry_aliases.append(alternative['points_px'])
            if mapping['id'] == chosen:
                points = alternative['points_px']
        truth = targets[row['id']]['expected_points_px']
        if chosen and truth:
            def distance(order):
                return max(math.dist(points[i], truth[j]) for i, j in enumerate(order))
            pixel_match = min(distance([0, 1, 2]), distance([0, 2, 1])) <= POLICY['candidate_point_tolerance_px']
        records.append({'id': row['id'], 'split': row['split'], 'chosen': chosen,
                        'reason': row['reply']['reason'], 'pixel_match': pixel_match,
                        'candidate_count': len(case['candidates']),
                        'unique_geometry_count': len({str(p) for p in geometry_aliases}),
                        'semantic_association_resolved': False, 'requires_review': True})
    return records


def component_links(root, config, outcomes):
    from .angular_text_components_v1 import group, evaluate, totals
    signs = read(checked(root, config['inputs']['signs']))
    contexts = read(checked(root, config['inputs']['component_cases']))
    output = []
    for case in contexts:
        current = [r for r in outcomes if r['corpus'] == 'components' and r['upstream_id'].split('/')[0] == case['id']]
        for mode in ('full_ocr', 'candidate_v3', 'given_crops', 'given_crops_plus_vlm'):
            baseline_mode = 'given_crops' if mode == 'given_crops_plus_vlm' else mode
            cached = next(r for r in signs['real_component_results'] if r['id'] == case['id'] and r['mode'] == baseline_mode)
            observations = deepcopy(cached['observations'])
            if mode == 'given_crops_plus_vlm':
                for row in current:
                    if row['reply']['status'] == 'propose':
                        for observation in observations:
                            if observation['id'] == row['observation_id']:
                                observation['raw'] = row['fused']
            # Cached OCR confidence is held constant solely to isolate literal changes.
            # It is not VLM confidence; none of these groups are accepted into CAD.
            report = group(observations, upstream_incomplete=case['upstream_incomplete'])
            metrics = evaluate(case, observations, report)
            output.append({'id': case['id'], 'split': case['split'], 'mode': mode,
                           'metrics': metrics, 'groups': report, 'accepted_graph_edges': 0})
    return {'records': output, 'known_train_totals': {
        mode: totals([v['metrics'] for v in output if v['mode'] == mode and v['split'] == 'train'])
        for mode in ('full_ocr', 'candidate_v3', 'given_crops', 'given_crops_plus_vlm')},
        'confidence_note': 'Cached OCR confidence held constant; not a calibrated VLM score. Given-crop localization is oracle.'}


def judge(primary, glyph, components, unknown, outcomes, repeat_equal, resource_pass, development=None):
    base, fused = primary['improved']['raw'], primary['fused']['raw']
    requested = [r for r in outcomes if r['call_planned']]
    valid = sum(r['reply']['reason'] in ('valid_proposal', 'model_abstention') for r in requested)
    groups0 = primary['improved']['by_source']; groups1 = primary['fused']['by_source']
    regressions = [g for g in groups0 if groups1[g]['raw']['distance'] > groups0[g]['raw']['distance']]
    checks = {
        'primary_raw_CER_gain': fused['cer'] <= base['cer'] * (1 - POLICY['primary_relative_raw_CER_reduction_min']) and fused['cer'] < base['cer'],
        'primary_exact_not_worse': fused['exact_matches'] >= base['exact_matches'],
        'no_false_corrections': all(not a['changes']['false_corrections'] for a in
                                    (primary, glyph, components, *([development] if development else []))),
        'no_primary_source_regressions': len(regressions) <= POLICY['maximum_primary_source_distance_regressions'],
        'glyph_not_worse': glyph['fused']['normalized']['distance'] <= glyph['improved']['normalized']['distance'],
        'components_not_worse': components['fused']['raw']['distance'] <= components['improved']['raw']['distance'],
        'schema_valid_fraction': bool(requested) and valid / len(requested) >= POLICY['minimum_schema_valid_fraction'],
        'unknown_proposals': unknown['proposals'] <= POLICY['maximum_unknown_proposals'],
        'repeat_equal': repeat_equal, 'resources': resource_pass,
    }
    return {'checks': checks, 'diagnostic_candidate_admitted': all(checks.values()),
            'decision': 'admit_for_further_validation' if all(checks.values()) else 'reject',
            'failed_checks': [k for k, v in checks.items() if not v], 'primary_source_regressions': regressions,
            'valid_responses': valid, 'requested': len(requested),
            'schema_valid_fraction': valid / len(requested) if requested else None,
            'production_admitted': False, 'graph_acceptance': False, 'physical_geometry_approved': False}


def execution_checks(result, runtime, requested):
    if not result or not runtime:
        return False
    replies = result['results']
    return (result['generation'] == {'do_sample': False, 'num_beams': 1, 'mode': 'greedy_search', 'max_new_tokens': 192}
            and result['training_executed'] is False and result['network_disabled'] is True
            and len(replies) == requested and result['model_calls'] <= requested
            and runtime['budget_pass'] and not runtime['gpu_memory']['errors']
            and runtime['gpu_memory']['peak_MiB'] is not None
            and runtime['gpu_memory']['peak_MiB'] <= ADAPTER_POLICY['device_MiB']
            and result['allocated_peak_MiB'] <= ADAPTER_POLICY['allocated_MiB']
            and result['reserved_peak_MiB'] <= ADAPTER_POLICY['reserved_MiB']
            and all(r.get('runtime') is not None and r['runtime']['seconds'] <= ADAPTER_POLICY['request_seconds']
                    and r['runtime']['input_tokens'] <= ADAPTER_POLICY['maximum_input_tokens']
                    and r['runtime']['image_tokens'] <= ADAPTER_POLICY['maximum_image_tokens']
                    and len(r['runtime']['token_ids']) == r['runtime']['new_tokens'] <= ADAPTER_POLICY['maximum_new_tokens']
                    for r in replies))


def evaluate_runs(root, config, cases, rows, runs, runtimes):
    targets = {r['id']: r for r in read(checked(root, config['targets']))}
    combined = [combine(rows, cases, run) for run in runs]
    signatures = [[(r['case_id'], r['status'], r['raw'], r.get('runtime', {}).get('token_ids'))
                   for r in (run or {}).get('results', [])] for run in runs]
    n = sum(r['call_planned'] for r in rows)
    complete = all(run and len(run['results']) == n for run in runs)
    same = complete and signatures[0] == signatures[1]
    resources = all(execution_checks(run, runtime, n) for run, runtime in zip(runs, runtimes))
    reports = []
    for outcomes in combined:
        primary = literal_scores(outcomes, targets, 'ocr', 'validation')
        glyph = literal_scores(outcomes, targets, 'glyph', 'validation')
        development = literal_scores(outcomes, targets, 'glyph', 'development')
        component = literal_scores(outcomes, targets, 'components')
        unknown_rows = [r for r in outcomes if r['corpus'] in ('glyph', 'components') and not targets[r['id']]['exact_target']]
        unknown = {'count': len(unknown_rows), 'proposals': sum(r['reply']['status'] == 'propose' for r in unknown_rows),
                   'records': [{'id': r['id'], 'reason': r['reply']['reason'], 'literal': r['reply']['literal']} for r in unknown_rows],
                   'truth': 'unresolved; proposals are risk, not asserted wrong answers'}
        reports.append({'primary': primary, 'glyph_validation': glyph,
                        'glyph_development': development,
                        'components': component, 'unknown': unknown,
                        'candidates': candidate_scores(root, outcomes, cases, targets),
                        'component_links': component_links(root, config, outcomes),
                        'reason_counts': dict(Counter(r['reply']['reason'] for r in outcomes)),
                        'call_reason_counts': dict(Counter(r['reply']['reason'] for r in outcomes if r['call_planned'])),
                        'decision': judge(primary, glyph, component, unknown, outcomes, same, resources, development),
                        'outcomes': outcomes})
    return {'schema_version': 'vlm-comparison-results/1.0', 'task': 'P16.4b',
            'assigned_per_repeat': len(rows), 'requested_per_repeat': n,
            'fresh_runs': 2, 'greedy_repeat_equal': same, 'resources_pass': resources,
            'model_calls': sum((r or {}).get('model_calls', 0) for r in runs),
            'repeats': reports, 'training_executed': False, 'production_admitted': False,
            'independent_evaluation': False, 'graph_acceptance': False, 'physical_geometry_approved': False}


def run(root, record, output):
    from .vlm_protocol_v1 import execute_requests
    require(not output.exists(), 'Choose a new comparison result folder')
    config, cases, rows = load(root, record)
    protected(root)
    output.mkdir(parents=True)
    (output / 'started.json').write_bytes(encoded({'protocol': record, 'requests': config['request'],
                                                  'repeats': POLICY['repeats']}))
    runs = []; runtimes = []
    for i in range(POLICY['repeats']):
        folder = output / ('repeat-' + str(i))
        try:
            result = execute_requests(root, ADAPTER, config['request'], folder)
        except (ValueError, RuntimeError, OSError) as error:
            result = None
            (output / ('repeat-' + str(i) + '-failure.json')).write_bytes(encoded({'error': str(error), 'retry': False}))
        runs.append(result)
        runtimes.append(read(folder / 'runtime.json') if (folder / 'runtime.json').exists() else None)
        print('Finished fixed repeat', i, 'model_calls', (result or {}).get('model_calls'), flush=True)
    report = evaluate_runs(root, config, cases, rows, runs, runtimes)
    (output / 'report.json').write_bytes(encoded(report))
    protected(root)
    manifest = {'schema_version': 'vlm-comparison-manifest/1.0', 'protocol': record,
                'files': [pin(output, p.relative_to(output).as_posix()) for p in sorted(output.rglob('*')) if p.is_file()]}
    (output / 'manifest.json').write_bytes(encoded(manifest))
    return {'decision': report['repeats'][0]['decision'], 'model_calls': report['model_calls'],
            'repeat_equal': report['greedy_repeat_equal'], 'resources_pass': report['resources_pass']}


def verify(root, record, output):
    """Authenticate logs and re-score raw replies with no model loading."""
    config, cases, rows = load(root, record)
    manifest = read(output / 'manifest.json')
    require(manifest['protocol'] == record and len({p['path'] for p in manifest['files']}) == len(manifest['files']), 'Changed manifest')
    for value in manifest['files']:
        checked(output, value)
    runs = []; runtimes = []
    for i in range(POLICY['repeats']):
        folder = output / ('repeat-' + str(i))
        failed = (output / ('repeat-' + str(i) + '-failure.json')).exists()
        result = read(folder / 'result/result.json') if not failed else None
        if result:
            require(result['request'] == config['request'], 'Different worker request')
        runs.append(result)
        runtimes.append(read(folder / 'runtime.json') if (folder / 'runtime.json').exists() else None)
    expected = evaluate_runs(root, config, cases, rows, runs, runtimes)
    require((output / 'report.json').read_bytes() == encoded(expected), 'Recomputed quality/decision differs')
    count = len(protected(root))
    return {'verified': True, 'model_loaded': False, 'new_model_calls': 0, 'protected_pins': count,
            'replayed_model_calls': expected['model_calls'], 'assigned_per_repeat': len(rows),
            'decision': expected['repeats'][0]['decision']['decision']}
