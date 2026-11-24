"""Frozen head-only comparison; optional cached-geometry binding is scored separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/local-width-heads-v1'
PREVIOUS = {'path': 'artifacts/local-role-guard-release-v1.json', 'sha256': '43187d5e412e00730cd87378727e85b8fd71be3bca64ce31eaa48bbc0368d213'}
CONTROLS = {'path': 'artifacts/local-width-head-controls-v1/contract.json', 'sha256': 'f1bfbb4d6c6c3d70ca165ccd0f9f26cc5f476c5160a037fe1684c4fa9a220315'}
CORE = {'path': 'artifacts/cv-comparison-protocol-v1/contract.json', 'sha256': '815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}


def read(path):
    return json.loads(path.read_text('utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def freeze():
    from cad_recovery.dataset_layout_v1 import checked, pin
    from cad_recovery.local_width_heads_v1 import POLICY
    from verify_cv_comparison_contract_v1 import verify_contract
    verify_contract(ROOT, CORE); checked(ROOT, PREVIOUS)
    controls = read(checked(ROOT, CONTROLS)); checked(ROOT, controls['generator'])
    old = read(ROOT/'artifacts/local-role-guard-v1/plan.json')
    index = {(r['case'], r['mode']): r for r in read(ROOT/'artifacts/local-role-guard-v1/results-index.json') if not r['repeat']}
    jobs = []
    for job in old['jobs']:
        views = []
        for view in job['views']:
            r = index[job['case']['id'], view['mode']]; folder = (ROOT/r['artifact']).parent
            views.append({'mode': view['mode'], 'gray': view['gray'], 'input_to_native': view['input_to_native'],
                'cache': pin(ROOT, r['artifact']), 'geometry': pin(ROOT, (folder/'geometry.json').relative_to(ROOT).as_posix()),
                'text_mask': pin(ROOT, (folder/'text-mask.png').relative_to(ROOT).as_posix()),
                'baseline_roles': pin(ROOT, (folder/'old_context-roles.png').relative_to(ROOT).as_posix())})
        jobs.append({'case': job['case'], 'views': views, 'new_control': False})
    jobs.extend(controls['jobs'])
    expanded_pin = pin(ROOT, 'artifacts/pixel-role-expansion-v3/records.json')
    labels = {r['id']: r for r in read(checked(ROOT, expanded_pin)) if 'source_id' in r}
    legacy_pin = pin(ROOT, 'configs/reviews/stroke-curve-control-reading-v1.json')
    legacy = {r['patch_id']: r for r in read(checked(ROOT, legacy_pin))['patches']}
    regression_pin = pin(ROOT, 'artifacts/arrow-head-validation-v1/report.json')
    for row in read(checked(ROOT, regression_pin))['rows']:
        cohort, name = row['cohort'], row['id']
        heads_pin = pin(ROOT, f'artifacts/arrow-head-validation-v1/{cohort}/{name}-heads.json')
        prior = read(checked(ROOT, heads_pin))
        if cohort == 'expanded':
            controls0 = labels[name]['review']['arrow_controls']
            refs = [{'id': a['id'], 'tip': a['tip_px'], 'direction': a['direction']} for a in controls0 if a['state'] == 'visible_head']
            negatives = [a for a in controls0 if a['state'] == 'negative_region']
        else:
            refs, negatives = legacy[name]['arrows'], legacy[name]['negatives']
        jobs.append({'case': {'id': f'historical-{cohort}-{name}', 'group_id': 'historical-'+cohort+'-'+name,
            'scope': 'historical_directed_regression', 'split': 'development', 'cohort': cohort,
            'labels': {'arrows': refs, 'negative_regions': negatives, 'dense_heads': False}},
            'new_control': False, 'views': [{'mode': 'historical_native', 'gray': prior['source'],
                'input_to_native': prior['processed_to_patch'], 'historical_cache': heads_pin}]})
    names = ['scripts/run_local_width_heads_v1.py', 'scripts/prepare_local_width_head_controls_v1.py',
        'tests/test_local_width_heads_v1.py', 'src/cad_recovery/local_width_heads_v1.py',
        'src/cad_recovery/local_role_guard_v1.py', 'src/cad_recovery/local_chain_evidence_v1.py',
        'src/cad_recovery/arrow_head_completion_v1.py', 'src/cad_recovery/arrow_path_recovery_v1.py',
        'src/cad_recovery/native_head_bounds_v1.py', 'src/cad_recovery/stroke_roles_v1.py',
        'src/cad_recovery/anchored_curve_fit_v1.py', 'src/cad_recovery/stroke_role_metrics_v1.py',
        'src/cad_recovery/stroke_curve_control_v1.py', 'src/cad_recovery/hard_raster_metrics_v1.py',
        'scripts/run_stroke_roles_v1.py', 'scripts/run_stroke_ownership_v1.py',
        'scripts/resume_centerline_comparison_v2.py']
    plan = {'task': 'P10.2c.2b.7b', 'previous': PREVIOUS, 'core': CORE, 'controls': CONTROLS,
        'code': [pin(ROOT, n) for n in names], 'policy': POLICY, 'jobs': jobs, 'reading': old['reading'],
        'historical_readings': [expanded_pin, legacy_pin, regression_pin],
        'new_controls_before_implementation': True, 'prepared_before_corpus_predictions': True,
        'parameters_selected_on_viewed_validation': False, 'repeats': 2, 'gates': controls['gates'],
        'baseline_policy': 'Cached frozen completion heads on old role inputs; anchored LINE/ARC + safe frozen completion on new controls; historical validated observations retained separately.',
        'detector_inputs': 'gray, baseline observations, optional saved text exclusions only; no fitted geometry, GT, OCR, DXF, scale or nominal',
        'binding_policy': 'Separate diagnostic hypotheses only on cached old role geometry; new head-only controls have no LINE requirement or binding claim.',
        'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 1800, 'CPU_threads': 1},
        'new_OCR_ML_calls': 0, 'training': False, 'closed_test_used': False, 'production_changed': False,
        'physical_geometry_approved': False, 'DT_automatic_admission': False}
    write(BASE/'plan.json', plan); print(json.dumps(pin(ROOT, 'artifacts/local-width-heads-v1/plan.json')))


def child(request_path):
    sys.path.insert(0, str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.morphology import skeletonize
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.local_width_heads_v1 import search
    from cad_recovery.arrow_head_completion_v1 import search as baseline_search
    from cad_recovery.native_head_bounds_v1 import guarded_native_heads
    from cad_recovery.anchored_curve_fit_v1 import extract
    from cad_recovery.local_role_guard_v1 import analyze as bind
    from cad_recovery.stroke_roles_v1 import foreground, stroke_floor
    from cad_recovery.stroke_role_metrics_v1 import head_scores
    from run_stroke_roles_v1 import refs_for
    from run_stroke_ownership_v1 import widths_for
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    request = read(request_path); plan = read(checked(ROOT, request['plan']))
    for p in plan['code']: checked(ROOT, p)
    job, case = request['job'], request['job']['case']; rows = []
    folder = request_path.parent/'output'; folder.mkdir()
    for view in job['views']:
        gray = np.array(Image.open(checked(ROOT, view['gray'])).convert('L'))
        source_sha = hashlib.sha256(gray.tobytes()).hexdigest(); cached = None
        if view.get('cache'):
            cached = read(checked(ROOT, view['cache']))
            if cached['input_pixel_sha256'] != source_sha: raise ValueError('Mixed cached raster')
            baseline = cached['head_candidates']['baseline']; baseline_incomplete = cached['head_search_incomplete']
        elif view.get('historical_cache'):
            historical = read(checked(ROOT, view['historical_cache']))
            baseline = historical['candidates']; baseline_incomplete = historical['search_incomplete']
        else:
            ink = foreground(gray); skeleton = skeletonize(ink > 0, method='zhang').astype(np.uint8)*255
            geometry = extract(ink, skeleton, conditioned=False, input_to_native=view['input_to_native'])['candidates']
            stroke = stroke_floor(widths_for(geometry)) or 3.
            with guarded_native_heads():
                base = baseline_search(gray, lines=[g for g in geometry if g['kind'] == 'line'],
                    arcs=[g for g in geometry if g['kind'] == 'arc'], stroke_px=stroke)
            baseline, baseline_incomplete = base['candidates'], base['search_incomplete']
        old_sha = fingerprint(baseline)
        started = time.perf_counter(); result = search(gray, baseline=baseline); elapsed = time.perf_counter()-started
        binding = None
        if cached:
            geometry = read(checked(ROOT, view['geometry']))
            text_mask = np.array(Image.open(checked(ROOT, view['text_mask'])))
            roles = np.array(Image.open(checked(ROOT, view['baseline_roles'])))
            evidence = cached['previous_interval_details']
            evidence_sha = fingerprint([geometry, evidence])
            attached = bind(gray, evidence, roles, text_mask, result['candidates'], diagnostic_heads=cached['head_candidates']['DT_novel'])['hybrid']
            binding = {'previous_chains': cached['details']['hybrid']['dimension_chains'],
                'dimension_chains': attached['dimension_chains'], 'head_ids_available': attached['head_ids_used'],
                'search_incomplete': attached['search_incomplete'], 'source_geometry_unchanged': evidence_sha == fingerprint([geometry, evidence]),
                'semantic_CAD_promotions': 0, 'physical_geometry_approved': False}
        # Labels enter only after head discovery and optional geometric binding.
        labels = case.get('labels', {})
        if 'arrows' in labels:
            refs, negatives = labels['arrows'], labels.get('negative_regions', [])
        else:
            refs, negatives = refs_for(case, read(checked(ROOT, plan['reading'])))
        dense = case['scope'].endswith('analytical_control') or case['scope'] == 'local_width_head_control'
        scores = {name: head_scores(refs, negatives, heads, view['input_to_native'], dense=dense) for name, heads in
            [('baseline', baseline), ('local_only', result['added']), ('combined', result['candidates'])]}
        row = {'case': case['id'], 'scope': case['scope'], 'group_id': case['group_id'], 'split': case['split'],
            'cohort': case.get('cohort'), 'mode': view['mode'], 'repeat': request['repeat'], 'new_control': job['new_control'],
            'input_to_native': view['input_to_native'], 'input_pixel_sha256': source_sha,
            'source_pixel_sha256_after': hashlib.sha256(gray.tobytes()).hexdigest(),
            'baseline_sha256_before': old_sha, 'baseline_sha256_after': fingerprint(baseline),
            'baseline_exact_prefix_preserved': result['candidates'][:len(baseline)] == baseline,
            'baseline_search_incomplete': baseline_incomplete, 'heads': scores,
            'baseline_candidates': baseline, 'local_search': result, 'binding': binding,
            'timing': {'local_search_seconds': elapsed}, 'new_OCR_ML_calls': 0, 'CAD_promotions': 0,
            'production_changed': False, 'physical_geometry_approved': False}
        write(folder/view['mode']/'result.json', row); rows.append(row)
    write(folder/'result.json', {'rows': rows})


def canonical(row):
    return {k: v for k, v in row.items() if k not in ('repeat', 'timing', 'artifact', 'runtime')}


def run(expected):
    from cad_recovery.dataset_layout_v1 import checked, pin
    from cad_recovery.service_profiles_v2.budget import AttemptBudget
    from resume_centerline_comparison_v2 import StableOutputCounter
    plan = read(checked(ROOT, {'path': 'artifacts/local-width-heads-v1/plan.json', 'sha256': expected}))
    index, previous, repeats, spent = [], {}, [], 0.
    for split in ('train', 'validation', 'development'):
        for item in plan['code']: checked(ROOT, item)
        write(BASE/(split+'-parameter-gate.json'), {'plan_sha256': expected, 'parameter_changes': False})
        for job in [j for j in plan['jobs'] if j['case']['split'] == split]:
            for repeat in range(2):
                if spent >= 1800: raise RuntimeError('Total child budget exhausted')
                folder = BASE/'runs'/f'{job["case"]["id"]}-r{repeat}'; folder.mkdir(parents=True, exist_ok=False)
                write(folder/'request.json', {'job': job, 'repeat': repeat, 'plan': pin(ROOT, 'artifacts/local-width-heads-v1/plan.json')})
                env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                    'XDG_CACHE_HOME': str(ROOT/'artifacts/.cache')}
                with (folder/'child.log').open('xb') as log:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'child', '--request', str(folder/'request.json')],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0, start_new_session=os.name != 'nt')
                    budget = AttemptBudget(process, folder, StableOutputCounter(), SimpleNamespace(timeout_seconds=min(60, 1800-spent),
                        max_output_bytes=64*1024**2, max_output_files=128), memory_MiB=1024)
                    try:
                        while process.poll() is None: budget.check(); time.sleep(.05)
                    finally: budget.stop()
                runtime = budget.measurements(); spent += runtime['walltime_seconds']; write(folder/'process.json', runtime)
                if process.returncode != 0 or runtime['budget_error']: raise RuntimeError('Child failed: '+folder.name)
                for row in read(folder/'output/result.json')['rows']:
                    key = row['case'], row['mode']
                    if repeat: repeats.append({'case': row['case'], 'mode': row['mode'], 'identical': canonical(row) == previous[key]})
                    else: previous[key] = canonical(row)
                    index.append({k: row[k] for k in ('case', 'scope', 'group_id', 'split', 'mode', 'repeat', 'new_control')} |
                        {'artifact': (folder/'output'/row['mode']/'result.json').relative_to(ROOT).as_posix(), 'runtime': runtime})
            print(json.dumps({'completed': job['case']['id'], 'split': split, 'child_seconds': round(spent, 2)}), flush=True)
    write(BASE/'results-index.json', index); write(BASE/'repeat-checks.json', repeats)
    new_views = sum(len(j['views']) for j in plan['jobs'] if j['new_control'])*2
    write(BASE/'execution.json', {'cases': len(plan['jobs']), 'children': 2*len(plan['jobs']), 'views': len(index),
        'canonical_views': len(repeats), 'all_repeat_identical': all(r['identical'] for r in repeats),
        'child_wall_seconds': spent, 'local_search_calls': len(index), 'fresh_completion_search_calls': new_views,
        'new_geometry_extractions': new_views, 'new_skeleton_views': new_views,
        'cached_baseline_views': len(index)-new_views, 'binding_guard_calls': sum(r['scope'] != 'historical_directed_regression'
            and not r['new_control'] for r in index),
        'new_OCR_ML_calls': 0, 'training': False, 'closed_test_used': False, 'production_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=['freeze', 'run', 'child'])
    parser.add_argument('--sha256'); parser.add_argument('--request', type=Path); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'child': child(args.request)
    else:
        if not args.sha256: parser.error('--sha256 required')
        run(args.sha256)
