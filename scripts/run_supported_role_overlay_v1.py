"""Compare a source-owned dimension overlay with frozen baseline and broad hybrid."""
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
BASE = ROOT/'artifacts/supported-role-overlay-v1'
PREVIOUS = {'path': 'artifacts/local-width-heads-release-v1.json', 'sha256': '9e90574854b568b5ee00c0c444619a02a00582876e0ee94e5de69651a20c0e6d'}
CONTROLS = {'path': 'artifacts/supported-role-controls-v1/contract.json', 'sha256': 'ffffe97edeabce2dae6176d3643f2c73b10ce354ef45a205b717436227183e50'}
DIAGNOSIS = {'path': 'artifacts/role-overlay-diagnosis-v1/diagnosis.json', 'sha256': '919bde78049958f340e8dfda7e379d96ff275b14a64e7d167ab5a3fac741a43b'}
CORE = {'path': 'artifacts/cv-comparison-protocol-v1/contract.json', 'sha256': '815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}
VARIANTS = ('old_context', 'old_hybrid', 'supported_overlay')


def read(path): return json.loads(path.read_text('utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def freeze():
    from cad_recovery.dataset_layout_v1 import checked, pin
    from cad_recovery.supported_role_overlay_v1 import POLICY
    from verify_cv_comparison_contract_v1 import verify_contract
    verify_contract(ROOT, CORE); checked(ROOT, PREVIOUS); checked(ROOT, DIAGNOSIS)
    controls = read(checked(ROOT, CONTROLS)); checked(ROOT, controls['generator'])
    prior = read(ROOT/'artifacts/local-role-guard-v1/plan.json')
    index = {(r['case'], r['mode']): r for r in read(ROOT/'artifacts/local-role-guard-v1/results-index.json') if not r['repeat']}
    jobs = []
    for job in prior['jobs']:
        views = []
        for view in job['views']:
            entry = index[job['case']['id'], view['mode']]; folder = (ROOT/entry['artifact']).parent
            cache = {name: pin(ROOT, (folder/file).relative_to(ROOT).as_posix()) for name, file in
                [('row', 'result.json'), ('source', 'source.json'), ('geometry', 'geometry.json'),
                 ('baseline_roles', 'old_context-roles.png'), ('hybrid_roles', 'hybrid-roles.png'),
                 ('local_roles', 'local_verified-roles.png'), ('text_mask', 'text-mask.png')]}
            views.append({'mode': view['mode'], 'gray': view['gray'], 'input_to_native': view['input_to_native'],
                'reviewed_mask': view.get('reviewed_mask'), 'cache': cache})
        jobs.append({**job, 'new_control': False, 'views': views})
    jobs.extend({**j, 'views': [{**v, 'cache': None, 'reviewed_mask': None} for v in j['views']]} for j in controls['jobs'])
    names = ['scripts/run_supported_role_overlay_v1.py', 'scripts/prepare_supported_role_controls_v1.py',
        'src/cad_recovery/supported_role_overlay_v1.py', 'tests/test_supported_role_overlay_v1.py',
        'src/cad_recovery/local_role_guard_v1.py', 'src/cad_recovery/local_chain_evidence_v1.py',
        'src/cad_recovery/interval_roles_v1.py', 'src/cad_recovery/native_head_bounds_v1.py',
        'src/cad_recovery/stroke_roles_v1.py', 'src/cad_recovery/stroke_roles_v2.py',
        'src/cad_recovery/stroke_ownership_metrics_v1.py', 'src/cad_recovery/stroke_role_metrics_v1.py',
        'src/cad_recovery/joint_shape_v1.py', 'src/cad_recovery/anchored_curve_fit_v1.py',
        'scripts/run_stroke_roles_v1.py', 'scripts/run_stroke_ownership_v1.py', 'scripts/resume_centerline_comparison_v2.py']
    plan = {'task': 'P10.2c.3b.2a.1', 'previous': PREVIOUS, 'core': CORE, 'diagnosis': DIAGNOSIS, 'controls': CONTROLS,
        'code': [pin(ROOT, n) for n in names], 'policy': POLICY, 'jobs': jobs, 'reading': prior['reading'],
        'variants': VARIANTS, 'repeats': 2, 'gates': controls['gates'], 'new_controls_before_implementation': True,
        'prepared_before_corpus_predictions': True, 'parameters_selected_on_viewed_validation': False,
        'baseline_policy': 'Identical cached baseline/head/geometry on old cases; same fresh extraction and frozen head/role producers on new controls.',
        'scope': 'Unknown-pixel dimension/extension overlay only. Bundle/boundary/leader remain review hypotheses. All known baseline classes immutable.',
        'head_policy': 'Same frozen completion observations and local guard; new local-width detector and DT additions not admitted.',
        'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 1800, 'CPU_threads': 1},
        'new_OCR_ML_calls': 0, 'training': False, 'closed_test_used': False, 'production_changed': False,
        'physical_geometry_approved': False, 'source_erasure_allowed': False}
    write(BASE/'plan.json', plan); print(json.dumps(pin(ROOT, 'artifacts/supported-role-overlay-v1/plan.json')))


def child(request_path):
    sys.path.insert(0, str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.morphology import skeletonize
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.supported_role_overlay_v1 import analyze
    from cad_recovery.local_role_guard_v1 import analyze as local_analyze
    from cad_recovery.interval_roles_v1 import analyze as interval_analyze
    from cad_recovery.joint_shape_v1 import extract
    from cad_recovery.anchored_curve_fit_v1 import extract as anchored_extract
    from cad_recovery.stroke_roles_v2 import analyze as safe_heads
    from cad_recovery.stroke_roles_v1 import foreground, stroke_floor, text_components, guard_heads, role_proposals, append_heads
    from cad_recovery.stroke_ownership_metrics_v1 import score_roles, ink_ledger
    from cad_recovery.stroke_role_metrics_v1 import to_native, synthetic_truth, head_scores, mask_effect
    from run_stroke_roles_v1 import refs_for
    from run_stroke_ownership_v1 import widths_for
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    request = read(request_path); plan = read(checked(ROOT, request['plan']))
    for p in plan['code']: checked(ROOT, p)
    job, case = request['job'], request['job']['case']; folder = request_path.parent/'output'; folder.mkdir()
    rows, truth, negative, native_gray = [], None, None, None
    for view in job['views']:
        gray = np.array(Image.open(checked(ROOT, view['gray'])).convert('L'))
        if native_gray is None: native_gray = gray.copy()
        source_sha = hashlib.sha256(gray.tobytes()).hexdigest()
        if view['cache']:
            cache = view['cache']; row = read(checked(ROOT, cache['row']))
            if row['input_pixel_sha256'] != source_sha: raise ValueError('Mixed cached raster')
            source, geometry = read(checked(ROOT, cache['source'])), read(checked(ROOT, cache['geometry']))
            baseline = np.array(Image.open(checked(ROOT, cache['baseline_roles'])))
            text = np.array(Image.open(checked(ROOT, cache['text_mask'])))
            previous = row['previous_interval_details']; heads, extra = row['head_candidates']['baseline'], row['head_candidates']['DT_novel']
            local = {name: {**row['details'][key], 'pixel_roles': np.array(Image.open(checked(ROOT, cache[map_key]))), 'text_mask': text}
                for name, key, map_key in [('local', 'local_verified', 'local_roles'), ('hybrid', 'hybrid', 'hybrid_roles')]}
        else:
            ink = foreground(gray); skeleton = skeletonize(ink > 0, method='zhang').astype(np.uint8)*255
            f = extract(ink, skeleton, input_to_native=view['input_to_native'], move_nodes=False)
            source, geometry = f['source'], f['candidates']
            anchored = anchored_extract(ink, skeleton, conditioned=False, input_to_native=view['input_to_native'])['candidates']
            widths = widths_for(anchored); h = safe_heads(gray, anchored, widths, stroke_px=stroke_floor(widths) or 3.)
            heads = h['base_heads']['candidates']; extra = [c for c in append_heads(heads, h['DT_proposals']['candidates']) if c['id'].startswith('DT-added-')]
            widths = widths_for(geometry); t = text_components(gray, stroke_floor(widths) or 3.)
            kept, _ = guard_heads(heads, t['component_mask'])
            baseline = role_proposals(gray, geometry, widths, kept, t['component_mask'], context=True)['pixel_roles']
            raw = interval_analyze(gray, geometry, source, heads, input_to_native=view['input_to_native'], diagnostic_heads=extra)
            previous = {k: v for k, v in raw.items() if k not in ('pixel_roles', 'text_mask')}; text = raw['text_mask']
            local = local_analyze(gray, previous, baseline, text, heads, diagnostic_heads=extra)
        before = fingerprint([source, geometry, previous, heads, extra])
        started = time.perf_counter(); result = analyze(gray, previous, baseline, local, heads=heads, diagnostic_heads=extra)
        elapsed = time.perf_counter()-started
        variants = {'old_context': baseline, 'old_hybrid': local['hybrid']['pixel_roles'], 'supported_overlay': result['pixel_roles']}
        # No GT/negative masks have entered either classifier.
        if truth is None:
            if job.get('truth'): truth = np.array(Image.open(checked(ROOT, job['truth'])))
            elif case['scope'] == 'analytical_control':
                truth = synthetic_truth(case, native_gray, np.array(Image.open(checked(ROOT, view['reviewed_mask']))))
        if negative is None and job.get('negative'): negative = np.array(Image.open(checked(ROOT, job['negative'])))
        if truth is None and negative is not None: truth = np.full(tuple(case['size_px'][::-1]), 255, np.uint8)
        if 'arrows' in case.get('labels', {}): refs, negatives = case['labels']['arrows'], case['labels'].get('negative_regions', [])
        else: refs, negatives = refs_for(case, read(checked(ROOT, plan['reading'])))
        sub = folder/view['mode']; sub.mkdir(); scores, ledgers = {}, {}
        for name, labels in variants.items():
            Image.fromarray(labels).save(sub/(name+'-roles.png'))
            scores[name] = score_roles(truth, to_native(labels, case['size_px']), negative) if truth is not None else None
            ledgers[name] = ink_ledger(gray, labels, foreground)
        Image.fromarray(result['pixel_interval_owner']).save(sub/'pixel-interval-owner.png')
        Image.fromarray(text).save(sub/'text-mask.png')
        available = set(result['head_ids_used'])
        metrics = {name: head_scores(refs, negatives, hh, view['input_to_native'], dense=case['scope'].endswith('analytical_control'))
            for name, hh in [('baseline', heads), ('local_guard', [h for h in heads if h['id'] in available])]}
        delta = variants['supported_overlay'] != baseline
        added_truth = None
        if truth is not None:
            native_delta = to_native(delta.astype(np.uint8), case['size_px']) > 0
            native_roles = to_native(result['pixel_roles'], case['size_px'])
            known = native_delta & (truth > 0) & (truth < 10)
            added_truth = {'known_added_pixels': int(known.sum()), 'correct_added_pixels': int((known & (native_roles == truth)).sum()),
                'incorrect_added_pixels': int((known & (native_roles != truth)).sum()),
                'unknown_truth_added_pixels': int((native_delta & (truth == 255)).sum())}
        row = {'case': case['id'], 'scope': case['scope'], 'group_id': case['group_id'], 'split': case['split'],
            'mode': view['mode'], 'repeat': request['repeat'], 'new_control': job['new_control'],
            'input_to_native': view['input_to_native'], 'input_pixel_sha256': source_sha,
            'source_pixel_sha256_after': hashlib.sha256(gray.tobytes()).hexdigest(),
            'source_observations_sha256_before': before, 'source_observations_sha256_after': fingerprint([source, geometry, previous, heads, extra]),
            'roles': scores, 'ink_ledgers': ledgers, 'heads': metrics, 'added_truth': added_truth,
            'details': {k: v for k, v in result.items() if k not in ('pixel_roles', 'pixel_interval_owner', 'text_mask')},
            'previous_local_details': {k: v for k, v in local['hybrid'].items() if k not in ('pixel_roles', 'text_mask')},
            'head_candidates': {'baseline': heads, 'DT_novel': extra},
            'text_mask_effect': mask_effect(truth, native_gray, to_native(text, case['size_px'])) if truth is not None else None,
            'timing': {'overlay_seconds': elapsed}, 'source_pixels_erased': 0, 'semantic_CAD_promotions': 0,
            'production_admitted': False, 'physical_geometry_approved': False}
        write(sub/'source.json', source); write(sub/'geometry.json', geometry); write(sub/'result.json', row); rows.append(row)
    write(folder/'result.json', {'rows': rows})


def canonical(row): return {k: v for k, v in row.items() if k not in ('repeat', 'timing', 'artifact', 'runtime')}


def run(expected):
    from cad_recovery.dataset_layout_v1 import checked, pin
    from cad_recovery.service_profiles_v2.budget import AttemptBudget
    from resume_centerline_comparison_v2 import StableOutputCounter
    plan = read(checked(ROOT, {'path': 'artifacts/supported-role-overlay-v1/plan.json', 'sha256': expected}))
    index, previous, repeats, spent = [], {}, [], 0.
    for split in ('train', 'validation', 'development'):
        for p in plan['code']: checked(ROOT, p)
        write(BASE/(split+'-parameter-gate.json'), {'plan_sha256': expected, 'parameter_changes': False})
        for job in [j for j in plan['jobs'] if j['case']['split'] == split]:
            for repeat in range(2):
                if spent >= 1800: raise RuntimeError('Total child budget exhausted')
                folder = BASE/'runs'/f'{job["case"]["id"]}-r{repeat}'; folder.mkdir(parents=True, exist_ok=False)
                write(folder/'request.json', {'job': job, 'repeat': repeat, 'plan': pin(ROOT, 'artifacts/supported-role-overlay-v1/plan.json')})
                env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'XDG_CACHE_HOME': str(ROOT/'artifacts/.cache')}
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
        'canonical_views': len(repeats), 'all_repeat_identical': all(r['identical'] for r in repeats), 'child_wall_seconds': spent,
        'overlay_calls': len(index), 'cached_head_and_local_guard_views': len(index)-new_views,
        'fresh_head_DT_skeleton_views': new_views, 'new_geometry_extractions': 2*new_views,
        'fresh_interval_calls': new_views, 'fresh_local_guard_calls': new_views,
        'fresh_legacy_role_calls': 2*new_views, 'new_OCR_ML_calls': 0, 'training': False, 'closed_test_used': False, 'production_changed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=['freeze', 'run', 'child'])
    parser.add_argument('--sha256'); parser.add_argument('--request', type=Path); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'child': child(args.request)
    else:
        if not args.sha256: parser.error('--sha256 required')
        run(args.sha256)
