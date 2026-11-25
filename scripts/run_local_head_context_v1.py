"""Frozen local repeat fields and source-derived shaft-axis recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from run_local_width_heads_v1 import read, write, fingerprint

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/local-head-context-v1'
PREVIOUS = {'path': 'artifacts/economical-local-heads-release-v1.json', 'sha256': '84584ff84dfa50a6cb83986a1fad9d8fa73aa2315eb436e370dd1d843a91d8ea'}
CONTROLS = {'path': 'artifacts/local-head-context-controls-v1/contract.json', 'sha256': '80437015dff3429bc7bfff01a59efda33a11411e89b05087e226b2d717a8d66c'}
CORE = {'path': 'artifacts/cv-comparison-protocol-v1/contract.json', 'sha256': '815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}
LARGE = ('historical-legacy-val149-upper', 'historical-legacy-val162-left')


def freeze():
    from cad_recovery.dataset_layout_v1 import checked, pin
    from cad_recovery.local_head_context_v1 import POLICY
    from verify_cv_comparison_contract_v1 import verify_contract
    verify_contract(ROOT, CORE); checked(ROOT, PREVIOUS)
    controls = read(checked(ROOT, CONTROLS)); checked(ROOT, controls['generator'])
    old = read(ROOT/'artifacts/economical-local-heads-v1/plan.json')
    index = {(r['case'], r['mode']): r for r in read(ROOT/'artifacts/economical-local-heads-v1/results-index.json') if not r['repeat']}
    jobs = []
    for job in old['jobs']:
        views = []
        for view in job['views']:
            entry = index[job['case']['id'], view['mode']]
            views.append({'mode': view['mode'], 'gray': view['gray'], 'input_to_native': view['input_to_native'],
                'comparison_cache': pin(ROOT, entry['artifact']), 'binding_cache': view.get('binding_cache')})
        jobs.append({'case': job['case'], 'views': views, 'new_control': False})
    jobs.extend(controls['jobs'])
    names = ['src/cad_recovery/local_head_axis_profiles_v1.py', 'src/cad_recovery/local_head_parallel_field_v1.py',
        'src/cad_recovery/local_head_context_v1.py', 'scripts/prepare_local_head_context_controls_v1.py',
        'scripts/run_local_head_context_v1.py', 'tests/test_local_head_context_v1.py',
        'src/cad_recovery/economical_head_profiles_v1.py', 'src/cad_recovery/head_parallel_context_v1.py',
        'src/cad_recovery/economical_local_heads_v1.py', 'scripts/prepare_economical_head_controls_v1.py',
        'scripts/run_economical_local_heads_v1.py', 'scripts/experiment_children_v1.py', 'tests/test_economical_local_heads_v1.py',
        'src/cad_recovery/sparse_head_regions_v1.py', 'src/cad_recovery/sparse_local_heads_v1.py',
        'src/cad_recovery/local_width_heads_v1.py', 'src/cad_recovery/arrow_head_completion_v1.py',
        'src/cad_recovery/arrow_path_recovery_v1.py', 'src/cad_recovery/native_head_bounds_v1.py',
        'src/cad_recovery/baseline_anchor_windows_v1.py', 'src/cad_recovery/local_role_guard_v1.py',
        'src/cad_recovery/local_chain_evidence_v1.py', 'src/cad_recovery/stroke_role_metrics_v1.py',
        'src/cad_recovery/stroke_curve_control_v1.py', 'scripts/run_local_width_heads_v1.py',
        'scripts/run_stroke_roles_v1.py', 'scripts/run_stroke_ownership_v1.py', 'scripts/resume_centerline_comparison_v2.py']
    value = {'task': 'P10.2c.2b.7c.2a', 'previous': PREVIOUS, 'core': CORE, 'controls': CONTROLS,
        'code': [pin(ROOT, n) for n in names], 'policy': POLICY, 'jobs': jobs, 'reading': old['reading'],
        'repeats': 2, 'gates': controls['gates'], 'new_controls_before_implementation': True,
        'prepared_before_corpus_predictions': True, 'parameters_selected_on_viewed_validation': False,
        'profile_policy': 'Same scalar 8M native-query budget, including shaft-axis retries. A short source run may propose an axis; final 6px/2width, 95percent shaft continuity is unchanged. Eager shares the new rules.',
        'proposal_policy': 'Raw source regions remain accounted with separate bounds. Head proposal counters count unique source-validated hypotheses, not rejected blobs; no raw tail silently discarded.',
        'contrast_policy': 'Both head and shaft contrast widths include the source pixel footprint; geometry and unchanged native fill/flank checks remain pixel-center based.',
        'baseline_policy': 'Immutable local-width v1 prefix; both earlier experimental outputs retained separately. Original completion remains global fallback until every gate passes.',
        'context_policy': 'Local source strokes must repeat with spacing uncertainty and co-support within the head occlusion footprint. A retained own tail must exit transversely; end-of-Hough-segment continuation is not proof. Observations and abstentions retained.',
        'eager_reference_policy': 'All new scenes and two old large pages; independent per-source algorithm ledgers, shared child wall/RSS/output caps.',
        'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 1800, 'CPU_threads': 1},
        'new_OCR_ML_calls': 0, 'training': False, 'closed_test_used': False, 'production_changed': False, 'physical_geometry_approved': False}
    write(BASE/'plan.json', value); print(json.dumps(pin(ROOT, 'artifacts/local-head-context-v1/plan.json')))


def child(request_path):
    sys.path.insert(0, str(ROOT/'artifacts/cv-skeleton-runtime-v1/deps'))
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.morphology import skeletonize
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.local_head_context_v1 import search
    from cad_recovery.economical_local_heads_v1 import search as previous_economical_search
    from cad_recovery.local_width_heads_v1 import search as v1_search
    from cad_recovery.sparse_local_heads_v1 import search as sparse_search
    from cad_recovery.baseline_anchor_windows_v1 import extract
    from cad_recovery.arrow_head_completion_v1 import search as completion
    from cad_recovery.native_head_bounds_v1 import guarded_native_heads
    from cad_recovery.local_role_guard_v1 import analyze as bind
    from cad_recovery.stroke_roles_v1 import foreground, stroke_floor
    from cad_recovery.stroke_role_metrics_v1 import head_scores
    from run_stroke_roles_v1 import refs_for
    from run_stroke_ownership_v1 import widths_for
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    request = read(request_path); plan = read(checked(ROOT, request['plan']))
    for p in plan['code']: checked(ROOT, p)
    job, case = request['job'], request['job']['case']; folder = request_path.parent/'output'; folder.mkdir()
    rows = []
    for view in job['views']:
        gray = np.array(Image.open(checked(ROOT, view['gray'])).convert('L')); source_sha = hashlib.sha256(gray.tobytes()).hexdigest()
        prep = None
        if view.get('comparison_cache'):
            old = read(checked(ROOT, view['comparison_cache']))
            if old['input_pixel_sha256'] != source_sha: raise ValueError('Mixed source raster')
            original, v1, sparse, previous = old['original_candidates'], old['v1_search'], old['previous_sparse'], old['economical_search']
        else:
            ink = foreground(gray); skeleton = skeletonize(ink > 0, method='zhang').astype(np.uint8)*255
            packet = extract(ink, skeleton, conditioned=False, input_to_native=view['input_to_native']); geometry = packet['candidates']
            prep = {'fit_calls': packet['analysis']['fit_calls'], 'windows': len(packet['analysis'].get('windows', [])),
                'search_incomplete': packet['analysis']['search_incomplete']}
            widths = widths_for(geometry); stroke = stroke_floor(widths) or 3.
            with guarded_native_heads():
                base = completion(gray, lines=[g for g in geometry if g['kind'] == 'line'],
                    arcs=[g for g in geometry if g['kind'] == 'arc'], stroke_px=stroke)
            original = base['candidates']; v1 = v1_search(gray, baseline=original)
            sparse = sparse_search(gray, baseline=v1['candidates'])
            previous = previous_economical_search(gray, baseline=v1['candidates'])
        baseline = v1['candidates']; before = fingerprint(baseline)
        started = time.perf_counter(); result = search(gray, baseline=baseline); elapsed = time.perf_counter()-started
        eager = search(gray, baseline=baseline, eager_reference=True) if job['new_control'] or case['id'] in LARGE else None
        binding = None
        if view.get('binding_cache'):
            saved = view['binding_cache']; parent = read(checked(ROOT, saved['cache']))
            geometry = read(checked(ROOT, saved['geometry'])); evidence = parent['previous_interval_details']
            text = np.array(Image.open(checked(ROOT, saved['text_mask']))); roles = np.array(Image.open(checked(ROOT, saved['baseline_roles'])))
            immutable = fingerprint([geometry, evidence])
            attached = bind(gray, evidence, roles, text, result['candidates'], diagnostic_heads=parent['head_candidates']['DT_novel'])['hybrid']
            binding = {'dimension_chains': attached['dimension_chains'], 'head_ids_available': attached['head_ids_used'],
                'search_incomplete': attached['search_incomplete'], 'source_geometry_unchanged': immutable == fingerprint([geometry, evidence]),
                'semantic_CAD_promotions': 0, 'physical_geometry_approved': False}
        # Ground truth never enters the algorithms above.
        labels = case.get('labels', {})
        if 'arrows' in labels: refs, negatives = labels['arrows'], labels.get('negative_regions', [])
        else: refs, negatives = refs_for(case, read(checked(ROOT, plan['reading'])))
        dense_labels = case['scope'].endswith('analytical_control') or case['scope'] in ('local_width_head_control',
            'sparse_head_control', 'sparse_head_large_control', 'economical_head_control', 'economical_head_large_control', 'local_head_context_control')
        variants = {'original': original, 'v1': baseline, 'previous_sparse': sparse['candidates'], 'previous_economical': previous['candidates'], 'context_local': result['candidates']}
        scores = {name: head_scores(refs, negatives, heads, view['input_to_native'], dense=dense_labels) for name, heads in variants.items()}
        if eager is not None: scores['eager_reference'] = head_scores(refs, negatives, eager['candidates'], view['input_to_native'], dense=dense_labels)
        row = {'case': case['id'], 'scope': case['scope'], 'group_id': case['group_id'], 'split': case['split'], 'cohort': case.get('cohort'),
            'mode': view['mode'], 'repeat': request['repeat'], 'new_control': job['new_control'], 'input_to_native': view['input_to_native'],
            'input_pixel_sha256': source_sha, 'source_pixel_sha256_after': hashlib.sha256(gray.tobytes()).hexdigest(),
            'baseline_sha256_before': before, 'baseline_sha256_after': fingerprint(baseline),
            'baseline_exact_prefix_preserved': result['candidates'][:len(baseline)] == baseline,
            'heads': scores, 'original_candidates': original, 'v1_search': v1, 'previous_sparse': sparse,
            'previous_economical': previous, 'context_search': result, 'eager_reference': eager, 'baseline_preparation': prep, 'binding': binding,
            'timing': {'context_search_seconds': elapsed}, 'new_OCR_ML_calls': 0, 'CAD_promotions': 0,
            'production_changed': False, 'physical_geometry_approved': False}
        write(folder/view['mode']/'result.json', row); rows.append(row)
    write(folder/'result.json', {'rows': rows})


def run(expected):
    from experiment_children_v1 import run as execute
    plan_pin = {'path': 'artifacts/local-head-context-v1/plan.json', 'sha256': expected}
    execution = execute(ROOT, BASE, plan_pin, Path(__file__).resolve())
    index = read(BASE/'results-index.json'); new_views = sum(r['new_control'] for r in index)
    prep = [read(ROOT/r['artifact'])['baseline_preparation'] for r in index if r['new_control']]
    extractions = sum(p['windows'] or 1 for p in prep)
    execution.update(context_search_calls=len(index), eager_reference_calls=new_views+4,
        fresh_completion_v1_sparse_economical_skeleton_calls=new_views, baseline_geometry_dispatches=new_views,
        actual_baseline_geometry_extractions=extractions, cached_previous_comparison_views=len(index)-new_views,
        binding_guard_calls=sum(bool(read(ROOT/r['artifact'])['binding']) for r in index), new_OCR_ML_calls=0, training=False, closed_test_used=False, production_changed=False)
    write(BASE/'execution.json', execution)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('mode', choices=['freeze', 'run', 'child'])
    parser.add_argument('--sha256'); parser.add_argument('--request', type=Path); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'child': child(args.request)
    else:
        if not args.sha256: parser.error('--sha256 required')
        run(args.sha256)
