"""Frozen comparison of source-backed part proposals against contact overlay v2."""
import argparse
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.boundary_role_overlay_v1 import analyze, POLICY
from boundary_role_inputs_v1 import prepare
from run_topology_robust_v1 import read, write, object_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/boundary-role-overlay-v1'
PREVIOUS = {'path': 'artifacts/pixel-export-qa-release-v1.json',
            'sha256': '1dd258c8a6e2d037c8ea9b27d6c153a9f4ddc90c8ef7f6790e64ff67d8a41cd7'}
CONTROLS = {'path': 'artifacts/boundary-role-controls-v1/contract.json',
            'sha256': '774f217a19142f7fe14a73a0f6ce40feed01a158a075bb00ad862a2e6859c6c0'}


def freeze():
    checked(ROOT, PREVIOUS); controls = read(checked(ROOT, CONTROLS)); checked(ROOT, controls['generator'])
    old = read(ROOT/'artifacts/contact-role-overlay-v2/plan.json')
    index = {(r['case'], r['mode']): r for r in read(ROOT/'artifacts/contact-role-overlay-v2/results-index.json') if not r['repeat']}
    jobs = []
    for job in old['jobs']:
        views = []
        for view in job['views']:
            folder = (ROOT/index[job['case']['id'], view['mode']]['artifact']).parent
            cached = {n: pin(ROOT, (folder/file).relative_to(ROOT).as_posix()) for n, file in
                      [('row', 'result.json'), ('source', 'source.json'), ('geometry', 'geometry.json'),
                       ('baseline', 'baseline-roles.png'), ('contact', 'contact_overlay-roles.png')]}
            views.append({**view, 'cached_contact': cached})
        jobs.append({**job, 'case': {**job['case'], 'original_split': job['case']['split'], 'split': 'development'},
                     'views': views, 'new_control': False})
    jobs += controls['jobs']
    names = ['src/cad_recovery/boundary_role_overlay_v1.py', 'scripts/boundary_role_inputs_v1.py',
             'scripts/run_boundary_roles_v1.py', 'scripts/prepare_boundary_role_controls_v1.py',
             'tests/test_boundary_role_overlay_v1.py']+[p['path'] for p in old['code']]
    write(BASE/'plan.json', {'task': 'P10.2c.3b.2b', 'previous': PREVIOUS, 'controls': CONTROLS,
                           'qualification': pin(ROOT, 'artifacts/boundary-role-qualification-v2/report.json'),
                           'old_plan': pin(ROOT, 'artifacts/contact-role-overlay-v2/plan.json'),
                           'code': [pin(ROOT, n) for n in dict.fromkeys(names)], 'policy': POLICY,
                           'jobs': jobs, 'repeats': 2, 'budgets': controls['budgets'],
                           'new_validation_opened_before_freeze': False, 'old_validation_reclassified_as_development': True,
                           'metrics_gates': controls['gates'], 'full_classifier_macro_F1_min': .90,
                           'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, (BASE/'plan.json').relative_to(ROOT).as_posix()), flush=True)


def child(request_path):
    from cad_recovery.stroke_roles_v1 import foreground
    from cad_recovery.stroke_role_metrics_v1 import synthetic_truth, to_native
    from cad_recovery.stroke_ownership_metrics_v1 import score_roles, ink_ledger
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    req = read(request_path); plan = read(checked(ROOT, req['plan'])); job = req['job']; case = job['case']
    for p in plan['code']: checked(ROOT, p)
    folder = request_path.parent/'output'; folder.mkdir(); rows = []
    truth = negative = native_gray = None
    for view in job['views']:
        gray, source, geometry, inherited, previous, local, heads, extra, preparation = prepare(ROOT, view)
        if native_gray is None: native_gray = gray.copy()
        sha = hashlib.sha256(gray.tobytes()).hexdigest(); inherited_sha = hashlib.sha256(inherited.tobytes()).hexdigest()
        evidence_before = object_hash([source, geometry, previous, heads, extra, local['hybrid']['dimension_chains']])
        started = time.perf_counter(); result = analyze(gray, previous, inherited, local, heads=heads, diagnostic_heads=extra)
        elapsed = time.perf_counter()-started
        # No annotation/GT reaches prepare() or analyze(); load targets after inference.
        if truth is None:
            if job.get('truth'): truth = np.array(Image.open(checked(ROOT, job['truth'])))
            elif case['scope'] == 'analytical_control':
                truth = synthetic_truth(case, native_gray, np.array(Image.open(checked(ROOT, view['reviewed_mask']))))
        if negative is None and job.get('negative'): negative = np.array(Image.open(checked(ROOT, job['negative'])))
        if truth is None and negative is not None: truth = np.full(tuple(case['size_px'][::-1]), 255, np.uint8)
        sub = folder/view['mode']; sub.mkdir()
        metrics, ledgers = {}, {}
        for name, labels in [('contact', inherited), ('boundary', result['pixel_roles'])]:
            Image.fromarray(labels).save(sub/(name+'-roles.png'))
            metrics[name] = score_roles(truth, to_native(labels, case['size_px']), negative) if truth is not None else None
            ledgers[name] = ink_ledger(gray, labels, foreground)
        baseline_matches = True
        if view.get('cached_contact'):
            old_row = read(checked(ROOT, view['cached_contact']['row']))
            baseline_matches = metrics['contact'] == old_row['roles']['contact_overlay']
        new = to_native(result['pixel_roles'], case['size_px']); old = to_native(inherited, case['size_px'])
        delta = new != old; known = delta & (truth > 0) & (truth < 10) if truth is not None else np.zeros_like(delta)
        additions = {'correct': int((known & (new == truth)).sum()) if truth is not None else 0,
                     'wrong': int((known & (new != truth)).sum()) if truth is not None else 0,
                     'unknown_truth': int((delta & (truth == 255)).sum()) if truth is not None else int(delta.sum()),
                     'negative': int((delta & (negative > 0) & (truth == 255)).sum()) if negative is not None else 0}
        Image.fromarray(result['pixel_interval_owner']).save(sub/'pixel-interval-owner.png')
        # Fixed image frame, no alignment to candidate bounds. Blue marks only the additions.
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        overlay[result['pixel_roles'] != inherited] = [20, 100, 230]
        Image.fromarray(overlay).save(sub/'additions-overlay.png')
        row = {k: case[k] for k in ('id', 'scope', 'group_id', 'split')}; row['case'] = row.pop('id')
        row.update(mode=view['mode'], repeat=req['repeat'], new_control=job['new_control'],
                   original_split=case.get('original_split'), roles=metrics, ink_ledgers=ledgers, additions=additions,
                   input_pixel_sha256=sha, source_pixel_sha256_after=hashlib.sha256(gray.tobytes()).hexdigest(),
                   inherited_map_unchanged=inherited_sha == hashlib.sha256(inherited.tobytes()).hexdigest(),
                   source_observations_unchanged=evidence_before == object_hash([source, geometry, previous, heads, extra, local['hybrid']['dimension_chains']]),
                   reconstructed_contact_metrics_match=baseline_matches,
                   details={k: v for k, v in result.items() if k not in ('pixel_roles', 'pixel_interval_owner')},
                   observed_boundary_count=len(previous['boundaries']['boundaries']),
                   observed_dimension_chains=len(local['hybrid']['dimension_chains']),
                   preparation=preparation, timing={'boundary_overlay_seconds': elapsed},
                   production_admitted=False, physical_geometry_approved=False, semantic_CAD_promotions=0)
        write(sub/'result.json', row); rows.append(row)
    write(folder/'result.json', {'rows': rows})


def run(sha):
    from experiment_children_v1 import run as execute
    execution = execute(ROOT, BASE, {'path': 'artifacts/boundary-role-overlay-v1/plan.json', 'sha256': sha}, Path(__file__).resolve())
    rows = [read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    execution.update(preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
                     boundary_overlay_calls=len(rows), new_CV_geometry_on_old_inputs=False,
                     new_OCR_ML_calls=0, new_DXF_exports=0, training=False, closed_test_used=False, production_changed=False)
    write(BASE/'execution.json', execution)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'run', 'child'])
    parser.add_argument('--sha256'); parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'child': child(args.request)
    elif args.sha256: run(args.sha256)
    else: parser.error('--sha256 required')
