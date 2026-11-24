"""Paired frozen role-contact comparison with independent source controls."""
import argparse
import hashlib
from pathlib import Path
import time

from run_supported_role_overlay_v1 import read, write, fingerprint

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/contact-role-overlay-v2'
PREVIOUS = {'path': 'artifacts/head-eligibility-release-v2.json',
            'sha256': '4bbe950ab117aa005aa53dde416effee6686aa39eeb99d73fae91ac1380fc006'}
CORE = {'path': 'artifacts/cv-comparison-protocol-v1/contract.json',
        'sha256': '815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}
CONTROLS = {'path': 'artifacts/contact-role-controls-v1/contract.json',
            'sha256': '428f8b2299a3abb18e5f0163bc9815c8372233c231f313d0d488f6cc641ea9f5'}


def freeze():
    from cad_recovery.dataset_layout_v1 import checked, pin
    from cad_recovery.contact_role_overlay_v2 import POLICY
    from contact_role_inputs_v1 import cached_jobs
    from verify_cv_comparison_contract_v1 import verify_contract
    verify_contract(ROOT, CORE)
    checked(ROOT, PREVIOUS)
    protocol = pin(ROOT, 'artifacts/contact-role-protocol-v1.json')
    old = read(checked(ROOT, read(checked(ROOT, protocol))['role_parent_plan']))
    controls = read(checked(ROOT, CONTROLS))
    checked(ROOT, controls['generator'])
    jobs = cached_jobs(ROOT)+controls['jobs']
    names = ['src/cad_recovery/contact_role_overlay_v2.py', 'scripts/contact_role_inputs_v1.py',
             'scripts/prepare_contact_role_controls_v1.py', 'scripts/run_contact_roles_v2.py',
             'tests/test_contact_role_overlay_v2.py']+[p['path'] for p in old['code']]
    plan = {'task': 'P10.2c.3b.2a.1a', 'previous': PREVIOUS, 'core': CORE, 'protocol': protocol,
            'controls': CONTROLS, 'jobs': jobs, 'reading': old['reading'], 'policy': POLICY,
            'code': [pin(ROOT, n) for n in dict.fromkeys(names)], 'repeats': 2,
            'budgets': read(checked(ROOT, protocol))['execution'], 'new_validation_used_before_freeze': False,
            'new_OCR_ML_calls': 0, 'training': False, 'closed_test_used': False, 'production_changed': False,
            'source_erasure_allowed': False, 'physical_geometry_approved': False}
    write(BASE/'plan.json', plan)
    print(pin(ROOT, (BASE/'plan.json').relative_to(ROOT).as_posix()))


def child(path):
    import cv2
    import numpy as np
    from PIL import Image
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.contact_role_overlay_v2 import analyze
    from cad_recovery.supported_role_overlay_v1 import analyze as old_analyze
    from cad_recovery.stroke_ownership_metrics_v1 import score_roles, ink_ledger
    from cad_recovery.stroke_roles_v1 import foreground
    from cad_recovery.stroke_role_metrics_v1 import to_native, synthetic_truth, head_scores, mask_effect
    from contact_role_inputs_v1 import prepare
    from run_stroke_roles_v1 import refs_for
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    request = read(path)
    plan = read(checked(ROOT, request['plan']))
    for p in plan['code']:
        checked(ROOT, p)
    job, case = request['job'], request['job']['case']
    folder = path.parent/'output'
    folder.mkdir()
    rows = []
    truth = negative = part = native_gray = None
    for view in job['views']:
        gray, source, geometry, baseline, text, previous, local, heads, extra, prep = prepare(ROOT, view)
        if native_gray is None:
            native_gray = gray.copy()
        sha = hashlib.sha256(gray.tobytes()).hexdigest()
        before = fingerprint([source, geometry, previous, heads, extra, local['hybrid']['dimension_chains']])
        if view.get('comparison'):
            old_map = np.array(Image.open(checked(ROOT, view['comparison']['old_overlay_roles'])))
            old_row = read(checked(ROOT, view['comparison']['row']))
            if old_row['input_pixel_sha256'] != sha:
                raise ValueError('Mixed cached role raster')
        else:
            old_map = old_analyze(gray, previous, baseline, local, heads=heads, diagnostic_heads=extra)['pixel_roles']
        started = time.perf_counter()
        result = analyze(gray, previous, baseline, local, heads=heads, diagnostic_heads=extra)
        elapsed = time.perf_counter()-started
        variants = {'baseline': baseline, 'previous_overlay': old_map, 'contact_overlay': result['pixel_roles']}
        # All truth is evaluated only after inference.
        if truth is None:
            if job.get('truth'):
                truth = np.array(Image.open(checked(ROOT, job['truth'])))
            elif case['scope'] == 'analytical_control':
                truth = synthetic_truth(case, native_gray, np.array(Image.open(checked(ROOT, view['reviewed_mask']))))
        if negative is None and job.get('negative'):
            negative = np.array(Image.open(checked(ROOT, job['negative'])))
        if part is None and job.get('part_contact'):
            part = np.array(Image.open(checked(ROOT, job['part_contact'])))
        if truth is None and negative is not None:
            truth = np.full(tuple(case['size_px'][::-1]), 255, np.uint8)
        sub = folder/view['mode']
        sub.mkdir()
        scores, ledgers, added, protected = {}, {}, {}, {}
        for name, labels in variants.items():
            Image.fromarray(labels).save(sub/(name+'-roles.png'))
            native = to_native(labels, case['size_px'])
            native_delta = to_native((labels != baseline).astype(np.uint8), case['size_px']) > 0
            scores[name] = score_roles(truth, native, negative) if truth is not None else None
            ledgers[name] = ink_ledger(gray, labels, foreground)
            if truth is not None:
                known = native_delta & (truth > 0) & (truth < 10)
                added[name] = {'known_added_pixels': int(known.sum()),
                    'correct_added_pixels': int((known & (native == truth)).sum()),
                    'incorrect_added_pixels': int((known & (native != truth)).sum()),
                    'unknown_truth_added_pixels': int((native_delta & (truth == 255)).sum())}
            if part is not None:
                protected[name] = int((native_delta & (part > 0) & np.isin(native, [2, 3])).sum())
        Image.fromarray(result['pixel_interval_owner']).save(sub/'pixel-interval-owner.png')
        Image.fromarray(text).save(sub/'text-mask.png')
        if 'arrows' in case.get('labels', {}):
            refs, negatives = case['labels']['arrows'], case['labels'].get('negative_regions', [])
        else:
            refs, negatives = refs_for(case, read(checked(ROOT, plan['reading'])))
        available = set(result['head_ids_used'])
        head_metrics = {name: head_scores(refs, negatives, hh, view['input_to_native'], dense=case['scope'].endswith('analytical_control'))
            for name, hh in [('baseline', heads), ('local_guard', [h for h in heads if h['id'] in available])]}
        row = {k: case[k] for k in ('scope', 'group_id', 'split')}
        row.update(case=case['id'], mode=view['mode'], repeat=request['repeat'], new_control=job['new_control'],
            input_to_native=view['input_to_native'], input_pixel_sha256=sha,
            source_pixel_sha256_after=hashlib.sha256(gray.tobytes()).hexdigest(),
            source_observations_sha256_before=before,
            source_observations_sha256_after=fingerprint([source, geometry, previous, heads, extra, local['hybrid']['dimension_chains']]),
            roles=scores, ink_ledgers=ledgers, added_truth=added, protected_part_additions=protected,
            heads=head_metrics, head_candidates={'baseline': heads, 'DT_novel': extra},
            details={k:v for k,v in result.items() if k not in ('pixel_roles', 'pixel_interval_owner', 'text_mask')},
            preparation=prep, text_mask_effect=mask_effect(truth, native_gray, to_native(text, case['size_px'])) if truth is not None else None,
            timing={'contact_overlay_seconds': elapsed}, source_pixels_erased=0, semantic_CAD_promotions=0,
            production_admitted=False, physical_geometry_approved=False)
        write(sub/'source.json', source)
        write(sub/'geometry.json', geometry)
        write(sub/'result.json', row)
        rows.append(row)
    write(folder/'result.json', {'rows': rows})


def run(sha):
    from experiment_children_v1 import run as execute
    result = execute(ROOT, BASE, {'path': 'artifacts/contact-role-overlay-v2/plan.json', 'sha256': sha}, Path(__file__).resolve())
    rows = [read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    result.update(contact_overlay_calls=len(rows), previous_overlay_fresh_calls=sum(r['new_control'] for r in rows),
        cached_previous_overlay_views=sum(not r['new_control'] for r in rows),
        preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
        source_partition_checks=3*len(rows), new_OCR_ML_calls=0, new_DXF_exports=0,
        training=False, closed_test_used=False, production_changed=False)
    write(BASE/'execution.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'run', 'child'])
    parser.add_argument('--sha256')
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'child': child(args.request)
    elif args.sha256: run(args.sha256)
    else: parser.error('--sha256 required')
