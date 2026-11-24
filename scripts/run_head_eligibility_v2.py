"""Frozen paired experiment: immutable source observations plus eligibility sidecar."""
import argparse
import hashlib
from pathlib import Path
import time

from run_local_width_heads_v1 import read, write, fingerprint

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/head-eligibility-v2'
PREVIOUS = {'path': 'artifacts/local-axis-heads-release-v2.json',
            'sha256': 'eb97b864c5d6f4f19e30c083e5b59395b095d0b06e9ca0bae0a7d1f69430c89d'}
CONTROLS = {'path': 'artifacts/head-eligibility-controls-v1/contract.json',
            'sha256': 'f0061500c2e2e26c6114524f579ee4725feb96bdbd8d16316996a17fd6e8f71a'}
CORE = {'path': 'artifacts/cv-comparison-protocol-v1/contract.json',
        'sha256': '815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}


def freeze():
    from cad_recovery.dataset_layout_v1 import checked, pin
    from cad_recovery.head_eligibility_v2 import POLICY
    from verify_cv_comparison_contract_v1 import verify_contract
    verify_contract(ROOT, CORE)
    checked(ROOT, PREVIOUS)
    old = read(checked(ROOT, {'path': 'artifacts/local-axis-heads-v2/plan.json',
        'sha256': '42b86bf7fac7fcb9855d3765ab33c1af5193ae16898c45234c9eea60c30b8411'}))
    controls = read(checked(ROOT, CONTROLS))
    checked(ROOT, controls['generator'])
    index = {(r['case'], r['mode']): r for r in read(ROOT/'artifacts/local-axis-heads-v2/results-index.json') if not r['repeat']}
    jobs = []
    for job in old['jobs']:
        views = [{'mode': v['mode'], 'gray': v['gray'], 'input_to_native': v['input_to_native'],
                  'comparison_cache': pin(ROOT, index[job['case']['id'], v['mode']]['artifact']),
                  'binding_cache': v.get('binding_cache')} for v in job['views']]
        jobs.append({'case': job['case'], 'views': views, 'new_control': False})
    jobs.extend(controls['jobs'])
    names = ['src/cad_recovery/head_eligibility_v2.py', 'scripts/head_eligibility_inputs_v1.py',
             'scripts/run_head_eligibility_v2.py', 'scripts/prepare_head_eligibility_controls_v1.py',
             'tests/test_head_eligibility_v2.py']+[p['path'] for p in old['code']]
    protocol = pin(ROOT, 'artifacts/head-eligibility-protocol-v1.json')
    value = {'task': 'P10.2c.2b.7c.2a.3b', 'previous': PREVIOUS, 'core': CORE,
             'controls': CONTROLS, 'protocol': protocol, 'code': [pin(ROOT, n) for n in dict.fromkeys(names)],
             'jobs': jobs, 'reading': old['reading'], 'policy': POLICY, 'repeats': 2,
             'budgets': read(checked(ROOT, protocol))['execution'],
             'new_validation_used_before_freeze': False, 'post_validation_parameter_changes': False,
             'production_changed': False, 'training': False, 'closed_test_used': False,
             'physical_geometry_approved': False, 'new_OCR_ML_calls': 0, 'new_DXF_exports': 0}
    write(BASE/'plan.json', value)
    print(pin(ROOT, (BASE/'plan.json').relative_to(ROOT).as_posix()))


def child(path):
    import cv2
    import numpy as np
    from PIL import Image
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.head_eligibility_v2 import apply
    from cad_recovery.stroke_role_metrics_v1 import head_scores
    from head_eligibility_inputs_v1 import prepare
    from run_stroke_roles_v1 import refs_for
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    request = read(path)
    plan = read(checked(ROOT, request['plan']))
    for p in plan['code']:
        checked(ROOT, p)
    job = request['job']
    case = job['case']
    folder = path.parent/'output'
    folder.mkdir()
    rows = []
    for view in job['views']:
        gray, packet, prep = prepare(ROOT, view)
        pixel_sha = hashlib.sha256(gray.tobytes()).hexdigest()
        before = fingerprint(packet)
        started = time.perf_counter()
        sidecar = apply(gray, packet['candidates'], inherited_morphology_pixels=packet['analysis']['morphology_pixels'])
        ids = set(sidecar['diagnostic_candidate_ids'])
        retained = [h for h in packet['candidates'] if h['id'] in ids]
        elapsed = time.perf_counter()-started
        binding = None
        if view.get('binding_cache'):
            from cad_recovery.local_role_guard_v1 import analyze as bind
            saved = view['binding_cache']
            parent = read(checked(ROOT, saved['cache']))
            geometry = read(checked(ROOT, saved['geometry']))
            evidence = parent['previous_interval_details']
            text = np.array(Image.open(checked(ROOT, saved['text_mask'])))
            roles = np.array(Image.open(checked(ROOT, saved['baseline_roles'])))
            immutable = fingerprint([geometry, evidence])
            attached = bind(gray, evidence, roles, text, retained, diagnostic_heads=parent['head_candidates']['DT_novel'])['hybrid']
            binding = {'dimension_chains': attached['dimension_chains'], 'search_incomplete': attached['search_incomplete'],
                       'source_geometry_unchanged': immutable == fingerprint([geometry, evidence]), 'CAD_promotions': 0}
        # Evaluation only. Neither labels nor coordinates enter the sidecar.
        labels = case.get('labels', {})
        refs, negative = (labels['arrows'], labels.get('negative_regions', [])) if 'arrows' in labels else refs_for(case, read(checked(ROOT, plan['reading'])))
        dense = case['scope'].endswith('analytical_control') or case['scope'] in (
            'local_width_head_control', 'sparse_head_control', 'sparse_head_large_control', 'economical_head_control',
            'economical_head_large_control', 'local_head_context_control', 'local_head_context_control_v2',
            'head_support_control', 'streamed_head_control', 'streamed_head_large_control', 'curved_head_control', 'head_eligibility_control')
        scores = {n: head_scores(refs, negative, heads, view['input_to_native'], dense=dense)
                  for n, heads in [('observed', packet['candidates']), ('retained', retained)]}
        row = {k: case.get(k) for k in ('scope', 'group_id', 'split', 'cohort')}
        row.update(case=case['id'], mode=view['mode'], repeat=request['repeat'], new_control=job['new_control'],
            source=view['gray'], input_to_native=view['input_to_native'], input_pixel_sha256=pixel_sha,
            source_pixel_sha256_after=hashlib.sha256(gray.tobytes()).hexdigest(),
            observations_sha256_before=before, observations_sha256_after=fingerprint(packet),
            source_packet=packet, sidecar=sidecar, heads=scores, preparation=prep, binding=binding,
            timing={'sidecar_seconds': elapsed}, new_OCR_ML_calls=0, CAD_promotions=0,
            production_changed=False, physical_geometry_approved=False)
        write(folder/view['mode']/'result.json', row)
        rows.append(row)
    write(folder/'result.json', {'rows': rows})


def run(sha):
    from experiment_children_v1 import run as execute
    result = execute(ROOT, BASE, {'path': 'artifacts/head-eligibility-v2/plan.json', 'sha256': sha}, Path(__file__).resolve())
    index = read(BASE/'results-index.json')
    fresh = sum(r['new_control'] for r in index)
    result.update(fresh_eligibility_calls=len(index), fresh_source_pipeline_calls_each=fresh,
                  cached_source_views=len(index)-fresh, binding_calls=sum(bool(read(ROOT/r['artifact'])['binding']) for r in index),
                  training=False, closed_test_used=False, new_OCR_ML_calls=0, new_DXF_exports=0, production_changed=False)
    write(BASE/'execution.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'run', 'child'])
    parser.add_argument('--sha256')
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze()
    elif args.mode == 'child':
        child(args.request)
    elif args.sha256:
        run(args.sha256)
    else:
        parser.error('--sha256 required')
