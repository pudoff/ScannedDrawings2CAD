"""Independent paired comparison of thick-frame repair and junction abstention."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_frame_context_v1 import ROOT, inputs, frame_score, bind_job_source
from run_source_region_v1 import truth_for, changes
from run_topology_robust_v1 import read, write, object_hash
from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.page_frame_context_v2 import detect
from cad_recovery.frame_grid_guard_v2 import bind_context, analyze
from cad_recovery.grid_role_revision_v1 import sha
from cad_recovery.joint_role_metrics_v1 import score
from cad_recovery.stroke_role_metrics_v1 import to_native

BASE = ROOT/'artifacts/frame-contact-profile-v1'
QUAL = ROOT/'artifacts/frame-contact-qualification-v3'
BANK = ROOT/'artifacts/frame-contact-page-bank-v3'
PREVIOUS = {'path': 'artifacts/frame-context-release-v1.json', 'sha256': '3e6ec41cc35faea8235e0896b4bcc7e5a63b343814e328393f9a9ee3ca8dd41d'}
CONTROLS = {'path': 'artifacts/frame-context-controls-v2/contract.json', 'sha256': '55a8b34ea07aa760ba86b3ad521da4b426101b8fa56e17678cabfbaa8461fa24'}
ARMS = ('previous', 'base', 'v1_page', 'coarse', 'uncertainty')
NEW_CODE = ['src/cad_recovery/page_frame_context_v2.py', 'src/cad_recovery/frame_grid_guard_v2.py',
            'scripts/run_frame_context_v2.py', 'scripts/prepare_frame_context_controls_v2.py', 'tests/test_frame_context_v2.py']


def old_jobs():
    prior = read(ROOT/'artifacts/frame-context-profile-v1/plan.json')
    index = {(r['case'], r['mode']): r for r in read(ROOT/'artifacts/frame-context-profile-v1/results-index.json') if not r['repeat']}
    jobs = []
    for job in prior['jobs']:
        views = []
        for v in job['views']:
            folder = (ROOT/index[job['case']['id'], v['mode']]['artifact']).parent; source = read(folder/'source-inputs.json')
            record = source.get('structural_inputs')
            if not record:
                parent = ROOT/source['baseline_source']
                record = {k: pin(ROOT, (parent/f).relative_to(ROOT).as_posix()) for k, f in
                          [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
            views.append({**v, 'structural_inputs': record, 'admission_cache': source['cache'],
                          'v1_page': pin(ROOT, (folder/'page_context-roles.png').relative_to(ROOT).as_posix()),
                          'v1_row': pin(ROOT, (folder/'result.json').relative_to(ROOT).as_posix())})
        jobs.append({k: v for k, v in job.items() if k != 'page_bank'} |
                    {'case': {**job['case'], 'split': 'development'}, 'new_control': False, 'views': views})
    return jobs


def page_packet(job, target):
    gray = np.array(Image.open(checked(ROOT, job['page_source']['source'])).convert('L'))
    packet = {'source_file': job['page_source']['source'], 'decoded_sha256': sha(gray), 'detection': detect(gray),
              'detector': pin(ROOT, 'src/cad_recovery/page_frame_context_v2.py')}
    write(target, packet); return packet, gray


def get_page(job, target=None):
    if not job.get('page_source'): return None, 0
    if job.get('page_bank'):
        packet = read(checked(ROOT, job['page_bank'])); checked(ROOT, packet['detector'])
        if packet['source_file'] != job['page_source']['source']: raise ValueError('Parent context rebound')
        return packet, 0
    packet, gray = page_packet(job, target)
    bind_job_source(job, packet, gray)  # Check fresh ROI source binding before inference.
    return packet, 1


def predict(job, view, packet, target=None):
    data = inputs(job, view, target)
    gray, graph, raw, heads, links, authority, old, base, owners, cache, _, calls = data
    context = bind_context(gray, graph, old, base, owners, job['page_source'], packet['detection'], packet['decoded_sha256'],
                           view['input_to_native'], job.get('page_bank') or job['page_source']['source']) if packet else None
    before = object_hash([graph, heads, links]); hashes = [sha(a) for a in (gray, raw, old, base, owners, authority['mask'])]
    result = analyze(gray, graph, raw, old, base, owners, authority=authority, context=context)
    if view.get('v1_page'): v1 = np.array(Image.open(checked(ROOT, view['v1_page'])))
    else: v1 = base.copy() if packet is None or packet['detection']['fallback_used'] else result['coarse'].copy()
    return {'data': data, 'result': result, 'context': context,
            'maps': {'previous': old, 'base': base, 'v1_page': v1, **{a: result[a] for a in ARMS[3:]}},
            'source_exact': before == object_hash([graph, heads, links]),
            'maps_exact': hashes == [sha(a) for a in (gray, raw, old, base, owners, authority['mask'])]}


def qualify():
    cv2.setNumThreads(1); cv2.setRNGSeed(0); BANK.mkdir(exist_ok=False); QUAL.mkdir(exist_ok=False)
    old = old_jobs(); fresh = [j for j in read(checked(ROOT, CONTROLS))['jobs'] if j['case']['split'] == 'train']
    prepared = {j['case']['id']: j for j in read(ROOT/'artifacts/frame-contact-qualification-v1/train-index.json')['jobs']}
    fresh = [{**j, 'views': prepared[j['case']['id']]['views']} for j in fresh]
    packets = {}; bindings = []; bound = []
    for job in old+fresh:
        if job.get('page_source'):
            key = job['page_source']['source']['sha256']
            if key not in packets:
                target = BANK/(key+'.json'); packet, gray = page_packet(job, target)
                packets[key] = pin(ROOT, target.relative_to(ROOT).as_posix())
            else: packet, gray = read(checked(ROOT, packets[key])), None
            bindings.append(bind_job_source(job, packet, gray)); job = {**job, 'page_bank': packets[key]}
        bound.append(job)
    write(BANK/'bindings.json', {'rows': bindings, 'GT_used': False, 'all_native_and_half_source_bindings_exact': True})
    write(BANK/'old-jobs.json', {'jobs': [j for j in bound if not j['new_control']]})
    train = []
    for job in [j for j in bound if j['new_control']]:
        views = []
        for v in job['views']:
            if v.get('admission_cache'):
                inputs(job, v); views.append(v); continue
            folder = QUAL/'bank'/job['case']['id']/v['mode']/'producer'; data = inputs(job, v, folder)
            cache = data[9]; source = folder/'baseline-source'
            record = {k: pin(ROOT, (source/f).relative_to(ROOT).as_posix()) for k, f in
                      [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
            views.append({**v, 'structural_inputs': record, 'admission_cache': cache})
        train.append({**job, 'views': views}); print('prepared', job['case']['id'], flush=True)
    write(QUAL/'train-index.json', {'jobs': train, 'fresh_validation_opened': False})
    rows = []
    for job in train+[j for j in bound if not j['new_control'] and j.get('page_source')]:
        packet, _ = get_page(job); native = None
        for view in job['views']:
            p = predict(job, view, packet); gray = p['data'][0]; old, base = p['data'][6:8]
            if native is None: native = gray.copy()
            truth, negative = truth_for(job, native)
            rows.append({'case': job['case']['id'], 'mode': view['mode'], 'split': job['case']['split'],
                         'frames': {a: frame_score(job, truth, old, base, m) for a, m in p['maps'].items()},
                         'changes': {a: changes(job, truth, negative, old, m) for a, m in p['maps'].items()},
                         'complete': p['result']['complete'], 'uncertain_input_pixels': int(np.count_nonzero(p['result']['uncertainty_mask']))})
    write(QUAL/'report.json', {'rows': rows, 'code': [pin(ROOT, n) for n in NEW_CODE], 'fresh_validation_opened': False})
    for split in ('train', 'development'):
        print(split, {a: {k: sum(r['frames'][a][k] for r in rows if r['split'] == split) for k in
              ('remaining_new_wrong_grid_pixels', 'correct_grid_candidates_withdrawn', 'frame_to_grid_errors_removed')}
              for a in ARMS[2:]}, 'incomplete', sum(not r['complete'] for r in rows if r['split'] == split), flush=True)


def freeze():
    checked(ROOT, PREVIOUS); control = read(checked(ROOT, CONTROLS)); prior = read(ROOT/'artifacts/frame-context-profile-v1/plan.json')
    qualification = read(QUAL/'report.json')
    for p in qualification['code']+prior['code']: checked(ROOT, p)
    old = read(BANK/'old-jobs.json')['jobs']; fresh = read(QUAL/'train-index.json')['jobs']+[j for j in control['jobs'] if j['case']['split'] == 'validation']
    batches = [{'id': j['case']['id'], 'split': j['case']['split'], 'members': [j]} for j in fresh]
    for start in range(0, len(old), 8): batches.append({'id': 'development-batch-'+str(start//8).zfill(3), 'split': 'development', 'members': old[start:start+8]})
    names = list(dict.fromkeys([p['path'] for p in prior['code']]+NEW_CODE)); BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json', {'task': 'P10.2c.3b.2h.3b.1', 'previous': PREVIOUS, 'controls': CONTROLS,
          'source_binding': pin(ROOT, (BANK/'bindings.json').relative_to(ROOT).as_posix()),
          'qualification': pin(ROOT, (QUAL/'report.json').relative_to(ROOT).as_posix()),
          'diagnostics': pin(ROOT, 'artifacts/frame-contact-diagnostics-v1/report.json'),
          'jobs': old+fresh, 'batches': batches, 'code': [pin(ROOT, n) for n in names], 'criteria': control['criteria'],
          'budgets': control['budgets'], 'arms': ARMS, 'fixed_selection_order': ['uncertainty', 'coarse', 'base'],
          'fresh_validation_predictions_before_freeze': False, 'uncertain_pixels_excluded_from_metrics': False,
          'native_resolution_refinement_implemented': False, 'batch_membership_not_model_input': True,
          'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, (BASE/'plan.json').relative_to(ROOT).as_posix()), flush=True)


def child(path):
    cv2.setNumThreads(1); cv2.setRNGSeed(0); request = read(path); plan = read(checked(ROOT, request['plan']))
    for p in plan['code']: checked(ROOT, p)
    out = path.parent/'output'; out.mkdir(); rows = []
    for job in request['batch']['members']:
        case_out = out/job['case']['id']; case_out.mkdir(); packet, detector_calls = get_page(job, case_out/'page-context.json'); native = None
        for view in job['views']:
            folder = case_out/view['mode']; folder.mkdir(); p = predict(job, view, packet, folder/'producer')
            gray, graph, raw, heads, links, authority, old, base, owners, cache, _, calls = p['data']; result = p['result']
            if native is None: native = gray.copy()
            truth, negative = truth_for(job, native)
            metrics = {a: score(truth, to_native(m, job['case']['size_px']), negative) if truth is not None else None for a, m in p['maps'].items()}
            checks = {}
            for a in ARMS[3:]:
                m = p['maps'][a]; changed = m != base
                checks[a] = {'only_new_grid_to_previous': bool(np.all((base[changed] == 8) & (old[changed] != 8) & (m[changed] == old[changed]))),
                             'protected_exact': bool(np.array_equal(m[authority['mask'] == 2], base[authority['mask'] == 2]))}
                Image.fromarray(m).save(folder/(a+'-roles.png'))
            Image.fromarray(result['uncertainty_mask']).save(folder/'uncertainty-mask.png')
            write(folder/'guard.json', {'context': p['context'], 'decisions': result['decisions'], 'audit': result['audit'], 'coarse_audit': result['coarse_audit']})
            source = {'cache': cache}
            if view.get('structural_inputs'): source['structural_inputs'] = view['structural_inputs']
            else: source['baseline_source'] = (folder/'producer'/'baseline-source').relative_to(ROOT).as_posix()
            write(folder/'source-inputs.json', source)
            former = read(checked(ROOT, view['v1_row'])) if view.get('v1_row') else None
            row = {**{k: job['case'][k] for k in ('scope', 'group_id', 'split')}, 'case': job['case']['id'], 'mode': view['mode'],
                   'repeat': request['repeat'], 'new_control': job['new_control'], 'roles': metrics,
                   'changes': {a: changes(job, truth, negative, old, m) for a, m in p['maps'].items()},
                   'frames': {a: frame_score(job, truth, old, base, m) for a, m in p['maps'].items()},
                   'source_exact': p['source_exact'], 'input_maps_exact': p['maps_exact'], 'view_checks': checks,
                   'historical_metrics_exact': former is None or all(metrics[a] == former['roles'][b] for a, b in [('previous', 'previous'), ('base', 'base'), ('v1_page', 'page_context')]),
                   'complete': result['complete'], 'page_detector_complete': packet['detection']['complete'] if packet else True,
                   'page_frames': len(packet['detection']['frames']) if packet else 0,
                   'fallback_used': packet['detection']['fallback_used'] if packet else False,
                   'uncertain_input_pixels': int(np.count_nonzero(result['uncertainty_mask'])),
                   'preparation': {**calls, 'page_detector_calls': detector_calls}, 'GT_used_for_inference': False,
                   'uncertain_pixels_excluded_from_metrics': False, 'source_pixels_erased': 0,
                   'production_changed': False, 'physical_geometry_approved': False}
            detector_calls = 0; write(folder/'result.json', row); rows.append(row)
    write(out/'result.json', {'rows': rows})


def run(sha256):
    from experiment_case_batches_v1 import run as execute
    e = execute(ROOT, BASE, {'path': (BASE/'plan.json').relative_to(ROOT).as_posix(), 'sha256': sha256}, Path(__file__).resolve())
    rows = [read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    e.update(guard_calls=len(rows), preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
             new_OCR_ML_calls=0, new_DXF_exports=0, training=False, closed_test_used=False)
    write(BASE/'execution.json', e); print(e, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['qualify', 'freeze', 'run', 'child'])
    p.add_argument('--sha256'); p.add_argument('--request', type=Path); a = p.parse_args()
    if a.mode == 'qualify': qualify()
    elif a.mode == 'freeze': freeze()
    elif a.mode == 'child': child(a.request)
    elif a.sha256: run(a.sha256)
    else: p.error('--sha256 required')
