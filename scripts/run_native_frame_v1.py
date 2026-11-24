"""Bounded native source-mask refinement, independent of native predictions."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_frame_context_v1 import ROOT, inputs, frame_score, bind_job_source
from run_frame_context_v2 import page_packet, get_page
from run_source_region_v1 import truth_for, changes
from run_topology_robust_v1 import read, write, object_hash
from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.raster_regions import read_prepared_region
from cad_recovery.native_frame_v1 import source_masks, refine
from cad_recovery.frame_grid_guard_v2 import bind_context
from cad_recovery.frame_grid_guard_v1 import analyze as coarse_guard
from cad_recovery.grid_role_revision_v1 import sha
from cad_recovery.part_admission_v1 import object_sha
from cad_recovery.joint_role_metrics_v1 import score
from cad_recovery.stroke_role_metrics_v1 import to_native

BASE = ROOT/'artifacts/native-frame-profile-v1'
QUAL = ROOT/'artifacts/native-frame-qualification-v3'
BANK = ROOT/'artifacts/native-frame-bank-v3'
PREVIOUS = {'path': 'artifacts/frame-contact-release-v1.json', 'sha256': '7ee8231fec2a60ad0533d8c6dceaab956f8fbb1d2aaf6be2a842a37a0efbb04d'}
PRIOR_PLAN = {'path': 'artifacts/frame-contact-profile-v1/plan.json', 'sha256': 'd736ab10db648819f680b854cb254908f7d63919923f41d38ccc18a06bec278f'}
CONTROLS = {'path': 'artifacts/native-frame-controls-v1/contract.json', 'sha256': '2b4fb4be3d1b3f69811b7d314ff91ce95aa759dbf3a52a712ca7da8cae7cb575'}
ARMS = ('previous', 'base', 'coarse', 'native_mid', 'native_safe')
NEW_CODE = ['src/cad_recovery/native_frame_v1.py', 'scripts/run_native_frame_v1.py',
            'scripts/prepare_native_frame_controls_v1.py', 'tests/test_native_frame_v1.py']
MASKS = ('mid', 'safe', 'possible', 'coverage', 'failed_coverage')


def old_jobs():
    prior = read(checked(ROOT, PRIOR_PLAN)); index = {(r['case'], r['mode']): r for r in read(ROOT/'artifacts/frame-contact-profile-v1/results-index.json') if not r['repeat']}
    jobs = []
    for job in prior['jobs']:
        views = []; page_bank = job.get('page_bank')
        for v in job['views']:
            folder = (ROOT/index[job['case']['id'], v['mode']]['artifact']).parent; source = read(folder/'source-inputs.json')
            record = source.get('structural_inputs')
            if record is None:
                parent = ROOT/source['baseline_source']
                record = {k: pin(ROOT, (parent/f).relative_to(ROOT).as_posix()) for k, f in
                          [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
            if job.get('page_source') and page_bank is None: page_bank = pin(ROOT, (folder.parent/'page-context.json').relative_to(ROOT).as_posix())
            views.append({k: v[k] for k in ('mode', 'gray', 'input_to_native')} |
                         {'structural_inputs': record, 'admission_cache': source['cache'],
                          'coarse': pin(ROOT, (folder/'coarse-roles.png').relative_to(ROOT).as_posix()),
                          'coarse_row': pin(ROOT, (folder/'result.json').relative_to(ROOT).as_posix())})
        jobs.append({**job, 'views': views, 'case': {**job['case'], 'split': 'development'}, 'new_control': False, 'page_bank': page_bank})
    return jobs


def original_region(job):
    page = job['page_source']; box = page['patch_bbox']
    if page['kind'] == 'authenticated_page_overview':
        checked(ROOT, page['original_source']); preparation = checked(ROOT, page['preparation'])
        rgb, proof = read_prepared_region(preparation.parent, box, max_pixels=4194304)
        gray = np.array(Image.fromarray(rgb).convert('L'))
    else:
        whole = np.array(Image.open(checked(ROOT, page['source'])).convert('L'))
        gray = whole[box[1]:box[3], box[0]:box[2]].copy(); proof = {'source': page['source'], 'original_pixel_crop': True}
    native = np.array(Image.open(checked(ROOT, job['views'][0]['gray'])).convert('L'))
    if not np.array_equal(gray, native): raise ValueError('Original ROI pixels differ from recorded native view')
    return gray, proof


def prepare_fit(job, parent, target):
    gray, proof = original_region(job); binding = bind_job_source(job, parent)
    page = job['page_source']; whole = None
    if max(page['source_size_px']) <= 4096 and int(np.prod(page['source_size_px'])) <= 4194304:
        full_pin = page.get('original_source', page['source'])
        whole = np.array(Image.open(checked(ROOT, full_pin)).convert('L'))
        proof = {**proof, 'whole_original_source': full_pin, 'whole_source_decoded_sha256': sha(whole)}
    result = source_masks(gray, page, parent['detection'], context_gray=whole); target.mkdir(parents=True)
    Image.fromarray(gray).save(target/'source-gray.png')
    paths = {}
    for name in MASKS:
        Image.fromarray(result[name]).save(target/(name+'.png')); paths[name] = pin(ROOT, (target/(name+'.png')).relative_to(ROOT).as_posix())
    record = {k: v for k, v in result.items() if k not in MASKS} | {'mask_paths': paths,
              'gray': pin(ROOT, (target/'source-gray.png').relative_to(ROOT).as_posix()), 'source_proof': proof,
              'binding': binding, 'model': pin(ROOT, 'src/cad_recovery/native_frame_v1.py')}
    write(target/'source.json', record); return pin(ROOT, (target/'source.json').relative_to(ROOT).as_posix())


def load_fit(job, parent, target=None):
    if parent is None: return None, None, None, 0
    record_pin = job.get('native_source') or prepare_fit(job, parent, target)
    record = read(checked(ROOT, record_pin)); checked(ROOT, record['model'])
    if record['page_sha256'] != object_sha(job['page_source']) or record['detector_sha256'] != object_sha(parent['detection']): raise ValueError('Native mask rebound to another page')
    gray = np.array(Image.open(checked(ROOT, record['gray'])))
    result = {**record, **{name: np.array(Image.open(checked(ROOT, record['mask_paths'][name]))) for name in MASKS}}
    return gray, result, record_pin, int(not job.get('native_source'))


def predict(job, view, parent, native_gray, source, folder=None):
    data = inputs(job, view, folder)
    gray, graph, raw, heads, links, authority, previous, base, owners, cache, _, calls = data
    if view.get('coarse'): coarse = np.array(Image.open(checked(ROOT, view['coarse'])))
    else:
        context = bind_context(gray, graph, previous, base, owners, job['page_source'], parent['detection'], parent['decoded_sha256'], view['input_to_native'], job['page_source']['source']) if parent else None
        coarse = coarse_guard(gray, graph, raw, previous, base, owners, authority=authority, context=context)['page_context']
    before = object_hash([graph, heads, links]); hashes = [sha(a) for a in (gray, raw, previous, base, coarse, owners, authority['mask'])]
    result = refine(gray, graph, raw, previous, base, coarse, owners, authority=authority,
                    input_to_native=view['input_to_native'], source_gray=native_gray, source=source)
    return {'data': data, 'result': result, 'maps': {'previous': previous, 'base': base, 'coarse': coarse, **{a: result[a] for a in ARMS[3:]}},
            'source_exact': before == object_hash([graph, heads, links]),
            'input_maps_exact': hashes == [sha(a) for a in (gray, raw, previous, base, coarse, owners, authority['mask'])]}


def qualify():
    cv2.setNumThreads(1); cv2.setRNGSeed(0); checked(ROOT, PREVIOUS); BANK.mkdir(exist_ok=False); QUAL.mkdir(exist_ok=False)
    old = old_jobs(); fresh = [j for j in read(checked(ROOT, CONTROLS))['jobs'] if j['case']['split'] == 'train']; bound = []
    cached = {j['case']['id']: j for j in read(ROOT/'artifacts/native-frame-qualification-v1/train-index.json')['jobs']}
    fresh = [{**j, 'views': cached[j['case']['id']]['views']} for j in fresh]
    for job in old+fresh:
        if job.get('page_source'):
            if not job.get('page_bank'):
                page_target = BANK/(job['case']['id']+'-page.json'); parent, _ = page_packet(job, page_target)
                job = {**job, 'page_bank': pin(ROOT, page_target.relative_to(ROOT).as_posix())}
            parent, _ = get_page(job)
            job = {**job, 'native_source': prepare_fit(job, parent, BANK/job['case']['id'])}
            print('bound source', job['case']['id'], flush=True)
        bound.append(job)
    write(BANK/'old-jobs.json', {'jobs': [j for j in bound if not j['new_control']]})
    train = []
    for job in [j for j in bound if j['new_control']]:
        views = []
        for v in job['views']:
            if v.get('admission_cache'):
                inputs(job, v); views.append(v); continue
            folder = QUAL/'bank'/job['case']['id']/v['mode']/'producer'; data = inputs(job, v, folder); source = folder/'baseline-source'
            record = {k: pin(ROOT, (source/f).relative_to(ROOT).as_posix()) for k, f in
                      [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
            views.append({**v, 'structural_inputs': record, 'admission_cache': data[9]})
        train.append({**job, 'views': views}); print('prepared', job['case']['id'], flush=True)
    write(QUAL/'train-index.json', {'jobs': train, 'fresh_validation_opened': False})
    rows = []
    for job in train+[j for j in bound if not j['new_control'] and j.get('page_source')]:
        parent, _ = get_page(job); gray, source, _, _ = load_fit(job, parent); native = None
        for view in job['views']:
            p = predict(job, view, parent, gray, source)
            if native is None: native = p['data'][0].copy()
            truth, negative = truth_for(job, native); previous, base = p['data'][6:8]
            rows.append({'case': job['case']['id'], 'mode': view['mode'], 'split': job['case']['split'],
                         'frames': {a: frame_score(job, truth, previous, base, m) for a, m in p['maps'].items()},
                         'changes': {a: changes(job, truth, negative, previous, m) for a, m in p['maps'].items()},
                         'complete': p['result']['complete'], 'source_fitted_sides': sum(d['fit'] for d in source['decisions'])})
    write(QUAL/'report.json', {'rows': rows, 'code': [pin(ROOT, n) for n in NEW_CODE], 'fresh_validation_opened': False})
    for split in ('train', 'development'):
        print(split, {a: {k: sum(r['frames'][a][k] for r in rows if r['split'] == split) for k in
              ('remaining_new_wrong_grid_pixels', 'correct_grid_candidates_withdrawn', 'frame_to_grid_errors_removed')} for a in ARMS[2:]},
              'incomplete', sum(not r['complete'] for r in rows if r['split'] == split), flush=True)


def freeze():
    prior = read(checked(ROOT, PRIOR_PLAN)); control = read(checked(ROOT, CONTROLS)); q = read(QUAL/'report.json')
    for p in prior['code']+q['code']: checked(ROOT, p)
    old = read(BANK/'old-jobs.json')['jobs']; fresh = read(QUAL/'train-index.json')['jobs']+[j for j in control['jobs'] if j['case']['split'] == 'validation']
    batches = [{'id': j['case']['id'], 'split': j['case']['split'], 'members': [j]} for j in fresh]
    for start in range(0, len(old), 8): batches.append({'id': 'development-batch-'+str(start//8).zfill(3), 'split': 'development', 'members': old[start:start+8]})
    names = list(dict.fromkeys([p['path'] for p in prior['code']]+NEW_CODE)); BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json', {'task': 'P10.2c.3b.2h.3b.2', 'previous': PREVIOUS, 'prior_plan': PRIOR_PLAN, 'controls': CONTROLS,
          'qualification': pin(ROOT, (QUAL/'report.json').relative_to(ROOT).as_posix()), 'jobs': old+fresh, 'batches': batches,
          'code': [pin(ROOT, n) for n in names], 'criteria': control['criteria'], 'budgets': control['budgets'],
          'coordinate_contract': control['coordinate_contract'], 'fixed_selection_order': ['native_safe', 'native_mid', 'coarse'],
          'source_context_contract': pin(ROOT, 'artifacts/native-frame-design-v1/source-context-contract.json'),
          'fresh_validation_predictions_before_freeze': False, 'native_prediction_transfer': False,
          'uncertain_pixels_excluded_from_metrics': False, 'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, (BASE/'plan.json').relative_to(ROOT).as_posix()), flush=True)


def child(path):
    cv2.setNumThreads(1); cv2.setRNGSeed(0); request = read(path); plan = read(checked(ROOT, request['plan']))
    for p in plan['code']: checked(ROOT, p)
    out = path.parent/'output'; out.mkdir(); rows = []
    for job in request['batch']['members']:
        case_out = out/job['case']['id']; case_out.mkdir(); parent, page_calls = get_page(job, case_out/'page-context.json')
        native_gray, source, source_pin, fit_calls = load_fit(job, parent, case_out/'native-source'); native = None
        for view in job['views']:
            folder = case_out/view['mode']; folder.mkdir(); p = predict(job, view, parent, native_gray, source, folder/'producer')
            gray, graph, raw, heads, links, authority, previous, base, owners, cache, _, calls = p['data']; result = p['result']
            if native is None: native = gray.copy()
            truth, negative = truth_for(job, native)
            metrics = {a: score(truth, to_native(m, job['case']['size_px']), negative) if truth is not None else None for a, m in p['maps'].items()}
            checks = {}; previous_n = to_native(previous, job['case']['size_px']); base_n = to_native(base, job['case']['size_px']); protection = to_native(authority['mask'], job['case']['size_px'])
            for a in ARMS[3:]:
                m = p['maps'][a]; changed = m != base_n
                checks[a] = {'native_shape_exact': list(m.shape[::-1]) == job['case']['size_px'],
                    'only_new_grid_to_previous': bool(np.all((base_n[changed] == 8) & (previous_n[changed] != 8) & (m[changed] == previous_n[changed]))),
                    'protected_exact': bool(np.array_equal(m[protection == 2], base_n[protection == 2]))}
                Image.fromarray(m).save(folder/(a+'-roles.png'))
            Image.fromarray(result['uncertainty_mask']).save(folder/'uncertainty-mask.png')
            Image.fromarray(p['maps']['coarse']).save(folder/'coarse-reference.png')
            write(folder/'native-output.json', {'coordinate_system': result['output_coordinate_system'], 'input_to_native': view['input_to_native'],
                  'native_to_page': job['page_source']['patch_to_source'] if parent else None, 'source': source_pin,
                  'original_source_context_used': result['source_context_used'], 'native_prediction_transfer': False})
            record = {'cache': cache, 'coarse': view.get('coarse'), 'native_source': source_pin}
            if view.get('structural_inputs'): record['structural_inputs'] = view['structural_inputs']
            else: record['baseline_source'] = (folder/'producer'/'baseline-source').relative_to(ROOT).as_posix()
            write(folder/'source-inputs.json', record)
            former = read(checked(ROOT, view['coarse_row'])) if view.get('coarse_row') else None
            row = {**{k: job['case'][k] for k in ('scope', 'group_id', 'split')}, 'case': job['case']['id'], 'mode': view['mode'],
                   'repeat': request['repeat'], 'new_control': job['new_control'], 'roles': metrics,
                   'changes': {a: changes(job, truth, negative, previous, m) for a, m in p['maps'].items()},
                   'frames': {a: frame_score(job, truth, previous, base, m) for a, m in p['maps'].items()},
                   'view_checks': checks, 'source_exact': p['source_exact'], 'input_maps_exact': p['input_maps_exact'],
                   'historical_metrics_exact': former is None or all(metrics[a] == former['roles'][a] for a in ARMS[:3]),
                   'complete': result['complete'], 'fitted_source_sides': sum(d['fit'] for d in source['decisions']) if source else 0,
                   'uncertain_native_pixels': int(np.count_nonzero(result['uncertainty_mask'])),
                   'preparation': {**calls, 'page_detector_calls': page_calls, 'native_source_fit_calls': fit_calls},
                   'GT_used_for_inference': False, 'native_prediction_transfer': False, 'uncertain_pixels_excluded_from_metrics': False,
                   'production_changed': False, 'source_pixels_erased': 0, 'physical_geometry_approved': False}
            page_calls = fit_calls = 0; write(folder/'result.json', row); rows.append(row)
    write(out/'result.json', {'rows': rows})


def run(sha256):
    from experiment_case_batches_v1 import run as execute
    e = execute(ROOT, BASE, {'path': (BASE/'plan.json').relative_to(ROOT).as_posix(), 'sha256': sha256}, Path(__file__).resolve())
    rows = [read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    e.update(native_refinement_calls=len(rows), preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
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
