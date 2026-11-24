"""Source-bound full-page context carried into independent native/half ROIs."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_part_admission_v1 import ROOT, inputs as part_inputs
from run_source_region_v1 import truth_for, changes
from run_topology_robust_v1 import read, write, object_hash
from cad_recovery.dataset_layout_v1 import pin, checked
from cad_recovery.raster_regions import read_prepared_region
from cad_recovery.page_frame_context_v1 import detect
from cad_recovery.frame_grid_guard_v1 import analyze, bind_context
from cad_recovery.part_admission_v1 import analyze as part_gate
from cad_recovery.grid_role_revision_v1 import sha
from cad_recovery.joint_role_metrics_v1 import score
from cad_recovery.stroke_role_metrics_v1 import to_native

BASE = ROOT/'artifacts/frame-context-profile-v1'
QUAL = ROOT/'artifacts/frame-context-qualification-v2'
BANK = ROOT/'artifacts/frame-page-bank-v2'
CONTROLS = {'path': 'artifacts/frame-context-controls-v1/contract.json',
            'sha256': '7e1e933f3ea5b0daca2d7cb89ba4b0f50b821612245f0369d1fd692035c86e5a'}
SOURCES = {'path': 'artifacts/frame-page-sources-v1/catalog.json',
           'sha256': '4f5e4997cabe267b42cfe9d02363fbf2e047a706b7331ce59af622dc36d567ce'}
PREVIOUS = {'path': 'artifacts/part-admission-release-v1.json',
            'sha256': '31edf470d66193ab61b2ff8cf66d23ca1f7a6de7db2b59304d87878cee97a729'}
ARMS = ('previous', 'base', 'local_edge', 'page_context')
NEW_CODE = ['src/cad_recovery/page_frame_context_v1.py', 'src/cad_recovery/frame_grid_guard_v1.py',
            'scripts/run_frame_context_v1.py', 'scripts/prepare_frame_context_controls_v1.py',
            'scripts/prepare_frame_page_sources_v1.py', 'tests/test_page_frame_context_v1.py', 'tests/test_frame_grid_guard_v1.py']


def old_jobs():
    prior = read(ROOT/'artifacts/part-admission-profile-v1/plan.json')
    index = {(r['case'], r['mode']): r for r in read(ROOT/'artifacts/part-admission-profile-v1/results-index.json') if not r['repeat']}
    page_rows = {r['case']: r for r in read(checked(ROOT, SOURCES))['rows']}
    result = []
    for job in prior['jobs']:
        views = []
        for v in job['views']:
            folder = (ROOT/index[job['case']['id'], v['mode']]['artifact']).parent
            source = read(folder/'source-inputs.json')
            if source.get('structural_inputs'): record = source['structural_inputs']
            else:
                parent = ROOT/source['baseline_source']
                record = {k: pin(ROOT, (parent/f).relative_to(ROOT).as_posix()) for k, f in
                          [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
            cache = {**source['admission_cache'], 'base': pin(ROOT, (folder/'no_part-roles.png').relative_to(ROOT).as_posix()),
                     'frame_row': pin(ROOT, (folder/'result.json').relative_to(ROOT).as_posix())}
            views.append({**v, 'structural_inputs': record, 'admission_cache': cache})
        result.append({**job, 'case': {**job['case'], 'split': 'development'}, 'views': views, 'new_control': False,
                       'page_source': page_rows.get(job['case']['id'], {}).get('page_source')})
    return result


def page_packet(page, folder=None):
    gray = np.array(Image.open(checked(ROOT, page['source'])).convert('L'))
    decoded = sha(gray)
    detection = detect(gray)
    packet = {'source_file': page['source'], 'decoded_sha256': decoded, 'detection': detection,
              'detector': pin(ROOT, 'src/cad_recovery/page_frame_context_v1.py')}
    if folder is not None: write(folder, packet)
    return packet, gray


def get_page(job, folder=None):
    if not job.get('page_source'): return None, 0
    if job.get('page_bank'):
        packet = read(checked(ROOT, job['page_bank']))
        checked(ROOT, packet['detector'])
        if packet['source_file'] != job['page_source']['source']: raise ValueError('Parent-page rebind')
        return packet, 0
    if folder is None: raise ValueError('Fresh page detector output required')
    packet, _ = page_packet(job['page_source'], folder)
    return packet, 1


def inputs(job, view, folder=None):
    data = part_inputs(job, view, folder)
    gray, graph, raw, heads, canonical, authority, previous, candidate, owners, evidence, cache, _, calls = data
    if cache.get('base'):
        base = np.array(Image.open(checked(ROOT, cache['base'])))
        old = read(checked(ROOT, cache['frame_row'])) if cache.get('frame_row') else None
        part_calls = 0
    else:
        base = part_gate(gray, graph, raw, previous, candidate, owners, canonical, authority=authority, evidence=evidence)['no_part']
        if folder is None: raise ValueError('New base output required')
        Image.fromarray(base).save(folder/'base.png')
        cache = {**cache, 'base': pin(ROOT, (folder/'base.png').relative_to(ROOT).as_posix())}
        old, part_calls = None, 1
    return gray, graph, raw, heads, canonical, authority, previous, base, owners, cache, old, {**calls, 'part_reference_calls': part_calls}


def frame_score(job, truth, previous, base, new):
    p, b, n = [to_native(a, job['case']['size_px']) for a in (previous, base, new)]
    delta = n != b; known = (truth >= 1) & (truth <= 9) if truth is not None else np.zeros_like(delta)
    return {'withdrawn_new_grid_pixels': int(delta.sum()),
            'wrong_grid_roles_removed': int((delta & known & (b != truth)).sum()) if truth is not None else 0,
            'frame_to_grid_errors_removed': int((delta & (truth == 7) & (b == 8)).sum()) if truth is not None else 0,
            'correct_grid_candidates_withdrawn': int((delta & known & (b == truth)).sum()) if truth is not None else 0,
            'unscored_withdrawals': int((delta & ~known).sum()),
            'remaining_new_wrong_grid_pixels': int(((n == 8) & (p != 8) & known & (truth != 8)).sum()) if truth is not None else 0}


def bind_job_source(job, packet, page_gray=None):
    page = job['page_source']; native_view = job['views'][0]
    native = np.array(Image.open(checked(ROOT, native_view['gray'])).convert('L'))
    box = page['patch_bbox']
    if page['kind'] == 'authenticated_page_overview':
        preparation = checked(ROOT, page['preparation'])
        pixels, proof = read_prepared_region(preparation.parent, box, max_pixels=4194304)
        expected = np.array(Image.fromarray(pixels).convert('L'))
    else:
        if page_gray is None: page_gray = np.array(Image.open(checked(ROOT, page['source'])).convert('L'))
        expected = page_gray[box[1]:box[3], box[0]:box[2]]
        proof = {'source': page['source'], 'unannotated_page_pixels': True}
    if not np.array_equal(native, expected): raise ValueError('ROI pixels do not match their declared original page region')
    for view in job['views']:
        gray = np.array(Image.open(checked(ROOT, view['gray'])).convert('L'))
        if gray.shape == native.shape: resized = native
        elif page['kind'] == 'authenticated_page_overview':
            resized = np.array(Image.fromarray(native).resize(gray.shape[::-1], Image.Resampling.BOX))
        else: resized = cv2.resize(native, gray.shape[::-1], interpolation=cv2.INTER_AREA)
        if not np.array_equal(gray, resized): raise ValueError('ROI resize does not match the bound source crop')
    return {'case': job['case']['id'], 'native_gray_sha256': sha(native), 'page_decoded_sha256': packet['decoded_sha256'],
            'patch_to_source': page['patch_to_source'], 'source_to_patch': page['source_to_patch'],
            'source_region_proof': proof, 'native_and_half_pixels_verified': True, 'GT_used': False}


def qualify():
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    # Resume only source-binding work interrupted before any qualification index.
    if (QUAL/'train-index.json').exists() or (BANK/'old-jobs.json').exists():
        raise ValueError('Completed qualification banks are immutable')
    BANK.mkdir(exist_ok=True); QUAL.mkdir(exist_ok=True)
    old = old_jobs(); fresh_train = [j for j in read(checked(ROOT, CONTROLS))['jobs'] if j['case']['split'] == 'train']
    prepared = read(ROOT/'artifacts/frame-context-qualification-v1/train-index.json')['jobs']
    prepared = {j['case']['id']: j for j in prepared}
    fresh_train = [{**j, 'views': prepared[j['case']['id']]['views']} for j in fresh_train]
    cache = {}; bindings = []; bound_old = []
    for job in old+fresh_train:
        if job.get('page_source'):
            key = job['page_source']['source']['sha256']
            if key not in cache:
                path = BANK/(key+'.json')
                if path.exists():
                    packet, page_gray = read(path), None
                    checked(ROOT, packet['source_file']); checked(ROOT, packet['detector'])
                else: packet, page_gray = page_packet(job['page_source'], path)
                cache[key] = pin(ROOT, path.relative_to(ROOT).as_posix())
            else: packet, page_gray = read(checked(ROOT, cache[key])), None
            bindings.append(bind_job_source(job, packet, page_gray))
            job = {**job, 'page_bank': cache[key]}
        if not job['new_control']: bound_old.append(job)
        else: fresh_train[fresh_train.index(next(j for j in fresh_train if j['case']['id'] == job['case']['id']))] = job
    write(BANK/'bindings.json', {'rows': bindings, 'source_catalog': SOURCES, 'semantic_frame_GT_read': False})
    write(BANK/'old-jobs.json', {'jobs': bound_old})
    train = []
    for job in fresh_train:
        views = []
        for v in job['views']:
            if v.get('admission_cache'):
                inputs(job, v)  # Verify every reused structural and producer pin.
                views.append(v); continue
            folder = QUAL/'bank'/job['case']['id']/v['mode']/'producer'
            data = inputs(job, v, folder)
            gray, graph, raw, heads, canonical, authority, previous, base, owners, cache_fields, old_row, calls = data
            source = folder/'baseline-source'
            record = {k: pin(ROOT, (source/f).relative_to(ROOT).as_posix()) for k, f in
                      [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
            views.append({**v, 'structural_inputs': record, 'admission_cache': cache_fields, 'capture_preparation': calls})
        train.append({**job, 'views': views}); print('prepared', job['case']['id'], flush=True)
    write(QUAL/'train-index.json', {'jobs': train, 'validation_predictions_opened': False})
    rows = []
    for job in train+[j for j in bound_old if j.get('page_source')]:
        packet, _ = get_page(job); native = None
        for view in job['views']:
            data = inputs(job, view)
            gray, graph, raw, heads, canonical, authority, previous, base, owners, _, _, _ = data
            context = bind_context(gray, graph, previous, base, owners, job['page_source'], packet['detection'], packet['decoded_sha256'],
                                   view['input_to_native'], job['page_bank']) if packet else None
            result = analyze(gray, graph, raw, previous, base, owners, authority=authority, context=context)
            if native is None: native = gray.copy()
            truth, negative = truth_for(job, native)
            rows.append({'case': job['case']['id'], 'mode': view['mode'], 'split': job['case']['split'],
                         'frames': {a: frame_score(job, truth, previous, base, result[a]) for a in ARMS[2:]},
                         'changes': {a: changes(job, truth, negative, previous, result[a]) for a in ARMS[2:]},
                         'complete': result['complete'], 'decisions': result['decisions'], 'audit': result['audit']})
    write(QUAL/'report.json', {'rows': rows, 'model': pin(ROOT, 'src/cad_recovery/frame_grid_guard_v1.py'),
          'page_detector': pin(ROOT, 'src/cad_recovery/page_frame_context_v1.py'), 'validation_predictions_opened': False})
    for split in ('train', 'development'):
        print(split, {a: {k: sum(r['frames'][a][k] for r in rows if r['split'] == split) for k in
              ('wrong_grid_roles_removed', 'frame_to_grid_errors_removed', 'correct_grid_candidates_withdrawn', 'remaining_new_wrong_grid_pixels')}
              for a in ARMS[2:]}, 'incomplete', sum(not r['complete'] for r in rows if r['split'] == split), flush=True)


def freeze():
    checked(ROOT, PREVIOUS); control = read(checked(ROOT, CONTROLS)); prior = read(ROOT/'artifacts/part-admission-profile-v1/plan.json')
    for p in prior['code']: checked(ROOT, p)
    qualification = read(QUAL/'report.json'); checked(ROOT, qualification['model']); checked(ROOT, qualification['page_detector'])
    old = read(BANK/'old-jobs.json')['jobs']
    fresh = read(QUAL/'train-index.json')['jobs']+[j for j in control['jobs'] if j['case']['split'] == 'validation']
    batches = [{'id': j['case']['id'], 'split': j['case']['split'], 'members': [j]} for j in fresh]
    for start in range(0, len(old), 8): batches.append({'id': 'development-batch-'+str(start//8).zfill(3), 'split': 'development', 'members': old[start:start+8]})
    names = list(dict.fromkeys([p['path'] for p in prior['code']]+NEW_CODE)); BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json', {'task': 'P10.2c.3b.2h.3b', 'previous': PREVIOUS, 'controls': CONTROLS, 'page_sources': SOURCES,
          'source_binding': pin(ROOT, (BANK/'bindings.json').relative_to(ROOT).as_posix()),
          'qualification': pin(ROOT, (QUAL/'report.json').relative_to(ROOT).as_posix()),
          'jobs': old+fresh, 'batches': batches, 'code': [pin(ROOT, n) for n in names],
          'criteria': control['criteria'], 'budgets': control['budgets'], 'arms': ARMS,
          'fixed_selection_order': ['page_context', 'local_edge', 'base'], 'fresh_validation_predictions_before_freeze': False,
          'parent_context_uses_pixels_not_frame_GT': True, 'batch_membership_not_model_input': True,
          'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, 'artifacts/frame-context-profile-v1/plan.json'), flush=True)


def child(path):
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    request = read(path); plan = read(checked(ROOT, request['plan']))
    for p in plan['code']: checked(ROOT, p)
    out = path.parent/'output'; out.mkdir(); rows = []
    for job in request['batch']['members']:
        case_out = out/job['case']['id']; case_out.mkdir()
        packet, detector_calls = get_page(job, case_out/'page-context.json'); native = None
        for view in job['views']:
            folder = case_out/view['mode']; folder.mkdir()
            data = inputs(job, view, folder/'producer')
            gray, graph, raw, heads, canonical, authority, previous, base, owners, cache, old, calls = data
            before = object_hash([graph, heads, canonical]); hashes = [sha(gray), sha(raw), sha(previous), sha(base), sha(owners), sha(authority['mask'])]
            context = bind_context(gray, graph, previous, base, owners, job['page_source'], packet['detection'], packet['decoded_sha256'],
                                   view['input_to_native'], job.get('page_bank') or pin(ROOT, (case_out/'page-context.json').relative_to(ROOT).as_posix())) if packet else None
            result = analyze(gray, graph, raw, previous, base, owners, authority=authority, context=context)
            if native is None: native = gray.copy()
            truth, negative = truth_for(job, native)
            maps = {'previous': previous, 'base': base, **{a: result[a] for a in ARMS[2:]}}
            metrics = {a: score(truth, to_native(m, job['case']['size_px']), negative) if truth is not None else None for a, m in maps.items()}
            delta = {a: changes(job, truth, negative, previous, m) for a, m in maps.items()}
            frames = {a: frame_score(job, truth, previous, base, m) for a, m in maps.items()}
            checks = {}
            for a in ARMS[2:]:
                m = maps[a]; changed = m != base
                checks[a] = {'only_new_grid_to_previous': bool(np.all((base[changed] == 8) & (previous[changed] != 8) & (m[changed] == previous[changed]))),
                             'known_base_nongrid_exact': bool(np.array_equal(m[base != 8], base[base != 8])),
                             'protected_exact': bool(np.array_equal(m[authority['mask'] == 2], base[authority['mask'] == 2]))}
                Image.fromarray(m).save(folder/(a+'-roles.png'))
            write(folder/'guard.json', {'decisions': result['decisions'], 'audit': result['audit'], 'bound_context': context})
            record = {'cache': cache}
            if view.get('structural_inputs'): record['structural_inputs'] = view['structural_inputs']
            else: record['baseline_source'] = (folder/'producer'/'baseline-source').relative_to(ROOT).as_posix()
            write(folder/'source-inputs.json', record)
            row = {**{k: job['case'][k] for k in ('scope', 'group_id', 'split')}, 'case': job['case']['id'], 'mode': view['mode'],
                   'repeat': request['repeat'], 'new_control': job['new_control'], 'roles': metrics, 'changes': delta, 'frames': frames,
                   'view_checks': checks, 'complete': result['complete'], 'page_context_available': packet is not None,
                   'page_frames': len(packet['detection']['frames']) if packet else 0,
                   'page_detector_complete': packet['detection']['complete'] if packet else True,
                   'source_geometry_heads_links_exact': before == object_hash([graph, heads, canonical]),
                   'input_maps_owners_authority_exact': hashes == [sha(gray), sha(raw), sha(previous), sha(base), sha(owners), sha(authority['mask'])],
                   'historical_metrics_exact': old is None or (metrics['previous'] == old['roles']['previous'] and metrics['base'] == old['roles']['no_part']),
                   'preparation': {**calls, 'page_detector_calls': detector_calls}, 'parent_or_GT_frame_labels_used_for_inference': False,
                   'source_pixels_erased': 0, 'production_changed': False, 'physical_geometry_approved': False}
            detector_calls = 0; write(folder/'result.json', row); rows.append(row)
    write(out/'result.json', {'rows': rows})


def run(sha256):
    from experiment_case_batches_v1 import run as execute
    execution = execute(ROOT, BASE, {'path': 'artifacts/frame-context-profile-v1/plan.json', 'sha256': sha256}, Path(__file__).resolve())
    rows = [read(ROOT / r['artifact']) for r in read(BASE/'results-index.json')]
    execution.update(frame_guard_calls=len(rows), preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
                     new_OCR_ML_calls=0, new_DXF_exports=0, training=False, closed_test_used=False)
    write(BASE/'execution.json', execution); print(execution, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['qualify', 'freeze', 'run', 'child'])
    p.add_argument('--sha256'); p.add_argument('--request', type=Path); a = p.parse_args()
    if a.mode == 'qualify': qualify()
    elif a.mode == 'freeze': freeze()
    elif a.mode == 'child': child(a.request)
    elif a.sha256: run(a.sha256)
    else: p.error('--sha256 required')
