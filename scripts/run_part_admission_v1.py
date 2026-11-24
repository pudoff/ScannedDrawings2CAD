"""Compare safe abstention and source-context admission over frozen CV proposals."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_source_region_v1 import ROOT, load, truth_for, changes
from run_topology_robust_v1 import read, write, object_hash
from run_grid_role_v2 import save_owner
from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.source_region_roles_v1 import analyze as propose
from cad_recovery.part_admission_v1 import analyze, provenance
from cad_recovery.grid_role_revision_v1 import sha
from cad_recovery.joint_role_metrics_v1 import score
from cad_recovery.stroke_role_metrics_v1 import to_native

BASE = ROOT/'artifacts/part-admission-profile-v1'
QUAL = ROOT/'artifacts/part-admission-qualification-v2'
PREVIOUS = {'path': 'artifacts/source-region-release-v1.json',
            'sha256': '90a29c65d2f91e9a3fbdf6240592ccc3e601f015d454ae380ee7db7618ebf1c3'}
CONTROLS = {'path': 'artifacts/part-admission-controls-v1/contract.json',
            'sha256': '76c9a2675c2d40ba8995be346336cac64eaedd6f1a91bf88efef7b5e9fb0db49'}
ARMS = ('previous', 'v1_candidate', 'no_part', 'canonical')
NEW_CODE = ['src/cad_recovery/part_admission_v1.py', 'scripts/run_part_admission_v1.py',
            'scripts/prepare_part_admission_controls_v1.py', 'scripts/experiment_case_batches_v1.py',
            'tests/test_part_admission_v1.py', 'tests/test_experiment_case_batches_v1.py']


def old_job(job):
    views = []
    for v in job['views']:
        folder = ROOT/'artifacts/source-region-profile-v1/runs'/(job['case']['id']+'-r0')/'output'/v['mode']
        if v.get('structural_inputs'):
            record = {k: v['structural_inputs'][k] for k in ('graph', 'roles', 'heads', 'links')}
        else:
            record = {k: pin(ROOT, (folder/'baseline-source'/f).relative_to(ROOT).as_posix()) for k, f in
                      [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
        cache = {k: pin(ROOT, (folder/f).relative_to(ROOT).as_posix()) for k, f in
                 [('previous', 'previous-roles.png'), ('candidate', 'guarded-roles.png'), ('owners', 'guarded-owner.png'), ('row', 'result.json')]}
        views.append({**v, 'structural_inputs': record, 'admission_cache': cache})
    return {**job, 'views': views, 'case': {**job['case'], 'split': 'development'}, 'new_control': False}


def inputs(job, view, folder=None):
    if not view.get('structural_inputs') and folder is None: raise ValueError('Fresh source output required')
    gray, graph, raw, heads, canonical, _, authority, calls = load(job, view, None if view.get('structural_inputs') else folder/'baseline-source')
    if view.get('admission_cache'):
        cache = view['admission_cache']
        previous, candidate, owners = (np.array(Image.open(checked(ROOT, cache[k]))) for k in ('previous', 'candidate', 'owners'))
        old_row = read(checked(ROOT, cache['row'])) if cache.get('row') else None
        producer_calls = 0
    else:
        result = propose(gray, graph, raw, authority=authority)
        if not result['complete']: raise ValueError('Incomplete frozen role producer')
        previous, candidate, owners = result['previous'], result['guarded'], result['owners']['guarded']
        if folder is None: raise ValueError('Fresh producer requires an output folder')
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(previous).save(folder/'previous.png'); Image.fromarray(candidate).save(folder/'candidate.png')
        save_owner(folder/'owners.png', owners, len(graph['proposals']))
        cache = {k: pin(ROOT, (folder/f).relative_to(ROOT).as_posix()) for k, f in
                 [('previous', 'previous.png'), ('candidate', 'candidate.png'), ('owners', 'owners.png')]}
        old_row, producer_calls = None, 1
    evidence = provenance(gray, graph, raw, previous, candidate, owners, canonical, cache['candidate'])
    return gray, graph, raw, heads, canonical, authority, previous, candidate, owners, evidence, cache, old_row, {**calls, 'positive_role_producer_calls': producer_calls}


def part_score(job, truth, previous, candidate, repaired):
    p, c, r = [to_native(a, job['case']['size_px']) for a in (previous, candidate, repaired)]
    proposed = (c == 1) & (p != 1)
    valid = (truth >= 1) & (truth <= 9) if truth is not None else np.zeros_like(proposed)
    true_part = truth == 1 if truth is not None else np.zeros_like(proposed)
    correct_hatch = (p == 9) & (truth == 9) if truth is not None else np.zeros_like(proposed)
    return {'new_part_pixels': int((proposed & (r == 1)).sum()),
            'wrong_new_part_pixels': int((proposed & (r == 1) & valid & ~true_part).sum()),
            'correct_new_part_pixels': int((proposed & (r == 1) & true_part).sum()),
            'correct_candidate_part_withdrawn': int((proposed & (r != 1) & true_part).sum()),
            'prior_hatching_losses_restored': int((proposed & correct_hatch & (r == 9)).sum()),
            'unscored_part_pixels': int((proposed & (r == 1) & ~valid).sum())}


def qualify():
    cv2.setNumThreads(1); cv2.setRNGSeed(0); QUAL.mkdir(exist_ok=False)
    train = []
    prepared = ROOT/'artifacts/part-admission-qualification-v1/train-index.json'
    cached = {j['case']['id']: j for j in read(prepared)['jobs']} if prepared.exists() else {}
    for job in read(checked(ROOT, CONTROLS))['jobs']:
        if job['case']['split'] != 'train': continue
        if job['case']['id'] in cached:
            train.append(cached[job['case']['id']]); continue
        views = []
        for v in job['views']:
            folder = QUAL/'bank'/job['case']['id']/v['mode']
            data = inputs(job, v, folder/'producer')
            gray, graph, raw, heads, canonical, authority, previous, candidate, owners, evidence, cache, _, calls = data
            # load() has saved newly computed geometry beside the producer.
            source = folder/'producer'/'baseline-source'
            record = {k: pin(ROOT, (source/f).relative_to(ROOT).as_posix()) for k, f in
                      [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
            views.append({**v, 'structural_inputs': record, 'admission_cache': cache, 'capture_preparation': calls})
        train.append({**job, 'views': views})
        print('prepared', job['case']['id'], flush=True)
    write(QUAL/'train-index.json', {'jobs': train, 'validation_predictions_opened': False})
    prior = read(ROOT/'artifacts/source-region-profile-v1/plan.json')
    dev = [old_job(j) for j in prior['jobs'] if j['new_control'] or j['case']['scope'] in
           ('partial_real_reviewed_ink', 'open_reference_analytical', 'reference_context_analytical_control')]
    # Always include all prior error cases via a separately pinned evaluation ledger.
    error_ids = {e['case'] for e in read(ROOT/'artifacts/source-region-profile-v1/error-audit.json')['rows']}
    present = {j['case']['id'] for j in dev}
    dev += [old_job(j) for j in prior['jobs'] if j['case']['id'] in error_ids-present]
    rows = []
    for job in train+dev:
        native = None
        for v in job['views']:
            data = inputs(job, v)
            gray, graph, raw, heads, canonical, authority, previous, candidate, owners, evidence, _, _, _ = data
            result = analyze(gray, graph, raw, previous, candidate, owners, canonical, authority=authority, evidence=evidence)
            if native is None: native = gray.copy()
            truth, negative = truth_for(job, native)
            rows.append({'case': job['case']['id'], 'mode': v['mode'], 'split': job['case']['split'],
                         'parts': {a: part_score(job, truth, previous, candidate, result[a]) for a in ARMS[2:]},
                         'changes': {a: changes(job, truth, negative, previous, result[a]) for a in ARMS[2:]},
                         'complete': result['complete'], 'decisions': result['decisions'], 'audit': result['audit']})
    write(QUAL/'report.json', {'rows': rows, 'model': pin(ROOT, 'src/cad_recovery/part_admission_v1.py'),
                              'validation_predictions_opened': False})
    for split in ('train', 'development'):
        print(split, {a: {k: sum(r['parts'][a][k] for r in rows if r['split'] == split) for k in
             ('new_part_pixels', 'wrong_new_part_pixels', 'correct_new_part_pixels', 'correct_candidate_part_withdrawn', 'prior_hatching_losses_restored')}
             for a in ARMS[2:]}, 'incomplete', sum(not r['complete'] for r in rows if r['split'] == split), flush=True)


def freeze():
    checked(ROOT, PREVIOUS)
    control = read(checked(ROOT, CONTROLS)); old = read(ROOT/'artifacts/source-region-profile-v1/plan.json')
    for p in old['code']: checked(ROOT, p)
    checked(ROOT, read(QUAL/'report.json')['model'])
    prior_jobs = [old_job(j) for j in old['jobs']]
    fresh = read(QUAL/'train-index.json')['jobs']+[j for j in control['jobs'] if j['case']['split'] == 'validation']
    batches = [{'id': j['case']['id'], 'split': j['case']['split'], 'members': [j]} for j in fresh]
    for start in range(0, len(prior_jobs), 8):
        batches.append({'id': 'development-batch-'+str(start//8).zfill(3), 'split': 'development', 'members': prior_jobs[start:start+8]})
    names = list(dict.fromkeys([p['path'] for p in old['code']]+NEW_CODE))
    BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json', {'task': 'P10.2c.3b.2h.3a', 'previous': PREVIOUS, 'controls': CONTROLS,
          'jobs': prior_jobs+fresh, 'batches': batches, 'code': [pin(ROOT, n) for n in names],
          'qualification': pin(ROOT, 'artifacts/part-admission-qualification-v2/report.json'),
          'criteria': control['criteria'], 'budgets': control['budgets'], 'arms': ARMS,
          'fixed_selection_order': ['canonical', 'no_part'], 'fresh_validation_predictions_before_freeze': False,
          'batch_membership_not_model_input': True, 'GT_or_case_names_used_for_inference': False,
          'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, 'artifacts/part-admission-profile-v1/plan.json'), flush=True)


def evaluate(job, view, folder, repeat):
    folder.mkdir(parents=True)
    data = inputs(job, view, folder/'producer')
    gray, graph, raw, heads, canonical, authority, previous, candidate, owners, evidence, cache, old, calls = data
    before = object_hash([graph, heads, canonical]); pixels = [sha(gray), sha(raw), sha(previous), sha(candidate), sha(owners), sha(authority['mask'])]
    result = analyze(gray, graph, raw, previous, candidate, owners, canonical, authority=authority, evidence=evidence)
    return data, before, pixels, result


def child(path):
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    request = read(path); plan = read(checked(ROOT, request['plan']))
    for p in plan['code']: checked(ROOT, p)
    rows = []; out = path.parent/'output'; out.mkdir()
    for job in request['batch']['members']:
        native = None
        for view in job['views']:
            folder = out/job['case']['id']/view['mode']
            data, before, pixels, result = evaluate(job, view, folder, request['repeat'])
            gray, graph, raw, heads, canonical, authority, previous, candidate, owners, evidence, cache, old, calls = data
            if native is None: native = gray.copy()
            truth, negative = truth_for(job, native)
            maps = {'previous': previous, 'v1_candidate': candidate, **{a: result[a] for a in ARMS[2:]}}
            metrics = {a: score(truth, to_native(m, job['case']['size_px']), negative) if truth is not None else None for a, m in maps.items()}
            deltas = {a: changes(job, truth, negative, previous, maps[a]) for a in ARMS}
            parts = {a: part_score(job, truth, previous, candidate, maps[a]) for a in ARMS}
            checks = {}
            for a in ARMS[2:]:
                m = maps[a]
                new_part = (candidate == 1) & (previous != 1)
                checks[a] = {'nonpart_producer_exact': bool(np.array_equal(m[~new_part], candidate[~new_part])),
                             'protected_exact': bool(np.array_equal(m[authority['mask'] == 2], previous[authority['mask'] == 2])),
                             'new_part_source_owned': bool(np.all(owners[(m == 1) & (previous != 1)] > 0))}
                Image.fromarray(m).save(folder/(a+'-roles.png'))
            retained = np.where(result['canonical'] != previous, owners, 0)
            owner_exact = save_owner(folder/'canonical-owner.png', retained, len(graph['proposals']))
            write(folder/'admission.json', {'decisions': result['decisions'], 'audit': result['audit']})
            # Producer paths may differ between repeats; only bound decoded hashes
            # enter the canonical comparison. Paths remain in a separate record.
            if view.get('admission_cache'): source_record = {'admission_cache': cache, 'structural_inputs': view['structural_inputs']}
            else: source_record = {'admission_cache': cache, 'baseline_source': (folder/'producer'/'baseline-source').relative_to(ROOT).as_posix()}
            write(folder/'source-inputs.json', source_record)
            row = {**{k: job['case'][k] for k in ('scope', 'group_id', 'split')}, 'case': job['case']['id'],
                   'mode': view['mode'], 'repeat': request['repeat'], 'new_control': job['new_control'],
                   'roles': metrics, 'changes': deltas, 'parts': parts, 'view_checks': checks,
                   'source_graph_heads_links_exact': before == object_hash([graph, heads, canonical]),
                   'source_role_owner_authority_exact': pixels == [sha(gray), sha(raw), sha(previous), sha(candidate), sha(owners), sha(authority['mask'])],
                   'historical_metrics_exact': old is None or (metrics['previous'] == old['roles']['previous'] and metrics['v1_candidate'] == old['roles']['guarded']),
                   'owner_PNG_exact': owner_exact, 'complete': result['complete'], 'preparation': calls,
                   'source_packet_hashes': {k: v for k, v in evidence.items() if k not in ('producer_kind', 'producer')},
                   'new_part_candidate_pixels': result['new_part_candidate_pixels'],
                   'production_changed': False, 'physical_geometry_approved': False, 'source_pixels_erased': 0}
            write(folder/'result.json', row); rows.append(row)
    write(out/'result.json', {'rows': rows})


def run(sha256):
    from experiment_case_batches_v1 import run as execute
    execution = execute(ROOT, BASE, {'path': 'artifacts/part-admission-profile-v1/plan.json', 'sha256': sha256}, Path(__file__).resolve())
    rows = [read(ROOT / e['artifact']) for e in read(BASE/'results-index.json')]
    execution.update(admission_calls=len(rows), preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
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
