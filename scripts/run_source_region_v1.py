"""Source-owned positive roles; frozen blinded inference and native scoring."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_grid_role_v1 import ROOT, load, truth_for
from run_grid_role_v2 import save_owner
from run_topology_robust_v1 import read, write, object_hash
from cad_recovery.dataset_layout_v1 import pin, checked
from cad_recovery.source_region_roles_v1 import analyze, ARMS
from cad_recovery.grid_role_revision_v1 import sha
from cad_recovery.stroke_role_metrics_v1 import to_native
from cad_recovery.joint_role_metrics_v1 import score
from cad_recovery.stroke_ownership_metrics_v1 import ink_ledger
from cad_recovery.stroke_roles_v1 import foreground

BASE = ROOT/'artifacts/source-region-profile-v1'
QUAL = ROOT/'artifacts/source-region-qualification-v2'
CONTROLS = {'path': 'artifacts/source-region-controls-v1/contract.json',
            'sha256': '39cdd657234914eb99387ba821ed9b043748e2dbafcf254b40de5ad04fb8e1be'}
PREVIOUS = {'path': 'artifacts/grid-context-release-v1.json',
            'sha256': '2e5bdb178c149a52eb9591fc96b008647a3821e8d22ec01f175dcf5fb0366bc6'}
NEW_CODE = ['src/cad_recovery/source_region_roles_v1.py', 'scripts/run_source_region_v1.py',
            'scripts/prepare_source_region_controls_v1.py', 'scripts/audit_source_region_labels_v1.py',
            'tests/test_source_region_roles_v1.py']


def old_job(job):
    views = []
    for v in job['views']:
        folder = ROOT/'artifacts/grid-context-profile-v1/runs'/(job['case']['id']+'-r0')/'output'/v['mode']
        if v.get('structural_inputs'):
            record = {k: v['structural_inputs'][k] for k in ('graph', 'roles', 'heads', 'links')}
        else:
            record = {k: pin(ROOT, (folder/'baseline-source'/f).relative_to(ROOT).as_posix()) for k, f in
                      [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
        views.append({**v, 'structural_inputs': record,
                      'historical_previous': pin(ROOT, (folder/'previous-roles.png').relative_to(ROOT).as_posix()),
                      'historical_result': pin(ROOT, (folder/'result.json').relative_to(ROOT).as_posix())})
    return {**job, 'case': {**job['case'], 'split': 'development'}, 'views': views,
            'new_control': False, 'previous_control': job['new_control']}


def changes(job, truth, negative, previous, new):
    before, after = (to_native(p, job['case']['size_px']) for p in (previous, new))
    changed = before != after
    valid = (truth >= 1) & (truth <= 9) if truth is not None else np.zeros_like(changed)
    noise = (truth == 255) & (negative > 0) if truth is not None and negative is not None else np.zeros_like(changed)
    known_correct = changed & valid & (after == truth) if truth is not None else np.zeros_like(changed)
    return {'changed_pixels': int(changed.sum()), 'new_correct_assignments': int(known_correct.sum()),
            'new_correct_from_unknown': int((known_correct & (before == 255)).sum()),
            'corrected_hatching_to_positive': int((known_correct & (before == 9)).sum()),
            'new_wrong_assignments': int((changed & valid & (after != truth) & (after != 255)).sum()) if truth is not None else 0,
            'known_correct_roles_lost': int((changed & valid & (before == truth)).sum()) if truth is not None else 0,
            'correct_hatching_lost': int((changed & valid & (before == 9) & (truth == 9)).sum()) if truth is not None else 0,
            'new_noise_assignments': int((changed & noise & (after != 255)).sum()),
            'unscored_positive_assignments': int((changed & ~valid & ~noise & (after != 255)).sum()),
            'by_new_role': {str(c): int((changed & (after == c)).sum()) for c in (1, 6, 8)}}


def qualify():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    QUAL.mkdir(exist_ok=False)
    train = []
    prepared = ROOT/'artifacts/source-region-qualification-v1/train-index.json'
    cached = {j['case']['id']: j for j in read(prepared)['jobs']} if prepared.exists() else {}
    for j in read(checked(ROOT, CONTROLS))['jobs']:
        if j['case']['split'] != 'train': continue
        if j['case']['id'] in cached:
            train.append(cached[j['case']['id']])
            continue
        views = []
        for v in j['views']:
            _, _, _, _, _, record, _, calls = load(j, v, QUAL/'train-bank'/j['case']['id']/v['mode'])
            views.append({**v, 'structural_inputs': record, 'capture_preparation': calls})
        train.append({**j, 'views': views})
        print('prepared train', j['case']['id'], flush=True)
    write(QUAL/'train-index.json', {'jobs': train, 'validation_predictions_opened': False})
    old = read(ROOT/'artifacts/grid-context-profile-v1/plan.json')
    dev = [old_job(j) for j in old['jobs'] if j['new_control'] or
           j['case']['scope'] in ('partial_real_reviewed_ink', 'grid_role_analytical')]
    rows = []
    for j in train+dev:
        native = None
        for v in j['views']:
            gray, graph, roles, _, _, _, authority, _ = load(j, v)
            result = analyze(gray, graph, roles, authority=authority)
            if native is None: native = gray.copy()
            truth, negative = truth_for(j, native)
            rows.append({'case': j['case']['id'], 'mode': v['mode'], 'split': j['case']['split'],
                         'changes': {a: changes(j, truth, negative, result['previous'], result[a]) for a in ARMS},
                         'decisions': result['decisions'], 'part_decisions': result['part_decisions'],
                         'complete': result['complete'], 'audit': result['audit']})
    write(QUAL/'report.json', {'rows': rows, 'model': pin(ROOT, 'src/cad_recovery/source_region_roles_v1.py'),
                              'validation_predictions_opened': False})
    for split in ('train', 'development'):
        print(split, {a: {k: sum(r['changes'][a][k] for r in rows if r['split'] == split) for k in
                          ('new_correct_assignments', 'new_wrong_assignments', 'known_correct_roles_lost', 'unscored_positive_assignments')}
                      for a in ARMS[1:]}, 'incomplete', sum(not r['complete'] for r in rows if r['split'] == split), flush=True)


def freeze():
    checked(ROOT, PREVIOUS)
    control = read(checked(ROOT, CONTROLS))
    previous = read(ROOT/'artifacts/grid-context-profile-v1/plan.json')
    for p in previous['code']: checked(ROOT, p)
    checked(ROOT, read(QUAL/'report.json')['model'])
    jobs = [old_job(j) for j in previous['jobs']]+read(QUAL/'train-index.json')['jobs']
    jobs += [j for j in control['jobs'] if j['case']['split'] == 'validation']
    names = list(dict.fromkeys([p['path'] for p in previous['code']]+NEW_CODE))
    BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json', {'task': 'P10.2c.3b.2h.3', 'previous': PREVIOUS, 'controls': CONTROLS,
          'data_audit': pin(ROOT, 'artifacts/source-region-data-audit-v1/report.json'),
          'qualification': pin(ROOT, 'artifacts/source-region-qualification-v2/report.json'),
          'jobs': jobs, 'code': [pin(ROOT, n) for n in names], 'arms': ARMS, 'criteria': control['criteria'],
          'budgets': control['budgets'], 'fixed_selection_order': ['guarded', 'structure', 'previous'],
          'fresh_validation_predictions_before_freeze': False, 'annotation_boxes_or_targets_used_for_inference': False,
          'rejected_source_fit_merged_enabled': False, 'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, 'artifacts/source-region-profile-v1/plan.json'), flush=True)


def child(path):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    request = read(path)
    plan = read(checked(ROOT, request['plan']))
    for p in plan['code']: checked(ROOT, p)
    j, out = request['job'], path.parent/'output'
    case = j['case']
    out.mkdir()
    rows, native = [], None
    for v in j['views']:
        folder = out/v['mode']
        folder.mkdir()
        gray, graph, roles, heads, links, _, authority, calls = load(j, v, folder/'baseline-source')
        before = object_hash([graph, heads, links])
        raw_hashes = [sha(gray), sha(roles), sha(authority['mask'])]
        # No filenames, split, family, target masks, bbox labels or native view.
        result = analyze(gray, graph, roles, authority=authority)
        if native is None: native = gray.copy()
        truth, negative = truth_for(j, native)
        metrics, delta, checks, ledgers = {}, {}, {}, {}
        protected = authority['mask'] == 2
        nonhatch_known = (result['previous'] != 255) & (result['previous'] != 9)
        for arm in ARMS:
            labels = result[arm]
            metrics[arm] = score(truth, to_native(labels, case['size_px']), negative) if truth is not None else None
            delta[arm] = changes(j, truth, negative, result['previous'], labels)
            changed = labels != result['previous']
            checks[arm] = {'protected_exact': bool(np.array_equal(labels[protected], result['previous'][protected])),
                           'nonhatch_known_exact': bool(np.array_equal(labels[nonhatch_known], result['previous'][nonhatch_known])),
                           'new_roles_only_1_6_8': bool(np.isin(labels[changed], [1, 6, 8]).all()),
                           'new_pixels_have_observed_owners': arm == 'previous' or bool(np.all(result['owners'][arm][changed] > 0))}
            Image.fromarray(labels).save(folder/(arm+'-roles.png'))
            ledgers[arm] = ink_ledger(gray, labels, foreground)
        owner_checks = [save_owner(folder/(a+'-owner.png'), result['owners'][a], len(graph['proposals'])) for a in ARMS[1:]]
        write(folder/'context.json', {k: result[k] for k in ('decisions', 'part_decisions', 'audit')})
        Image.fromarray(authority['mask']).save(folder/'authority.png')
        write(folder/'authority.json', {k: value for k, value in authority.items() if k != 'mask'})
        old = read(checked(ROOT, v['historical_result'])) if v.get('historical_result') else None
        old_map = np.array(Image.open(checked(ROOT, v['historical_previous']))) if old else result['previous']
        row = {**{k: case[k] for k in ('scope', 'group_id', 'split')}, 'case': case['id'], 'mode': v['mode'],
               'repeat': request['repeat'], 'new_control': j['new_control'], 'roles': metrics, 'changes': delta,
               'view_checks': checks, 'ink_ledgers': ledgers, 'complete': result['complete'],
               'source_graph_heads_links_exact': before == object_hash([graph, heads, links]),
               'gray_raw_roles_authority_exact': raw_hashes == [sha(gray), sha(roles), sha(authority['mask'])],
               'historical_map_exact': bool(np.array_equal(old_map, result['previous'])),
               'historical_metrics_exact': old is None or metrics['previous'] == old['roles']['previous'],
               'owner_PNGs_exact': all(owner_checks), 'preparation': calls,
               'manual_review_protected_pixels': int(protected.sum()), 'GT_or_annotation_boxes_used_for_inference': False,
               'production_changed': False, 'physical_geometry_approved': False, 'source_pixels_erased': 0}
        write(folder/'result.json', row)
        rows.append(row)
    write(out/'result.json', {'rows': rows})


def run(sha256):
    from experiment_children_v1 import run as execute
    execution = execute(ROOT, BASE, {'path': 'artifacts/source-region-profile-v1/plan.json', 'sha256': sha256}, Path(__file__).resolve())
    rows = [read(ROOT / entry['artifact']) for entry in read(BASE/'results-index.json')]
    execution.update(positive_role_calls=len(rows), preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
                     new_OCR_ML_calls=0, new_DXF_exports=0, training=False, closed_test_used=False)
    write(BASE/'execution.json', execution)
    print(execution, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['qualify', 'freeze', 'run', 'child'])
    parser.add_argument('--sha256')
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.mode == 'qualify': qualify()
    elif args.mode == 'freeze': freeze()
    elif args.mode == 'child': child(args.request)
    elif args.sha256: run(args.sha256)
    else: parser.error('--sha256 required')
