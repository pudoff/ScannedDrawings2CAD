"""Frozen, isolated comparison of source-fit and source-merged grid contexts."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from diagnose_grid_context_v1 import ROOT, CONTROLS, previous_job, load
from run_grid_role_v1 import truth_for, revision_score
from run_grid_role_v2 import save_owner
from run_topology_robust_v1 import read, write, object_hash
from cad_recovery.dataset_layout_v1 import pin, checked
from cad_recovery.grid_context_v1 import analyze
from cad_recovery.grid_role_revision_v1 import sha
from cad_recovery.stroke_role_metrics_v1 import to_native
from cad_recovery.joint_role_metrics_v1 import score
from cad_recovery.stroke_ownership_metrics_v1 import ink_ledger
from cad_recovery.stroke_roles_v1 import foreground

BASE = ROOT/'artifacts/grid-context-profile-v1'
QUAL = ROOT/'artifacts/grid-context-qualification-v1'
PREVIOUS = {'path': 'artifacts/grid-role-release-v1.json',
            'sha256': 'af352392e6a5d40861872aa34019c5c0b827a5647ce1a8be946960340368bbaf'}
ARMS = ('baseline', 'previous', 'source_fit', 'source_merged')
NEW_CODE = ['scripts/prepare_grid_context_controls_v1.py', 'scripts/diagnose_grid_context_v1.py',
            'scripts/run_grid_context_v1.py', 'src/cad_recovery/grid_context_v1.py', 'tests/test_grid_context_v1.py']


def qualify():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    QUAL.mkdir(exist_ok=False)
    jobs = read(ROOT/'artifacts/grid-context-diagnosis-v1/train-index.json')['jobs']
    dev = [previous_job(j) for j in read(ROOT/'artifacts/grid-role-profile-v2/plan.json')['jobs']
           if j['new_control'] or j['case']['scope'] == 'partial_real_reviewed_ink']
    rows = []
    for j in jobs+dev:
        native = None
        for v in j['views']:
            gray, graph, roles, _, _, _, authority, _ = load(j, v)
            result = analyze(gray, graph, roles, authority=authority)
            if native is None:
                native = gray.copy()
            truth, negative = truth_for(j, native)
            rows.append({'case': j['case']['id'], 'mode': v['mode'], 'split': j['case']['split'],
                         'revisions': {a: revision_score(j, truth, negative, result['previous'], result[a]) for a in ARMS[2:]},
                         'complete': result['complete'], 'context_evidence': result['context_evidence'],
                         'structures': result['structures'], 'details': result['details']})
    write(QUAL/'report.json', {'rows': rows, 'fresh_validation_opened': False,
                              'model': pin(ROOT, 'src/cad_recovery/grid_context_v1.py')})
    for split in ('train', 'development'):
        print(split, {a: {k: sum(r['revisions'][a][k] for r in rows if r['split'] == split)
                          for k in ('withdrawn_pixels', 'wrong_roles_removed', 'known_correct_roles_lost', 'unscored_withdrawals')}
                      for a in ARMS[2:]}, 'incomplete', sum(not r['complete'] for r in rows if r['split'] == split), flush=True)


def freeze():
    checked(ROOT, PREVIOUS)
    control = read(checked(ROOT, CONTROLS))
    old = read(ROOT/'artifacts/grid-role-profile-v2/plan.json')
    for p in old['code']:
        checked(ROOT, p)
    checked(ROOT, read(QUAL/'report.json')['model'])
    jobs = [previous_job(j) for j in old['jobs']]
    jobs += read(ROOT/'artifacts/grid-context-diagnosis-v1/train-index.json')['jobs']
    jobs += [j for j in control['jobs'] if j['case']['split'] == 'validation']
    names = list(dict.fromkeys([p['path'] for p in old['code']]+NEW_CODE))
    BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json', {'task': 'P10.2c.3b.2h.2', 'previous': PREVIOUS, 'controls': CONTROLS,
          'jobs': jobs, 'code': [pin(ROOT, n) for n in names], 'arms': ARMS,
          'criteria': control['criteria'], 'budgets': control['budgets'],
          'qualification': pin(ROOT, 'artifacts/grid-context-qualification-v1/report.json'),
          'fresh_validation_predictions_before_freeze': False,
          'fixed_selection_order_if_qualified': ['source_merged', 'source_fit', 'previous'],
          'historical_maps_immutable': True, 'half_model_uses_native_or_GT': False,
          'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, 'artifacts/grid-context-profile-v1/plan.json'), flush=True)


def child(path):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    request = read(path)
    plan = read(checked(ROOT, request['plan']))
    for p in plan['code']:
        checked(ROOT, p)
    j = request['job']
    case = j['case']
    out = path.parent/'output'
    out.mkdir()
    native, rows = None, []
    for v in j['views']:
        folder = out/v['mode']
        folder.mkdir()
        gray, graph, roles, heads, links, _, authority, calls = load(j, v, folder/'baseline-source')
        before = object_hash([graph, heads, links])
        pixels, original_roles, original_mask = sha(gray), sha(roles), sha(authority['mask'])
        # Inference receives this raster, source intervals and authority only.
        # Native images, targets, heads and dimension links are not model inputs.
        result = analyze(gray, graph, roles, authority=authority)
        if native is None:
            native = gray.copy()
        truth, negative = truth_for(j, native)
        metrics, revisions, incremental, checks, ledgers = {}, {}, {}, {}, {}
        protected = authority['mask'] == 2
        previous_row = read(checked(ROOT, v['previous_result'])) if v.get('previous_result') else None
        old_map = np.array(Image.open(checked(ROOT, v['previous_guarded']))) if v.get('previous_guarded') else result['previous']
        for a in ARMS:
            labels = result[a]
            metrics[a] = score(truth, to_native(labels, case['size_px']), negative) if truth is not None else None
            revisions[a] = revision_score(j, truth, negative, roles, labels)
            incremental[a] = revision_score(j, truth, negative, result['previous'], labels) if a != 'baseline' else None
            delta = labels != roles
            checks[a] = {'only_hatching_to_unknown': bool(np.all((roles[delta] == 9) & (labels[delta] == 255))),
                         'protected_pixels_exact': bool(np.array_equal(labels[protected], roles[protected])),
                         'previous_revisions_retained': a == 'baseline' or bool(np.all(labels[result['previous'] == 255] == 255))}
            ledgers[a] = ink_ledger(gray, labels, foreground)
            Image.fromarray(labels).save(folder/(a+'-roles.png'))
        owners_exact = all(save_owner(folder/(a+'-owner.png'), result['owners'][a], len(graph['proposals'])) for a in ARMS[1:])
        write(folder/'context.json', {k: result[k] for k in ('context_evidence', 'structures', 'details')})
        write(folder/'authority.json', {k: value for k, value in authority.items() if k != 'mask'})
        Image.fromarray(authority['mask']).save(folder/'authority.png')
        row = {**{k: case[k] for k in ('scope', 'group_id', 'split')}, 'case': case['id'], 'mode': v['mode'],
               'repeat': request['repeat'], 'new_control': j['new_control'], 'roles': metrics, 'revisions': revisions,
               'incremental': incremental, 'view_checks': checks, 'ink_ledgers': ledgers,
               'source_graph_heads_links_exact': before == object_hash([graph, heads, links]),
               'gray_roles_authority_exact': (pixels, original_roles, original_mask) == (sha(gray), sha(roles), sha(authority['mask'])),
               'old_guarded_map_exact': np.array_equal(old_map, result['previous']),
               'historical_metrics_exact': previous_row is None or (
                   metrics['baseline'] == previous_row['roles']['baseline'] and metrics['previous'] == previous_row['roles']['guarded']),
               'owner_pngs_exact': owners_exact, 'complete': result['complete'], 'preparation': calls,
               'authority': {'protected_pixels': int(protected.sum()), 'GT_used_for_authority': False},
               'structure_counts': {a: len(result['structures'][a]) for a in ARMS[1:]},
               'fit_counts': {a: {'tracks': len(result['context_evidence'][a].get('tracks', [])),
                                 'merges': len(result['context_evidence'][a].get('merges', []))} for a in ARMS[2:]},
               'production_changed': False, 'physical_geometry_approved': False, 'source_pixels_erased': 0,
               'half_inference_read_native_or_GT': result['native_or_GT_inputs']}
        write(folder/'result.json', row)
        rows.append(row)
    write(out/'result.json', {'rows': rows})


def run(sha256):
    from experiment_children_v1 import run as execute
    execution = execute(ROOT, BASE, {'path': 'artifacts/grid-context-profile-v1/plan.json', 'sha256': sha256}, Path(__file__).resolve())
    rows = [read(ROOT[r['artifact']]) for r in read(BASE/'results-index.json')]
    execution.update(context_calls=len(rows), preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
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
