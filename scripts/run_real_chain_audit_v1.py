"""Frozen model replay with target-free stage traces, followed by local GT scoring."""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.source_chain_profile_v2 import analyze, diagnostic_heads
from cad_recovery.head_junction_ownership_v1 import apply
from cad_recovery.local_role_guard_v1 import hard_head_gate
from cad_recovery.local_chain_evidence_v2 import verify
from cad_recovery.stroke_roles_v1 import foreground
from cad_recovery.stroke_role_metrics_v1 import to_native
from cad_recovery.stroke_ownership_metrics_v1 import score_roles, ink_ledger
from cad_recovery.real_chain_metrics_v1 import head_metrics, chain_metrics, POLICY
from head_junction_inputs_v1 import prepare
from prepare_real_chain_audit_v1 import ROOT, BASE, PREVIOUS, read, write
from run_topology_robust_v1 import object_hash

LABELS = {'path': 'artifacts/real-chain-audit-v1/labels-v2.json',
          'sha256': '525bd50aa9ae50588524f8df21f0d86dc1f12f0d6b1579e15398c4abe231bbf8'}


def freeze():
    selection = read(BASE / 'selection.json'); labels = read(checked(ROOT, LABELS))
    previous = read(checked(ROOT, selection['previous_plan']))
    index = {(r['case'], r['mode']): r for r in read(ROOT / 'artifacts/head-junction-profile-v1/results-index.json') if not r['repeat']}
    jobs = []
    for old in selection['old_jobs']:
        views = []
        for view in old['views']:
            folder = (ROOT / index[old['case']['id'], view['mode']]['artifact']).parent
            reference = {n: pin(ROOT, (folder / file).relative_to(ROOT).as_posix())
                         for n, file in [('row', 'result.json'), ('heads', 'heads.json'),
                                         ('routes', 'routes-roles.png'), ('joint', 'joint-roles.png'), ('guarded', 'guarded-roles.png')]}
            views.append({**view, 'audit_reference': reference})
        jobs.append({**old, 'views': views, 'case': {**old['case'], 'split': 'development'}, 'new_control': False})
    for crop in read(BASE / 'crop-recipe.json'):
        target = next(r for r in labels['records'] if r['id'] == crop['id'])
        views = [{'mode': mode, 'gray': crop[key], 'input_to_native': matrix}
                 for mode, key, matrix in [('native_crop', 'native', [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                           ('half_resolution_crop', 'half', [[2, 0, .5], [0, 2, .5], [0, 0, 1]])]]
        jobs.append({'case': {'id': crop['id'], 'scope': 'real_chain_new_diagnostic', 'group_id': crop['group_id'],
                              'size_px': target['size_px'], 'split': 'validation', 'original_split': 'validation',
                              'source': crop['source'], 'source_box': crop['source_box']},
                     'views': views, 'new_control': True})
    names = [p['path'] for p in previous['code']] + ['src/cad_recovery/real_chain_metrics_v1.py',
             'scripts/run_real_chain_audit_v1.py', 'tests/test_real_chain_metrics_v1.py']
    write(BASE / 'plan.json', {'task': 'P10.2c.3b.2e', 'previous': PREVIOUS,
          'selection': pin(ROOT, 'artifacts/real-chain-audit-v1/selection.json'), 'labels': LABELS,
          'jobs': jobs, 'code': [pin(ROOT, n) for n in dict.fromkeys(names)], 'repeats': 2,
          'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 600},
          'policy': POLICY, 'fresh_predictions_before_label_freeze': False,
          'model_changes': False, 'physical_geometry_approved': False, 'production_changed': False})
    print(pin(ROOT, 'artifacts/real-chain-audit-v1/plan.json'), flush=True)


def stage_trace(gray, previous, heads, text):
    available, events = hard_head_gate(heads, text)
    boundaries = [b for b in previous['boundaries']['boundaries']
                  if max(b['box'][2]-b['box'][0], b['box'][3]-b['box'][1]) >= .15 * min(gray.shape)]
    result = verify(previous['proposals'], previous['tracks'], boundaries, available, foreground(gray))
    return {**result, 'hard_head_events': events, 'all_boundary_count': len(previous['boundaries']['boundaries']),
            'large_boundary_count': len(boundaries)}


def child(path):
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    request = read(path); plan = read(checked(ROOT, request['plan'])); job = request['job']; case = job['case']
    for p in plan['code']: checked(ROOT, p)
    output = path.parent / 'output'; output.mkdir(); rows = []
    for view in job['views']:
        gray, source, geometry, inherited, previous, local, heads, extra, calls = prepare(ROOT, view)
        source_hash = object_hash([source, geometry, previous, heads, extra])
        pixel_hash = hashlib.sha256(gray.tobytes()).hexdigest()
        legacy_packet = analyze(gray, previous, inherited, local['local']['text_mask'], heads, diagnostic_heads=extra)
        advanced = diagnostic_heads(gray, heads)
        joint = analyze(gray, previous, inherited, local['local']['text_mask'], advanced['candidates'], diagnostic_heads=extra)
        guarded = apply(gray, previous, inherited, joint, advanced['candidates'])
        traces = {name: stage_trace(gray, previous, hh, local['local']['text_mask'])
                  for name, hh in [('legacy', heads), ('advanced', advanced['candidates'])]}
        # No annotation or target mask is opened before all proposals and traces exist.
        labels = read(checked(ROOT, plan['labels']))
        review = next((r for r in labels['records'] if r['id'] == case['id']), None)
        if case['id'] in labels['fully_reviewed_negative_head_cases']:
            w, h = case['size_px']; review = {'heads': [], 'head_scopes': [[0, 0, w, h]], 'chains': []}
        folder = output / view['mode']; folder.mkdir()
        maps = {'routes': inherited, 'joint': joint['pixel_roles'], 'guarded': guarded['pixel_roles']}
        role_scores = {}; original_scores = {}; ledgers = {}; old_maps_identical = True; old_heads_identical = True
        original_truth = np.array(Image.open(checked(ROOT, job['truth']))) if job.get('truth') else None
        target = np.array(Image.open(checked(ROOT, review['roles_mask']))) if review and review.get('roles_mask') else None
        for name, values in maps.items():
            Image.fromarray(values).save(folder / (name + '-roles.png'))
            native = to_native(values, case['size_px']); ledgers[name] = ink_ledger(gray, values, foreground)
            role_scores[name] = score_roles(target, native) if target is not None else None
            original_scores[name] = score_roles(original_truth, native) if original_truth is not None else None
        old_reference = view.get('audit_reference')
        if old_reference:
            old_row = read(checked(ROOT, old_reference['row']))
            old_maps_identical = all(np.array_equal(values, np.array(Image.open(checked(ROOT, old_reference[name])))) for name, values in maps.items())
            old_heads_identical = advanced == read(checked(ROOT, old_reference['heads']))
            assert original_scores == old_row['roles'], 'Original role scores changed'
            assert joint['dimension_chains'] == old_row['dimension_chains'], 'Frozen chains changed'
        hm = {}; cm = {}
        if review is not None:
            for name, hh, packet in [('legacy', heads, legacy_packet), ('advanced', advanced['candidates'], guarded)]:
                hm[name] = head_metrics(review, hh, view['input_to_native'])
                cm[name] = chain_metrics(review, hh, view['input_to_native'], packet, previous, traces[name], hm[name])
        write(folder / 'heads.json', advanced)
        write(folder / 'observations.json', {'tracks': previous['tracks'], 'source_intervals': previous['proposals'],
              'boundaries': previous['boundaries'], 'stage_traces': traces,
              'legacy_chains': legacy_packet['dimension_chains'], 'joint_chains': joint['dimension_chains']})
        overlay = np.repeat(gray[:, :, None], 3, axis=2)
        for cls, color in [(1, [20, 100, 220]), (2, [225, 110, 20]), (3, [160, 55, 180])]:
            overlay[guarded['pixel_roles'] == cls] = color
        Image.fromarray(overlay).save(folder / 'guarded-overlay.png')
        row = {k: case[k] for k in ('id', 'scope', 'group_id', 'split')}; row['case'] = row.pop('id')
        row.update(mode=view['mode'], repeat=request['repeat'], new_control=job['new_control'],
          original_split=case.get('original_split', 'development'), roles=role_scores, original_roles=original_scores,
          ink_ledgers=ledgers, heads=hm, chains=cm, total_heads=len(advanced['candidates']),
          total_chains=len(joint['dimension_chains']), trace_incomplete=any(t['incomplete'] for t in traces.values()),
          model_incomplete=advanced['search_incomplete'] or guarded['search_incomplete'] or legacy_packet['search_incomplete'],
          source_observations_unchanged=source_hash == object_hash([source, geometry, previous, heads, extra]),
          source_pixels_unchanged=pixel_hash == hashlib.sha256(gray.tobytes()).hexdigest(),
          old_maps_identical=old_maps_identical, old_heads_identical=old_heads_identical,
          head_prefix_unchanged=advanced['prefix_unchanged'], guard_chains_unchanged=joint['dimension_chains'] == guarded['dimension_chains'],
          preparation=calls, new_OCR_ML_calls=0, new_DXF_exports=0, production_changed=False, physical_geometry_approved=False)
        write(folder / 'result.json', row); rows.append(row)
    write(output / 'result.json', {'rows': rows})


def run(sha):
    from experiment_children_v1 import run as execute
    result = execute(ROOT, BASE, {'path': 'artifacts/real-chain-audit-v1/plan.json', 'sha256': sha}, Path(__file__).resolve())
    result.update(model_changes=False, production_changed=False, training=False, closed_test_used=False)
    write(BASE / 'execution.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'run', 'child'])
    parser.add_argument('--request', type=Path); parser.add_argument('--sha256'); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'child': child(args.request)
    elif args.sha256: run(args.sha256)
    else: parser.error('--sha256 required')
