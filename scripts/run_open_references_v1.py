"""Full-corpus paired local-reference experiment on immutable source observations."""
import argparse
from pathlib import Path
import hashlib
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.open_reference_chains_v1 import extend, contacts, POLICY
from cad_recovery.real_chain_metrics_v1 import head_metrics, chain_metrics
from cad_recovery.source_chain_profile_v2 import analyze, diagnostic_heads
from cad_recovery.head_junction_ownership_v1 import apply
from head_junction_inputs_v1 import prepare
from open_reference_inputs_v1 import reconstruct
from qualify_open_references_v1 import CONTROLS
from run_topology_robust_v1 import read, write, object_hash

ROOT = Path(__file__).resolve().parents[1]; BASE = ROOT / 'artifacts/open-reference-profile-v1'
PREVIOUS = {'path': 'artifacts/real-chain-audit-release-v1.json',
            'sha256': '41388a6eda1e5ba3021a70ef432f540c8ac0b898d5e5ea8413ed99f6edbc1a82'}
LABELS = {'path': 'artifacts/real-chain-audit-v1/admitted-labels.json',
          'sha256': '0803a073e62ff7cc74355766d268ebd0bd6dede625f92deffe0d2949f5967589'}


def freeze():
    checked(ROOT, PREVIOUS); contract = read(checked(ROOT, CONTROLS))
    old = read(ROOT / 'artifacts/head-junction-profile-v1/plan.json')
    old_index = {(r['case'], r['mode']): r for r in read(ROOT / 'artifacts/head-junction-profile-v1/results-index.json') if not r['repeat']}
    real = read(ROOT / 'artifacts/real-chain-audit-v1/plan.json')
    real_index = {(r['case'], r['mode']): r for r in read(ROOT / 'artifacts/real-chain-audit-v1/results-index.json') if not r['repeat']}
    train = {(r['case']['id'], r['view']['mode']): r for r in read(ROOT / 'artifacts/open-reference-qualification-v1/index.json')}
    jobs = []
    for job in old['jobs'] + [j for j in real['jobs'] if j['new_control']]:
        fresh_real = job['case']['id'].startswith('real-chain-new-'); views = []
        for v in job['views']:
            source = (real_index if fresh_real else old_index)[job['case']['id'], v['mode']]
            folder = (ROOT / source['artifact']).parent
            cache = {k: pin(ROOT, (folder / name).relative_to(ROOT).as_posix()) for k, name in
                     [('row', 'result.json'), ('heads', 'heads.json'), ('roles', 'guarded-roles.png'),
                      ('source', 'observations.json' if fresh_real else 'source-intervals.json')]}
            if (job['case']['id'], v['mode']) in real_index:
                rf = (ROOT / real_index[job['case']['id'], v['mode']]['artifact']).parent
                cache['real_graph_reference'] = pin(ROOT, (rf / 'observations.json').relative_to(ROOT).as_posix())
            views.append({'mode': v['mode'], 'gray': v['gray'], 'input_to_native': v['input_to_native'],
                          'frozen_source': cache, 'rich_source': fresh_real})
        c = job['case']; case = {k: c[k] for k in ('id', 'scope', 'group_id', 'size_px')}
        case.update(split='development', original_split=c.get('original_split', c.get('previous_split', c['split'])))
        if c.get('labels'): case['labels'] = c['labels']
        jobs.append({'case': case, 'views': views, 'new_control': False})
    for job in contract['jobs']:
        views = []
        for v in job['views']:
            q = train.get((job['case']['id'], v['mode']))
            if q:
                folder = ROOT / q['folder']
                cache = {k: pin(ROOT, (folder / name).relative_to(ROOT).as_posix()) for k, name in
                         [('graph', 'graph.json'), ('heads', 'heads.json'), ('roles', 'roles.png'), ('base', 'base.json')]}
                views.append({**v, 'qualification_source': cache})
            else: views.append(v)
        jobs.append({**job, 'views': views})
    names = [p['path'] for p in real['code']] + ['src/cad_recovery/open_reference_chains_v1.py',
            'scripts/open_reference_inputs_v1.py', 'scripts/run_open_references_v1.py',
            'scripts/prepare_open_reference_controls_v1.py', 'tests/test_open_reference_chains_v1.py']
    write(BASE / 'plan.json', {'task': 'P10.2c.3b.2f', 'previous': PREVIOUS, 'controls': CONTROLS, 'labels': LABELS,
          'code': [pin(ROOT, n) for n in dict.fromkeys(names)], 'jobs': jobs, 'policy': POLICY,
          'criteria': contract['criteria'], 'budgets': contract['budgets'], 'repeats': 2,
          'qualification': pin(ROOT, 'artifacts/open-reference-qualification-v1/report-v2.json'),
          'new_validation_opened_before_freeze': False, 'pixel_roles_changed': False,
          'head_or_geometry_models_changed': False, 'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, 'artifacts/open-reference-profile-v1/plan.json'), flush=True)


def inputs(view):
    gray = np.array(Image.open(checked(ROOT, view['gray'])).convert('L')); original_exact = True
    calls = {'graph_reconstructions': 0, 'geometry_extractions': 0, 'advanced_head_calls': 0, 'baseline_profile_calls': 0}
    if view.get('frozen_source'):
        cache = view['frozen_source']; row = read(checked(ROOT, cache['row']))
        source = read(checked(ROOT, cache['source'])); intervals = source['source_intervals'] if view['rich_source'] else source
        previous = reconstruct(gray, intervals, view['input_to_native']); calls['graph_reconstructions'] = 1
        heads = read(checked(ROOT, cache['heads']))['candidates']; roles = np.array(Image.open(checked(ROOT, cache['roles'])))
        if view['rich_source']:
            chains = source['joint_chains']; excluded = {e['id'] for e in source['stage_traces']['advanced']['hard_head_events']}
            allowed = {h['id'] for h in heads} - excluded
        else: chains = row['dimension_chains']; allowed = set(row['details']['head_ids_used'])
        base = {'pixel_roles': roles, 'dimension_chains': chains, 'search_incomplete': previous['search_incomplete']}
        if cache.get('real_graph_reference'):
            reference = read(checked(ROOT, cache['real_graph_reference']))
            original_exact = previous['tracks'] == reference['tracks'] and previous['boundaries'] == reference['boundaries']
    elif view.get('qualification_source'):
        cache = view['qualification_source']; previous = read(checked(ROOT, cache['graph']))
        heads = read(checked(ROOT, cache['heads']))['candidates']; base = read(checked(ROOT, cache['base']))
        base['pixel_roles'] = np.array(Image.open(checked(ROOT, cache['roles']))); allowed = set(base['head_ids_used'])
    else:
        gray, source, geometry, inherited, previous, local, legacy_heads, extra, prep = prepare(ROOT, view)
        advanced = diagnostic_heads(gray, legacy_heads); heads = advanced['candidates']
        raw = analyze(gray, previous, inherited, local['local']['text_mask'], heads, diagnostic_heads=extra)
        base = apply(gray, previous, inherited, raw, heads); allowed = set(base['head_ids_used'])
        calls.update(geometry_extractions=prep['geometry_extractions'], advanced_head_calls=1, baseline_profile_calls=2)
        base['search_incomplete'] |= advanced['search_incomplete']
    return gray, previous, base, heads, [h for h in heads if h['id'] in allowed], original_exact, calls


def control_review(case):
    refs = case['labels']['arrows']; a, b = np.asarray(refs[0]['tip']), np.asarray(refs[1]['tip'])
    heads = [{'id': h['id'], 'tip_px': h['tip'], 'direction': h['direction']} for h in refs]
    chains = []
    if case['labels']['expected_dimension_chains']:
        chains = [{'id': 'dimension', 'head_ids': [h['id'] for h in refs], 'shaft': [(.75*a+.25*b).tolist(), (.25*a+.75*b).tolist()],
                   'feet_px': case['labels']['feet_px'], 'head_scope': [0, 0, *case['size_px']],
                   'topology': 'two_extensions', 'reference_kinds': ['contour', 'contour'], 'whole_part_contour_visible': False}]
    return {'heads': heads, 'head_scopes': [[0, 0, *case['size_px']]], 'chains': chains}


def child(path):
    cv2.setNumThreads(1); cv2.setRNGSeed(0)
    request = read(path); plan = read(checked(ROOT, request['plan'])); job = request['job']; case = job['case']
    for p in plan['code']: checked(ROOT, p)
    out = path.parent / 'output'; out.mkdir(); rows = []
    for view in job['views']:
        gray, graph, base, heads, used, graph_exact, calls = inputs(view)
        source_sha = object_hash([graph, heads, base['dimension_chains']]); pixels = hashlib.sha256(gray.tobytes()).hexdigest()
        labels_sha = hashlib.sha256(base['pixel_roles'].tobytes()).hexdigest()
        result = extend(gray, graph, base, used)
        prefix = result['dimension_chains'][:len(base['dimension_chains'])] == base['dimension_chains']
        # No target enters inference or graph reconstruction; only score completed outputs.
        targets = read(checked(ROOT, plan['labels']))
        review = next((r for r in targets['records'] if r['id'] == case['id']), None)
        if job['new_control']: review = control_review(case)
        hh = {}; cc = {}; trace = {'contacts': {k: [h['id'] for h in v] for k, v in contacts(graph['tracks'], used, gray < 240).items()},
                                  'eligible_boundaries': [b for b in graph['boundaries']['boundaries'] if not b['frame_hypothesis']]}
        if review:
            hh = head_metrics(review, heads, view['input_to_native'])
            for arm, packet in [('base', base), ('candidate', result),
                                ('new_only', {**result, 'dimension_chains': result['open_reference_chains']})]:
                cc[arm] = chain_metrics(review, heads, view['input_to_native'], packet, graph, trace, hh)
        folder = out / view['mode']; folder.mkdir()
        write(folder / 'new-chains.json', result['open_reference_chains'])
        write(folder / 'features.json', {k: v for k, v in graph.items() if k != 'proposals'})
        if not view.get('frozen_source') and not view.get('qualification_source'):
            write(folder / 'source-intervals.json', graph['proposals']); write(folder / 'heads.json', heads)
        row = {k: case[k] for k in ('scope', 'group_id', 'split')}
        row.update(case=case['id'], mode=view['mode'], repeat=request['repeat'], new_control=job['new_control'],
          original_split=case.get('original_split', case['split']), heads=hh, chains=cc,
          expected_dimension_chains=case.get('labels', {}).get('expected_dimension_chains'),
          previous_chains=len(base['dimension_chains']), new_chains=len(result['open_reference_chains']),
          original_chain_prefix_unchanged=prefix, source_observations_unchanged=source_sha == object_hash([graph, heads, base['dimension_chains']]),
          source_pixels_unchanged=pixels == hashlib.sha256(gray.tobytes()).hexdigest(),
          pixel_roles_unchanged=labels_sha == hashlib.sha256(result['pixel_roles'].tobytes()).hexdigest(),
          reconstructed_real_graph_exact=graph_exact, audit=result['open_reference_audit'], junction_audit=result['open_reference_junction_audit'],
          rejections=result['open_reference_rejections'], incomplete=result['search_incomplete'], preparation=calls,
          production_changed=False, physical_geometry_approved=False, source_pixels_erased=0)
        write(folder / 'result.json', row); rows.append(row)
    write(out / 'result.json', {'rows': rows})


def run(sha):
    from experiment_children_v1 import run as execute
    execution = execute(ROOT, BASE, {'path': 'artifacts/open-reference-profile-v1/plan.json', 'sha256': sha}, Path(__file__).resolve())
    rows = [read(ROOT / r['artifact']) for r in read(BASE / 'results-index.json')]
    execution['preparation_calls'] = {k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']}
    execution.update(open_reference_calls=len(rows), new_OCR_ML_calls=0, new_DXF_exports=0, training=False, closed_test_used=False)
    write(BASE / 'execution.json', execution); print(execution, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'run', 'child'])
    parser.add_argument('--sha256'); parser.add_argument('--request', type=Path); args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    elif args.mode == 'child': child(args.request)
    elif args.sha256: run(args.sha256)
    else: parser.error('--sha256 required')
