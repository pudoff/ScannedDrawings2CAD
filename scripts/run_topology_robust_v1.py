"""Freeze and evaluate robust QA on unchanged geometry, never re-run fitting."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/topology-robust-v1'
CONTROLS = {'path': 'artifacts/topology-robust-v1/controls-contract.json', 'sha256': 'bad424b99cd6e7ccc134088fe9e9e123e1bcdd60f051ac40d2f14acc10f61219'}
PREVIOUS = {'path': 'artifacts/joint-shape-release-v1.json', 'sha256': 'dbc0d49dd34913e62ed86b9ff583ac69c511f615a853ef24fdfc2a2bf6078ba3'}
CORE = {'path': 'artifacts/cv-comparison-protocol-v1/contract.json', 'sha256': '815c8a5ee7f47714066206b2a7a55cbdd94328bb75fdc2ce5383b02ca300d901'}
VARIANTS = ('legacy', 'topology_only', 'compatible', 'provenance')


def read(path):
    return json.loads(path.read_text('utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def freeze():
    from cad_recovery.dataset_layout_v1 import checked, pin
    from cad_recovery.robust_topology_v1 import POLICY as topology
    from cad_recovery.robust_graph_qa_v1 import POLICY as matching
    from verify_cv_comparison_contract_v1 import verify_contract
    verify_contract(ROOT, CORE)
    checked(ROOT, PREVIOUS)
    controls = read(checked(ROOT, CONTROLS))
    checked(ROOT, controls['generator'])
    old_plan = read(ROOT/'artifacts/joint-shape-v1/plan.json')
    old_index = read(ROOT/'artifacts/joint-shape-v1/results-index.json')
    lookup = {(r['case'], r['mode']): r for r in old_index if not r['repeat']}
    jobs = []
    for old in old_plan['jobs']:
        views = []
        for view in old['views']:
            row = lookup[old['case']['id'], view['mode']]
            folder = (ROOT/row['artifact']).parent
            views.append({'mode': view['mode'], 'row': pin(ROOT, row['artifact']),
                          'candidates': pin(ROOT, (folder/'candidates.json').relative_to(ROOT).as_posix()),
                          'source': pin(ROOT, (folder/'source.json').relative_to(ROOT).as_posix())})
        jobs.append({'id': old['case']['id'], 'group_id': old['case']['group_id'], 'split': old['case']['split'],
                     'new_control': False, 'views': views})
    jobs.extend({'id': c['id'], 'group_id': c['family'], 'split': c['split'], 'new_control': True, 'control': c} for c in controls['cases'])
    names = ['src/cad_recovery/robust_topology_v1.py', 'src/cad_recovery/robust_graph_qa_v1.py',
             'scripts/run_topology_robust_v1.py', 'scripts/prepare_topology_robust_controls_v1.py',
             'tests/test_robust_topology_v1.py', 'src/cad_recovery/primitive_graph_qa_v1.py',
             'src/cad_recovery/simplification_qa_v1.py', 'src/cad_recovery/cad_validation_topology_v1.py',
             'src/cad_recovery/stroke_curve_control_v1.py', 'scripts/resume_centerline_comparison_v2.py']
    plan = {'task': 'P15.2d.2', 'previous': PREVIOUS, 'core': CORE, 'controls': CONTROLS,
        'code': [pin(ROOT, p) for p in names], 'policy': {'topology': topology, 'matching': matching}, 'jobs': jobs,
        'variants': VARIANTS, 'repeats': 2, 'new_controls_before_implementation': True,
        'prepared_before_corpus_predictions': True, 'parameter_selection_on_viewed_validation': False,
        'old_geometry': '219 prior unmasked views, source vs native_fixed and joint; no geometry changes/extractions',
        'distance_policy': 'same owner distances and 0.25 native-pixel sampling bound, threshold 2; joint reuses pinned evidence, fixed measured once per child',
        'provenance_policy': 'interval coverage, original path IDs/endpoints and actual model nodes must independently agree; bad/ambiguous provenance rejects',
        'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 1800, 'CPU_threads': 1},
        'new_geometry_extractions': 0, 'new_OCR_ML_calls': 0, 'training': False, 'closed_test_used': False,
        'physical_geometry_approved': False, 'production_changed': False}
    write(BASE/'plan.json', plan)
    print(json.dumps(pin(ROOT, 'artifacts/topology-robust-v1/plan.json')))


def add_distance(decision, errors):
    result = copy.deepcopy(decision)
    if not errors['complete']:
        result['reasons'].append('incomplete_geometric_error')
    if errors['max_px'] is not None and errors['max_px']+.25 > 2:
        result['reasons'].append('geometric_error_with_sampling_bound')
    result['accepted'] = not result['reasons']
    result['distance'] = errors
    result['selection'] = 'proposal' if result['accepted'] else 'unchanged_source'
    return result


def evaluate(source, candidates, old_graphs, graphs, distance, evidence):
    from cad_recovery.simplification_qa_v1 import compare_graphs as legacy_compare
    from cad_recovery.robust_graph_qa_v1 import compare_graphs, verified_provenance
    proof = verified_provenance(*graphs, source, candidates, evidence) if evidence is not None else None
    decisions = {'legacy': add_distance(legacy_compare(*old_graphs), distance),
                 'topology_only': add_distance(legacy_compare(*graphs), distance),
                 'compatible': add_distance(compare_graphs(*graphs), distance),
                 'provenance': add_distance(compare_graphs(*graphs, provenance=proof), distance)}
    return decisions, proof


def child(request_path):
    from cad_recovery.dataset_layout_v1 import checked
    from cad_recovery.robust_topology_v1 import build
    from cad_recovery.primitive_graph_qa_v1 import build as legacy_build
    from cad_recovery.curve_graph_metrics_v1 import native
    from cad_recovery.simplification_qa_v1 import owner_errors
    request = read(request_path)
    plan = read(checked(ROOT, request['plan']))
    for pin in plan['code']:
        checked(ROOT, pin)
    job = request['job']
    folder = request_path.parent/'output'
    folder.mkdir()
    rows = []
    if job['new_control']:
        control = job['control']
        source, proposal = control['source'], control['proposal']
        original = object_hash([source, proposal])
        old_graphs = legacy_build(source), legacy_build(proposal)
        graphs = build(source), build(proposal)
        distance = owner_errors(source, proposal)
        decisions, proof = evaluate(source, proposal, old_graphs, graphs, distance, None)
        rows.append({'case': job['id'], 'group_id': job['group_id'], 'split': job['split'], 'mode': 'analytic', 'proposal': 'analytic',
            'new_control': True, 'repeat': request['repeat'], 'graphs': {'source': graphs[0], 'proposal': graphs[1]},
            'old_graphs': {'source': old_graphs[0], 'proposal': old_graphs[1]}, 'decisions': decisions,
            'proof': proof, 'expected': control['expected'], 'geometry_sha256_before': original,
            'geometry_sha256_after': object_hash([source, proposal]), 'geometry_changed': False, 'physical_geometry_approved': False})
    else:
        for view in job['views']:
            old = read(checked(ROOT, view['row']))
            source = [native(c, old['input_to_native']) for c in read(checked(ROOT, view['source']))]
            packet = read(checked(ROOT, view['candidates']))
            before_graph = build(source)
            for variant in ('native_fixed', 'joint'):
                proposal = [native(c, old['input_to_native']) for c in packet[variant]]
                original = object_hash([source, proposal])
                after_graph = build(proposal)
                graphs = before_graph, after_graph
                old_graphs = old['source_graph'], old['graphs'][variant]
                distance = old['guard']['distance'] if variant == 'joint' else owner_errors(source, proposal)
                evidence = old['nodes'] if variant == 'joint' else [{**n, 'estimated_native_px': n['observed_native_px']} for n in old['nodes']]
                decisions, proof = evaluate(source, proposal, old_graphs, graphs, distance, evidence)
                sub = folder/(view['mode']+'-'+variant)
                sub.mkdir()
                selection = decisions['provenance']['accepted']
                selected = copy.deepcopy(proposal if selection else source)
                write(sub/'selected.json', selected)
                row = {'case': job['id'], 'group_id': job['group_id'], 'split': job['split'], 'mode': view['mode'],
                    'proposal': variant, 'repeat': request['repeat'], 'new_control': False,
                    'input': view, 'graphs': {'source': before_graph, 'proposal': after_graph},
                    'decisions': decisions, 'proof': proof, 'old_joint_accepted': old['guard']['accepted'] if variant == 'joint' else None,
                    'geometry_sha256_before': original, 'geometry_sha256_after': object_hash([source, proposal]),
                    'selection': 'proposal' if selection else 'unchanged_source',
                    'selected_geometry_sha256': object_hash(selected), 'selected_equals_exact_source': selected == source,
                    'geometry_changed': False, 'new_geometry_extractions': 0, 'semantic_ownership_confirmed': False,
                    'physical_geometry_approved': False, 'production_admitted': False}
                write(sub/'result.json', row)
                rows.append(row)
    write(folder/'result.json', {'rows': rows})


def canonical(row):
    return {k: v for k, v in row.items() if k not in ('repeat', 'artifact', 'runtime')}


def run(expected):
    from cad_recovery.dataset_layout_v1 import checked, pin
    from cad_recovery.service_profiles_v2.budget import AttemptBudget
    from resume_centerline_comparison_v2 import StableOutputCounter
    from verify_cv_comparison_contract_v1 import verify_contract
    plan = read(checked(ROOT, {'path': 'artifacts/topology-robust-v1/plan.json', 'sha256': expected}))
    verify_contract(ROOT, CORE)
    checked(ROOT, PREVIOUS)
    rows, prior, repeats, spent = [], {}, [], 0.
    for split in ('train', 'validation', 'development'):
        for pin in plan['code']:
            checked(ROOT, pin)
        write(BASE/(split+'-parameter-gate.json'), {'plan_sha256': expected, 'QA_parameter_changes': False})
        for job in [j for j in plan['jobs'] if j['split'] == split]:
            for repeat in range(2):
                if spent >= 1800:
                    raise RuntimeError('Total QA budget exhausted')
                folder = BASE/'runs'/f'{job["id"]}-r{repeat}'
                folder.mkdir(parents=True, exist_ok=False)
                write(folder/'request.json', {'job': job, 'repeat': repeat, 'plan': pin(ROOT, 'artifacts/topology-robust-v1/plan.json')})
                env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '1',
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'XDG_CACHE_HOME': str(ROOT/'artifacts/.cache')}
                with (folder/'child.log').open('xb') as log:
                    p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'child', '--request', str(folder/'request.json')],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0, start_new_session=os.name != 'nt')
                    budget = AttemptBudget(p, folder, StableOutputCounter(), SimpleNamespace(timeout_seconds=min(60, 1800-spent),
                        max_output_bytes=64*1024**2, max_output_files=128), memory_MiB=1024)
                    try:
                        while p.poll() is None:
                            budget.check()
                            time.sleep(.05)
                    finally:
                        budget.stop()
                runtime = budget.measurements()
                spent += runtime['walltime_seconds']
                write(folder/'process.json', runtime)
                if p.returncode != 0 or runtime['budget_error']:
                    raise RuntimeError('Child failed: '+folder.name)
                output = read(folder/'output/result.json')['rows']
                for row in output:
                    key = row['case'], row['mode'], row['proposal']
                    if repeat:
                        repeats.append({'case': row['case'], 'mode': row['mode'], 'proposal': row['proposal'], 'identical': canonical(row) == prior[key]})
                    else:
                        prior[key] = canonical(row)
                    # Analytic controls share their packet; corpus pairs have individual result files.
                    relative = folder/'output/result.json' if job['new_control'] else folder/'output'/(row['mode']+'-'+row['proposal'])/'result.json'
                    rows.append({k: row[k] for k in ('case', 'mode', 'proposal', 'repeat', 'new_control', 'split', 'group_id')} |
                                {'artifact': relative.relative_to(ROOT).as_posix(), 'runtime': runtime})
            print(json.dumps({'completed': job['id'], 'split': split, 'child_seconds': round(spent, 2)}), flush=True)
    write(BASE/'results-index.json', rows)
    write(BASE/'repeat-checks.json', repeats)
    write(BASE/'execution.json', {'cases': len(plan['jobs']), 'children': 2*len(plan['jobs']), 'comparison_rows': len(rows),
        'canonical_comparisons': len(repeats), 'all_repeat_identical': all(r['identical'] for r in repeats),
        'child_wall_seconds': spent, 'new_robust_graph_builds': 1410, 'new_legacy_graph_builds': 96,
        'new_owner_distance_evaluations': 486, 'cached_joint_distance_evaluations': 438,
        'new_geometry_extractions': 0, 'new_OCR_ML_calls': 0, 'training': False, 'closed_test_used': False, 'production_changed': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['freeze', 'run', 'child'])
    p.add_argument('--sha256')
    p.add_argument('--request', type=Path)
    args = p.parse_args()
    if args.mode == 'freeze':
        freeze()
    elif args.mode == 'child':
        child(args.request)
    else:
        if not args.sha256:
            p.error('--sha256 required')
        run(args.sha256)
