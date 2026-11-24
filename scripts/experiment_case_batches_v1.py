"""Bounded case-independent batches; validate every case/view and both repeats."""
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from cad_recovery.dataset_layout_v1 import checked
from cad_recovery.service_profiles_v2.budget import AttemptBudget
from resume_centerline_comparison_v2 import StableOutputCounter
from run_topology_robust_v1 import read, write
from run_local_width_heads_v1 import canonical


def expected_pairs(batch):
    return {(j['case']['id'], v['mode']) for j in batch['members'] for v in j['views']}


def validate_rows(batch, rows):
    actual = [(r['case'], r['mode']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected_pairs(batch):
        raise ValueError('Missing, duplicate or unexpected case/view in child batch')


def run(root, base, plan_pin, entry):
    plan = read(checked(root, plan_pin)); policy = plan['budgets']
    index, repeats, previous, spent, children = [], [], {}, 0., 0
    for split in ('train', 'validation', 'development'):
        for p in plan['code']: checked(root, p)
        write(base/(split+'-parameter-gate.json'), {'plan': plan_pin, 'parameter_changes': False})
        for batch in [b for b in plan['batches'] if b['split'] == split]:
            for repeat in range(2):
                if spent >= policy['total_child_seconds']: raise RuntimeError('Total child budget exhausted')
                folder = base/'runs'/(batch['id']+'-r'+str(repeat))
                folder.mkdir(parents=True, exist_ok=False)
                write(folder/'request.json', {'batch': batch, 'repeat': repeat, 'plan': plan_pin})
                env = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONDONTWRITEBYTECODE': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'XDG_CACHE_HOME': str(root/'artifacts/.cache')}
                with (folder/'child.log').open('xb') as log:
                    process = subprocess.Popen([sys.executable, str(entry), 'child', '--request', str(folder/'request.json')],
                        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0, start_new_session=os.name != 'nt')
                    budget = AttemptBudget(process, folder, StableOutputCounter(), SimpleNamespace(
                        timeout_seconds=min(policy['child_seconds'], policy['total_child_seconds']-spent),
                        max_output_bytes=policy['output_MiB']*1024**2, max_output_files=policy['files']), memory_MiB=policy['RSS_MiB'])
                    try:
                        while process.poll() is None: budget.check(); time.sleep(.05)
                    finally: budget.stop()
                runtime = budget.measurements(); spent += runtime['walltime_seconds']; children += 1
                write(folder/'process.json', runtime)
                if process.returncode or runtime['budget_error']: raise RuntimeError('Child failed: '+folder.name)
                rows = read(folder/'output/result.json')['rows']
                validate_rows(batch, rows)
                for row in rows:
                    key = row['case'], row['mode']
                    if repeat: repeats.append({'case': row['case'], 'mode': row['mode'], 'identical': canonical(row) == previous[key]})
                    else:
                        if key in previous: raise ValueError('Case/view repeated in multiple batches')
                        previous[key] = canonical(row)
                    index.append({k: row[k] for k in ('case', 'scope', 'group_id', 'split', 'mode', 'repeat', 'new_control')} |
                        {'artifact': (folder/'output'/row['case']/row['mode']/'result.json').relative_to(root).as_posix(), 'runtime': runtime})
            print({'completed_batch': batch['id'], 'cases': len(batch['members']), 'split': split, 'child_seconds': round(spent, 2)}, flush=True)
    write(base/'results-index.json', index); write(base/'repeat-checks.json', repeats)
    return {'cases': len(plan['jobs']), 'batches': len(plan['batches']), 'children': children,
            'views': len(index), 'canonical_views': len(repeats), 'all_repeat_identical': all(r['identical'] for r in repeats),
            'child_wall_seconds': spent, 'batch_membership_not_model_input': True}
