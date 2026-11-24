"""Bounded isolated children for immutable paired-raster comparison plans."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.service_profiles_v2.budget import AttemptBudget
from resume_centerline_comparison_v2 import StableOutputCounter
from run_local_width_heads_v1 import read, write, canonical


def run(root, base, plan_pin, entry):
    root, base, entry = Path(root), Path(base), Path(entry)
    plan = read(checked(root, plan_pin)); policy = plan['budgets']
    index, previous, repeats, spent, children = [], {}, [], 0., 0
    for split in ('train', 'validation', 'development'):
        for p in plan['code']: checked(root, p)
        write(base/(split+'-parameter-gate.json'), {'plan': plan_pin, 'parameter_changes': False})
        for job in [j for j in plan['jobs'] if j['case']['split'] == split]:
            for repeat in range(2):
                if spent >= policy['total_child_seconds']: raise RuntimeError('Total child budget exhausted')
                folder = base/'runs'/f'{job["case"]["id"]}-r{repeat}'; folder.mkdir(parents=True, exist_ok=False)
                write(folder/'request.json', {'job': job, 'repeat': repeat, 'plan': plan_pin})
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
                if process.returncode != 0 or runtime['budget_error']: raise RuntimeError('Child failed: '+folder.name)
                rows = read(folder/'output/result.json')['rows']
                if len(rows) != len(job['views']): raise RuntimeError('Incomplete child views')
                for row in rows:
                    key = row['case'], row['mode']
                    if repeat: repeats.append({'case': row['case'], 'mode': row['mode'], 'identical': canonical(row) == previous[key]})
                    else: previous[key] = canonical(row)
                    index.append({k: row[k] for k in ('case', 'scope', 'group_id', 'split', 'mode', 'repeat', 'new_control')} |
                        {'artifact': (folder/'output'/row['mode']/'result.json').relative_to(root).as_posix(), 'runtime': runtime})
            print(json.dumps({'completed': job['case']['id'], 'split': split, 'child_seconds': round(spent, 2)}), flush=True)
    write(base/'results-index.json', index); write(base/'repeat-checks.json', repeats)
    return {'cases': len(plan['jobs']), 'children': children, 'views': len(index), 'canonical_views': len(repeats),
        'all_repeat_identical': all(r['identical'] for r in repeats), 'child_wall_seconds': spent}
