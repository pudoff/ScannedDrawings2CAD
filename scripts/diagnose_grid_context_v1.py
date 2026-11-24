"""Separate native/half context failures on development and prospective train."""
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from run_grid_role_v1 import ROOT, load
from run_topology_robust_v1 import read, write
from cad_recovery.dataset_layout_v1 import checked, pin
from cad_recovery.grid_context_v1 import source_track_context
from cad_recovery.grid_role_revision_v1 import grids, analyze
from cad_recovery.stroke_roles_v1 import foreground

BASE = ROOT/'artifacts/grid-context-diagnosis-v1'
CONTROLS = {'path': 'artifacts/grid-context-controls-v1/contract.json',
            'sha256': '24535048a9ded5d856996b6bddd0b5645af26dfe15579d4ac69615bdbd796526'}


def previous_job(job):
    views = []
    for v in job['views']:
        folder = ROOT/'artifacts/grid-role-profile-v2/runs'/(job['case']['id']+'-r0')/'output'/v['mode']
        if v.get('structural_inputs'):
            record = {k: v['structural_inputs'][k] for k in ('graph', 'roles', 'heads', 'links')}
        else:
            record = {k: pin(ROOT, (folder/'baseline-source'/f).relative_to(ROOT).as_posix())
                      for k, f in [('graph', 'graph.json'), ('roles', 'roles.png'), ('heads', 'heads.json'), ('links', 'links.json')]}
        views.append({**v, 'structural_inputs': record,
                      'previous_guarded': pin(ROOT, (folder/'guarded-roles.png').relative_to(ROOT).as_posix()),
                      'previous_result': pin(ROOT, (folder/'result.json').relative_to(ROOT).as_posix())})
    return {**job, 'case': {**job['case'], 'split': 'development'}, 'views': views, 'new_control': False,
            'previous_control': job['new_control']}


def run():
    cv2.setNumThreads(1)
    BASE.mkdir(exist_ok=False)
    jobs = []
    for j in read(checked(ROOT, CONTROLS))['jobs']:
        if j['case']['split'] != 'train':
            continue
        views = []
        for v in j['views']:
            _, _, _, _, _, record, _, calls = load(j, v, BASE/'train-bank'/j['case']['id']/v['mode'])
            views.append({**v, 'structural_inputs': record, 'capture_preparation': calls})
        jobs.append({**j, 'views': views})
    write(BASE/'train-index.json', {'jobs': jobs, 'fresh_validation_opened': False})
    old = [previous_job(j) for j in read(ROOT/'artifacts/grid-role-profile-v2/plan.json')['jobs']
           if j['new_control'] or j['case']['scope'] == 'partial_real_reviewed_ink']
    rows = []
    for j in jobs+old:
        for v in j['views']:
            gray, graph, roles, _, _, _, authority, _ = load(j, v)
            ink = foreground(gray)
            context, work = source_track_context(ink, graph)
            row = {'case': j['case']['id'], 'mode': v['mode'], 'split': j['case']['split'], 'fit': work}
            for label, view in [('frozen_tracks', graph), ('source_fit', context)]:
                result = analyze(gray, view, roles, authority=authority)
                row[label] = {'complete': result['complete'], 'audit': result['audit'],
                              'decisions': result['decisions'], 'withdrawals': int((result['guarded'] != roles).sum())}
            rows.append(row)
    write(BASE/'report.json', {'controls': CONTROLS, 'rows': rows,
          'fresh_validation_opened': False, 'model': pin(ROOT, 'src/cad_recovery/grid_context_v1.py')})
    for split in ('train', 'development'):
        for mode in ('native_crop', 'half_resolution_crop'):
            selected = [r for r in rows if r['split'] == split and r['mode'] == mode]
            print(split, mode, {a: {'grid_views': sum(bool(r[a]['decisions']) for r in selected),
                  'text_grids': sum(d['guard_context_supported'] for r in selected for d in r[a]['decisions']),
                  'withdrawals': sum(r[a]['withdrawals'] for r in selected)} for a in ('frozen_tracks', 'source_fit')}, flush=True)


if __name__ == '__main__':
    run()
