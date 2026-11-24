"""Metadata-only repair of odd-size resize transforms; frozen model unchanged."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import run_native_frame_v1 as original
from run_native_frame_v1 import ROOT, ARMS, MASKS
from run_topology_robust_v1 import read, write
from cad_recovery.dataset_layout_v1 import checked, pin

BASE = ROOT/'artifacts/native-frame-profile-v2'
OLD_PLAN = {'path': 'artifacts/native-frame-profile-v1/plan.json', 'sha256': 'efa933c3b0ed3be12137f67b09ab63c0f644dd5ea605b709c6035e065c80809b'}
CORRECTED = ROOT/'artifacts/native-frame-coordinate-repair-v1'


def prepare():
    old = read(checked(ROOT, OLD_PLAN))
    for p in old['code']: checked(ROOT, p)
    control = read(checked(ROOT, old['controls'])); corrected = []; repairs = []
    for job in control['jobs']:
        views = []
        for v in job['views']:
            width, height = Image.open(checked(ROOT, v['gray'])).size
            sx, sy = np.asarray(job['case']['size_px'])/np.array([width, height])
            matrix = [[float(sx), 0, float(.5*sx-.5)], [0, float(sy), float(.5*sy-.5)], [0, 0, 1]]
            if not np.allclose(matrix, v['input_to_native']):
                repairs.append({'case': job['case']['id'], 'mode': v['mode'], 'before': v['input_to_native'], 'after': matrix,
                                'image': v['gray'], 'truth': job['truth'], 'pixels_and_labels_changed': False})
            views.append({**v, 'input_to_native': matrix})
        corrected.append({**job, 'views': views})
    CORRECTED.mkdir(exist_ok=False)
    write(CORRECTED/'controls.json', {**control, 'jobs': corrected, 'original_controls': old['controls'],
          'control_pixels_prepared_before_model': True, 'manifest_prepared_before_model': False,
          'metadata_only_coordinate_repair': True, 'pixels_and_labels_changed': False,
          'reason': 'cv2.resize uses actual source/output sizes, including odd native heights'})
    runs = original.BASE/'runs'; processes = [read(p) for p in runs.glob('*/process.json')]
    write(CORRECTED/'repair.json', {'original_plan': OLD_PLAN, 'controls': pin(ROOT, (CORRECTED/'controls.json').relative_to(ROOT).as_posix()),
          'repairs': repairs, 'all_model_and_original_runner_pins_exact': True, 'algorithm_parameters_changed': False,
          'images_labels_splits_and_thresholds_changed': False, 'validation_used_for_parameter_selection': False,
          'initial_attempt': {'children': len(processes), 'child_wall_seconds': sum(p['walltime_seconds'] for p in processes),
              'completed_view_rows': len(list(runs.glob('*/output/*/*/result.json'))),
              'completed_validation_native_views_before_repair': 1, 'failure': 'odd-size resize metadata mismatch before half refinement',
              'failed_case': 'native-frame-acute_contact-w3-r37', 'model_quality_metrics_used_to_choose_repair': False}})
    by_id = {j['case']['id']: j for j in corrected}; jobs = []
    for job in old['jobs']:
        matrices = {v['mode']: v['input_to_native'] for v in by_id[job['case']['id']]['views']} if job['new_control'] else {}
        jobs.append({**job, 'views': [{**v, 'input_to_native': matrices.get(v['mode'], v['input_to_native'])} for v in job['views']]})
    index = {j['case']['id']: j for j in jobs}
    batches = [{**b, 'members': [index[j['case']['id']] for j in b['members']]} for b in old['batches']]
    BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json', {**old, 'controls': pin(ROOT, (CORRECTED/'controls.json').relative_to(ROOT).as_posix()),
          'original_freeze': OLD_PLAN, 'coordinate_repair': pin(ROOT, (CORRECTED/'repair.json').relative_to(ROOT).as_posix()),
          'jobs': jobs, 'batches': batches, 'code': old['code']+[pin(ROOT, 'scripts/run_native_frame_corrected_v1.py')],
          'fresh_validation_predictions_before_freeze': True, 'model_freeze_preceded_first_validation_prediction': True,
          'metadata_repair_after_first_validation_native_prediction': True, 'validation_used_for_parameter_selection': False})
    print({'repairs': len(repairs), 'plan': pin(ROOT, (BASE/'plan.json').relative_to(ROOT).as_posix())}, flush=True)


def run(sha256):
    from experiment_case_batches_v1 import run as execute
    p = {'path': (BASE/'plan.json').relative_to(ROOT).as_posix(), 'sha256': sha256}
    e = execute(ROOT, BASE, p, Path(__file__).resolve())
    rows = [read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]
    attempt = read(CORRECTED/'repair.json')['initial_attempt']
    e.update(native_refinement_calls=len(rows), preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
             initial_attempt=attempt, all_attempt_child_wall_seconds=e['child_wall_seconds']+attempt['child_wall_seconds'],
             all_attempt_children=e['children']+attempt['children'], new_OCR_ML_calls=0, new_DXF_exports=0, training=False, closed_test_used=False)
    write(BASE/'execution.json', e); print(e, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'run', 'child']); p.add_argument('--sha256'); p.add_argument('--request', type=Path); a = p.parse_args()
    if a.mode == 'prepare': prepare()
    elif a.mode == 'child': original.child(a.request)
    elif a.sha256: run(a.sha256)
    else: p.error('--sha256 required')
