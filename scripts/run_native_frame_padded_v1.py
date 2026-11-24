"""Preserve source scenes and scoring pixels while making ROI sizes even."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import run_native_frame_v1 as original
from run_native_frame_v1 import ROOT, ARMS, MASKS
from run_native_frame_corrected_v1 import OLD_PLAN
from run_topology_robust_v1 import read, write
from cad_recovery.dataset_layout_v1 import checked, pin

BASE = ROOT/'artifacts/native-frame-profile-v3'
CORRECTED = ROOT/'artifacts/native-frame-even-roi-repair-v1'


def even_case(job):
    width, height = job['case']['size_px']; dx, dy = width % 2, height % 2
    if not (dx or dy): return job, None
    page = job['page_source']; x, y, r, b = page['patch_bbox']
    if x < dx or y < dy or page['kind'] != 'synthetic_full_page_pixels': raise ValueError('Verified source margin required for even ROI')
    if job['case']['labels']['arrows'] or job['case']['labels']['chains']: raise ValueError('Coordinate labels require explicit translation')
    whole = np.array(Image.open(checked(ROOT, page['source'])).convert('L')); gray = whole[y-dy:b, x-dx:r].copy()
    old_gray = np.array(Image.open(checked(ROOT, job['views'][0]['gray'])).convert('L'))
    if not np.array_equal(gray[dy:, dx:], old_gray): raise ValueError('Original ROI samples changed')
    folder = CORRECTED/job['case']['id']; folder.mkdir()
    labels = {}
    for name, fill in [('truth', 255), ('negative', 0)]:
        old = np.array(Image.open(checked(ROOT, job[name]))); padded = np.pad(old, ((dy, 0), (dx, 0)), constant_values=fill)
        assert np.array_equal(padded[dy:, dx:], old)
        if name == 'truth': assert int(((old >= 1) & (old <= 9)).sum()) == int(((padded >= 1) & (padded <= 9)).sum())
        Image.fromarray(padded).save(folder/(name+'.png')); labels[name] = pin(ROOT, (folder/(name+'.png')).relative_to(ROOT).as_posix())
    views = []
    for v in job['views']:
        scale = 1 if v['mode'] == 'native_crop' else 2
        pixels = gray if scale == 1 else cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2), interpolation=cv2.INTER_AREA)
        path = folder/(v['mode']+'.png'); Image.fromarray(pixels).save(path)
        views.append({**v, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                      'input_to_native': [[scale, 0, .5*scale-.5], [0, scale, .5*scale-.5], [0, 0, 1]]})
    changed = {**job, 'case': {**job['case'], 'size_px': list(gray.shape[::-1])}, **labels, 'views': views,
               'page_source': {**page, 'patch_bbox': [x-dx, y-dy, r, b],
                   'patch_to_source': [[1, 0, x-dx], [0, 1, y-dy], [0, 0, 1]],
                   'source_to_patch': [[1, 0, -x+dx], [0, 1, -y+dy], [0, 0, 1]]}}
    record = {'case': job['case']['id'], 'original_size': [width, height], 'even_size': list(gray.shape[::-1]),
              'added_source_margin_left_top': [dx, dy], 'original_source': page['source'],
              'old_bbox': page['patch_bbox'], 'new_bbox': changed['page_source']['patch_bbox'],
              'existing_image_samples_and_known_targets_preserved': True, 'new_margin_labels': 'unknown; outside original scoring domain'}
    return changed, record


def prepare():
    old = read(checked(ROOT, OLD_PLAN))
    for p in old['code']: checked(ROOT, p)
    controls = read(checked(ROOT, old['controls'])); CORRECTED.mkdir(exist_ok=False); jobs = []; repairs = []
    for job in controls['jobs']:
        changed, record = even_case(job); jobs.append(changed)
        if record: repairs.append(record)
    write(CORRECTED/'controls.json', {**controls, 'jobs': jobs, 'original_controls': old['controls'],
          'source_scenes_prepared_before_model': True, 'derived_ROI_repair_after_model_freeze': True,
          'source_scenes_labels_splits_and_scoring_domain_changed': False,
          'derived_image_samples_changed_only_by_declared_source_margin': True,
          'original_known_target_pixels_removed': 0, 'model_parameters_changed': False})
    attempts = []
    for name in ('native-frame-profile-v1', 'native-frame-profile-v2'):
        folder = ROOT/'artifacts'/name/'runs'; processes = [read(p) for p in folder.glob('*/process.json')]
        attempts.append({'profile': name, 'children': len(processes), 'child_wall_seconds': sum(p['walltime_seconds'] for p in processes),
                         'completed_view_rows': len(list(folder.glob('*/output/*/*/result.json'))),
                         'completed_validation_native_views_before_repair': 1})
    write(CORRECTED/'repair.json', {'original_model_freeze': OLD_PLAN, 'repairs': repairs, 'prior_attempts': attempts,
          'algorithm_parameters_changed': False, 'validation_used_for_parameter_selection': False,
          'original_source_images_and_known_targets_changed': False, 'original_known_target_pixels_removed': 0,
          'reason': 'Existing CV contract requires uniform scaling; odd ROI is expanded in original source coordinates',
          'prior_failures': ['nominal half transform mismatched odd height', 'legacy CV rejects anisotropic resize transform']})
    by_id = {j['case']['id']: j for j in jobs}; planned = []
    for job in old['jobs']:
        if job['new_control'] and any(r['case'] == job['case']['id'] for r in repairs): planned.append(by_id[job['case']['id']])
        else: planned.append(job)
    index = {j['case']['id']: j for j in planned}
    batches = [{**b, 'members': [index[j['case']['id']] for j in b['members']]} for b in old['batches']]
    BASE.mkdir(exist_ok=False)
    write(BASE/'plan.json', {**old, 'controls': pin(ROOT, (CORRECTED/'controls.json').relative_to(ROOT).as_posix()),
          'original_freeze': OLD_PLAN, 'coordinate_repair': pin(ROOT, (CORRECTED/'repair.json').relative_to(ROOT).as_posix()),
          'jobs': planned, 'batches': batches, 'code': old['code']+[pin(ROOT, 'scripts/run_native_frame_padded_v1.py')],
          'fresh_validation_predictions_before_freeze': True, 'model_freeze_preceded_first_validation_prediction': True,
          'validation_native_results_produced_before_repair': 2, 'validation_used_for_parameter_selection': False,
          'original_known_target_pixels_removed': 0})
    print({'repairs': len(repairs), 'plan': pin(ROOT, (BASE/'plan.json').relative_to(ROOT).as_posix())}, flush=True)


def run(sha256):
    from experiment_case_batches_v1 import run as execute
    p = {'path': (BASE/'plan.json').relative_to(ROOT).as_posix(), 'sha256': sha256}; e = execute(ROOT, BASE, p, Path(__file__).resolve())
    rows = [read(ROOT/r['artifact']) for r in read(BASE/'results-index.json')]; attempts = read(CORRECTED/'repair.json')['prior_attempts']
    e.update(native_refinement_calls=len(rows), preparation_calls={k: sum(r['preparation'][k] for r in rows) for k in rows[0]['preparation']},
             prior_attempts=attempts, all_attempt_child_wall_seconds=e['child_wall_seconds']+sum(a['child_wall_seconds'] for a in attempts),
             all_attempt_children=e['children']+sum(a['children'] for a in attempts), new_OCR_ML_calls=0, new_DXF_exports=0, training=False, closed_test_used=False)
    write(BASE/'execution.json', e); print(e, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['prepare', 'run', 'child']); p.add_argument('--sha256'); p.add_argument('--request', type=Path); a = p.parse_args()
    if a.mode == 'prepare': prepare()
    elif a.mode == 'child': original.child(a.request)
    elif a.sha256: run(a.sha256)
    else: p.error('--sha256 required')
