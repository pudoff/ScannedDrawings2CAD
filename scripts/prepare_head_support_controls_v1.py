"""Freeze positive neighbouring arrows and negative reused-support scenes first."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import pin

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/head-support-controls-v1'
FAMILIES = [('shared_short_pair', 'train'), ('single_hook_pointer', 'train'),
    ('rounded_slot_noise', 'train'), ('offset_opposed_pair', 'validation'),
    ('single_arc_pointer', 'validation'), ('parallel_loop_noise', 'validation')]


def scenes():
    rng = np.random.default_rng(936148)
    for family, split in FAMILIES:
        for width in (2, 4, 6):
            for angle in (0, 33, 90, 147):
                image = np.full((480, 480), 255, np.uint8); refs = []
                theta = np.deg2rad(angle)
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-6, 7, 2)

                def transform(points): return (np.asarray(points, float)-240)@rotation.T+240+shift

                def line(points, pen=width):
                    cv2.polylines(image, [np.rint(transform(points)).astype(np.int32)], False, 0, pen)

                def arrow(tip, unit, length):
                    tip, unit = np.asarray(tip, float), np.asarray(unit, float)
                    normal = np.array([-unit[1], unit[0]]); base = tip-length*unit
                    triangle = transform([tip, base+(4+width)*normal, base-(4+width)*normal])
                    cv2.fillConvexPoly(image, np.rint(triangle).astype(np.int32), 0)
                    refs.append({'id': 'head-'+str(len(refs)), 'tip': transform([tip])[0].tolist(),
                        'direction': (rotation@unit).tolist(), 'triangle': triangle.tolist()})
                    return base

                length = 18+2*width
                if family == 'shared_short_pair':
                    tip = np.array([164., 224.]); gap = max(14, 3*width+4)
                    left = arrow(tip, [-1, 0], length)
                    right = arrow(tip+[2*length+gap, 0], [1, 0], length)
                    line([left, right])
                elif family == 'offset_opposed_pair':
                    left = arrow([162, 211], [-1, 0], length)
                    right = arrow([162+2*length+30, 236], [1, 0], length)
                    line([left, left+[15, 0], right-[15, 0], right])
                elif family in ('single_hook_pointer', 'single_arc_pointer'):
                    base = arrow([174, 214], [-1, 0], length)
                    t = np.linspace(0, 1, 80)
                    bend = 27 if family == 'single_hook_pointer' else -38
                    points = base+np.c_[74*t, bend*t*t]
                    line(points)
                    if family == 'single_arc_pointer': line([points[-1], points[-1]+[18, -17]])
                else:
                    t = np.linspace(0, 2*np.pi, 180)
                    for offset in (-34, 34):
                        points = np.c_[240+74*np.cos(t), 240+offset+17*np.sin(t)]
                        line(points)
                    if family == 'parallel_loop_noise':
                        for y in (184, 240, 296): line([[134, y], [347, y]])
                case = {'id': f'head-support-{family}-w{width}-r{angle}',
                    'scope': 'head_support_control', 'group_id': 'head-support-'+family,
                    'split': split, 'size_px': [480, 480], 'labels': {'family': family,
                        'width_px': width, 'rotation_deg': angle, 'arrows': refs,
                        'negative_regions': [], 'dense_heads': True,
                        'authority': 'new procedural source controls; no physical CAD approval'}}
                yield case, image


if __name__ == '__main__':
    BASE.mkdir(exist_ok=False); jobs = []
    for case, gray in scenes():
        folder = BASE/case['id']; folder.mkdir(); views = []
        for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
            image = gray if scale == 1 else cv2.resize(gray, (240, 240), interpolation=cv2.INTER_AREA)
            path = folder/(mode+'.png'); Image.fromarray(image).save(path)
            views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
        jobs.append({'case': case, 'views': views, 'new_control': True})
    contract = {'task': 'P10.2c.2b.7c.2a.1', 'seed': 936148, 'families': FAMILIES,
        'generator': pin(ROOT, 'scripts/prepare_head_support_controls_v1.py'),
        'prepared_before_guard_implementation': True, 'jobs': jobs,
        'gates': {'lost_raw_true_targets_max': 0, 'additional_dense_FP_against_raw_max': 0,
            'old_additional_dense_FP_against_local_width_v1_max': 0,
            'known_reverse_false_head_removed': True, 'source_and_baseline_unchanged': True,
            'no_new_incomplete_searches': True, 'all_guard_decisions_complete': True,
            'historical_20_of_21_and_13_of_13_preserved': True,
            'old_width4_half': [12, 12], 'old_interval_all': [72, 72]},
        'training': False, 'closed_test_used': False, 'production_changed': False}
    path = BASE/'contract.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(pin(ROOT, path.relative_to(ROOT).as_posix())))
