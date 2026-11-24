"""Freeze loop/wedge counterexamples and genuine nearby arrows before rules."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import pin
from run_local_width_heads_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/head-eligibility-controls-v1'
FAMILIES = [('tangent_ring_rails', 'train'), ('solid_wedge_marks', 'train'),
            ('arrow_on_ring_leader', 'train'), ('overlapped_ovals_grid', 'validation'),
            ('skew_wedge_marks', 'validation'), ('curved_pointer_beside_ring', 'validation')]


def scenes():
    rng = np.random.default_rng(638412)
    for family, split in FAMILIES:
        angles = (7, 49, 98, 157) if split == 'train' else (19, 74, 127, 173)
        for width in (3, 5, 7):
            for angle in angles:
                gray = np.full((512, 512), 255, np.uint8)
                theta = np.deg2rad(angle)
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-8, 9, 2)
                refs = []

                def transform(points):
                    return (np.asarray(points, float)-256)@rotation.T+256+shift

                def line(points):
                    cv2.polylines(gray, [np.rint(transform(points)).astype(np.int32)], False, 0, width)

                def oval(center, radii):
                    t = np.linspace(0, 2*np.pi, 241)
                    line(np.asarray(center)+np.c_[radii[0]*np.cos(t), radii[1]*np.sin(t)])

                def triangle(tip, direction, length, breadth, label=False):
                    tip, direction = np.asarray(tip, float), np.asarray(direction, float)
                    normal = np.array([-direction[1], direction[0]])
                    base = tip-length*direction
                    points = transform([tip, base+breadth/2*normal, base-breadth/2*normal])
                    cv2.fillConvexPoly(gray, np.rint(points).astype(np.int32), 0)
                    if label:
                        refs.append({'id': 'head-0', 'tip': transform([tip])[0].tolist(),
                                     'direction': (rotation@direction).tolist(), 'triangle': points.tolist()})
                    return base

                if family in ('tangent_ring_rails', 'overlapped_ovals_grid'):
                    for y in (220, 288):
                        oval([256, y], [81, 21 if split == 'train' else 24])
                        line([[139, y-21], [373, y-21]])
                    if split == 'validation':
                        oval([289, 257], [51, 31])
                        line([[139, 257], [373, 257]])
                elif family in ('solid_wedge_marks', 'skew_wedge_marks'):
                    triangle([181, 215], [-1, 0], 24+2*width, 10+2*width)
                    direction = [1, 0] if split == 'train' else np.array([1., .18])/np.linalg.norm([1., .18])
                    base = triangle([308, 293], direction, 28+2*width, 12+2*width)
                    # A disconnected dash must not become the wedge's shaft.
                    line([base-[24, 0], base-[57, 0]])
                else:
                    base = triangle([160, 222], [-1, 0], 25+2*width, 11+2*width, label=True)
                    oval([291, 256], [65, 28])
                    if split == 'train':
                        line([base, [291, 222], [291, 228]])
                    else:
                        t = np.linspace(0, 1, 91)
                        points = base+np.c_[102*t, 18*t*t]
                        line(points)
                        line([points[-1], [291, 228]])
                case = {'id': f'head-eligibility-{family}-w{width}-r{angle}',
                        'scope': 'head_eligibility_control', 'group_id': 'head-eligibility-'+family,
                        'split': split, 'size_px': [512, 512], 'labels': {
                            'family': family, 'width_px': width, 'rotation_deg': angle,
                            'arrows': refs, 'negative_regions': [], 'dense_heads': True,
                            'authority': 'procedural source truth; no physical CAD approval'}}
                yield case, gray


if __name__ == '__main__':
    BASE.mkdir(exist_ok=False)
    jobs = []
    for case, gray in scenes():
        folder = BASE/case['id']
        folder.mkdir()
        views = []
        for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
            image = gray if scale == 1 else cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
            path = folder/(mode+'.png')
            Image.fromarray(image).save(path)
            views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                          'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
        jobs.append({'case': case, 'views': views, 'new_control': True})
    write(BASE/'contract.json', {'task': 'P10.2c.2b.7c.2a.3b', 'seed': 638412,
        'generator': pin(ROOT, 'scripts/prepare_head_eligibility_controls_v1.py'), 'families': FAMILIES,
        'prepared_before_new_rules': True, 'predictor_calls': 0, 'jobs': jobs,
        'training': False, 'closed_test_used': False, 'production_changed': False})
    print(pin(ROOT, (BASE/'contract.json').relative_to(ROOT).as_posix()))
