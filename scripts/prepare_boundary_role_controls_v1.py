"""Freeze full-pipeline boundary-role controls before the new classifier exists."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/boundary-role-controls-v1'
FAMILIES = [('stepped_body', 'train'), ('rounded_body', 'train'), ('glyph_loops', 'train'),
            ('title_cells', 'train'), ('notched_body', 'validation'), ('body_with_hole', 'validation'),
            ('frame_and_text', 'validation'), ('fold_loops', 'validation')]


def scenes():
    rng = np.random.default_rng(481907)
    for family, split in FAMILIES:
        for width in (3, 5):
            for angle in ((0, 35, 90) if split == 'train' else (13, 59, 137)):
                size = 512; gray = np.full((size, size), 255, np.uint8)
                truth = np.full_like(gray, 255); negative = np.zeros_like(gray)
                conflict = np.zeros_like(gray, bool)
                theta = np.deg2rad(angle); rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                offset = rng.integers(-4, 5, 2)

                def xy(p): return (np.asarray(p, float)-256)@rot.T+256+offset

                def add(mask, cls):
                    selected = mask > 0
                    gray[selected] = 0
                    if cls is None: negative[selected & (truth == 255)] = 255
                    else:
                        conflict[:] |= selected & (truth != 255) & (truth != cls)
                        truth[selected & ~conflict] = cls

                def stroke(p, cls, pen, closed=False):
                    mask = np.zeros_like(gray)
                    cv2.polylines(mask, [np.rint(xy(p)).astype(np.int32)], closed, 255, pen)
                    add(mask, cls)

                if family == 'stepped_body':
                    body = [[136, 232], [376, 232], [376, 292], [348, 292], [348, 394], [136, 394]]
                elif family == 'rounded_body':
                    body = np.r_[[[136, 232], [330, 232]], np.c_[330+46*np.cos(np.linspace(-np.pi/2, np.pi/2, 65)), 313+81*np.sin(np.linspace(-np.pi/2, np.pi/2, 65))], [[136, 394]]]
                elif family == 'notched_body':
                    body = [[136, 232], [376, 232], [376, 394], [296, 394], [296, 351], [218, 351], [218, 394], [136, 394]]
                else: body = [[136, 232], [376, 232], [376, 394], [136, 394]]
                positive = family not in ('glyph_loops', 'title_cells', 'frame_and_text', 'fold_loops')
                if positive or family in ('title_cells', 'frame_and_text', 'fold_loops'):
                    cls = 1 if positive else 8 if family == 'title_cells' else 7 if family == 'frame_and_text' else None
                    stroke(body, cls, 2*width+1, True)
                if positive:
                    # A real observed dimension, with both extensions and heads.
                    stroke([[136, 113], [376, 113]], 2, width)
                    for x in (136, 376): stroke([[x, 95], [x, 232]], 3, width)
                    for x, direction in ((136, -1), (376, 1)):
                        p = np.array([x, 113.]); base = p-np.array([direction*(24+width), 0])
                        mask = np.zeros_like(gray)
                        cv2.fillConvexPoly(mask, np.rint(xy([p, base+[0, 5+width], base-[0, 5+width]])).astype(np.int32), 255)
                        add(mask, 255); conflict[:] |= mask > 0
                if family == 'body_with_hole':
                    a = np.linspace(0, 2*np.pi, 129)
                    stroke(np.c_[256+35*np.cos(a), 308+35*np.sin(a)], 1, width, True)
                if family == 'title_cells':
                    for y in (274, 316, 358): stroke([[136, y], [376, y]], 8, width)
                    for x in (196, 256, 316): stroke([[x, 232], [x, 394]], 8, width)
                if family == 'frame_and_text':
                    stroke([[106, 208], [406, 208], [406, 414], [106, 414]], 7, width, True)
                if family == 'fold_loops':
                    stroke([[120, 222], [360, 244], [390, 385], [145, 410]], None, 1, True)
                    stroke([[95, 178], [390, 185]], None, 1)
                # Text glyphs, closed O/0/8, and unrelated thin diagonal marks.
                mask = np.zeros_like(gray)
                cv2.putText(mask, 'O08 R25', (172, 181 if positive else 314), cv2.FONT_HERSHEY_SIMPLEX, .8, 255, 2)
                affine = np.c_[rot, np.array([256, 256])+offset-rot@np.array([256, 256])]
                mask = cv2.warpAffine(mask, affine, (size, size), flags=cv2.INTER_NEAREST)
                add(mask, 6)
                for x in (204, 289): stroke([[x, 426], [x+14, 435]], None, 1)
                truth[conflict] = 255; negative[truth != 255] = 0
                yield {'case': {'id': f'boundary-role-{family}-w{width}-r{angle}',
                               'scope': 'boundary_role_analytical_control', 'group_id': 'boundary-role-'+family,
                               'split': split, 'size_px': [size, size], 'labels': {'family': family,
                                   'positive_part': positive, 'authority': 'procedural pixel-role truth; no physical approval'}},
                       'gray': gray, 'truth': truth, 'negative': negative}


def main():
    BASE.mkdir(exist_ok=False); jobs = []
    for scene in scenes():
        folder = BASE/scene['case']['id']; folder.mkdir()
        for name in ('truth', 'negative'): Image.fromarray(scene[name]).save(folder/(name+'.png'))
        views = []
        for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
            image = scene['gray'] if scale == 1 else cv2.resize(scene['gray'], (256, 256), interpolation=cv2.INTER_AREA)
            path = folder/(mode+'.png'); Image.fromarray(image).save(path)
            views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                          'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
        jobs.append({'case': scene['case'], 'views': views, 'new_control': True,
                     **{n: pin(ROOT, (folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth', 'negative')}})
    write(BASE/'contract.json', {'task': 'P10.2c.3b.2b', 'generator': pin(ROOT, 'scripts/prepare_boundary_role_controls_v1.py'),
                                'families': FAMILIES, 'seed': 481907, 'jobs': jobs, 'prepared_before_implementation': True,
                                'gates': ['zero new known-wrong roles', 'zero new negative assignments', 'positive new validation part pixels',
                                          'real-family F1 nondegradation', 'all inherited roles/heads/source unchanged',
                                          'source interval for each addition', 'all budgets complete and repeats exact'],
                                'full_classifier_macro_F1_min': .90, 'full_classifier_is_separate': True,
                                'production_changed': False, 'physical_geometry_approved': False,
                                'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128,
                                            'total_child_seconds': 1800, 'OpenCV_threads': 1, 'RNG_seed': 0}})
    print(pin(ROOT, (BASE/'contract.json').relative_to(ROOT).as_posix()))


if __name__ == '__main__': main()
