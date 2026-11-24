"""Freeze independent contact/tangency adversaries before joint-owner rules."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import pin
from run_supported_role_overlay_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/contact-role-controls-v1'
FAMILIES = [('wide_T_contact', 'train'), ('slanted_wall_contact', 'train'),
            ('two_boundary_corner', 'train'), ('arc_tangent_contacts', 'validation'),
            ('crossing_rib_contact', 'validation'), ('offset_frame_touch', 'validation')]


def scenes():
    rng = np.random.default_rng(274618)
    for family, split in FAMILIES:
        angles = (0, 31, 90, 143) if split == 'train' else (13, 57, 103, 167)
        for width in (3, 5, 7):
            for angle in angles:
                size = 512
                gray = np.full((size, size), 255, np.uint8)
                truth = np.full_like(gray, 255)
                conflicts = np.zeros_like(gray, bool)
                part = np.zeros_like(gray)
                negative = np.zeros_like(gray)
                theta = np.deg2rad(angle)
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-5, 6, 2)

                def transform(points):
                    return (np.asarray(points, float)-256)@rotation.T+256+shift

                def draw(points, cls, pen, closed=False):
                    mask = np.zeros_like(gray)
                    cv2.polylines(mask, [np.rint(transform(points)).astype(np.int32)], closed, 255, pen)
                    selected = mask > 0
                    conflicts[:] |= selected & (truth != 255) & (truth != cls)
                    truth[selected & ~conflicts] = cls
                    gray[selected] = 96 if cls == 1 else 0
                    if cls == 1:
                        part[selected] = 255

                left, right, top, bottom, shaft_y = 132, 380, 244, 420, 120
                if family == 'slanted_wall_contact':
                    wall = [[132, top], [380, top], [354, bottom], [153, bottom]]
                elif family == 'two_boundary_corner':
                    wall = [[132, top], [380, top], [380, bottom], [132, bottom], [132, 310], [162, 283]]
                elif family == 'offset_frame_touch':
                    wall = [[132, top], [380, top], [380, bottom], [132, bottom], [132, 310], [117, 294], [117, 260]]
                else:
                    wall = [[132, top], [380, top], [380, bottom], [132, bottom]]
                if family == 'arc_tangent_contacts':
                    t = np.linspace(0, 2*np.pi, 321)
                    wall = np.c_[256+124*np.cos(t), 320+84*np.sin(t)]
                    top = 320
                draw(wall, 1, 2*width+5, closed=True)
                if family == 'crossing_rib_contact':
                    draw([[112, top+15], [155, top-15]], 1, width+2)
                    draw([[359, top-12], [397, top+16]], 1, width+2)
                arrows = []
                for tip, unit in [([left, shaft_y], [-1, 0]), ([right, shaft_y], [1, 0])]:
                    tip, unit = np.asarray(tip, float), np.asarray(unit, float)
                    normal = np.array([-unit[1], unit[0]])
                    base = tip-(23+2*width)*unit
                    triangle = transform([tip, base+(5+width)*normal, base-(5+width)*normal])
                    mask = np.zeros_like(gray)
                    cv2.fillConvexPoly(mask, np.rint(triangle).astype(np.int32), 255)
                    gray[mask > 0] = 0
                    conflicts[:] |= mask > 0
                    arrows.append({'id': 'head-'+str(len(arrows)), 'tip': transform([tip])[0].tolist(),
                                   'direction': (rotation@unit).tolist(), 'triangle': triangle.tolist()})
                draw([[left+23+2*width, shaft_y], [right-23-2*width, shaft_y]], 2, width)
                draw([[left, shaft_y-24], [left, top]], 3, width)
                draw([[right, shaft_y-24], [right, top]], 3, width)
                # Unrelated marks are not input authority for either role.
                for x in (201, 281):
                    mask = np.zeros_like(gray)
                    cv2.line(mask, tuple(np.rint(transform([[x, 162]])[0]).astype(int)),
                             tuple(np.rint(transform([[x+13, 174]])[0]).astype(int)), 255, 1)
                    selected = (mask > 0) & (gray == 255)
                    negative[selected] = 255
                    gray[selected] = 0
                truth[conflicts] = 255
                yield {'case': {'id': f'contact-role-{family}-w{width}-r{angle}',
                    'scope': 'contact_role_analytical_control', 'group_id': 'contact-role-'+family,
                    'split': split, 'size_px': [size, size], 'labels': {'family': family,
                        'thin_width_px': width, 'rotation_deg': angle, 'arrows': arrows,
                        'negative_regions': [], 'expected_dimension_chains': 1,
                        'authority': 'procedural role truth and protected part support; no CAD authority'}},
                    'gray': gray, 'truth': truth, 'negative': negative, 'part_contact': part}


if __name__ == '__main__':
    BASE.mkdir(exist_ok=False)
    jobs = []
    for scene in scenes():
        folder = BASE/scene['case']['id']
        folder.mkdir()
        for name in ('truth', 'negative', 'part_contact'):
            Image.fromarray(scene[name]).save(folder/(name+'.png'))
        views = []
        for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
            gray = scene['gray'] if scale == 1 else cv2.resize(scene['gray'], (256, 256), interpolation=cv2.INTER_AREA)
            path = folder/(mode+'.png')
            Image.fromarray(gray).save(path)
            views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]],
                'reviewed_mask': None, 'cache': None})
        jobs.append({'case': scene['case'], 'views': views, 'new_control': True,
            **{name: pin(ROOT, (folder/(name+'.png')).relative_to(ROOT).as_posix()) for name in ('truth', 'negative', 'part_contact')}})
    write(BASE/'contract.json', {'task': 'P10.2c.3b.2a.1a', 'seed': 274618,
        'generator': pin(ROOT, 'scripts/prepare_contact_role_controls_v1.py'),
        'families': FAMILIES, 'prepared_before_contact_guard_implementation': True,
        'jobs': jobs, 'training': False, 'closed_test_used': False, 'physical_geometry_approved': False})
    print(pin(ROOT, (BASE/'contract.json').relative_to(ROOT).as_posix()))
