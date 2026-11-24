"""Freeze pixel-ownership adversaries before implementing role overlay admission."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import pin
from cad_recovery.stroke_roles_v1 import CLASSES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/supported-role-controls-v1'
FAMILIES = [('detached_specks', 'train'), ('contacting_blot', 'train'),
    ('hatch_pockets', 'train'), ('title_contact', 'train'),
    ('parallel_unknown_strokes', 'validation'), ('interrupted_hatch_next_to_dimension', 'validation'),
    ('leader_ambiguous', 'validation'), ('closed_pockets', 'validation')]


def scenes():
    rng = np.random.default_rng(671302)
    for family, split in FAMILIES:
        for width in (2, 6):
            for angle in (0, 90):
                size = 448; shift = rng.integers(-5, 6, 2)
                gray = np.full((size, size), 255, np.uint8); truth = np.full_like(gray, 255)
                negative = np.zeros_like(gray); conflicts = np.zeros_like(gray, bool); arrows, geometry = [], []

                def transform(p):
                    p = np.asarray(p, float)+shift
                    return np.c_[size-1-p[:, 1], p[:, 0]] if angle else p

                def add(mask, role):
                    selected = mask > 0; cls = CLASSES.index(role)
                    conflicts[:] |= selected & (truth != 255) & (truth != cls)
                    truth[selected & ~conflicts] = cls; gray[selected] = 0

                def line(name, points, role, thickness=None, closed=False):
                    p = transform(points); mask = np.zeros_like(gray)
                    cv2.polylines(mask, [np.rint(p).astype(np.int32)], closed, 255, thickness or width)
                    add(mask, role); geometry.append({'id': name, 'points': p.tolist(), 'role': role, 'closed': closed})

                def head(name, tip, direction):
                    p, u = np.asarray(tip, float), np.asarray(direction, float); n = np.array([-u[1], u[0]])
                    base = p-(22+2*width)*u; triangle = transform([p, base+(5+width)*n, base-(5+width)*n])
                    mask = np.zeros_like(gray); cv2.fillConvexPoly(mask, np.rint(triangle).astype(np.int32), 255)
                    gray[mask > 0] = 0; conflicts[:] |= mask > 0
                    arrows.append({'id': name, 'tip': transform([p])[0].tolist(),
                        'direction': [-u[1], u[0]] if angle else u.tolist(), 'triangle': triangle.tolist()})
                    return base

                def noise(points, thickness=1, filled=False):
                    mask = np.zeros_like(gray); p = np.rint(transform(points)).astype(np.int32)
                    if filled: cv2.fillConvexPoly(mask, p, 255)
                    else: cv2.polylines(mask, [p], False, 255, thickness)
                    negative[(mask > 0) & (truth == 255) & ~conflicts] = 255; gray[mask > 0] = 0

                line('part', [[88, 202], [350, 202], [350, 353], [88, 353]], 'part_contour', 2*width+3, True)
                dimension = family in ('detached_specks', 'contacting_blot', 'parallel_unknown_strokes', 'interrupted_hatch_next_to_dimension')
                if dimension:
                    base = head('dimension-head', [88, 110], [-1, 0])
                    line('dimension', [base, [350, 110]], 'dimension_line')
                    line('extension-left', [[88, 90], [88, 202]], 'extension_line')
                    line('extension-right', [[350, 90], [350, 202]], 'extension_line')
                if family == 'detached_specks':
                    for j in range(4):
                        x = 176+j*30
                        noise([[x, 110+width+5], [x+3, 110+width+6]], 2)
                if family == 'contacting_blot':
                    noise([[230, 111], [247, 112], [252, 133], [232, 130]], filled=True)
                if family == 'parallel_unknown_strokes':
                    noise([[150, 110+width+4], [285, 110+width+4]])
                    noise([[350+width+4, 140], [350+width+4, 192]])
                if family in ('interrupted_hatch_next_to_dimension', 'hatch_pockets'):
                    for j in range(6):
                        x = 114+j*31
                        line(f'hatch-{j}-a', [[x, 229], [x+10, 260]], 'hatching')
                        line(f'hatch-{j}-b', [[x+14, 275], [x+28, 322]], 'hatching')
                    noise([[208, 261], [223, 243], [237, 274], [221, 285]], filled=True)
                if family == 'title_contact':
                    for j in range(3): line(f'grid-{j}', [[120, 240+j*28], [318, 240+j*28]], 'title_grid')
                    for j in range(4): line(f'grid-v-{j}', [[120+j*66, 240], [120+j*66, 296]], 'title_grid')
                    mask = np.zeros_like(gray); cv2.putText(mask, 'A4 R12', (117, 241), cv2.FONT_HERSHEY_SIMPLEX, .7, 255, 2)
                    mask = cv2.warpAffine(mask, np.array([[1, 0, shift[0]], [0, 1, shift[1]]], float), (size, size))
                    if angle: mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
                    add(mask, 'text')
                if family == 'leader_ambiguous':
                    base = head('leader-head', [88, 271], [1, 0]); line('leader', [[27, 271], base], 'leader_line')
                if family == 'closed_pockets':
                    for j in range(3):
                        y = 226+j*34
                        line(f'pocket-{j}', [[122, y], [304, y], [304, y+14], [122, y+14]], 'part_contour', width, True)
                truth[conflicts] = 255; negative[truth != 255] = 0
                yield {'case': {'id': f'supported-control-{family}-w{width}-r{angle}',
                    'scope': 'supported_role_analytical_control', 'group_id': 'supported-control-'+family,
                    'split': split, 'size_px': [size, size], 'labels': {'family': family,
                        'thin_width_px': width, 'rotation_deg': angle, 'arrows': arrows, 'geometry': geometry,
                        'negative_regions': [], 'expected_dimension_chains': 1 if dimension else 0,
                        'authority': 'analytic role/negative ink; no physical CAD authority'}},
                    'gray': gray, 'truth': truth, 'negative': negative}


if __name__ == '__main__':
    BASE.mkdir(exist_ok=False); jobs = []
    for scene in scenes():
        folder = BASE/scene['case']['id']; folder.mkdir(); views = []
        for name in ('truth', 'negative'): Image.fromarray(scene[name]).save(folder/(name+'.png'))
        for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
            gray = scene['gray'] if scale == 1 else cv2.resize(scene['gray'], (224, 224), interpolation=cv2.INTER_AREA)
            path = folder/(mode+'.png'); Image.fromarray(gray).save(path)
            views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
        jobs.append({'case': scene['case'], 'views': views, 'new_control': True,
            'truth': pin(ROOT, (folder/'truth.png').relative_to(ROOT).as_posix()),
            'negative': pin(ROOT, (folder/'negative.png').relative_to(ROOT).as_posix())})
    value = {'task': 'P10.2c.3b.2a.1', 'generator': pin(ROOT, 'scripts/prepare_supported_role_controls_v1.py'),
        'prepared_before_overlay_implementation': True, 'seed': 671302, 'families': FAMILIES, 'jobs': jobs,
        'gates': {'baseline_known_roles_preserved': True, 'lost_assigned_heads_max': 0,
            'false_chains_on_negative_controls_max': 0, 'additional_negative_role_assignments_max': 0,
            'real_family_F1_must_not_degrade_against_baseline': True, 'new_validation_positive_dimension_additions_required': True,
            'all_new_assignments_have_source_interval_and_chain_evidence': True,
            'whole_role_quality_F1_min': .90, 'source_erasure_max': 0, 'semantic_CAD_promotions_max': 0},
        'closed_test_used': False, 'training': False, 'physical_geometry_approved': False}
    path = BASE/'contract.json'
    with path.open('x', encoding='utf-8') as stream: json.dump(value, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(pin(ROOT, path.relative_to(ROOT).as_posix())))
