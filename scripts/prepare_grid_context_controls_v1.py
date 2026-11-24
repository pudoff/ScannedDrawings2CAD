"""Prospective scale-context controls; construction labels are evaluation only."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/grid-context-controls-v1'
FAMILIES = [('unequal_columns', 'train'), ('open_schedule', 'train'),
            ('engraved_mesh', 'train'), ('machined_cells', 'train'),
            ('merged_schedule', 'validation'), ('interrupted_columns', 'validation'),
            ('noted_mesh', 'validation'), ('numbered_windows', 'validation')]


def build():
    BASE.mkdir(exist_ok=False)
    rng = np.random.default_rng(714239)
    jobs = []
    for family, split in FAMILIES:
        for width in (3, 5):
            for angle in ((17, 73) if split == 'train' else (29, 119)):
                size = 576
                gray = np.full((size, size), 255, np.uint8)
                truth = np.full_like(gray, 255)
                conflict = np.zeros_like(gray, bool)
                negative = np.zeros_like(gray)
                theta = np.deg2rad(angle)
                rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-4, 5, 2)

                def transform(points):
                    return (np.asarray(points, float) - size/2) @ rot.T + size/2 + shift

                def apply(mask, role):
                    selected = mask > 0
                    gray[selected] = 0
                    if role is None:
                        negative[selected & (truth == 255)] = 255
                    else:
                        conflict[:] |= selected & (truth != 255) & (truth != role)
                        truth[selected & ~conflict] = role

                def line(points, role, pen=width, closed=False):
                    mask = np.zeros_like(gray)
                    cv2.polylines(mask, [np.rint(transform(points)).astype(np.int32)], closed, 255, pen)
                    apply(mask, role)

                def text(value, origin, scale=1.1):
                    raw = np.zeros_like(gray)
                    cv2.putText(raw, value, origin, cv2.FONT_HERSHEY_COMPLEX, scale, 255, width, cv2.LINE_8)
                    matrix = np.c_[rot, size/2 + shift - rot @ np.array([size/2, size/2])]
                    apply(cv2.warpAffine(raw, matrix, (size, size), flags=cv2.INTER_NEAREST), 6)

                if family in ('unequal_columns', 'open_schedule', 'merged_schedule', 'interrupted_columns'):
                    xs = (86, 246, 484)
                    ys = (107, 213, 329, 455)
                    for y in ys:
                        line([[xs[0], y], [xs[-1], y]], 8, 2*width+1)
                    for x in xs:
                        if family == 'interrupted_columns' and x == xs[1]:
                            line([[x, ys[0]], [x, 258]], 8, 2*width+1)
                            line([[x, 272], [x, ys[-1]]], 8, 2*width+1)
                        elif family in ('open_schedule', 'merged_schedule') and x == xs[1]:
                            line([[x, ys[1]], [x, ys[-1]]], 8, 2*width+1)
                        else:
                            line([[x, ys[0]], [x, ys[-1]]], 8, 2*width+1)
                    for value, y in [('RP 38', 183), ('PR 64', 300), ('AP 79', 424)]:
                        text(value, (109, y))
                elif family in ('engraved_mesh', 'noted_mesh'):
                    line([[90, 105], [481, 105], [481, 461], [90, 461]], 1, 2*width+1, True)
                    for x in (125, 205, 285, 365, 445):
                        line([[x, 125], [x, 439]], 9)
                    for y in (148, 228, 308, 388):
                        line([[110, y], [462, y]], 9)
                    # Genuine hatch with nearby text is a critical negative.
                    for value, y in [('RP 38', 200), ('PR 64', 280), ('AP 79', 360)]:
                        text(value, (140, y), .85)
                else:
                    line([[90, 105], [481, 105], [481, 461], [90, 461]], 1, 2*width+1, True)
                    for x in (217, 351):
                        line([[x, 105], [x, 461]], 1, 2*width+1)
                    for y in (222, 341):
                        line([[90, y], [481, y]], 1, 2*width+1)
                    for x in (150, 280, 415):
                        for y in (160, 280, 402):
                            t = np.linspace(0, 2*np.pi, 97)
                            line(np.array([x, y]) + 17*np.c_[np.cos(t), np.sin(t)], 1, width, True)
                    if family == 'numbered_windows':
                        for value, x, y in [('R1', 110, 205), ('P2', 240, 326), ('A3', 375, 445)]:
                            text(value, (x, y), .7)
                line([[129, 494], [147, 500]], None, 1)
                truth[conflict] = 255
                negative[(truth != 255) | (gray == 255)] = 0
                identity = f'grid-context-{family}-w{width}-r{angle}'
                folder = BASE / identity
                folder.mkdir()
                for name, pixels in [('truth', truth), ('negative', negative)]:
                    Image.fromarray(pixels).save(folder / (name+'.png'))
                views = []
                for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
                    image = gray if scale == 1 else cv2.resize(gray, (size//2, size//2), interpolation=cv2.INTER_AREA)
                    path = folder / (mode+'.png')
                    Image.fromarray(image).save(path)
                    views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                                  'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
                jobs.append({'case': {'id': identity, 'scope': 'grid_context_analytical',
                                     'group_id': 'grid-context-'+family, 'split': split, 'size_px': [size, size],
                                     'labels': {'family': family, 'arrows': [], 'chains': []}},
                             'views': views, 'new_control': True,
                             **{n: pin(ROOT, (folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth', 'negative')}})
    write(BASE/'contract.json', {'task': 'P10.2c.3b.2h.2', 'seed': 714239, 'jobs': jobs, 'families': FAMILIES,
          'generator': pin(ROOT, 'scripts/prepare_grid_context_controls_v1.py'),
          'prepared_before_context_algorithm': True, 'previous_validation_is_development': True,
          'criteria': {'minimum_fresh_half_wrong_roles_removed_increment': 32,
                       'minimum_old_half_wrong_roles_removed_increment': 1,
                       'maximum_lost_known_correct_roles': 0, 'maximum_new_wrong_or_noise_roles': 0,
                       'previous_guarded_revisions_preserved': True, 'source_and_review_protection_exact': True,
                       'half_inference_never_reads_native_or_GT': True, 'full_role_macro_F1_min': .90},
          'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 2400},
          'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, 'artifacts/grid-context-controls-v1/contract.json'), flush=True)


if __name__ == '__main__':
    build()
