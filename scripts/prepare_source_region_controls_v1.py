"""Prospective positive-role controls with similar-looking semantic negatives."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/source-region-controls-v1'
FAMILIES = [('ruled_register', 'train'), ('open_register', 'train'),
            ('lettered_windows', 'train'), ('engraved_mesh', 'train'),
            ('broken_schedule', 'validation'), ('captioned_form', 'validation'),
            ('marked_grid_part', 'validation'), ('crosshatch_notes', 'validation')]


def build():
    BASE.mkdir(exist_ok=False)
    rng = np.random.default_rng(815317)
    jobs = []
    for family, split in FAMILIES:
        for width in (3, 5):
            for angle in ((23, 71) if split == 'train' else (37, 127)):
                size = 640
                gray = np.full((size, size), 255, np.uint8)
                truth = np.full_like(gray, 255)
                conflict = np.zeros_like(gray, bool)
                negative = np.zeros_like(gray)
                theta = np.deg2rad(angle)
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-3, 4, 2)

                def transform(p):
                    return (np.asarray(p, float)-size/2) @ rotation.T+size/2+shift

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

                def text(value, origin, pen=None, scale=.9):
                    raw = np.zeros_like(gray)
                    cv2.putText(raw, value, origin, cv2.FONT_HERSHEY_COMPLEX, scale, 255,
                                max(1, width//2) if pen is None else pen, cv2.LINE_8)
                    matrix = np.c_[rotation, size/2+shift-rotation @ np.array([size/2, size/2])]
                    apply(cv2.warpAffine(raw, matrix, (size, size), flags=cv2.INTER_NEAREST), 6)

                xs, ys = (130, 330, 525), (150, 245, 350, 455)
                is_table = family in ('ruled_register', 'open_register', 'broken_schedule', 'captioned_form')
                is_part = family in ('lettered_windows', 'marked_grid_part')
                if is_table or is_part:
                    role = 8 if is_table else 1
                    for y in ys:
                        line([[xs[0], y], [xs[-1], y]], role, 2*width+1)
                    for x in xs:
                        if family == 'broken_schedule' and x == xs[1]:
                            line([[x, ys[0]], [x, 285]], role, 2*width+1)
                            line([[x, 301], [x, ys[-1]]], role, 2*width+1)
                        elif family in ('open_register', 'captioned_form') and x == xs[1]:
                            line([[x, ys[1]], [x, ys[-1]]], role, 2*width+1)
                        else:
                            line([[x, ys[0]], [x, ys[-1]]], role, 2*width+1)
                    if is_part:
                        for x in (275, 460):
                            for y in (205, 302, 407):
                                t = np.linspace(0, 2*np.pi, 101)
                                line(np.array([x, y])+14*np.c_[np.cos(t), np.sin(t)], 1, width, True)
                        # Deliberately the same aligned complex glyphs as a table.
                        # Grid regularity and text texture alone cannot decide.
                    for value, y in [('R E P 4', 220), ('E R P 4', 320), ('P E R 4', 425)]:
                        text(value, (148, y), scale=.85)
                else:
                    line([[130, 150], [525, 150], [525, 455], [130, 455]], 1, 2*width+1, True)
                    for x in (165, 245, 325, 405, 485):
                        line([[x, 170], [x, 435]], 9)
                    for y in (188, 268, 348, 428):
                        line([[150, y], [505, y]], 9)
                    for value, y in [('R E P 4', 230), ('P E R 4', 310), ('E R P 4', 390)]:
                        text(value, (180, y), scale=.75)
                line([[195, 502], [211, 511]], None, 1)
                line([[405, 102], [416, 98]], None, 1)
                truth[conflict] = 255
                negative[(truth != 255) | (gray == 255)] = 0
                identity = f'source-region-{family}-w{width}-r{angle}'
                folder = BASE/identity
                folder.mkdir()
                for name, pixels in [('truth', truth), ('negative', negative)]:
                    Image.fromarray(pixels).save(folder/(name+'.png'))
                views = []
                for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
                    image = gray if scale == 1 else cv2.resize(gray, (size//2, size//2), interpolation=cv2.INTER_AREA)
                    p = folder/(mode+'.png')
                    Image.fromarray(image).save(p)
                    views.append({'mode': mode, 'gray': pin(ROOT, p.relative_to(ROOT).as_posix()),
                                  'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
                jobs.append({'case': {'id': identity, 'scope': 'source_region_analytical',
                                     'group_id': 'source-region-'+family, 'split': split, 'size_px': [size, size],
                                     'labels': {'family': family, 'arrows': [], 'chains': []}},
                             'views': views, 'new_control': True,
                             **{n: pin(ROOT, (folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth', 'negative')}})
    write(BASE/'contract.json', {'task': 'P10.2c.3b.2h.3', 'seed': 815317, 'families': FAMILIES, 'jobs': jobs,
          'generator': pin(ROOT, 'scripts/prepare_source_region_controls_v1.py'),
          'prepared_before_positive_role_algorithm': True, 'previous_validation_now_development': True,
          'criteria': {'minimum_fresh_correct_positive_assignments_per_resolution': 32,
                       'minimum_real_correct_positive_assignments': 1, 'maximum_new_wrong_or_noise_assignments': 0,
                       'maximum_lost_known_correct_roles': 0, 'positive_coverage_and_accuracy_with_unknown_must_increase': True,
                       'minimum_incremental_precision_on_scored_pixels': .98,
                       'source_ownership_review_and_canonical_geometry_preserved': True, 'full_role_macro_F1_min': .90},
          'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 2400},
          'closed_test_used': False, 'production_changed': False, 'physical_geometry_approved': False})
    print(pin(ROOT, 'artifacts/source-region-controls-v1/contract.json'), flush=True)


if __name__ == '__main__':
    build()
