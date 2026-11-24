"""Fresh dimensioned parts and misleading references on repeated hatch rows."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/part-admission-controls-v1'
FAMILIES = [('solid_tab', 'train'), ('notched_bracket', 'train'),
            ('parallel_strip_reference', 'train'), ('closed_mesh_reference', 'train'),
            ('stepped_shell', 'validation'), ('arc_shell', 'validation'),
            ('offset_hatch_reference', 'validation'), ('headed_hatch_reference', 'validation')]


def build():
    BASE.mkdir(exist_ok=False)
    rng = np.random.default_rng(916453)
    jobs = []
    for family, split in FAMILIES:
        for width in (3, 5):
            for angle in ((17, 59) if split == 'train' else (43, 121)):
                gray = np.full((640, 640), 255, np.uint8)
                truth = np.full_like(gray, 255)
                conflict = np.zeros_like(gray, bool)
                negative = np.zeros_like(gray)
                theta = np.deg2rad(angle)
                rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-3, 4, 2)

                def xy(points): return (np.asarray(points, float)-320) @ rotation.T+320+shift

                def paint(points, role, pen, filled=False):
                    mask = np.zeros_like(gray)
                    p = np.rint(xy(points)).astype(np.int32)
                    if filled: cv2.fillConvexPoly(mask, p, 255)
                    else: cv2.polylines(mask, [p], False, 255, pen)
                    selected = mask > 0
                    gray[selected] = 0
                    if role is None: negative[selected & (truth == 255)] = 255
                    elif role == 255: conflict[:] |= selected
                    else:
                        conflict[:] |= selected & (truth != 255) & (truth != role)
                        truth[selected & ~conflict] = role

                is_part = family in ('solid_tab', 'notched_bracket', 'stepped_shell', 'arc_shell')
                body_pen = 2*width+3
                feet = [[190, 235], [450, 235]]
                if is_part:
                    body = [[130, 235], [530, 235], [530, 410], [130, 410], [130, 235]]
                    if family in ('notched_bracket', 'stepped_shell'):
                        body = [[130, 235], [290, 235], [290, 270], [530, 270], [530, 410], [130, 410], [130, 235]]
                        feet[1][1] = 270
                        if family == 'stepped_shell':
                            body = [[130, 235], [265, 235], [305, 270], [530, 270], [530, 410], [130, 410], [130, 235]]
                    if family == 'arc_shell':
                        t = np.linspace(0, np.pi, 97)
                        arc = np.c_[530-400*t/np.pi, 410+25*np.sin(t)]
                        body = np.r_[[[130, 235], [530, 235]], arc, [[130, 235]]]
                    paint(body, 1, body_pen)
                    for x in range(165, 480, 43): paint([[x, 405], [x+39, 352]], 9, 1)
                else:
                    ys = (235, 265, 295, 325, 355, 385)
                    feet = [[190, 295], [450, 295]]
                    for i, y in enumerate(ys):
                        left = 130+(i % 2)*12 if family == 'offset_hatch_reference' else 130
                        right = 530-(i % 3)*9 if family == 'offset_hatch_reference' else 530
                        paint([[left, y], [right, y]], 9, body_pen)
                    paint([[130, 235], [130, 385]], 9, body_pen)
                    if family == 'closed_mesh_reference': paint([[530, 235], [530, 385]], 9, body_pen)
                    if family == 'headed_hatch_reference': paint([[275, 325], [300, 314], [300, 336]], 255, 1, True)
                paint([[190, 145], [450, 145]], 2, width)
                for foot in feet: paint([[foot[0], 115], foot], 3, width)
                for x, sign in ((190, -1), (450, 1)):
                    tip = np.array([x, 145]); back = tip-[sign*(28+width), 0]
                    paint([tip, back+[0, 6+width], back-[0, 6+width]], 255, 1, True)
                paint([[180, 475], [198, 482]], None, 1)
                truth[conflict] = 255
                negative[(truth != 255) | (gray == 255)] = 0
                identity = f'part-admission-{family}-w{width}-r{angle}'
                folder = BASE/identity
                folder.mkdir()
                for n, a in [('truth', truth), ('negative', negative)]: Image.fromarray(a).save(folder/(n+'.png'))
                views = []
                for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
                    pixels = gray if scale == 1 else cv2.resize(gray, (320, 320), interpolation=cv2.INTER_AREA)
                    path = folder/(mode+'.png'); Image.fromarray(pixels).save(path)
                    views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                                  'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
                jobs.append({'case': {'id': identity, 'scope': 'part_admission_analytical', 'split': split,
                                     'group_id': 'part-admission-'+family, 'size_px': [640, 640],
                                     'labels': {'family': family, 'arrows': [], 'chains': []}},
                             'views': views, 'new_control': True,
                             **{n: pin(ROOT, (folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth', 'negative')}})
    write(BASE/'contract.json', {'task': 'P10.2c.3b.2h.3a', 'seed': 916453, 'families': FAMILIES, 'jobs': jobs,
          'generator': pin(ROOT, 'scripts/prepare_part_admission_controls_v1.py'),
          'prepared_before_admission_algorithm': True, 'previous_validation_now_development': True,
          'criteria': {'maximum_wrong_new_part_pixels': 0, 'maximum_lost_previously_correct_roles': 0,
                       'all_13850_prior_hatch_to_part_errors_removed': True, 'all_23_prior_hatching_losses_restored': True,
                       'text_and_grid_candidates_exact': True, 'minimum_fresh_correct_new_part_per_resolution_for_canonical': 16,
                       'no_part_is_safe_abstention_reference_not_positive_part_gain': True,
                       'frame_864_residual_is_separate_P10_2c_3b_2h_3b': True,
                       'source_geometry_heads_links_and_review_preserved': True, 'full_role_macro_F1_min': .90},
          'batching': {'old_cases_per_child': 8, 'fresh_cases_per_child': 1,
                       'each_case_inferred_independently': True, 'batch_membership_not_model_input': True},
          'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 1800},
          'production_changed': False, 'physical_geometry_approved': False, 'closed_test_used': False})
    print(pin(ROOT, 'artifacts/part-admission-controls-v1/contract.json'), flush=True)


if __name__ == '__main__': build()
