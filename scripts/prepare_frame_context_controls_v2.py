"""Prospective thick-frame, contact and distractor families, before model work."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/frame-context-controls-v2'
FAMILIES = [('shallow_double', 'train'), ('landscape', 'train'), ('shared_contacts', 'train'),
            ('isolated_table', 'train'), ('padded_portrait', 'validation'),
            ('close_double', 'validation'), ('offset_contacts', 'validation'), ('rectangular_detail', 'validation')]


def build():
    BASE.mkdir(exist_ok=False); jobs = []; rng = np.random.default_rng(1581271)
    for family, split in FAMILIES:
        for width in (3, 5, 7):
            for angle in ((15, 75) if split == 'train' else (-19, 105)):
                gray = np.full((1024, 1024), 255, np.uint8); truth = np.full_like(gray, 255)
                conflict = np.zeros_like(gray, bool); negative = np.zeros_like(gray)
                theta = np.deg2rad(angle); rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                offset = rng.integers(-3, 4, 2)
                def xy(points): return (np.asarray(points, float)-512) @ rot.T+512+offset
                def apply(mask, role):
                    chosen = mask > 0; gray[chosen] = 0
                    if role is None: negative[chosen & (truth == 255)] = 255
                    else:
                        conflict[:] |= chosen & (truth != 255) & (truth != role)
                        truth[chosen & ~conflict] = role
                def line(points, role, pen=None, closed=False):
                    mask = np.zeros_like(gray)
                    cv2.polylines(mask, [np.rint(xy(points)).astype(np.int32)], closed, 255, pen or 2*width+1)
                    apply(mask, role)
                def rect(box, role, pen=None):
                    x, y, r, b = box; line([[x, y], [r, y], [r, b], [x, b]], role, pen, True)
                def text(value, origin):
                    mask = np.zeros_like(gray)
                    cv2.putText(mask, value, origin, cv2.FONT_HERSHEY_COMPLEX, .72, 255, max(1, width//2), cv2.LINE_8)
                    matrix = np.c_[rot, 512+offset-rot @ np.array([512, 512])]
                    apply(cv2.warpAffine(mask, matrix, (1024, 1024), flags=cv2.INTER_NEAREST), 6)
                has_frame = family not in ('isolated_table', 'rectangular_detail')
                frame = [145, 160, 875, 875]
                if family in ('shallow_double', 'landscape'): frame = [125, 420, 895, 875]
                if family == 'padded_portrait': frame = [400, 145, 880, 880]
                left, top, right, bottom = frame
                if has_frame:
                    if family in ('shallow_double', 'close_double'):
                        gap = 13 if family == 'shallow_double' else width+4
                        rect([left-gap, top-gap, right+gap, bottom+gap], None, 1)
                    rect(frame, 7)
                    cx, cy = left+.25*(right-left), top+.27*(bottom-top)
                    radius = .105*min(right-left, bottom-top)
                    t = np.linspace(0, 2*np.pi, 121)
                    line(np.array([cx, cy])+radius*np.c_[np.cos(t), np.sin(t)], 1, width, True)
                    line([[cx-radius, cy+radius+25], [cx+radius, cy+radius+25]], 2, width)
                elif family == 'rectangular_detail':
                    rect([160, 185, 850, 845], 1)
                    for x in (300, 510, 705):
                        t = np.linspace(0, 2*np.pi, 91)
                        line(np.array([x, 380])+37*np.c_[np.cos(t), np.sin(t)], 1, width, True)
                    for x in range(220, 730, 42): line([[x, 650], [x+22, 600]], 9, 1)
                x0, x1 = int(left+.44*(right-left)), right
                y0, y1 = bottom-200, bottom
                if family == 'isolated_table': x0, x1, y0, y1 = 390, 875, 610, 880
                if family == 'rectangular_detail': x0, x1, y0, y1 = 460, 790, 675, 805
                detached = not has_frame
                if family == 'offset_contacts': x0 -= 13; y0 -= 11
                if family == 'isolated_table':
                    # A separate large symbol does not turn a table into a sheet.
                    t = np.linspace(0, 2*np.pi, 91)
                    line(np.array([260, 340])+62*np.c_[np.cos(t), np.sin(t)], 1, width, True)
                for row in range(3):
                    y = round(y0+row*(y1-y0)/3); line([[x0, y], [x1, y]], 8)
                    text(('E R P 6', 'R P E 6', 'P E R 6')[row], (x0+18, round(y0+(row+1)*(y1-y0)/3)-17))
                if detached: line([[x0, y1], [x1, y1]], 8)
                for x in (x0, round(x0+.47*(x1-x0))): line([[x, y0], [x, y1]], 8)
                if detached: line([[x1, y0], [x1, y1]], 8)
                crop = [x0-30, y0-30, min(960, x1+30), min(960, y1+30)]
                corners = xy([[crop[0], crop[1]], [crop[2], crop[1]], [crop[2], crop[3]], [crop[0], crop[3]]])
                lo = np.maximum(0, np.floor(corners.min(0))).astype(int); hi = np.minimum(1024, np.ceil(corners.max(0))).astype(int)
                hi = np.minimum(1024, hi+(hi-lo) % 2); x, y = lo; r, b = hi
                truth[conflict] = 255; negative[(truth != 255) | (gray == 255)] = 0
                roi = gray[y:b, x:r]; identity = f'frame-contact-{family}-w{width}-r{angle}'
                folder = BASE/identity; folder.mkdir(); Image.fromarray(gray).save(folder/'page.png')
                Image.fromarray(truth[y:b, x:r]).save(folder/'truth.png'); Image.fromarray(negative[y:b, x:r]).save(folder/'negative.png')
                views = []
                for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
                    a = roi if scale == 1 else cv2.resize(roi, (roi.shape[1]//2, roi.shape[0]//2), interpolation=cv2.INTER_AREA)
                    path = folder/(mode+'.png'); Image.fromarray(a).save(path)
                    views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                                  'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
                page = {'source': pin(ROOT, (folder/'page.png').relative_to(ROOT).as_posix()), 'size_px': [1024, 1024],
                        'source_size_px': [1024, 1024], 'overview_to_source': np.eye(3).tolist(),
                        'patch_to_source': [[1, 0, int(x)], [0, 1, int(y)], [0, 0, 1]],
                        'source_to_patch': [[1, 0, -int(x)], [0, 1, -int(y)], [0, 0, 1]],
                        'patch_bbox': [int(x), int(y), int(r), int(b)], 'kind': 'synthetic_full_page_pixels', 'source_verified': True}
                jobs.append({'case': {'id': identity, 'scope': 'frame_contact_analytical', 'split': split,
                             'group_id': 'frame-contact-'+family, 'size_px': list(roi.shape[::-1]),
                             'labels': {'family': family, 'arrows': [], 'chains': []}},
                             'views': views, 'page_source': page, 'new_control': True,
                             **{name: pin(ROOT, (folder/(name+'.png')).relative_to(ROOT).as_posix()) for name in ('truth', 'negative')}})
    write(BASE/'contract.json', {'task': 'P10.2c.3b.2h.3b.1', 'seed': 1581271, 'families': FAMILIES, 'jobs': jobs,
          'generator': pin(ROOT, 'scripts/prepare_frame_context_controls_v2.py'), 'prepared_before_model': True,
          'previous_validation_now_development': True, 'criteria': {'maximum_new_wrong_or_noise_roles': 0,
          'maximum_lost_previously_correct_roles': 0, 'maximum_withdrawn_correct_grid_candidates': 0,
          'minimum_fresh_wrong_grid_removed_per_resolution': 1, 'minimum_fresh_positive_retention': .9,
          'all_known_864_real_errors_removed': True, 'no_new_part_or_frame_roles': True, 'full_role_macro_F1_min': .9},
          'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 1800},
          'closed_test_used': False, 'production_changed': False, 'physical_geometry_approved': False})
    print({'cases': len(jobs), 'contract': pin(ROOT, (BASE/'contract.json').relative_to(ROOT).as_posix())}, flush=True)


if __name__ == '__main__': build()
