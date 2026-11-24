"""Prospective pixel-phase and unequal-width frame contacts, before refinement."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/native-frame-controls-v1'
FAMILIES = [('integer_contact', 'train'), ('near_parallel', 'train'), ('offset_t', 'train'),
            ('isolated_table', 'train'), ('acute_contact', 'validation'), ('thin_on_thick', 'validation'),
            ('adjacent_part', 'validation'), ('nested_sheet', 'validation')]


def build():
    BASE.mkdir(exist_ok=False); rng = np.random.default_rng(1611397); jobs = []
    for family, split in FAMILIES:
        for width in (3, 6):
            for angle in ((23, 67) if split == 'train' else (37, 113)):
                gray = np.full((1024, 1024), 255, np.uint8); truth = np.full_like(gray, 255)
                conflict = np.zeros_like(gray, bool); negative = np.zeros_like(gray)
                theta = np.deg2rad(angle); rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-3, 4, 2)
                def xy(points): return (np.asarray(points, float)-512) @ rot.T+512+shift
                def apply(mask, role):
                    chosen = mask > 0; gray[chosen] = 0
                    if role is None: negative[chosen & (truth == 255)] = 255
                    else:
                        conflict[:] |= chosen & (truth != 255) & (truth != role)
                        truth[chosen & ~conflict] = role
                def line(points, role, pen, closed=False):
                    mask = np.zeros_like(gray)
                    cv2.polylines(mask, [np.rint(xy(points)).astype(np.int32)], closed, 255, pen)
                    apply(mask, role)
                def rectangle(box, role, pen):
                    x, y, r, b = box; line([[x, y], [r, y], [r, b], [x, b]], role, pen, True)
                def text(value, origin):
                    mask = np.zeros_like(gray)
                    cv2.putText(mask, value, origin, cv2.FONT_HERSHEY_COMPLEX, .75, 255, max(1, width//2), cv2.LINE_8)
                    apply(cv2.warpAffine(mask, np.c_[rot, 512+shift-rot @ np.array([512, 512])], (1024, 1024), flags=cv2.INTER_NEAREST), 6)
                frame = [165, 180, 855, 855]; left, top, right, bottom = frame
                has_frame = family != 'isolated_table'
                if has_frame:
                    rectangle(frame, 7, 2*width+1)
                    if family == 'nested_sheet': rectangle([left-17, top-17, right+17, bottom+17], None, 1)
                    t = np.linspace(0, 2*np.pi, 101)
                    line(np.array([345, 365])+62*np.c_[np.cos(t), np.sin(t)], 1, width, True)
                x0, x1, y0, y1 = 485, right, 640, bottom
                if family == 'offset_t': x0 -= 11; y0 -= 7
                if family == 'isolated_table': x0, x1, y0, y1 = 470, 855, 600, 850
                grid_width = max(1, width//2) if family == 'thin_on_thick' else 2*width+1
                for i, value in enumerate(('P E R 7', 'R P E 7', 'E R P 7')):
                    y = round(y0+i*(y1-y0)/3); line([[x0, y], [x1, y]], 8, grid_width)
                    text(value, (x0+18, round(y0+(i+1)*(y1-y0)/3)-18))
                for x in (x0, round(x0+.49*(x1-x0))): line([[x, y0], [x, y1]], 8, grid_width)
                if not has_frame:
                    line([[x0, y1], [x1, y1]], 8, grid_width); line([[x1, y0], [x1, y1]], 8, grid_width)
                if family == 'near_parallel':
                    y = bottom-3*(2*width+1)-5; line([[left+40, y], [x0-35, y]], 2, width)
                if family == 'adjacent_part':
                    rectangle([330, 670, x0-35, bottom-2*width-11], 1, width)
                    for y in range(695, 790, 22): line([[345, y], [375, y-13]], 9, 1)
                crop = [x0-37, y0-33, x1+27, y1+29]
                if family == 'adjacent_part': crop[0] = 310
                corners = xy([[crop[0], crop[1]], [crop[2], crop[1]], [crop[2], crop[3]], [crop[0], crop[3]]])
                lo = np.maximum(0, np.floor(corners.min(0))).astype(int); hi = np.minimum(1024, np.ceil(corners.max(0))).astype(int)
                hi = np.minimum(1024, hi+(hi-lo) % 2); x, y = lo; r, b = hi
                truth[conflict] = 255; negative[(truth != 255) | (gray == 255)] = 0
                roi = gray[y:b, x:r]; identity = f'native-frame-{family}-w{width}-r{angle}'; folder = BASE/identity; folder.mkdir()
                Image.fromarray(gray).save(folder/'page.png')
                for name, a in [('truth', truth[y:b, x:r]), ('negative', negative[y:b, x:r])]: Image.fromarray(a).save(folder/(name+'.png'))
                views = []
                for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
                    a = roi if scale == 1 else cv2.resize(roi, (roi.shape[1]//2, roi.shape[0]//2), interpolation=cv2.INTER_AREA)
                    path = folder/(mode+'.png'); Image.fromarray(a).save(path)
                    views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                                  'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
                jobs.append({'case': {'id': identity, 'scope': 'native_frame_analytical', 'split': split,
                                     'group_id': 'native-frame-'+family, 'size_px': list(roi.shape[::-1]),
                                     'labels': {'family': family, 'arrows': [], 'chains': []}}, 'views': views, 'new_control': True,
                             'page_source': {'source': pin(ROOT, (folder/'page.png').relative_to(ROOT).as_posix()), 'size_px': [1024, 1024],
                                 'source_size_px': [1024, 1024], 'overview_to_source': np.eye(3).tolist(),
                                 'patch_to_source': [[1, 0, int(x)], [0, 1, int(y)], [0, 0, 1]],
                                 'source_to_patch': [[1, 0, -int(x)], [0, 1, -int(y)], [0, 0, 1]],
                                 'patch_bbox': [int(x), int(y), int(r), int(b)], 'kind': 'synthetic_full_page_pixels', 'source_verified': True},
                             **{name: pin(ROOT, (folder/(name+'.png')).relative_to(ROOT).as_posix()) for name in ('truth', 'negative')}})
    write(BASE/'contract.json', {'task': 'P10.2c.3b.2h.3b.2', 'jobs': jobs, 'families': FAMILIES, 'seed': 1611397,
          'generator': pin(ROOT, 'scripts/prepare_native_frame_controls_v1.py'), 'prepared_before_model': True,
          'coordinate_contract': {'output': 'native_source_pixel_centres', 'input': 'native_or_half_pixel_centres',
              'source_context': 'authenticated_unannotated_original_ROI_pixels', 'native_prediction_transfer': False,
              'nearest_projection_is_reference_only': True, 'uncertainty_excluded_from_metrics': False},
          'criteria': {'maximum_new_wrong_or_noise_roles': 0, 'maximum_lost_previously_correct_roles': 0,
              'maximum_withdrawn_correct_grid_candidates': 0, 'minimum_fresh_wrong_grid_removed_per_resolution': 1,
              'minimum_fresh_positive_retention': .9, 'all_known_864_real_errors_removed': True,
              'no_new_part_or_frame_roles': True, 'full_role_macro_F1_min': .9},
          'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 1800},
          'production_changed': False, 'closed_test_used': False, 'physical_geometry_approved': False})
    print({'cases': len(jobs), 'contract': pin(ROOT, (BASE/'contract.json').relative_to(ROOT).as_posix())}, flush=True)


if __name__ == '__main__': build()
