"""Prospective whole pages and ROI crops; frame GT is never model input."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/frame-context-controls-v1'
FAMILIES = [('single_sheet', 'train'), ('double_sheet', 'train'), ('standalone_table', 'train'),
            ('rectangular_part', 'train'), ('shifted_canvas', 'validation'), ('gapped_grid_sheet', 'validation'),
            ('inset_table_with_frame', 'validation'), ('framed_hatched_part', 'validation')]


def build():
    BASE.mkdir(exist_ok=False); jobs = []; rng = np.random.default_rng(1017593)
    for family, split in FAMILIES:
        for width in (3, 5):
            for angle in ((0, 11) if split == 'train' else (-7, 90)):
                size = 1024; gray = np.full((size, size), 255, np.uint8)
                truth = np.full_like(gray, 255); conflict = np.zeros_like(gray, bool); negative = np.zeros_like(gray)
                theta = np.deg2rad(angle); rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-2, 3, 2)
                def xy(points): return (np.asarray(points, float)-512) @ rotation.T+512+shift
                def apply(mask, role):
                    chosen = mask > 0; gray[chosen] = 0
                    if role is None: negative[chosen & (truth == 255)] = 255
                    else:
                        conflict[:] |= chosen & (truth != 255) & (truth != role)
                        truth[chosen & ~conflict] = role
                def line(points, role, pen=width, closed=False):
                    mask = np.zeros_like(gray); cv2.polylines(mask, [np.rint(xy(points)).astype(np.int32)], closed, 255, pen); apply(mask, role)
                def text(value, origin):
                    mask = np.zeros_like(gray); cv2.putText(mask, value, origin, cv2.FONT_HERSHEY_COMPLEX, .9, 255, max(1, width//2), cv2.LINE_8)
                    affine = np.c_[rotation, 512+shift-rotation @ np.array([512, 512])]
                    apply(cv2.warpAffine(mask, affine, (size, size), flags=cv2.INTER_NEAREST), 6)
                has_frame = family not in ('standalone_table', 'rectangular_part')
                frame = [110, 120, 910, 900] if family != 'shifted_canvas' else [90, 410, 930, 950]
                left, top, right, bottom = frame
                if has_frame:
                    if family in ('double_sheet', 'shifted_canvas'):
                        line([[left-10, top-10], [right+10, top-10], [right+10, bottom+10], [left-10, bottom+10]], None, 1, True)
                    line([[left, top], [right, top], [right, bottom], [left, bottom]], 7, 2*width+1, True)
                if family == 'rectangular_part':
                    line([[125, 150], [890, 150], [890, 870], [125, 870]], 1, 2*width+1, True)
                    for x in (270, 500, 730):
                        t = np.linspace(0, 2*np.pi, 101); line(np.array([x, 390])+45*np.c_[np.cos(t), np.sin(t)], 1, width, True)
                    crop = [520, 615, 930, 915]
                else:
                    # Independent drawing content, not text dots or table cells.
                    if has_frame:
                        cx, cy = left+.27*(right-left), top+.32*(bottom-top)
                        radius = .12*min(right-left, bottom-top)
                        t = np.linspace(0, 2*np.pi, 121)
                        line(np.array([cx, cy])+radius*np.c_[np.cos(t), np.sin(t)], 1, 2*width+1, True)
                        line([[cx-radius, cy+radius+30], [cx+radius, cy+radius+30]], 2, width)
                    if family == 'framed_hatched_part':
                        x0, y0, x1, y1 = 450, 580, 790, 810
                        line([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 1, 2*width+1, True)
                        for x in range(470, 750, 35): line([[x, 802], [x+35, 740]], 9, 1)
                        crop = [425, 555, 945, 935]
                    else:
                        x0, x1 = int(left+.48*(right-left)), right
                        y0, y1 = bottom-210, bottom
                        detached = family in ('standalone_table', 'gapped_grid_sheet', 'inset_table_with_frame')
                        if family == 'standalone_table': x0, x1, y0, y1 = 490, 910, 630, 900
                        if family == 'gapped_grid_sheet': x1 -= 22; y1 -= 24; y0 -= 24
                        if family == 'inset_table_with_frame': x0, x1, y0, y1 = 420, 815, 540, 810
                        ys = (y0, y0+(y1-y0)//3, y0+2*(y1-y0)//3, y1)
                        for y in ys[:-1]: line([[x0, y], [x1, y]], 8, 2*width+1)
                        if detached: line([[x0, y1], [x1, y1]], 8, 2*width+1)
                        for x in (x0, x0+190): line([[x, y0], [x, y1]], 8, 2*width+1)
                        if detached: line([[x1, y0], [x1, y1]], 8, 2*width+1)
                        for value, y in zip(('R E P 4', 'E R P 4', 'P E R 4'), [ys[i+1]-18 for i in range(3)]): text(value, (x0+18, y))
                        crop = [x0-25, y0-25, min(right+18, 985), min(bottom+22, 985)]
                line([[170, 95], [185, 102]], None, 1)
                truth[conflict] = 255; negative[(truth != 255) | (gray == 255)] = 0
                corners = xy([[crop[0], crop[1]], [crop[2], crop[1]], [crop[2], crop[3]], [crop[0], crop[3]]])
                lo = np.maximum(0, np.floor(corners.min(0))).astype(int); hi = np.minimum(size, np.ceil(corners.max(0))).astype(int)
                # Exact-double half inputs keep evaluation and centre mappings explicit.
                hi = np.minimum(size, hi+(hi-lo) % 2)
                x, y = lo; r, b = hi
                roi, target, neg = gray[y:b, x:r], truth[y:b, x:r], negative[y:b, x:r]
                identity = f'frame-context-{family}-w{width}-r{angle}'; folder = BASE/identity; folder.mkdir()
                Image.fromarray(gray).save(folder/'page.png')
                for n, a in [('truth', target), ('negative', neg)]: Image.fromarray(a).save(folder/(n+'.png'))
                page = {'source': pin(ROOT, (folder/'page.png').relative_to(ROOT).as_posix()), 'size_px': [size, size],
                        'overview_to_source': np.eye(3).tolist(), 'source_to_patch': [[1, 0, -int(x)], [0, 1, -int(y)], [0, 0, 1]],
                        'source_size_px': [size, size], 'patch_to_source': [[1, 0, int(x)], [0, 1, int(y)], [0, 0, 1]],
                        'patch_bbox': [int(x), int(y), int(r), int(b)], 'kind': 'synthetic_full_page_pixels', 'source_verified': True}
                views = []
                for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
                    pixels = roi if scale == 1 else cv2.resize(roi, (roi.shape[1]//2, roi.shape[0]//2), interpolation=cv2.INTER_AREA)
                    path = folder/(mode+'.png'); Image.fromarray(pixels).save(path)
                    views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                                  'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
                jobs.append({'case': {'id': identity, 'scope': 'frame_context_analytical', 'split': split,
                                     'group_id': 'frame-context-'+family, 'size_px': list(roi.shape[::-1]),
                                     'labels': {'family': family, 'arrows': [], 'chains': []}},
                             'views': views, 'page_source': page, 'new_control': True,
                             **{n: pin(ROOT, (folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth', 'negative')}})
    write(BASE/'contract.json', {'task': 'P10.2c.3b.2h.3b', 'seed': 1017593, 'families': FAMILIES, 'jobs': jobs,
          'generator': pin(ROOT, 'scripts/prepare_frame_context_controls_v1.py'), 'prepared_before_frame_algorithm': True,
          'previous_validation_now_development': True, 'criteria': {'prior_864_frame_to_grid_errors_removed': True,
          'maximum_lost_known_correct_roles': 0, 'maximum_new_wrong_or_noise_roles': 0,
          'minimum_fresh_wrong_grid_roles_removed_per_resolution': 1, 'minimum_fresh_correct_positive_gain_retention': .9,
          'no_new_part_after_no_part_reference': True, 'no_GT_frame_or_bbox_inputs': True, 'source_and_review_preserved': True,
          'full_role_macro_F1_min': .90}, 'batching': {'old_cases_per_child': 8, 'fresh_cases_per_child': 1},
          'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 1800},
          'production_changed': False, 'physical_geometry_approved': False, 'closed_test_used': False})
    print(pin(ROOT, 'artifacts/frame-context-controls-v1/contract.json'), flush=True)


if __name__ == '__main__': build()
