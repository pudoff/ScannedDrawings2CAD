"""Prospective raster controls for local dimension references on open source paths."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/open-reference-controls-v1'
FAMILIES = [('open_u', 'train'), ('pass_through', 'train'), ('thick_transition', 'train'),
            ('equal_pen_open', 'train'), ('broken_extension', 'train'),
            ('stepped_open', 'validation'), ('angled_open', 'validation'),
            ('disconnected_refs', 'validation'), ('frame_reference', 'validation'), ('hatch_reference', 'validation')]


def build():
    BASE.mkdir(exist_ok=False); jobs = []; rng = np.random.default_rng(742619)
    for family, split in FAMILIES:
        for width in (3, 5):
            for angle in ((0, 61) if split == 'train' else (27, 137)):
                gray = np.full((512, 512), 255, np.uint8); truth = np.full_like(gray, 255)
                conflict = np.zeros(gray.shape, bool); negative = np.zeros_like(gray)
                theta = np.deg2rad(angle); rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-3, 4, 2)
                def xy(points): return (np.asarray(points, float)-256) @ rotation.T + 256 + shift
                def stroke(points, cls, pen, filled=False):
                    mask = np.zeros_like(gray); p = np.rint(xy(points)).astype(np.int32)
                    if filled: cv2.fillConvexPoly(mask, p, 255)
                    else: cv2.polylines(mask, [p], False, 255, pen)
                    selected = mask > 0; gray[selected] = 0
                    if cls == 255: conflict[:] |= selected
                    else:
                        conflict[:] |= selected & (truth != 255) & (truth != cls)
                        truth[selected & ~conflict] = cls
                part_pen = width if family == 'equal_pen_open' else 2*width+3
                feet = [[145, 280], [365, 280]]
                body = [[95, 190], [95, 280], [420, 280], [420, 190]]
                if family == 'stepped_open':
                    feet = [[145, 270], [365, 305]]
                    body = [[95, 195], [95, 270], [245, 270], [245, 305], [420, 305], [420, 205]]
                elif family == 'angled_open':
                    feet = [[145, 270], [365, 300]]
                    body = [[95, 190], [95, 270], [205, 270], [305, 300], [420, 300], [420, 190]]
                expected = family not in ('broken_extension', 'disconnected_refs', 'frame_reference', 'hatch_reference')
                if family == 'disconnected_refs':
                    for p in ([[95, 190], [95, 280], [210, 280]], [[300, 280], [420, 280], [420, 190]]): stroke(p, 1, part_pen)
                elif family == 'frame_reference':
                    feet = [[145, 477], [365, 477]]
                    stroke([[35, 35], [477, 35], [477, 477], [35, 477], [35, 35]], 7, part_pen)
                elif family == 'hatch_reference':
                    for y in (220, 240, 260, 280, 300): stroke([[90, y], [420, y]], 9, part_pen)
                    stroke([[90, 220], [90, 300]], 9, part_pen)
                else:
                    stroke(body, 1, part_pen)
                    if family == 'pass_through': stroke([[95, 210], [420, 210]], 1, part_pen)
                    if family == 'thick_transition':
                        for x in (145, 365): stroke([[x, 280], [x, 330]], 1, part_pen)
                stroke([[145, 120], [365, 120]], 2, width)
                for foot in feet: stroke([[foot[0], 99], foot], 3, width)
                arrows = []
                for x, sign in ((145, -1), (365, 1)):
                    tip = np.array([x, 120]); base = tip-[sign*(26+width), 0]
                    triangle = [tip, base+[0, 5+width], base-[0, 5+width]]
                    stroke(triangle, 255, 1, filled=True)
                    arrows.append({'id': 'head-'+str(len(arrows)), 'tip': xy([tip])[0].tolist(),
                                   'direction': (rotation @ [sign, 0]).tolist(), 'triangle': xy(triangle).tolist()})
                if family == 'broken_extension':
                    mask = np.zeros_like(gray)
                    cv2.fillConvexPoly(mask, np.rint(xy([[135, 182], [155, 182], [155, 200], [135, 200]])).astype(np.int32), 255)
                    gray[mask > 0] = 255; truth[mask > 0] = 255; conflict[mask > 0] = False
                # Explicit isolated scan specks, never used as source evidence.
                for p in ([[230, 410], [239, 414]], [[285, 75], [291, 79]]):
                    mask = np.zeros_like(gray); cv2.polylines(mask, [np.rint(xy(p)).astype(np.int32)], False, 255, 1)
                    gray[mask > 0] = 0; negative[(mask > 0) & (truth == 255)] = 255
                truth[conflict] = 255; negative[truth != 255] = 0
                name = f'open-reference-{family}-w{width}-r{angle}'; folder = BASE / name; folder.mkdir()
                for n, a in [('truth', truth), ('negative', negative)]: Image.fromarray(a).save(folder / (n+'.png'))
                views = []
                for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
                    image = gray if scale == 1 else cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
                    path = folder / (mode+'.png'); Image.fromarray(image).save(path)
                    views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                                  'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
                jobs.append({'case': {'id': name, 'scope': 'open_reference_analytical_control', 'group_id': 'open-reference-'+family,
                     'split': split, 'size_px': [512, 512], 'labels': {'family': family, 'arrows': arrows,
                     'feet_px': xy(feet).tolist(), 'expected_dimension_chains': int(expected)}},
                     'views': views, 'new_control': True,
                     **{n: pin(ROOT, (folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth', 'negative')}})
    write(BASE / 'contract.json', {'task': 'P10.2c.3b.2f', 'generator': pin(ROOT, 'scripts/prepare_open_reference_controls_v1.py'),
          'seed': 742619, 'families': FAMILIES, 'jobs': jobs, 'prepared_before_local_reference_algorithm': True,
          'criteria': ['zero false new chains on explicit negatives', 'at least four correct new chains on fresh validation views',
                       'new correct train-origin real native dimension 50', 'preserve both previously correct real dimension views',
                       'no inherited chain/map/head/source changes', 'no new known-wrong/noise pixel roles',
                       'all complete bounded searches and exact repeats', 'ambiguous reference choices abstain'],
          'full_role_macro_F1_min': .90, 'physical_geometry_approved': False, 'production_changed': False,
          'budgets': {'child_seconds': 60, 'RSS_MiB': 1024, 'output_MiB': 64, 'files': 128, 'total_child_seconds': 1800}})
    print(pin(ROOT, 'artifacts/open-reference-controls-v1/contract.json'))


if __name__ == '__main__': build()
