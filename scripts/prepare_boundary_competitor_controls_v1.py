"""Freeze new curved/contact families before adding a competing-stroke guard."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from cad_recovery.dataset_layout_v1 import pin
from run_topology_robust_v1 import write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/boundary-competitor-controls-v1'
FAMILIES = [('convex_bow', 'train'), ('reentrant_wave', 'train'), ('offset_shoulder', 'train'),
            ('double_bow', 'validation'), ('concave_arch', 'validation'), ('splayed_wall', 'validation')]


def main():
    BASE.mkdir(exist_ok=False); jobs = []; rng = np.random.default_rng(184095)
    for family, split in FAMILIES:
        for width in (2, 4, 6):
            for angle in ((17, 83) if split == 'train' else (41, 127)):
                gray = np.full((512, 512), 255, np.uint8); truth = np.full_like(gray, 255)
                conflict = np.zeros_like(gray, bool); negative = np.zeros_like(gray)
                theta = np.deg2rad(angle); rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-5, 6, 2)
                def xy(p): return (np.asarray(p, float)-256)@rot.T+256+shift
                def paint(p, cls, pen, closed=False, filled=False):
                    mask = np.zeros_like(gray); q = np.rint(xy(p)).astype(np.int32)
                    if filled: cv2.fillConvexPoly(mask, q, 255)
                    else: cv2.polylines(mask, [q], closed, 255, pen)
                    selected = mask > 0; gray[selected] = 0
                    if cls is None: negative[selected & (truth == 255)] = 255
                    elif cls == 255: conflict[:] |= selected
                    else:
                        conflict[:] |= selected & (truth != 255) & (truth != cls)
                        truth[selected & ~conflict] = cls
                t = np.linspace(0, 1, 161); x = 140+232*t
                if family == 'convex_bow': y = 240-30*np.sin(np.pi*t)
                elif family == 'reentrant_wave': y = 240+25*np.sin(2*np.pi*t)
                elif family == 'double_bow': y = 240-18*np.sin(2*np.pi*t)**2
                elif family == 'concave_arch': y = 240+45*np.sin(np.pi*t)
                else: y = 240+20*np.sin(3*np.pi*t)
                top = np.c_[x, y] if family != 'offset_shoulder' else np.array([[140,240],[182,240],[210,270],[330,270],[372,240]])
                bottom = [[350, 405], [157, 405]] if family == 'splayed_wall' else [[372, 405], [140, 405]]
                paint(np.r_[top, bottom], 1, 2*width+3, True)
                paint([[140, 116], [372, 116]], 2, width+1)
                for px, sign in ((140, -1), (372, 1)):
                    paint([[px, 97], [px, 240]], 3, width+1)
                    p = np.array([px, 116]); base = p-np.array([sign*(25+width), 0])
                    paint([p, base+[0, 6+width], base-[0, 6+width]], 255, 1, filled=True)
                # A close unrelated stroke must not inherit part ownership.
                paint([[201, 174], [216, 183]], None, 1)
                paint([[277, 431], [292, 437]], None, 1)
                truth[conflict] = 255; negative[truth != 255] = 0
                name = f'boundary-competitor-{family}-w{width}-r{angle}'; folder = BASE/name; folder.mkdir()
                for n, array in [('truth', truth), ('negative', negative)]: Image.fromarray(array).save(folder/(n+'.png'))
                views = []
                for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
                    image = gray if scale == 1 else cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
                    p = folder/(mode+'.png'); Image.fromarray(image).save(p)
                    views.append({'mode': mode, 'gray': pin(ROOT, p.relative_to(ROOT).as_posix()),
                                  'input_to_native': [[1/scale,0,.5/scale-.5],[0,1/scale,.5/scale-.5],[0,0,1]]})
                jobs.append({'case': {'id': name, 'scope': 'boundary_competitor_analytical_control', 'group_id': 'boundary-competitor-'+family,
                                     'split': split, 'size_px': [512,512], 'labels': {'family': family}},
                             'views': views, 'new_control': True,
                             **{n: pin(ROOT, (folder/(n+'.png')).relative_to(ROOT).as_posix()) for n in ('truth','negative')}})
    write(BASE/'contract.json', {'task':'P10.2c.3b.2b','generator':pin(ROOT,'scripts/prepare_boundary_competitor_controls_v1.py'),
          'seed':184095,'families':FAMILIES,'jobs':jobs,'prepared_before_competitor_guard':True,
          'gates':['zero new known-wrong/negative roles','old correct additions retained >=80% overall and on former holdout',
                   'known five tangent errors removed','inherited roles and inputs unchanged','exact repeats and budgets'],
          'budgets':{'child_seconds':60,'RSS_MiB':1024,'output_MiB':64,'files':128,'total_child_seconds':1800,'OpenCV_threads':1,'RNG_seed':0},
          'full_classifier_macro_F1_min':.90,'production_changed':False,'physical_geometry_approved':False})
    print(pin(ROOT,(BASE/'contract.json').relative_to(ROOT).as_posix()))


if __name__ == '__main__': main()
