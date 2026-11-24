"""Freeze independent hatch/arrow and text workload scenes before new rules."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import pin

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/economical-head-controls-v1'
FAMILIES = [('isolated_short_stub', 'train'), ('head_beside_hatch', 'train'),
    ('hatch_triangular_mark', 'train'), ('small_glyph_grid', 'train'),
    ('attached_compound_dimension', 'validation'), ('leader_from_hatch_field', 'validation'),
    ('fragmented_hatch_marks', 'validation'), ('touching_caption_negative', 'validation')]


def scenes():
    rng = np.random.default_rng(285174)
    for family, split in FAMILIES:
        for width in (2, 4, 6):
            for angle in (0, 43, 90, 137):
                size = 384; gray = np.full((size, size), 255, np.uint8); refs = []
                theta = np.deg2rad(angle); rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-7, 8, 2)

                def transform(p): return (np.asarray(p, float)-192)@rotation.T+192+shift

                def line(p, w=width): cv2.polylines(gray, [np.rint(transform(p)).astype(np.int32)], False, 0, w)

                def arrow(tip, direction, span):
                    p, u = np.asarray(tip, float), np.asarray(direction, float); normal = np.array([-u[1], u[0]])
                    base = p-(20+2*width)*u
                    triangle = transform([p, base+(4+width)*normal, base-(4+width)*normal])
                    cv2.fillConvexPoly(gray, np.rint(triangle).astype(np.int32), 0)
                    line([base, base-span*u])
                    refs.append({'id': 'head', 'tip': transform([p])[0].tolist(), 'direction': (rotation@u).tolist(), 'triangle': triangle.tolist()})

                if family == 'isolated_short_stub': arrow([126, 147], [-1, 0], max(14, 3*width))
                if family == 'attached_compound_dimension':
                    arrow([101, 127], [-1, 0], 150)
                    line([[101, 100], [101, 235]]); line([[283, 100], [283, 235]])
                    line([[101, 235], [283, 235]], 2*width+1)
                hatch = family in ('head_beside_hatch', 'hatch_triangular_mark', 'leader_from_hatch_field', 'fragmented_hatch_marks')
                if hatch:
                    for j in range(6):
                        x = 137+j*23
                        if family == 'fragmented_hatch_marks':
                            line([[x, 145], [x+7, 180]]); line([[x+11, 197], [x+25, 267]])
                        else: line([[x, 145], [x+25, 267]])
                    if family in ('head_beside_hatch', 'leader_from_hatch_field'):
                        # The shaft leaves the parallel field; nearby hatching alone must not veto it.
                        tip = [137, 198] if family == 'head_beside_hatch' else [145, 221]
                        arrow(tip, [1, 0], 65)
                    else:
                        triangle = transform([[207, 207], [188, 174], [215, 170]])
                        cv2.fillConvexPoly(gray, np.rint(triangle).astype(np.int32), 0)
                if family in ('small_glyph_grid', 'touching_caption_negative'):
                    canvas = np.full_like(gray, 255)
                    for row in range(3):
                        cv2.putText(canvas, 'A4 R18 24', (64, 127+row*49), cv2.FONT_HERSHEY_SIMPLEX, .7, 0, width)
                        if family == 'touching_caption_negative': cv2.line(canvas, (68, 129+row*49), (301, 129+row*49), 0, width)
                    affine = np.c_[rotation, np.array([192, 192])+shift-rotation@np.array([192, 192])]
                    gray = cv2.warpAffine(canvas, affine, (size, size), flags=cv2.INTER_NEAREST, borderValue=255)
                yield {'case': {'id': f'economy-head-{family}-w{width}-r{angle}', 'scope': 'economical_head_control',
                    'group_id': 'economy-head-'+family, 'split': split, 'size_px': [size, size],
                    'labels': {'family': family, 'width_px': width, 'rotation_deg': angle, 'arrows': refs,
                        'negative_regions': [], 'dense_heads': True, 'authority': 'analytic fresh adversaries, no physical CAD authority'}}, 'gray': gray}
    for k, split in enumerate(('train', 'train', 'validation', 'validation')):
        width, height = 2500, 1800; gray = np.full((height, width), 255, np.uint8); refs = []
        for row in range(9+k):
            for col in range(7):
                cv2.putText(gray, f'A{row} R{12+col}', (55+col*343, 77+row*99), cv2.FONT_HERSHEY_SIMPLEX, .65, 0, 2)
        for j in range(3):
            tip = np.array([407+j*691+k*11, 1390+(j%2)*130], float); base = tip+[31, 0]
            triangle = np.array([tip, base+[0, 9], base-[0, 9]])
            cv2.fillConvexPoly(gray, np.rint(triangle).astype(np.int32), 0)
            cv2.line(gray, tuple(base.astype(int)), tuple((base+[89, 0]).astype(int)), 0, 3)
            refs.append({'id': 'head-'+str(j), 'tip': tip.tolist(), 'direction': [-1., 0.], 'triangle': triangle.tolist()})
        yield {'case': {'id': f'economy-head-text-workload-{k}', 'scope': 'economical_head_large_control',
            'group_id': f'economy-head-text-workload-{k}', 'split': split, 'size_px': [width, height],
            'labels': {'family': f'text_workload_{k}', 'width_px': 3, 'rotation_deg': None, 'arrows': refs,
                'negative_regions': [], 'dense_heads': True, 'authority': 'analytic large raster workload, not full real precision'}}, 'gray': gray}


if __name__ == '__main__':
    BASE.mkdir(exist_ok=False); jobs = []
    for scene in scenes():
        case = scene['case']; folder = BASE/case['id']; folder.mkdir(); views = []
        for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
            gray = scene['gray'] if scale == 1 else cv2.resize(scene['gray'], tuple(v//2 for v in case['size_px']), interpolation=cv2.INTER_AREA)
            path = folder/(mode+'.png'); Image.fromarray(gray).save(path)
            views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
        jobs.append({'case': case, 'views': views, 'new_control': True})
    contract = {'task': 'P10.2c.2b.7c.2', 'generator': pin(ROOT, 'scripts/prepare_economical_head_controls_v1.py'),
        'prepared_before_detector_implementation': True, 'seed': 285174, 'families': FAMILIES, 'jobs': jobs,
        'gates': {'lost_previous_true_targets_max': 0, 'additional_dense_FP_against_v1_max': 0,
            'additional_negative_ROI_FP_max': 0, 'new_validation_recall_min_each_scope_scale': .90,
            'all_economical_searches_complete': True, 'profile_samples_max_per_source': 8000000,
            'morphology_pixels_max_per_source': 40000000, 'source_and_baseline_unchanged': True,
            'historical_20_of_21_and_13_of_13_preserved': True},
        'training': False, 'closed_test_used': False, 'physical_geometry_approved': False}
    path = BASE/'contract.json'
    with path.open('x', encoding='utf-8') as stream: json.dump(contract, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(pin(ROOT, path.relative_to(ROOT).as_posix())))
