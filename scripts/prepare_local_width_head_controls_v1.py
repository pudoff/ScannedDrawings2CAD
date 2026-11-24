"""Freeze unseen width/rotation arrow controls before implementing the detector."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cad_recovery.dataset_layout_v1 import pin

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/local-width-head-controls-v1'
FAMILIES = [('isolated_single', 'train'), ('paired_dimension', 'train'),
            ('hatch_wedges', 'train'), ('text_rectangles', 'train'),
            ('short_shaft_leader', 'validation'), ('curved_shaft', 'validation'),
            ('parallel_junctions', 'validation'), ('frame_stain', 'validation')]


def scenes():
    rng = np.random.default_rng(731508)
    for family, split in FAMILIES:
        for width in (2, 4, 6):
            for angle in (0, 37, 90, 143):
                size = 384
                gray = np.full((size, size), 255, np.uint8)
                theta = np.deg2rad(angle)
                matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                shift = rng.integers(-8, 9, 2)
                refs = []

                def transform(p):
                    return (np.asarray(p, float)-192)@matrix.T+192+shift

                def line(p, thickness=width):
                    cv2.polylines(gray, [np.rint(transform(p)).astype(np.int32)], False, 0, thickness, cv2.LINE_8)

                def head(identity, tip, direction):
                    tip, direction = np.asarray(tip, float), np.asarray(direction, float)
                    normal = np.array([-direction[1], direction[0]])
                    length, half = 20+3*width, 5+width
                    base = tip-length*direction
                    triangle = transform([tip, base+half*normal, base-half*normal])
                    cv2.fillConvexPoly(gray, np.rint(triangle).astype(np.int32), 0)
                    refs.append({'id': identity, 'tip': transform([tip])[0].tolist(),
                                 'direction': (matrix@direction).tolist(), 'triangle': triangle.tolist()})
                    return base

                if family == 'isolated_single':
                    base = head('single', [115, 190], [-1, 0]); line([base, [284, 190]])
                elif family == 'paired_dimension':
                    a = head('left', [100, 178], [-1, 0]); b = head('right', [284, 178], [1, 0]); line([a, b])
                    line([[100, 150], [100, 262]]); line([[284, 150], [284, 262]])
                    line([[100, 262], [284, 262]], 2*width+1)
                elif family == 'short_shaft_leader':
                    base = head('leader', [150, 175], [-1, 0]); line([base, base+[max(14, 4*width), 0]])
                    line([[90, 245], [280, 245]], 2*width+1)
                elif family == 'curved_shaft':
                    base = head('curved', [124, 154], [-1, 0])
                    t = np.linspace(0, 1, 80)
                    line(np.c_[base[0]+104*t, base[1]+43*t*t])
                elif family == 'hatch_wedges':
                    for j in range(7): line([[80+j*27, 112], [122+j*27, 272]])
                    # A solid wedge without its own shaft is not a head observation.
                    cv2.fillConvexPoly(gray, np.rint(transform([[181, 190], [223, 174], [223, 206]])).astype(np.int32), 0)
                elif family == 'text_rectangles':
                    mask = np.full_like(gray, 255)
                    cv2.putText(mask, 'A4 R12 O', (67, 156), cv2.FONT_HERSHEY_SIMPLEX, 1., 0, width, cv2.LINE_8)
                    cv2.rectangle(mask, (95, 212), (281, 251), 0, width)
                    affine = np.c_[matrix, np.array([192, 192])+shift-matrix@np.array([192, 192])]
                    gray = cv2.warpAffine(mask, affine, (size, size), flags=cv2.INTER_NEAREST, borderValue=255)
                elif family == 'parallel_junctions':
                    for j in range(4): line([[94, 130+j*35], [290, 130+j*35]])
                    line([[131, 92], [131, 292]]); line([[251, 92], [251, 292]])
                    line([[109, 110], [268, 276]])
                elif family == 'frame_stain':
                    line([[76, 94], [292, 94], [292, 282], [76, 282], [76, 94]])
                    line([[76, 224], [292, 224]]); line([[220, 224], [220, 282]])
                    cv2.fillConvexPoly(gray, np.rint(transform([[145, 170], [195, 153], [215, 189], [164, 201]])).astype(np.int32), 0)
                yield {'case': {'id': f'width-head-{family}-w{width}-r{angle}', 'group_id': 'width-head-'+family,
                    'split': split, 'scope': 'local_width_head_control', 'size_px': [size, size],
                    'labels': {'family': family, 'width_px': width, 'rotation_deg': angle, 'arrows': refs,
                        'negative_regions': [], 'dense_heads': True, 'authority': 'analytic, not engineering CAD'}}, 'gray': gray}


if __name__ == '__main__':
    BASE.mkdir(exist_ok=False)
    jobs = []
    for scene in scenes():
        case, gray = scene['case'], scene['gray']
        folder = BASE/case['id']; folder.mkdir()
        views = []
        for mode, scale in [('native_crop', 1.), ('half_resolution_crop', .5)]:
            image = gray if scale == 1 else cv2.resize(gray, (192, 192), interpolation=cv2.INTER_AREA)
            path = folder/(mode+'.png'); Image.fromarray(image).save(path)
            views.append({'mode': mode, 'gray': pin(ROOT, path.relative_to(ROOT).as_posix()),
                'input_to_native': [[1/scale, 0, .5/scale-.5], [0, 1/scale, .5/scale-.5], [0, 0, 1]]})
        jobs.append({'case': case, 'views': views, 'new_control': True})
    contract = {'task': 'P10.2c.2b.7b', 'families': FAMILIES, 'jobs': jobs,
        'generator': pin(ROOT, 'scripts/prepare_local_width_head_controls_v1.py'),
        'prepared_before_detector_implementation': True, 'seed': 731508,
        'metrics': {'tip_tolerance_native_px': 8, 'direction_tolerance_deg': 20,
            'dense_controls_all_unmatched_predictions_are_false': True,
            'partial_real_unmatched_predictions_are_unreviewed': True,
            'family_width_angle_and_scale_reported_separately': True},
        'gates': {'no_lost_baseline_targets': True, 'no_new_false_heads_on_dense_controls_or_negative_ROIs': True,
            'new_validation_recall_min_each_scale': .90, 'no_input_or_baseline_mutation': True,
            'source_checks_required': ['triangle', 'taper', 'own_shaft', 'background', 'bounds'],
            'DT_automatic_admission': False},
        'training': False, 'closed_test_used': False, 'physical_geometry_approved': False}
    path = BASE/'contract.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2); stream.write('\n')
    print(json.dumps(pin(ROOT, path.relative_to(ROOT).as_posix())))
