"""Pinned, already screened pixel cohort shared by the two arrow reviews."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .dataset_layout_v1 import checked, read
from .family_catalog_v2 import require
from .source_coallocation_v1 import package

CONTROL = {'path':'artifacts/stroke-curve-control-v1/manifest.json',
           'sha256':'388caa1b462135f96d0483815ea4288eb99e5aa1b459039b8ff569ea00cb0996'}
CODE = ['src/cad_recovery/arrow_review_cohort_v1.py',
        'scripts/review_stroke_curve_control_v1.py', 'src/cad_recovery/arrow_detection.py',
        'src/cad_recovery/stroke_curve_control_v1.py', 'src/cad_recovery/hard_raster_metrics_v1.py']
PROFILES = ['baseline','ranked_global','regional_masked']


def load(root):
    root = Path(root).resolve()
    folder,manifest = package(root,CONTROL,'stroke-curve-control-manifest/1.0')
    spec = importlib.util.spec_from_file_location('historical_arrow_controls',root/'scripts/review_stroke_curve_control_v1.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    config,reading,parents,geometry,cases,baseline = old.context(manifest['config'])
    require(read(folder/'report.json') == old.score_saved(config,reading,parents,geometry,cases,baseline,folder),
            'Historical arrow-control scores changed')
    rows = []
    for reference in reading['patches']:
        parent = parents[reference['patch_id']]
        case = cases[parent['case_id']]
        source = checked(root,case['source'])
        gray = np.asarray(Image.open(source).convert('L'))
        threshold,foreground = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        x,y = parent['patch_box_in_context'][:2]
        matrix = np.array([[1,0,-x],[0,1,-y],[0,0,1]]) @ np.asarray(case['processed_to_source'])
        ocr = read(baseline/'runs'/case['id']/'cli_analyze_dimensions/result/page-001-ocr.json')
        profiles = {}
        for profile in PROFILES:
            if profile == 'baseline':
                geometry_report = read(checked(root,case['baseline_geometry']))
                arrow_report = read(baseline/'runs'/case['id']/'cli_analyze_dimensions/result/page-001-arrows.json')
            else:
                geometry_report = read(geometry/'runs'/case['id']/profile/'result/page-001-geometry.json')
                arrow_report = read(folder/'runs'/case['id']/profile/'page-001-arrows.json')
            require(arrow_report['source_sha256'] == case['source']['sha256']
                    and geometry_report['source_sha256'] == case['source']['sha256'], 'Mixed arrow input bytes')
            profiles[profile] = {'geometry':geometry_report,'arrows':arrow_report}
        rows.append({'reference':reference,'parent':parent,'case':case,'profiles':profiles,
                     'matrix':matrix,'gray':gray,'foreground':foreground,'threshold':threshold,
                     'text_polygons':[line['polygon_px'] for line in ocr['lines']],
                     'original_crop':(root/config['draft']['path']).parent/parent['image']})
    return rows
