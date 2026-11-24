"""Analytical CV fixtures: geometry labels precede rasterization; no CAD source."""
import math

import cv2
import numpy as np

FAMILIES = {
    'parallel_gaps': 'train', 'junctions': 'train', 'circle_holes': 'train',
    'arrows_text': 'train', 'rounded_outline': 'validation', 'angular_dimension': 'validation',
}


def scene(family, width):
    if family not in FAMILIES or width not in (3, 7, 11):
        raise ValueError('Unknown fixture family or stroke width')
    ink = np.zeros((512, 768), np.uint8)
    text = np.zeros_like(ink)
    geometry = []; arrows = []; gaps = []; negatives = []; junctions = []

    def line(identifier, a, b, role='part_contour', weight=width):
        cv2.line(ink, tuple(a), tuple(b), 255, weight, cv2.LINE_8)
        geometry.append({'id': identifier, 'kind': 'line', 'points': [a, b],
                         'role': role, 'stroke_width_px': weight})

    def curve(identifier, center, radius, start=0, sweep=360, role='part_contour'):
        kind = 'circle' if sweep == 360 else 'arc'
        if kind == 'circle':
            cv2.circle(ink, tuple(center), radius, 255, width, cv2.LINE_8)
        else:
            cv2.ellipse(ink, tuple(center), (radius, radius), 0, start, start+sweep, 255, width, cv2.LINE_8)
        geometry.append({'id': identifier, 'kind': kind, 'center': center, 'radius': radius,
                         'start_angle_deg': start, 'sweep_angle_deg': sweep,
                         'role': role, 'stroke_width_px': width})

    def arrow(identifier, tip, direction):
        unit = np.asarray(direction, float); unit /= np.linalg.norm(unit)
        p = np.asarray(tip, float); base = p-unit*28; normal = np.array([-unit[1], unit[0]])
        triangle = np.rint([p, base+normal*9, base-normal*9]).astype(int)
        cv2.fillConvexPoly(ink, triangle, 255)
        arrows.append({'id': identifier, 'tip': tip, 'direction': unit.tolist(),
                       'triangle': triangle.tolist(), 'kind': 'filled'})

    if family == 'parallel_gaps':
        for i, angle in enumerate((0, 30, 75)):
            a = np.array([80+i*210, 70]); v = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            for j in range(2):
                p = a+np.array([0, j*70]); q = np.rint(p+v*160).astype(int)
                line(f'parallel-{i}-{j}', p.astype(int).tolist(), q.tolist())
        line('gap-left', [70, 380], [300, 380], 'axis_construction', 3)
        line('gap-right', [340, 380], [680, 380], 'axis_construction', 3)
        gaps.append({'id': 'intentional-axis-gap', 'points': [[310, 380], [330, 380]], 'must_remain_open': True})
    elif family == 'junctions':
        line('x-a', [80, 80], [330, 330]); line('x-b', [80, 330], [330, 80])
        line('t-top', [450, 110], [690, 110]); line('t-stem', [570, 110], [570, 350])
        line('near-upper', [430, 410], [700, 410]); line('near-lower', [430, 435], [700, 435])
        junctions = [{'point': [205, 205], 'degree': 4}, {'point': [570, 110], 'degree': 3}]
    elif family == 'circle_holes':
        for i, radius in enumerate((18, 45, 90)):
            curve(f'hole-{i}', [110+i*240, 210], radius)
        line('axis', [440, 210], [735, 210], 'axis_construction', 3)
        negatives.append({'id': 'circle-axis-not-head', 'box': [570, 190, 610, 230], 'kind': 'no_arrow'})
    elif family == 'arrows_text':
        for i, angle in enumerate((0, 30, 75)):
            d = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
            a = np.array([80+i*220, 70]); b = np.rint(a+d*170).astype(int)
            line(f'shaft-{i}', a.tolist(), b.tolist(), 'dimension_line', 3)
            arrow(f'head-{i}', b.tolist(), d.tolist())
        cv2.putText(text, 'XXX O12 +/- 0.5', (65, 390), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 2, cv2.LINE_8)
        ink = np.maximum(ink, text)
        negatives.append({'id': 'letter-head-negatives', 'box': [40, 330, 720, 425], 'kind': 'no_arrow'})
    elif family == 'rounded_outline':
        line('corner-top', [80, 100], [300, 100]); curve('corner', [300, 200], 100, 270, 90)
        line('corner-side', [400, 200], [400, 430]); curve('small-hole', [240, 240], 20)
        line('near-hole-axis', [160, 240], [320, 240], 'axis_construction', 3)
        junctions = [{'point': [300, 100], 'degree': 2}, {'point': [400, 200], 'degree': 2}]
    else:
        curve('angular-shaft', [350, 290], 180, 205, 110, 'dimension_line')
        for i, angle in enumerate((205, 315)):
            theta = math.radians(angle); p = np.rint([350+180*math.cos(theta), 290+180*math.sin(theta)]).astype(int)
            line(f'extension-{i}', [350, 290], p.tolist(), 'extension_line', 3)
            tangent = [-math.sin(theta), math.cos(theta)]
            arrow(f'curve-head-{i}', p.tolist(), tangent if i == 0 else [-v for v in tangent])
        curve('separate-open-arc', [630, 350], 65, 20, 200)
        gaps.append({'id': 'missing-sector-not-circle', 'kind': 'open_arc', 'geometry_id': 'separate-open-arc'})
    labels = {'authority': 'analytical_construction_before_rasterization', 'family': family,
        'split': FAMILIES[family], 'group_id': 'procedural-cv-'+family, 'geometry': geometry,
        'arrows': arrows, 'gaps': gaps, 'junctions': junctions, 'negative_regions': negatives,
        'pixel_center_geometry': True, 'rasterization': 'OpenCV LINE_8; cap/edge pixel deviations are not centerline GT',
        'physical_geometry_approved': False, 'dense_geometry_truth': True}
    return 255-ink, text, labels
