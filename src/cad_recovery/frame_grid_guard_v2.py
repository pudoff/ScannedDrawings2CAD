"""Keep ambiguous junction footprints explicit instead of hiding label losses."""
import math
import cv2
import numpy as np
from . import frame_grid_guard_v1 as previous
from .stroke_roles_v1 import foreground

POLICY = {'version': 'frame-grid-guard/2.0', 'maximum_owner_side_pairs': 250000,
          'maximum_source_points': 2000000, 'transverse_angle_deg': 8.,
          'junction_reach_in_strokes': 1.5, 'production_admitted': False}


def bind_context(*args, **kwargs):
    packet = previous.bind_context(*args, **kwargs)
    page, detector = args[5], args[6]
    overview = previous.affine(page['overview_to_source']); matrix = previous.affine(packet['input_to_page'])
    flat = [s for f in detector['frames'] for s in f['sides']]
    for side, observed in zip(packet['sides'], flat):
        a, b = np.asarray(observed['points_overview']); u = (b-a)/np.linalg.norm(b-a); n = np.array([-u[1], u[0]])
        a, b = np.asarray(side['points_page']); u = (b-a)/np.linalg.norm(b-a); normal = np.array([-u[1], u[0]])
        scale = abs(normal @ overview[:2, :2] @ n)
        side['stroke_width_page_px'] = float(observed['stroke_width_overview_px']*scale)
        side['position_uncertainty_page_px'] = float(observed['position_uncertainty_overview_px']*scale)
        side['input_normal_footprint_page_px'] = float(.5*np.abs(normal @ matrix[:2, :2]).sum())
    return packet


def analyze(gray, graph, raw, old, base, owners, *, authority=None, context=None):
    coarse = previous.analyze(gray, graph, raw, old, base, owners, authority=authority, context=context)
    result = coarse['page_context'].copy(); uncertain = np.zeros(gray.shape, np.uint8)
    audit = {'owner_side_pairs': 0, 'source_points': 0, 'limits': []}
    candidate = (base == 8) & (old != 8)
    if context is not None and coarse['complete'] and candidate.any():
        matrix = previous.affine(context['input_to_page']); ink = foreground(gray)
        distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        y, x = np.nonzero(candidate); pixels = previous.project(np.c_[x, y], matrix)
        for side in context['sides']:
            a, b = np.asarray(side['points_page']); u = (b-a)/np.linalg.norm(b-a); n = np.array([-u[1], u[0]])
            width = float(side['stroke_width_page_px']); error = float(side['position_uncertainty_page_px'])
            footprint = float(side['input_normal_footprint_page_px'])
            normal = np.abs((pixels-a) @ n); along = (pixels-a) @ u
            edge = (normal >= max(0., width/2-error-footprint)) & (normal <= width/2+error+footprint)
            edge &= (along >= -width) & (along <= np.linalg.norm(b-a)+width)
            if not edge.any(): continue
            for item in graph['proposals']:
                audit['owner_side_pairs'] += 1
                p = np.asarray(item['points_px'], float); audit['source_points'] += len(p)
                if audit['owner_side_pairs'] > POLICY['maximum_owner_side_pairs'] or audit['source_points'] > POLICY['maximum_source_points']:
                    audit['limits'].append('junction_context_budget'); break
                if len(p) < 2: continue
                points = previous.project(p, matrix)
                _, _, vectors = np.linalg.svd(points-points.mean(0), full_matrices=False)
                if abs(vectors[0] @ u) > math.sin(math.radians(POLICY['transverse_angle_deg'])): continue
                d = np.abs((points-a) @ n)
                if d.min() > width+error+footprint: continue
                q = np.rint(p).astype(int)
                if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any(): raise ValueError('Source owner outside image')
                radii = distance[q[:, 1], q[:, 0]]; radii = radii[radii > 0]
                if not len(radii): continue
                native_pen = max(1., float(2*np.quantile(radii, .25)-1))*np.linalg.norm(matrix[:2, :2], ord=2)
                location = float(np.median((points[d <= width+error+footprint]-a) @ u))
                near = edge & (np.abs(along-location) <= POLICY['junction_reach_in_strokes']*native_pen+footprint+error)
                uncertain[y[near], x[near]] = 255
            if audit['limits']: break
    complete = coarse['complete'] and not audit['limits']
    if complete: result[uncertain > 0] = base[uncertain > 0]
    else: result[:] = base; uncertain[:] = 0
    return {'coarse': coarse['page_context'], 'uncertainty': result, 'uncertainty_mask': uncertain,
            'complete': complete, 'coarse_complete': coarse['complete'], 'coarse_audit': coarse['audit'],
            'audit': audit, 'decisions': coarse['decisions'], 'new_frame_roles': 0, 'source_pixels_erased': 0,
            'uncertain_pixels_are_not_excluded_from_metrics': True, 'production_admitted': False, 'physical_geometry_approved': False}
