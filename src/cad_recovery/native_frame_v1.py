"""Native source-pixel frame masks with bounded digital-edge feasibility."""
import math
import cv2
import numpy as np
from PIL import Image
from .frame_grid_guard_v1 import affine, project
from .grid_role_revision_v1 import sha, check_authority
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray

POLICY = {'version': 'native-frame/1.0', 'maximum_sides': 96, 'maximum_columns': 4096,
          'maximum_cross_section_pixels': 256, 'maximum_source_samples': 4000000,
          'minimum_sections': 24, 'maximum_polygon_vertices': 8192,
          'production_admitted': False, 'native_prediction_transfer': False}


def clip_polygon(polygon, normal, bound):
    output = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = float(a @ normal-bound), float(b @ normal-bound)
        if da <= 0: output.append(a)
        if (da <= 0) != (db <= 0): output.append(a+(b-a)*(da/(da-db)))
    return np.asarray(output, float).reshape(-1, 2)


def digital_edge(x, y, all_x):
    """Solve y=floor(m*x+b) on source observations, without target labels."""
    origin = float(x.mean()); scale = max(1., float(np.ptp(x))/2)
    t = (x-origin)/scale; m, b = np.polyfit(t, y, 1)
    polygon = np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]])
    # Interleave observations across the segment so early constraints span it.
    order = np.r_[np.arange(0, len(x), 2), np.arange(1, len(x), 2)]
    for i in order:
        low = float(y[i]-(m*t[i]+b)); n = np.array([t[i], 1.])
        polygon = clip_polygon(polygon, -n, -low-1e-7)
        if not len(polygon): return None
        polygon = clip_polygon(polygon, n, low+1-1e-7)
        if not len(polygon) or len(polygon) > POLICY['maximum_polygon_vertices']: return None
    tt = (all_x-origin)/scale
    predictions = (m*tt+b)[:, None]+tt[:, None]*polygon[:, 0]+polygon[:, 1]
    midpoint = polygon.mean(0)
    mid = np.floor(m*tt+b+midpoint[0]*tt+midpoint[1]+1e-9).astype(int)
    lo = np.floor(predictions.min(1)+1e-9).astype(int); hi = np.floor(predictions.max(1)+1e-9).astype(int)
    return {'mid': mid, 'minimum': lo, 'maximum': hi, 'vertices': len(polygon),
            'models': polygon+np.array([m, b]), 'origin': origin, 'scale': scale,
            'source_points': len(x), 'source_fit_exact': bool(np.array_equal(np.floor(m*t+b+midpoint[0]*t+midpoint[1]+1e-9).astype(int), y))}


def couple_edges(top, bottom, x, lo, hi, all_x):
    """Both edges of one straight stroke must admit the same slope."""
    lower = max(top['models'][:, 0].min(), bottom['models'][:, 0].min())
    upper = min(top['models'][:, 0].max(), bottom['models'][:, 0].max())
    if lower > upper: return None
    slope = (lower+upper)/2; output = []
    for edge, observed in ((top, lo), (bottom, hi)):
        p = clip_polygon(edge['models'], np.array([-1., 0.]), -lower)
        if not len(p): return None
        p = clip_polygon(p, np.array([1., 0.]), upper)
        if not len(p): return None
        intercepts = []
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            if abs(a[0]-slope) < 1e-12: intercepts.append(a[1])
            if (a[0]-slope)*(b[0]-slope) < 0: intercepts.append(a[1]+(b[1]-a[1])*(slope-a[0])/(b[0]-a[0]))
        if not intercepts: return None
        intercept = (min(intercepts)+max(intercepts))/2
        t = (all_x-edge['origin'])/edge['scale']; values = t[:, None]*p[:, 0]+p[:, 1]
        observed_t = (x-edge['origin'])/edge['scale']
        exact = np.array_equal(np.floor(slope*observed_t+intercept+1e-9).astype(int), observed)
        output.append({'mid': np.floor(slope*t+intercept+1e-9).astype(int),
                       'minimum': np.floor(values.min(1)+1e-9).astype(int), 'maximum': np.floor(values.max(1)+1e-9).astype(int),
                       'vertices': len(p), 'source_fit_exact': bool(exact)})
    return output


def side_masks(ink, points, width, uncertainty, audit):
    """Read full source runs, omit junction outgrowths, and extrapolate edges."""
    delta = points[1]-points[0]; major = int(np.argmax(np.abs(delta))); minor = 1-major
    if abs(delta[major]) < 1: return None, 'short_side'
    slope = delta[minor]/delta[major]; intercept = points[0, minor]-slope*points[0, major]
    source = ink.T if major == 0 else ink
    length, across = source.shape
    if length > POLICY['maximum_columns']: return None, 'column_budget'
    normal_scale = abs(delta[major])/np.linalg.norm(delta)
    reach = max(4, math.ceil((1.5*width+uncertainty+2)/normal_scale))
    if 2*reach+1 > POLICY['maximum_cross_section_pixels']: return None, 'cross_section_budget'
    start = max(0, math.floor(min(points[:, major])-width)); stop = min(length, math.ceil(max(points[:, major])+width)+1)
    columns = np.arange(start, stop); sections = {}
    for x in columns:
        center = slope*x+intercept; low = max(0, math.floor(center-reach)); high = min(across, math.ceil(center+reach)+1)
        if high <= low: continue
        audit['source_samples'] += high-low
        if audit['source_samples'] > POLICY['maximum_source_samples']: return None, 'source_sample_budget'
        a = source[x, low:high]; change = np.diff(np.r_[False, a, False].astype(int))
        runs = [(lo+low, hi+low-1) for lo, hi in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)) if lo > 0 and hi < high-low]
        if not runs: continue
        lo, hi = min(runs, key=lambda r: abs((r[0]+r[1])/2-center))
        if abs((lo+hi)/2-center) > (width/2+uncertainty+1)/normal_scale: continue
        sections[int(x)] = (lo, hi)
    if len(sections) < POLICY['minimum_sections']: return None, 'few_complete_source_sections'
    sizes = np.array([hi-lo+1 for lo, hi in sections.values()]); median = float(np.median(sizes))
    valid = np.zeros(len(columns), np.uint8)
    for i, x in enumerate(columns):
        if x in sections:
            lo, hi = sections[x]
            if median-1 <= hi-lo+1 <= median+1: valid[i] = 1
    # Width changes expose transverse ink. Keep fitting observations outside it.
    invalid = (valid == 0).astype(np.uint8)[None, :]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*math.ceil(median)+1, 1))
    valid = cv2.dilate(invalid, kernel)[0] == 0
    xx = columns[valid]
    if len(xx) < POLICY['minimum_sections']: return None, 'few_uncontaminated_sections'
    lo = np.array([sections[x][0] for x in xx]); hi = np.array([sections[x][1] for x in xx])
    top = digital_edge(xx, lo, columns); bottom = digital_edge(xx, hi, columns)
    if top is None or bottom is None: return None, 'no_exact_digital_edge_model'
    pair = couple_edges(top, bottom, xx, lo, hi, columns)
    # Integer rasterization can give the two exact digital edges slightly
    # different slope intervals. Their observed run widths are already bounded.
    if pair is not None: top, bottom = pair
    if not top['source_fit_exact'] or not bottom['source_fit_exact']: return None, 'source_fit_mismatch'
    y = np.arange(across)[None, :]
    mid = (y >= top['mid'][:, None]) & (y <= bottom['mid'][:, None])
    safe = (y >= top['maximum'][:, None]) & (y <= bottom['minimum'][:, None])
    possible = (y >= top['minimum'][:, None]) & (y <= bottom['maximum'][:, None])
    outputs = []
    for band in (mid, safe, possible):
        mask = np.zeros_like(source, bool); mask[columns] = band & source[columns]
        outputs.append(mask.T if major == 0 else mask)
    return {'mid': outputs[0], 'safe': outputs[1], 'possible': outputs[2],
            'sections': len(xx), 'observed_width_across_px': median,
            'edge_model': 'shared_slope' if pair is not None else 'independent_quantized_edges',
            'edge_vertices': [top['vertices'], bottom['vertices']]}, None


def source_masks(gray, page, detector, *, context_gray=None):
    checked_gray(gray)
    if context_gray is not None:
        checked_gray(context_gray); box = page['patch_bbox']
        if list(context_gray.shape[::-1]) != page['source_size_px'] or not np.array_equal(gray, context_gray[box[1]:box[3], box[0]:box[2]]):
            raise ValueError('Expanded source context is not the same original raster')
        expanded = {**page, 'patch_bbox': [0, 0, context_gray.shape[1], context_gray.shape[0]], 'source_to_patch': np.eye(3).tolist()}
        result = source_masks(context_gray, expanded, detector)
        for name in ('mid', 'safe', 'possible', 'coverage', 'failed_coverage'): result[name] = result[name][box[1]:box[3], box[0]:box[2]].copy()
        result.update(source_gray_sha256=sha(gray), page_sha256=object_sha(page),
                      context_gray_sha256=sha(context_gray), bounded_whole_source_context_used=True,
                      mask_sha256={k: sha(result[k]) for k in ('mid', 'safe', 'possible', 'coverage', 'failed_coverage')})
        return result
    if not page['source_verified'] or not detector['complete']: raise ValueError('Authenticated complete page context required')
    box = page['patch_bbox']
    if list(gray.shape[::-1]) != [box[2]-box[0], box[3]-box[1]]: raise ValueError('Native source ROI dimensions disagree')
    matrix = affine(page['source_to_patch']) @ affine(page['overview_to_source'])
    sides = [s for f in detector['frames'] for s in f['sides']]
    if len(sides) > POLICY['maximum_sides']: raise ValueError('Frame side budget exceeded')
    if any(f['frame_semantics_confirmed'] for f in detector['frames']): raise ValueError('Manual frame targets are not inference inputs')
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1] > 0
    masks = {k: np.zeros_like(gray, np.uint8) for k in ('mid', 'safe', 'possible', 'coverage', 'failed_coverage')}
    audit = {'source_samples': 0, 'limits': []}; decisions = []; seen = set()
    yy, xx = np.indices(gray.shape)
    for side in sides:
        points = project(side['points_overview'], matrix); delta = points[1]-points[0]
        axis = delta/np.linalg.norm(delta); normal = np.array([-axis[1], axis[0]])
        old = np.asarray(side['points_overview']); old_axis = (old[1]-old[0])/np.linalg.norm(old[1]-old[0]); old_normal = np.array([-old_axis[1], old_axis[0]])
        scale = abs(normal @ matrix[:2, :2] @ old_normal)
        width = float(side['stroke_width_overview_px']*scale); error = float(side['position_uncertainty_overview_px']*scale)
        key = tuple(np.round(np.r_[points.ravel(), width, error], 6))
        if key in seen: continue
        seen.add(key)
        fitted, reason = side_masks(ink, points, width, error, audit)
        row = {'points_native': points.tolist(), 'width_native_px': width, 'error_native_px': error, 'fit': fitted is not None, 'reason': reason}
        if reason and reason.endswith('_budget'): audit['limits'].append(reason)
        along = (xx-points[0, 0])*axis[0]+(yy-points[0, 1])*axis[1]
        distance = np.abs((xx-points[0, 0])*normal[0]+(yy-points[0, 1])*normal[1])
        radius = width/2+error+2*math.sqrt(2.)
        region = (distance <= radius) & (along >= -radius) & (along <= np.linalg.norm(delta)+radius)
        if fitted is not None:
            for name in ('mid', 'safe', 'possible'): masks[name][fitted[name]] = 255
            masks['coverage'][region] = 255
            row.update(source_sections=fitted['sections'], edge_vertices=fitted['edge_vertices'], edge_model=fitted['edge_model'])
        else: masks['failed_coverage'][region] = 255
        decisions.append(row)
        if audit['limits']: break
    complete = not audit['limits']
    if not complete:
        for m in masks.values(): m[:] = 0
    return {**masks, 'source_gray_sha256': sha(gray), 'page_sha256': object_sha(page), 'detector_sha256': object_sha(detector),
            'mask_sha256': {k: sha(v) for k, v in masks.items()}, 'decisions': decisions, 'audit': audit, 'complete': complete,
            'fit_context_bbox': list(box), 'fit_decision_coordinate_system': 'fit_context_pixel_centres',
            'resize_method': 'PIL_BOX' if page['kind'] == 'authenticated_page_overview' else 'CV_AREA',
            'bounded_whole_source_context_used': False, 'GT_used': False, 'native_prediction_transfer': False}


def refine(gray, graph, raw, old, base, coarse, owners, *, authority, input_to_native, source_gray=None, source=None):
    checked_gray(gray); permitted = check_authority(gray, raw, authority)
    if any(a.shape != gray.shape or a.dtype != np.uint8 for a in (old, base, coarse)): raise ValueError('Aligned uint8 input role maps required')
    delta = coarse != base
    if not np.all((base[delta] == 8) & (old[delta] != 8) & (coarse[delta] == old[delta])): raise ValueError('Untrusted coarse revisions')
    if np.any(delta & ~permitted): raise ValueError('Coarse revisions touch protected or untrusted pixels')
    matrix = affine(input_to_native)
    if source_gray is None:
        # Native output requires explicit size inferred from the edge-preserving map.
        corners = project([[-.5, -.5], [gray.shape[1]-.5, gray.shape[0]-.5]], matrix)
        size = np.rint(corners[1]+.5).astype(int)
    else: size = np.array(source_gray.shape[::-1])
    corners = project([[-.5, -.5], [gray.shape[1]-.5, gray.shape[0]-.5]], matrix)
    if not np.allclose(corners, [[-.5, -.5], (size-.5).tolist()]): raise ValueError('Native pixel-centre transform is inconsistent')
    sx, sy = size/np.array(gray.shape[::-1])
    if not np.allclose(matrix, [[sx, 0, .5*sx-.5], [0, sy, .5*sy-.5], [0, 0, 1]]):
        raise ValueError('Role projection requires axis-aligned pixel-centre resizing')
    if int(np.prod(size)) > 4194304: raise ValueError('Bounded native output required')
    def native(a): return cv2.resize(a, tuple(size), interpolation=cv2.INTER_NEAREST)
    previous, baseline, reference = (native(a) for a in (old, base, coarse))
    allowed = native(permitted.astype(np.uint8)) > 0
    owner = native(owners.astype(np.float32)).astype(np.int64)
    candidate = allowed & (baseline == 8) & (previous != 8)
    if np.any(candidate & ((owner <= 0) | (owner > len(graph['proposals'])))): raise ValueError('Native proposal has no source owner')
    result = {a: reference.copy() for a in ('native_mid', 'native_safe')}; uncertainty = np.zeros_like(reference)
    used = source is not None
    if not used: uncertainty[candidate] = 255
    if used:
        if source_gray is None or source.get('source_gray_sha256') != sha(source_gray) or source.get('GT_used') is not False or source.get('native_prediction_transfer') is not False:
            raise ValueError('Native refinement source is not bound to its exact pixels')
        if gray.shape == source_gray.shape: expected = source_gray
        elif source.get('resize_method') == 'PIL_BOX': expected = np.array(Image.fromarray(source_gray).resize(gray.shape[::-1], Image.Resampling.BOX))
        elif source.get('resize_method') == 'CV_AREA': expected = cv2.resize(source_gray, gray.shape[::-1], interpolation=cv2.INTER_AREA)
        else: raise ValueError('Explicit source resize provenance required')
        if not np.array_equal(gray, expected): raise ValueError('Input pixels do not match the authenticated native source')
        for name in ('mid', 'safe', 'possible', 'coverage', 'failed_coverage'):
            m = source[name]
            if m.shape != source_gray.shape or m.dtype != np.uint8 or sha(m) != source['mask_sha256'][name]: raise ValueError('Stale native source mask')
        if source['complete']:
            region = candidate & (source['coverage'] > 0) & (source['failed_coverage'] == 0)
            for arm, mask in [('native_mid', 'mid'), ('native_safe', 'safe')]:
                result[arm][region] = baseline[region]
                withdraw = candidate & (source[mask] > 0)
                result[arm][withdraw] = previous[withdraw]
            uncertainty[candidate & (source['possible'] > 0) & (source['safe'] == 0)] = 255
            uncertainty[candidate & (source['coverage'] == 0) & (reference != baseline)] = 255
    return {**result, 'uncertainty_mask': uncertainty, 'complete': source is None or source['complete'],
            'source_context_used': used, 'output_coordinate_system': 'native_source_pixel_centres',
            'new_frame_roles': 0, 'source_pixels_erased': 0, 'production_changed': False}
