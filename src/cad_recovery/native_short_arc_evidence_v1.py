"""Compare a fixed independently supported circle with a free native straight fit."""
import math
import cv2
import numpy as np

from .anchored_curve_fit_v1 import POLICY as FIT
from .native_frame_v1 import POLICY as FRAME
from .source_arrow_observation_v1 import checked_matrix, transform


def assess(gray, origin, source_to_working, points, circle, width):
    matrix = checked_matrix(source_to_working); inverse = np.linalg.inv(matrix)
    if gray.ndim != 2 or gray.dtype != np.uint8 or max(gray.shape) > 1024:
        raise ValueError('Bounded original grayscale window required')
    native = transform(np.asarray(points, float), inverse)-origin
    scale = float(np.linalg.svd(matrix[:2, :2], compute_uv=False).min())
    result = dict(accepted=False, section_count=0, ratio_limit=FIT['curve_line_RMSE_ratio'],
        original_pixels=True, circle_parameters_changed=False)
    if scale > .5 or len(native) < 3:
        return dict(result, reason='insufficient_native_resolution')
    delta = native[-1]-native[0]; major = int(np.argmax(abs(delta))); minor = 1-major
    if abs(delta[major]) < 1:
        return dict(result, reason='short_native_extent')
    margin = max(1, math.ceil(width/(2*scale)))
    reach = math.ceil((width/2+FIT['fit_max_input_px'])/scale)
    slope = delta[minor]/delta[major]
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    rows = []
    for x in range(math.ceil(native[:,major].min())+margin, math.floor(native[:,major].max())-margin+1):
        if not 0 <= x < gray.shape[1-major]: continue
        center = native[0,minor]+(x-native[0,major])*slope
        section = gray[:,x] if major == 0 else gray[x,:]
        low, high = max(0,math.floor(center-reach)), min(len(section),math.ceil(center+reach)+1)
        if high <= low: continue
        weights = 255.-section[low:high].astype(float)
        dark = section[low:high] <= threshold
        change = np.diff(np.r_[False,dark,False].astype(int))
        runs = list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))
        if len(runs) != 1: continue
        a,b = runs[0]
        if a == 0 or b == high-low: continue
        weights[:max(0,a-1)] = 0; weights[min(len(weights),b+1):] = 0
        if not weights.sum(): continue
        point = np.zeros(2); point[major] = x
        point[minor] = float(np.arange(low,high)@weights/weights.sum())
        rows.append((point+origin, int(b-a)))
    if rows:
        median = float(np.median([width for _,width in rows]))
        rows = [row for row in rows if abs(row[1]-median) <= 1]
    result['section_count'] = len(rows)
    if len(rows) < FRAME['minimum_sections']:
        return dict(result, reason='few_native_cross_sections')
    work = transform(np.asarray([row[0] for row in rows]), matrix)
    centered = work-work.mean(0); _,_,axes = np.linalg.svd(centered, full_matrices=False)
    linear = centered@axes[-1]
    radial = np.linalg.norm(work-circle['center_px'],axis=1)-circle['radius_px']
    offset = float(np.median(radial)); radial -= offset
    line_error = float(np.sqrt(np.mean(linear**2))); arc_error = float(np.sqrt(np.mean(radial**2)))
    numerical = 64*np.finfo(float).eps*max(1.,float(abs(work).max()))
    accepted = (line_error > numerical and arc_error <= FIT['curve_line_RMSE_ratio']*line_error
        and abs(offset) <= FIT['fit_max_input_px'] and abs(radial).max() <= FIT['fit_max_input_px']
        and np.percentile(abs(radial),95) <= FIT['fit_p95_input_px'])
    result.update(accepted=bool(accepted), reason='native_circle_preferred' if accepted else 'straight_or_ambiguous_native_evidence',
        line_RMSE_working_px=line_error, circle_RMSE_working_px=arc_error, nuisance_offset_working_px=offset,
        section_margin_native_px=margin, section_reach_native_px=reach)
    return result
