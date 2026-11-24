"""Find internal table rules on verified, bounded native raster pixels."""
from collections import defaultdict

import cv2
import numpy as np

from .native_frame_v1 import side_masks
from .native_section_roles_v1 import axis_strokes
from .stroke_roles_v1 import checked_gray


MAX_PIXELS = 4_194_304
MAX_COMPONENTS = 8192
MAX_FACES = 512
MAX_FACE_PIXELS = 16_777_216
MAX_PEN = 96
POLICY = {'minimum_run_fraction': .045, 'minimum_rectangular_fill': .88,
          'minimum_text_cells': 3, 'minimum_active_cells': 4}


def _vertices(labels, size, along_rows):
    """Find the immediately adjacent faces on the two sides of a rule."""
    if not along_rows:
        labels = labels.T
    height, width = labels.shape
    positions = np.arange(height, dtype=np.int32)[:, None]
    above = np.maximum.accumulate(np.where(labels > 0, positions, -1), axis=0)
    below = np.minimum.accumulate(np.where(labels > 0, positions, height)[::-1], axis=0)[::-1]
    column = np.arange(width)
    both = (above >= 0) & (below < height) & (below - above <= size)
    first = labels[np.maximum(above, 0), column]
    second = labels[np.minimum(below, height - 1), column]
    return (both, first, second) if along_rows else (both.T, first.T, second.T)


def _rule_groups(candidate, first, second, faces, minimum_run):
    y, x = np.nonzero(candidate)
    keys = faces*first[y, x] + second[y, x]
    order = np.argsort(keys, kind='stable')
    changes, = np.nonzero(keys[order][1:] != keys[order][:-1])
    for group in np.split(order, changes+1):
        if not len(group):
            continue
        cols, rows = x[group], y[group]
        span = int(cols.max()-cols.min()+1)
        if span < minimum_run:
            continue
        unique, support = np.unique(rows, return_counts=True)
        accepted = unique[support >= .92*span]
        if not len(accepted) or np.any(np.diff(accepted) != 1):
            continue
        yield cols, rows, accepted


def _stable_rule_pixels(candidate, first, second, faces, minimum_run):
    """Retain stable cross-section rows and stop before perpendicular contacts."""
    keep = np.zeros_like(candidate)
    core = np.zeros_like(candidate)
    for cols, rows, accepted in _rule_groups(candidate, first, second, faces, minimum_run):
        if len(accepted) < 3:
            continue
        margin = max(2, len(accepted))
        inside = (cols >= cols.min()+margin) & (cols <= cols.max()-margin)
        valid = np.isin(rows, accepted) & inside
        keep[rows[valid], cols[valid]] = True
        # The outermost digital samples can include an attached vertical
        # glyph stroke. Only promote the interior of a resolved thick rule.
        accepted = accepted[1:-1]
        valid = np.isin(rows, accepted) & inside
        core[rows[valid], cols[valid]] = True
    return keep, core


def _thin_rule_pixels(candidate, first, second, faces, minimum_run, visible):
    """Corroborate a thin line across several independent pairs of cells."""
    rows_by_band = defaultdict(list)
    for cols, rows, accepted in _rule_groups(candidate, first, second, faces, minimum_run):
        if len(accepted) <= 2:
            band = tuple(accepted)
            rows_by_band[band].append((int(cols.min()), int(cols.max()), cols, rows))
    result = np.zeros_like(candidate)
    for band, runs in rows_by_band.items():
        runs.sort(key=lambda r: (r[0], r[1]))
        for left, right in zip(runs, runs[1:]):
            # Demand a real junction in between two different observed
            # separations. Both shape and width agree, and the bridge is ink.
            span = right[0]-left[1]
            if not 1 <= span <= MAX_PEN or not visible[np.asarray(band), left[1]:right[0]+1].all():
                continue
            for start, end, cols, rows in (left, right):
                keep = (cols >= start+2) & (cols <= end-2) & np.isin(rows, band)
                result[rows[keep], cols[keep]] = True
    return result


def _inward_table_boundary(ink, network, cells, exterior, eligible, directions, outward, audit):
    output = np.zeros_like(ink)
    exterior_cells = np.zeros_like(eligible)
    exterior_cells[list(exterior - {0})] = True
    for major in (0, 1):
        sign = outward[1-major]
        if not sign:
            continue
        valid, first, second = _vertices(cells, MAX_PEN, major == 0)
        outside, inside = (first, second) if sign > 0 else (second, first)
        allowed = valid & exterior_cells[outside] & eligible[inside] & network & directions[major]
        limit = network.shape[major]-1 if sign > 0 else 0
        strokes = list(axis_strokes(network, major, opening=43, minimum_length=80, maximum_band=24))
        rim = [span for _, span, center, pen in strokes if abs(center-limit) <= 3*pen]
        for points, (lo, hi), center, pen in strokes:
            if abs(center-limit) <= 3*pen or not any(min(hi, b)-max(lo, a) >= .55*(hi-lo) for a, b in rim):
                continue
            x, y = points.T
            at_cell = allowed[y, x]
            if at_cell.sum() < .5*len(points) or np.unique(inside[y[at_cell], x[at_cell]]).size < 2:
                continue
            audit['fit_pixels'] += ink.size
            if audit['fit_pixels'] > MAX_FACE_PIXELS:
                audit['limited'] = True
                return output
            ends = np.array([[lo, center], [hi, center]])
            if major:
                ends = ends[:, ::-1]
            fitted, reason = side_masks(ink, ends, pen, .7, audit)
            if reason and reason.endswith('_budget'):
                audit['limited'] = True
                return output
            if fitted is not None:
                along = np.arange(ink.shape[1])[None, :] if major == 0 else np.arange(ink.shape[0])[:, None]
                output |= fitted['safe'] & allowed & (along >= lo+2*pen) & (along <= hi-2*pen)
    return output


def _wide_cell_separators(ink, network, cells, eligible, has_text, directions, minimum_run, audit):
    output = np.zeros_like(ink)
    for major in (0, 1):
        valid, first, second = _vertices(cells, MAX_PEN, major == 0)
        supported = valid & eligible[first] & eligible[second] & (first != second)
        supported &= (has_text[first] | has_text[second]) & network & directions[major]
        for points, (lo, hi), center, pen in axis_strokes(network, major, opening=minimum_run | 1,
                                                         minimum_length=3*(minimum_run | 1), maximum_band=24):
            if pen > 4:
                continue
            x, y = points.T
            if supported[y, x].sum() < .55*len(points):
                continue
            audit['fit_pixels'] += ink.size
            if audit['fit_pixels'] > MAX_FACE_PIXELS:
                audit['limited'] = True
                return output
            ends = np.array([[lo, center], [hi, center]])
            if major:
                ends = ends[:, ::-1]
            fitted, reason = side_masks(ink, ends, pen, .7, audit)
            if reason and reason.endswith('_budget'):
                audit['limited'] = True
                return output
            if fitted is not None and fitted['observed_width_across_px'] <= 4:
                along = np.arange(ink.shape[1])[None, :] if major == 0 else np.arange(ink.shape[0])[:, None]
                output |= fitted['safe'] & supported & (along >= lo+pen) & (along <= hi-pen)
    return output


def analyze(gray, *, original_edge=True, thin_rules=False, outward=None, wide_cells=False):
    """Confirm interior rulings from two actual cells and a text-bearing grid.

    Opening proposes horizontal/vertical rules. Every pixel kept by the
    observer was foreground in the source. The outer rim and detached lines
    provide no positive GRID evidence.
    """
    checked_gray(gray)
    if gray.size > MAX_PIXELS:
        raise ValueError('Bounded original raster required')
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    minimum_run = max(16, round(POLICY['minimum_run_fraction']*min(gray.shape)))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, minimum_run | 1), np.uint8)) > 0
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((minimum_run | 1, 1), np.uint8)) > 0
    ruling = horizontal | vertical
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ruling.astype(np.uint8), connectivity=8)
    empty = np.zeros(gray.shape, bool)
    result = {'grid': empty, 'stable_grid': empty.copy(), 'stable_bands': empty.copy(),
              'table_context': empty.copy(), 'complete': count <= MAX_COMPONENTS,
              'networks': [], 'minimum_run_px': minimum_run, 'window_reaches_original_edge': original_edge}
    if thin_rules:
        result['thin_grid'] = empty.copy()
    if outward is not None:
        if len(outward) != 2 or any(side not in (-1, 0, 1) for side in outward):
            raise ValueError('Observed page side direction required for each axis')
        result['boundary_grid'] = empty.copy()
    if wide_cells:
        result['wide_cell_grid'] = empty.copy()
    if not result['complete']:
        return result
    configurations = []
    face_pixels = 0
    boundary_audit = {'source_samples': 0, 'fit_pixels': 0, 'limited': False}
    wide_audit = {'source_samples': 0, 'fit_pixels': 0, 'limited': False}
    attached_ink = cv2.dilate(((ink > 0) & ~ruling).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        if min(width, height) < 2*minimum_run or area < 4*minimum_run:
            continue
        face_pixels += width*height
        if face_pixels > MAX_FACE_PIXELS:
            return {**result, 'complete': False, 'reason': 'face_pixel_budget', 'networks': []}
        section = np.s_[y:y+height, x:x+width]
        network = labels[section] == component
        # Label free faces on the exact network. Text is checked separately.
        faces, cells, bounds, _ = cv2.connectedComponentsWithStats((~network).astype(np.uint8), connectivity=4)
        if faces > MAX_FACES:
            return {**result, 'complete': False, 'reason': 'face_budget', 'networks': []}
        exterior = set(np.unique(np.r_[cells[0], cells[-1], cells[:, 0], cells[:, -1]]))
        holes = []
        text_cells = []
        for identity in range(1, faces):
            cx, cy, w, h, cell_area = map(int, bounds[identity])
            if (identity in exterior or min(w, h) < max(14, minimum_run//3)
                    or cell_area < POLICY['minimum_rectangular_fill']*w*h):
                continue
            occupied = ((ink[y+cy:y+cy+h, x+cx:x+cx+w] > 0) & (cells[cy:cy+h, cx:cx+w] == identity))
            density = occupied.sum()/cell_area
            if density > .32:
                continue
            holes.append(identity)
            if .007 <= density and cv2.connectedComponents(occupied.astype(np.uint8), connectivity=8)[0] >= 3:
                text_cells.append(identity)
        if len(holes) < POLICY['minimum_active_cells'] or len(text_cells) < POLICY['minimum_text_cells']:
            continue
        # Keep only branches of one cell complex. A rectangle at a different
        # scale (such as the free sheet interior) does not certify a separator.
        median_area = float(np.median(bounds[text_cells, cv2.CC_STAT_AREA]))
        eligible = np.zeros(faces, bool)
        eligible[holes] = bounds[holes, cv2.CC_STAT_AREA] <= 10*median_area
        if sum(eligible[text_cells]) < POLICY['minimum_text_cells']:
            continue
        internal = np.zeros_like(network)
        stable = np.zeros_like(network)
        stable_bands = np.zeros_like(network)
        thin = np.zeros_like(network) if thin_rules else None
        for along_rows, direction in ((True, horizontal), (False, vertical)):
            valid, first, second = _vertices(cells, MAX_PEN, along_rows)
            candidate = valid & eligible[first] & eligible[second] & (first != second) & direction[section] & network
            internal |= candidate
            if original_edge:
                if along_rows:
                    bands, core = _stable_rule_pixels(candidate, first, second, faces, minimum_run)
                else:
                    vertical_bands, vertical_core = _stable_rule_pixels(candidate.T, first.T, second.T, faces, minimum_run)
                    bands, core = vertical_bands.T, vertical_core.T
                stable |= core
                stable_bands |= bands
                if thin_rules:
                    visible = (ink[section] > 0) & network
                    if along_rows:
                        thin |= _thin_rule_pixels(candidate, first, second, faces, minimum_run, visible)
                    else:
                        thin |= _thin_rule_pixels(candidate.T, first.T, second.T, faces, minimum_run, visible.T).T
        if internal.any():
            boundary = None
            wide = None
            if outward is not None and original_edge:
                boundary = _inward_table_boundary(ink[section] > 0, network, cells, exterior, eligible,
                    (horizontal[section], vertical[section]), outward, boundary_audit)
            if wide_cells and original_edge:
                extended = np.zeros_like(eligible)
                extended[holes] = bounds[holes, cv2.CC_STAT_AREA] <= 25*median_area
                text = np.zeros_like(eligible)
                text[text_cells] = True
                wide = _wide_cell_separators(ink[section] > 0, network, cells, extended, text,
                    (horizontal[section], vertical[section]), minimum_run, wide_audit)
            configurations.append((section, internal, stable, stable_bands, network, thin, boundary, wide))
            result['networks'].append({'box_px': [x, y, x+width, y+height], 'cells': int(eligible.sum()),
                                       'text_cells': len(text_cells), 'internal_pixels': int(internal.sum()),
                                       'stable_internal_pixels': int(stable.sum())})
    for section, accepted, stable, stable_bands, network, thin, boundary, wide in configurations:
        result['grid'][section] |= accepted
        result['stable_grid'][section] |= stable & ~attached_ink[section]
        result['stable_bands'][section] |= stable_bands & ~attached_ink[section]
        result['table_context'][section] |= network
        if thin_rules:
            result['thin_grid'][section] |= thin & ~attached_ink[section]
        if boundary is not None and not boundary_audit['limited']:
            result['boundary_grid'][section] |= boundary & ~attached_ink[section]
        if wide is not None and not wide_audit['limited']:
            result['wide_cell_grid'][section] |= wide & ~attached_ink[section]
    return result


def _row_runs(raster):
    transitions = np.diff(np.pad(raster, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, starts = np.nonzero(transitions == 1)
    _, stops = np.nonzero(transitions == -1)
    return zip(rows, starts, stops)


def _validate_support(ink, support):
    if (ink.ndim != 2 or ink.shape != support.shape or ink.dtype != bool or support.dtype != bool
            or np.any(support & ~ink)):
        raise ValueError('Aligned original ink and supported rule pixels required')


def connect_supported_runs(ink, support):
    """Keep observed pixels between confirmed endpoints of the same straight run."""
    _validate_support(ink, support)
    result = np.zeros_like(ink)
    for raster, anchors, output in ((ink, support, result), (ink.T, support.T, result.T)):
        for row, first, last in _row_runs(raster):
            span = last-first
            if span < 24:
                continue
            confirmed = np.flatnonzero(anchors[row, first:last])
            if len(confirmed) >= max(16, .5*span):
                output[row, first+confirmed[0]:first+confirmed[-1]+1] = True
    return result


def short_supported_gaps(ink, support):
    """Find short unassigned spans between supported samples on observed ink."""
    _validate_support(ink, support)
    result = np.zeros_like(ink)
    for raster, anchors, target in ((ink, support, result), (ink.T, support.T, result.T)):
        for row, lo, hi in _row_runs(raster):
            if hi-lo < 10:
                continue
            hits = np.flatnonzero(anchors[row, lo:hi])
            separation = np.diff(hits)
            for gap in np.flatnonzero((separation > 1) & (separation <= 24)):
                target[row, lo+hits[gap]+1:lo+hits[gap+1]] = True
    return result


def rules_between_crossings(ink, support):
    """Localize short interiors between two confirmed perpendicular rules."""
    _validate_support(ink, support)
    output = np.zeros_like(ink)
    if not support.any():
        return output
    for raster, anchors, target in ((ink, support, output), (ink.T, support.T, output.T)):
        y, x = np.nonzero(raster)
        top, bottom = int(y.min()), int(y.max())
        left, right = int(x.min()), int(x.max())
        axis = np.ones((25, 1), np.uint8)
        upright = cv2.morphologyEx(anchors.astype(np.uint8), cv2.MORPH_OPEN, axis)
        observed = cv2.morphologyEx(raster.astype(np.uint8), cv2.MORPH_OPEN, axis) > 0
        crossings = (cv2.dilate(upright, np.ones((35, 1), np.uint8)) > 0) & observed
        for row, first, last in _row_runs(raster):
            if not top+16 < row < bottom-16 or last-first < 50:
                continue
            hits = np.flatnonzero(crossings[row, first:last])
            if not len(hits) or hits[-1]-hits[0] < 48 or not np.any(np.diff(hits) > 1):
                continue
            start = max(left+16, first+hits[0])
            stop = min(right-16, first+hits[-1]+1)
            target[row, start:stop] = True
    return output


def supported_table_footprints(ink, table, accepted):
    """Recover the observed interior of a rule with both exact raster edges."""
    context = table['table_context']
    if any(mask.shape != ink.shape or mask.dtype != bool for mask in (context, accepted)) or ink.dtype != bool:
        raise ValueError('Aligned source and table masks required')
    result = np.zeros_like(ink)
    if not table['complete'] or not table['window_reaches_original_edge'] or not table['stable_grid'].any():
        return result
    supported = context & (table['stable_bands'] | table['thin_grid'] | accepted)
    y, x = np.nonzero(context)
    bounds = [(x.min(), x.max()), (y.min(), y.max())]
    audit = {'source_samples': 0}
    fit_pixels = 0
    for major in (0, 1):
        lower, upper = bounds[1-major]
        for points, (start, stop), center, pen in axis_strokes(context, major, opening=43, minimum_length=48, maximum_band=24):
            if not lower+2*pen < center < upper-2*pen:
                continue
            x, y = points.T
            if supported[y, x].sum() < max(55, .27*len(points)):
                continue
            fit_pixels += ink.size
            if fit_pixels > 256_000_000:
                return np.zeros_like(ink)
            ends = np.array([[start, center], [stop, center]])
            if major == 1:
                ends = ends[:, ::-1]
            fitted, reason = side_masks(ink, ends, pen, 1., audit)
            if reason and reason.endswith('_budget'):
                return np.zeros_like(ink)
            if fitted is not None:
                along = np.arange(ink.shape[1])[None, :] if major == 0 else np.arange(ink.shape[0])[:, None]
                result |= fitted['safe'] & (along >= start+2*pen) & (along <= stop-2*pen)
    return result


def crossing_supported_border(gray, accepted):
    """Retain a thin source ruling only between one-sided grid contacts."""
    checked_gray(gray)
    if accepted.dtype != bool or accepted.shape != gray.shape:
        raise ValueError('Aligned grid anchors required')
    result = np.zeros_like(accepted)
    if not accepted.any():
        return result
    visible = gray < 238
    for major in (0, 1):
        support = accepted if major == 0 else accepted.T
        source = visible if major == 0 else visible.T
        crossings = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_OPEN, np.ones((21, 1), np.uint8))
        for points, (lo, hi), center, pen in axis_strokes(visible, major, opening=35, minimum_length=43, maximum_band=3):
            middle = round(center)
            margin = max(2, int(pen))
            contacts = []
            for side in (-1, 1):
                offsets = middle+side*np.arange(margin+1, margin+7)
                offsets = offsets[(offsets >= 0) & (offsets < crossings.shape[0])]
                if offsets.size < 4:
                    continue
                count = crossings[offsets, lo:hi+1].sum(axis=0)
                inside = slice(min(middle, offsets[0]), max(middle, offsets[0])+1)
                contact = source[inside, lo:hi+1].all(axis=0)
                hits = np.flatnonzero((count >= 4) & contact)+lo
                clusters = 1+np.count_nonzero(np.diff(hits) > 1) if hits.size else 0
                if clusters >= 3 and hits[-1]-hits[0] >= 40:
                    contacts.append((hits[0], hits[-1]))
            if len(contacts) == 1:
                first, last = contacts[0]
                axis = points[:, major]
                x, y = points[(axis >= first) & (axis <= last)].T
                result[y, x] = True
    return result


def supported_rule_crossings(gray, accepted):
    """Find intersections when both observed directions have grid support."""
    checked_gray(gray)
    if accepted.dtype != bool or accepted.shape != gray.shape:
        raise ValueError('Aligned grid anchors required')
    visible = gray < 238
    bands = []
    for axis in (0, 1):
        band = np.zeros_like(visible)
        for points, _, _, pen in axis_strokes(visible, axis, opening=31, minimum_length=48, maximum_band=24):
            x, y = points.T
            if pen >= 3 and 4*accepted[y, x].sum() >= len(points):
                band[y, x] = True
        bands.append(band)
    return bands[0] & bands[1]
