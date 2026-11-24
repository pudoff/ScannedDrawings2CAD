"""Bounded, explicitly reviewed table regions; never a page classifier.

Coordinates are half-open boxes in the exact working raster. Open or damaged
faces remain unknown. OCR derivatives never replace the source image.
"""
from hashlib import sha256

import cv2
import numpy as np

POLICY = dict(max_pixels=4_194_304, max_regions=16, max_cells=2048,
              max_rules=4096, max_ocr_calls=96, minimum_rule_fraction=.025,
              minimum_face_fill=.92)


def pixel_sha(gray):
    return sha256(gray.tobytes()).hexdigest()


def validate_review(gray, review, source_sha):
    if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2
            or not gray.size or gray.size > POLICY['max_pixels']):
        raise ValueError('A bounded uint8 working grayscale image is required')
    if (review.get('schema_version') != 'table-region-review/1.0'
            or review.get('source_sha256') != source_sha
            or review.get('working_pixel_sha256') != pixel_sha(gray)
            or review.get('working_size_px') != list(gray.shape[::-1])
            or review.get('page_kind') not in ('table_document', 'mixed')
            or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()):
        raise ValueError('Explicit review/source/working image binding required')
    regions = review.get('regions', [])
    if not 1 <= len(regions) <= POLICY['max_regions']:
        raise ValueError('Reviewed region count outside budget')
    used = np.zeros(gray.shape, bool)
    identifiers = set()
    for row in regions:
        box = row.get('box_px', [])
        if (not isinstance(row.get('id'), str) or not row['id'] or row['id'] in identifiers
                or row.get('kind') not in ('table', 'title') or len(box) != 4
                or any(type(v) is not int for v in box)):
            raise ValueError('Unique region id, kind and integer box required')
        l, t, r, b = box
        if not (0 <= l < r <= gray.shape[1] and 0 <= t < b <= gray.shape[0]):
            raise ValueError('Region outside working image')
        if used[t:b, l:r].any():
            raise ValueError('Reviewed regions overlap')
        link = row.get('continuation')
        if link is not None and (not isinstance(link, dict)
                or not isinstance(link.get('source_sha256'), str) or len(link['source_sha256']) != 64
                or type(link.get('page_index')) is not int or link['page_index'] < 0
                or not isinstance(link.get('region_id'), str) or not link['region_id']
                or link.get('evidence') != 'explicit_visual_review'):
            raise ValueError('Continuation requires an explicit source/page/region reference')
        used[t:b, l:r] = True
        identifiers.add(row['id'])
    return used


def _runs(values):
    edges = np.flatnonzero(np.diff(np.r_[False, values, False].astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _rules(mask, horizontal, offset):
    a = mask if horizontal else mask.T
    bands = _runs(np.count_nonzero(a, axis=1) >= max(8, a.shape[1]*.02))
    result = []
    for top, bottom in bands:
        center = (top+bottom-1)/2
        for left, right in _runs(np.any(a[top:bottom], axis=0)):
            if right-left < max(8, a.shape[1]*POLICY['minimum_rule_fraction']):
                continue
            points = [[left, center], [right-1, center]]
            if not horizontal:
                points = [p[::-1] for p in points]
            result.append(dict(points_px=[[float(x+offset[0]), float(y+offset[1])] for x, y in points],
                               axis='horizontal' if horizontal else 'vertical',
                               observed_width_px=bottom-top))
    return result


def _recognize(gray, recognizer, budget):
    if recognizer is None:
        return [], 'not_requested'
    if budget['calls'] >= POLICY['max_ocr_calls']:
        return [], 'budget_exhausted'
    budget['calls'] += 1
    # White padding avoids clipping glyphs at the detector input boundary.
    pad = 8
    image = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    values = recognizer(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
    output = []
    for value in values:
        polygon = np.asarray(value['polygon_px'], float)-pad
        score = float(value['confidence'])
        if (polygon.shape != (4, 2) or not np.isfinite(polygon).all()
                or not np.isfinite(score) or not 0 <= score <= 1
                or not isinstance(value['raw'], str)):
            raise ValueError('Invalid OCR observation')
        if (not value['raw'].strip() or np.any(polygon < -.5)
                or np.any(polygon[:, 0] > gray.shape[1]-.5)
                or np.any(polygon[:, 1] > gray.shape[0]-.5)):
            continue
        output.append(dict(raw=value['raw'], confidence=score, polygon_px=polygon.tolist()))
    return output, 'observed'


def analyze(gray, review, source_sha, *, recognizer=None):
    """Return JSON graph plus exact source/grid/text/residual pixel partition.

    Missing edges are not filled. Enclosed rectangular faces represent merged
    cells directly; row/column spans are inferred from the observed rule axes.
    OCR uses the ORIGINAL cell, with a grid-free derivative only for inspection.
    """
    selected = validate_review(gray, review, source_sha)
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0
    grid = np.zeros_like(ink)
    text = np.zeros_like(ink)
    derivative = gray.copy()
    tables = []
    budget = dict(calls=0)
    total_cells = total_rules = 0
    for region in review['regions']:
        l, t, r, b = region['box_px']
        local = ink[t:b, l:r].astype(np.uint8)*255
        height, width = local.shape
        horizontal = cv2.morphologyEx(local, cv2.MORPH_OPEN,
            np.ones((1, max(8, round(width*.025))), np.uint8)) > 0
        vertical = cv2.morphologyEx(local, cv2.MORPH_OPEN,
            np.ones((max(8, round(height*.025)), 1), np.uint8)) > 0
        network = horizontal | vertical
        # A reviewed box denotes one table. Detached glyph stems must not
        # become column axes or little cells. Other networks remain residual.
        n, components, component_stats, _ = cv2.connectedComponentsWithStats(
            network.astype(np.uint8), connectivity=8)
        if n > 1:
            network &= components == (1+int(np.argmax(component_stats[1:, cv2.CC_STAT_AREA])))
        horizontal &= network
        vertical &= network
        grid[t:b, l:r] = network
        derivative[t:b, l:r][network] = 255
        rules = _rules(horizontal, True, (l, t))+_rules(vertical, False, (l, t))
        total_rules += len(rules)
        if total_rules > POLICY['max_rules']:
            raise ValueError('Rule budget exceeded; no partial DXF emitted')
        xs = sorted(set(p['points_px'][0][0] for p in rules if p['axis'] == 'vertical'))
        ys = sorted(set(p['points_px'][0][1] for p in rules if p['axis'] == 'horizontal'))
        count, labels, stats, _ = cv2.connectedComponentsWithStats((~network).astype(np.uint8), connectivity=4)
        if count > POLICY['max_cells']*8:
            raise ValueError('Face component budget exceeded')
        cells = []
        for k in range(1, count):
            x, y, w, h, area = map(int, stats[k])
            if (x == 0 or y == 0 or x+w == width or y+h == height or min(w, h) < 3
                    or area/(w*h) < POLICY['minimum_face_fill']):
                continue
            total_cells += 1
            if total_cells > POLICY['max_cells']:
                raise ValueError('Cell budget exceeded; no partial DXF emitted')
            face = labels[y:y+h, x:x+w] == k
            original = gray[t+y:t+y+h, l+x:l+x+w].copy()
            original[~face] = 255
            contents = ink[t+y:t+y+h, l+x:l+x+w] & face
            observations, status = ([], 'empty') if not contents.any() else _recognize(original, recognizer, budget)
            for value in observations:
                polygon = np.asarray(value['polygon_px']) + [l+x, t+y]
                value['polygon_px'] = polygon.tolist()
                mask = np.zeros_like(gray)
                cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
                text |= (mask > 0) & ink & ~grid & selected
            cell = dict(id=f"{region['id']}-cell-{len(cells)+1}", box_px=[l+x, t+y, l+x+w, t+y+h],
                        row=max(0, sum(v < t+y for v in ys)-1),
                        column=max(0, sum(v < l+x for v in xs)-1),
                        row_span=1+sum(t+y < v < t+y+h for v in ys),
                        column_span=1+sum(l+x < v < l+x+w for v in xs),
                        state='empty_observed' if not contents.any() else 'text_draft' if observations else 'unreadable',
                        ocr_status=status, tokens=observations, source_ink_pixels=int(contents.sum()))
            cells.append(cell)
        cells.sort(key=lambda c: (c['row'], c['column'], c['box_px'][1], c['box_px'][0]))
        for order, cell in enumerate(cells):
            cell['reading_order'] = order
        if not cells:
            # A detached glyph stroke is not a recovered table. With no
            # enclosed face, keep ALL source ink unresolved and emit no grid.
            rules = []
            grid[t:b, l:r] = False
            derivative[t:b, l:r] = gray[t:b, l:r]
        tables.append(dict(id=region['id'], kind=region['kind'], box_px=region['box_px'],
                           rules=rules, cells=cells, row_axes_px=ys, column_axes_px=xs,
                           structure_status='cells_observed' if cells else 'unknown',
                           open_faces_state='unknown_source_ink_retained',
                           continuation=region.get('continuation'),
                           semantic_mechanical_inference=False))
    # A crossing glyph can share grid pixels. The complete source is also
    # retained in a switchable source layer, independently of these owners.
    grid &= ink & selected
    text &= ~grid
    residual = ink & selected & ~grid & ~text
    if not np.array_equal(grid | text | residual, ink & selected):
        raise ValueError('Source ink partition failed')
    graph = dict(schema_version='table-structure/1.0', source_sha256=source_sha,
                 working_pixel_sha256=pixel_sha(gray), working_size_px=list(gray.shape[::-1]),
                 review=review, policy=POLICY, tables=tables, ocr_calls=budget['calls'],
                 source_pixels=int((ink & selected).sum()), grid_pixels=int(grid.sum()),
                 text_pixels=int(text.sum()), residual_pixels=int(residual.sum()),
                 source_partition_exact=True, automatic_page_routing=False,
                 production_admitted=False, native_resolution_claimed=False)
    return graph, dict(selected=selected, ink=ink, grid=grid, text=text,
                       residual=residual, gridless=derivative)
