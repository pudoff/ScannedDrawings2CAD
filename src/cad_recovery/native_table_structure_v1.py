"""Reviewed native table regions, bounded pixel windows and a sparse cell graph.

No full-page connected-component bitmap or downscaled table evidence. Missing
separators produce draft merges; partial separators remain explicitly unknown.
"""
from collections import defaultdict
from pathlib import Path
import math

import cv2
import numpy as np

from .dataset_layout_v1 import encoded, digest
from .table_structure_v1 import _runs

POLICY = dict(max_window_pixels=4_194_304, maximum_window_side=2048, core_side=1024,
    max_windows_per_region=1024, max_regions=16, max_rules=4096, max_band=128,
    max_rule_pairs=2_000_000, max_atomic_faces=200000, max_cells=4096,
    max_ocr_calls=192, ocr_side=1536, ocr_overlap=192,
    foreground_gray_below=240, rule_fraction=.025, projection_fraction=.02,
    automatic_page_kind=False, production_admitted=False)


def _disk(path, shape):
    with path.open('xb') as stream: stream.truncate(math.prod(shape))
    return np.memmap(path, dtype=np.uint8, mode='r+', shape=shape)


def validate_review(page, review, source_sha):
    if (review.get('schema_version') != 'native-table-review/1.0'
            or review.get('source_sha256') != source_sha
            or review.get('source_surface_sha256') != page.record['surface_sha256']
            or review.get('size_px') != [page.width, page.height]
            or review.get('page_kind') not in ('table_document', 'mixed')
            or not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
            or type(review.get('page_index')) is not int or review['page_index'] < 0):
        raise ValueError('Explicit native source/surface/page/reviewer binding required')
    regions = review.get('regions', [])
    if not 1 <= len(regions) <= POLICY['max_regions']: raise ValueError('Region budget')
    ids = set()
    for region in regions:
        box = region.get('box_px', [])
        if (not isinstance(region.get('id'), str) or not region['id'] or region['id'] in ids
                or region.get('kind') not in ('table', 'title') or len(box) != 4
                or any(type(v) is not int for v in box)):
            raise ValueError('Region id/kind/integer box required')
        l, t, r, b = box
        if not 0 <= l < r <= page.width or not 0 <= t < b <= page.height:
            raise ValueError('Region outside source')
        ids.add(region['id'])
        link = region.get('continuation')
        if link is not None and (not isinstance(link, dict)
                or not isinstance(link.get('source_sha256'), str) or len(link['source_sha256']) != 64
                or type(link.get('page_index')) is not int or link['page_index'] < 0
                or not link.get('region_id') or link.get('evidence') != 'explicit_visual_review'):
            raise ValueError('Explicit source/page/region continuation required')
    for i, a in enumerate(regions):
        l, t, r, b = a['box_px']
        for other in regions[i+1:]:
            x, y, xx, yy = other['box_px']
            if l < xx and x < r and t < yy and y < b: raise ValueError('Overlapping regions')


class Sets:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]; a = self.parent[a]
        return a
    def join(self, a, b): self.parent[self.find(a)] = self.find(b)


def _rules(mask, counts, horizontal, offset):
    length = mask.shape[1] if horizontal else mask.shape[0]
    bands = _runs(counts >= max(8, length*POLICY['projection_fraction']))
    rules = []
    for low, high in bands:
        if high-low > POLICY['max_band']: continue
        band = mask[low:high] if horizontal else mask[:, low:high].T
        active = np.any(band, axis=0)
        for first, last in _runs(active):
            if last-first < max(8, length*POLICY['rule_fraction']): continue
            center = (low+high-1)/2
            points = [[first, center], [last-1, center]]
            if not horizontal: points = [p[::-1] for p in points]
            rules.append(dict(axis='horizontal' if horizontal else 'vertical',
                points_px=[[float(x+offset[0]), float(y+offset[1])] for x, y in points],
                width_px=high-low, band=[low, high], run=[first, last]))
    return rules


def connected_rules(rules):
    horizontal = [i for i, r in enumerate(rules) if r['axis'] == 'horizontal']
    vertical = [i for i, r in enumerate(rules) if r['axis'] == 'vertical']
    if len(horizontal)*len(vertical) > POLICY['max_rule_pairs']: raise ValueError('Grid junction budget')
    sets = Sets(len(rules)); joined = set()
    for i in horizontal:
        h = rules[i]; a, b = h['points_px']
        for j in vertical:
            v = rules[j]; c, d = v['points_px']
            tolerance = (h['width_px']+v['width_px'])/2+.5
            if a[0]-tolerance <= c[0] <= b[0]+tolerance and c[1]-tolerance <= a[1] <= d[1]+tolerance:
                sets.join(i, j); joined.update([i, j])
    groups = defaultdict(list)
    for i in joined: groups[sets.find(i)].append(i)
    eligible = []
    for group in groups.values():
        h = {rules[i]['points_px'][0][1] for i in group if rules[i]['axis'] == 'horizontal'}
        v = {rules[i]['points_px'][0][0] for i in group if rules[i]['axis'] == 'vertical'}
        if len(h) >= 2 and len(v) >= 2: eligible.append(group)
    if not eligible: return []
    largest = max(eligible, key=lambda g: sum(rules[i]['run'][1]-rules[i]['run'][0] for i in g))
    return [rules[i] for i in sorted(largest)]


def _edge(rules, lo, hi, end_width):
    # Junction pixel footprints are excluded at the ends, not used to bridge
    # an unexplained internal break in a separator.
    margin = end_width/2+.5
    lo, hi = lo+margin, hi-margin
    if hi <= lo: return 'partial'
    spans = sorted((max(lo, r['along'][0]-.5), min(hi, r['along'][1]+.5)) for r in rules)
    spans = [(a, b) for a, b in spans if b > a]
    if not spans: return 'missing'
    cursor = lo
    for a, b in spans:
        if a > cursor+.5: return 'partial'
        cursor = max(cursor, b)
    return 'present' if cursor >= hi-.5 else 'partial'


def cells_from_rules(rules, region):
    hs, vs = defaultdict(list), defaultdict(list)
    for r in rules:
        a, b = r['points_px']; horizontal = r['axis'] == 'horizontal'
        axis = a[1] if horizontal else a[0]
        (hs if horizontal else vs)[axis].append(dict(along=[a[0], b[0]] if horizontal else [a[1], b[1]],
                                                    width=r['width_px']))
    ys, xs = sorted(hs), sorted(vs)
    nr, nc = len(ys)-1, len(xs)-1
    if min(nr, nc) < 1: return [], dict(status='unknown', atomic_faces=0)
    if nr*nc > POLICY['max_atomic_faces']: raise ValueError('Sparse cell graph budget')
    sets = Sets(nr*nc+1); exterior = nr*nc; invalid = set()
    hp = {y: max(r['width'] for r in hs[y]) for y in ys}
    vp = {x: max(r['width'] for r in vs[x]) for x in xs}
    for row in range(nr+1):
        for col in range(nc):
            state = _edge(hs[ys[row]], xs[col], xs[col+1], max(vp[xs[col]], vp[xs[col+1]]))
            above = (row-1)*nc+col if row else exterior
            below = row*nc+col if row < nr else exterior
            if state == 'missing': sets.join(above, below)
            elif state == 'partial': invalid.update([above, below])
    for col in range(nc+1):
        for row in range(nr):
            state = _edge(vs[xs[col]], ys[row], ys[row+1], max(hp[ys[row]], hp[ys[row+1]]))
            left = row*nc+col-1 if col else exterior
            right = row*nc+col if col < nc else exterior
            if state == 'missing': sets.join(left, right)
            elif state == 'partial': invalid.update([left, right])
    groups = defaultdict(list)
    for i in range(nr*nc): groups[sets.find(i)].append(divmod(i, nc))
    bad = {sets.find(i) for i in invalid} | {sets.find(exterior)}
    cells = []
    for root, units in groups.items():
        if root in bad: continue
        top, bottom = min(i for i, _ in units), max(i for i, _ in units)+1
        left, right = min(j for _, j in units), max(j for _, j in units)+1
        if len(units) != (bottom-top)*(right-left): continue
        box = [math.ceil(xs[left]+(vp[xs[left]]+1)/2), math.ceil(ys[top]+(hp[ys[top]]+1)/2),
               math.floor(xs[right]-(vp[xs[right]]-1)/2), math.floor(ys[bottom]-(hp[ys[bottom]]-1)/2)]
        if min(box[2]-box[0], box[3]-box[1]) < 1: continue
        cells.append(dict(id=f"{region['id']}-r{top}-c{left}", row=top, column=left,
            row_span=bottom-top, column_span=right-left, box_px=box,
            merge_evidence='no_observed_internal_separator', status='structure_draft'))
    cells.sort(key=lambda c: (c['row'], c['column']))
    for order, cell in enumerate(cells): cell['reading_order'] = order
    if len(cells) > POLICY['max_cells']: raise ValueError('Cell budget')
    return cells, dict(status='cells_observed' if cells else 'unknown', atomic_faces=nr*nc,
                       unknown_face_groups=len(groups)-len(cells), row_axes_px=ys, column_axes_px=xs)


def _ocr_cell(page, cell, engine, budget, text_mask, region):
    l, t, r, b = cell['box_px']; pixels = 0
    for y in range(t, b, 64):
        gray = page.crop([l, y, r, min(b, y+64)])
        pixels += int(np.count_nonzero(gray < POLICY['foreground_gray_below']))
    cell.update(source_ink_pixels=pixels, tokens=[], state='empty_observed' if not pixels else 'unreadable')
    if not pixels: cell['ocr_status'] = 'empty'; return
    if engine is None: cell['ocr_status'] = 'not_requested'; return
    side, overlap = POLICY['ocr_side'], POLICY['ocr_overlap']
    def starts(n):
        if n <= side: return [0]
        return sorted(set([*range(0, n-side+1, side-overlap), n-side]))
    xs, ys = starts(r-l), starts(b-t)
    if len(xs)*len(ys) > 16: cell['ocr_status'] = 'cell_window_budget'; return
    observations = []
    for yy in ys:
        for xx in xs:
            if budget['calls'] >= POLICY['max_ocr_calls']:
                cell['ocr_status'] = 'budget_exhausted'; break
            x, y = l+xx, t+yy
            gray = page.crop([x, y, min(r, x+side), min(b, y+side)])
            pad = 8
            supplied = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
            budget['calls'] += 1
            for token in engine.recognize(cv2.cvtColor(supplied, cv2.COLOR_GRAY2BGR)):
                p = np.asarray(token['polygon_px'], float)-pad
                raw, confidence = token['raw'], float(token['confidence'])
                if (p.shape != (4, 2) or not np.isfinite(p).all() or not isinstance(raw, str)
                        or not np.isfinite(confidence) or not 0 <= confidence <= 1):
                    raise ValueError('Invalid native cell OCR observation')
                if (not raw.strip() or p.min() < -.5 or p[:, 0].max() > gray.shape[1]-.5
                        or p[:, 1].max() > gray.shape[0]-.5): continue
                polygon = p+[x, y]
                observations.append(dict(raw=raw, confidence=confidence, polygon_px=polygon.tolist(),
                    uid=f"{cell['id']}-token-{len(observations)}"))
    from .native_merge_v1 import merge, conflicts
    merged, aliases, _ = merge(observations, 'text')
    cell['tokens'] = [{k: item[k] for k in ('raw', 'confidence', 'polygon_px')} for item in merged]
    cell['ocr_conflicts'] = conflicts(merged)
    cell['ocr_aliases'] = aliases
    cell['ocr_status'] = cell.get('ocr_status', 'observed')
    cell['state'] = 'text_draft' if merged else 'unreadable'
    rl, rt, _, _ = region['box_px']
    for token in cell['tokens']:
        p = np.asarray(token['polygon_px'])
        left, top = np.floor(p.min(0)).astype(int); right, bottom = np.ceil(p.max(0)).astype(int)+1
        left, top, right, bottom = max(l, left), max(t, top), min(r, right), min(b, bottom)
        if right <= left or bottom <= top: continue
        local = np.zeros((bottom-top, right-left), np.uint8)
        cv2.fillPoly(local, [np.rint(p-[left, top]).astype(np.int32)], 1)
        text_mask[top-rt:bottom-rt, left-rl:right-rl] |= local


def analyze_region(page, region, folder, engine, budget):
    folder = Path(folder); folder.mkdir()
    l, t, r, b = region['box_px']; w, h = r-l, b-t
    kx, ky = max(8, round(w*POLICY['rule_fraction'])), max(8, round(h*POLICY['rule_fraction']))
    hx, hy = kx-1, ky-1  # Full opening dependency: erosion AND dilation.
    core = min(POLICY['core_side'], POLICY['maximum_window_side']-2*max(hx, hy))
    if core < 64: raise ValueError('Morphology halo exceeds native window budget')
    count = math.ceil(w/core)*math.ceil(h/core)
    if count > POLICY['max_windows_per_region']: raise ValueError('Grid window count budget')
    masks = {key: _disk(folder/(key+'.mask'), (h, w)) for key in ('horizontal', 'vertical', 'grid', 'text')}
    windows = []; hcounts = np.zeros(h, np.int64); vcounts = np.zeros(w, np.int64)
    try:
        for y in range(0, h, core):
            for x in range(0, w, core):
                rr, bb = min(w, x+core), min(h, y+core)
                left, top, right, bottom = max(0, x-hx), max(0, y-hy), min(w, rr+hx), min(h, bb+hy)
                gray = page.crop([l+left, t+top, l+right, t+bottom])
                binary = (gray < POLICY['foreground_gray_below']).astype(np.uint8)
                horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, kx), np.uint8))
                vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((ky, 1), np.uint8))
                selected = np.s_[y-top:bb-top, x-left:rr-left]
                a, c = horizontal[selected], vertical[selected]
                masks['horizontal'][y:bb, x:rr] = a; masks['vertical'][y:bb, x:rr] = c
                hcounts[y:bb] += a.sum(1, dtype=np.int64)
                vcounts[x:rr] += c.sum(0, dtype=np.int64)
                windows.append(dict(box_px=[l+left, t+top, l+right, t+bottom],
                    core_px=[l+x, t+y, l+rr, t+bb], pixels=int(gray.size), resized=False))
        rules = _rules(masks['horizontal'], hcounts, True, (l, t))+_rules(masks['vertical'], vcounts, False, (l, t))
        if len(rules) > POLICY['max_rules']: raise ValueError('Rule count budget')
        rules = connected_rules(rules)
        cells, structure = cells_from_rules(rules, region)
        if not cells: rules = []
        for rule in rules:
            low, high = rule['band']; first, last = rule['run']
            if rule['axis'] == 'horizontal':
                masks['grid'][low:high, first:last] |= masks['horizontal'][low:high, first:last]
            else: masks['grid'][first:last, low:high] |= masks['vertical'][first:last, low:high]
        for cell in cells: _ocr_cell(page, cell, engine, budget, masks['text'], region)
        for mask in masks.values(): mask.flush()
        report = dict(id=region['id'], kind=region['kind'], box_px=region['box_px'],
            continuation=region.get('continuation'), rules=rules, cells=cells, structure=structure,
            kernel_px=[kx, ky], halo_px=[hx, hy], windows=windows,
            source_resolution_evidence=True, masks={k:dict(path=str((folder/(k+'.mask')).resolve()),
                sha256=digest(folder/(k+'.mask')), shape=[h, w]) for k in ('grid', 'text')})
        (folder/'structure.json').write_bytes(encoded(report))
        return report
    finally:
        for mask in masks.values(): mask._mmap.close()


def analyze(page, review, source_sha, output, *, engine=None):
    validate_review(page, review, source_sha)
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    budget = dict(calls=0); tables = []
    for i, region in enumerate(review['regions']):
        tables.append(analyze_region(page, region, output/f'region-{i:02d}', engine, budget))
    result = dict(schema_version='native-table-structure/1.0', source_sha256=source_sha,
        source_surface_sha256=page.record['surface_sha256'], size_px=[page.width, page.height],
        policy=POLICY, review=review, tables=tables, ocr_calls=budget['calls'],
        OCR_provenance=getattr(engine, 'provenance', None), automatic_routing=False,
        production_admitted=False, physical_geometry_approved=False)
    (output/'structure.json').write_bytes(encoded(result))
    return result
