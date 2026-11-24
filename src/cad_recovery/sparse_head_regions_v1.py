"""Sparse superlevel components with the same eight-connectivity as dense masks."""
import cv2
import numpy as np


def regions(mask, radii, audit, limits, *, dense=False):
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    audit['distance_preparation_pixels'] = int(mask.size)
    if dense:
        for radius in radii:
            if audit['dense_threshold_pixels']+mask.size > limits['maximum_morphology_pixels']:
                audit['limits'].add('dense_threshold_pixel_budget'); return
            audit['dense_threshold_pixels'] += mask.size
            n, labels, stats, _ = cv2.connectedComponentsWithStats((distance > radius).astype(np.uint8), 8)
            result = []
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if max(w, h)+2*radius > 192: continue
                yy, xx = np.where(labels[y:y+h, x:x+w] == i)
                result.append({'points': np.c_[xx+x, yy+y], 'box': [int(x), int(y), int(w), int(h)], 'area': int(area)})
            yield radius, sorted(result, key=lambda r: (-r['area'], r['box']))
        return
    flat = np.flatnonzero(distance.ravel() > min(radii)); count = len(flat)
    audit['sparse_active_pixels'] = count
    if count > limits['maximum_active_pixels']:
        audit['limits'].add('active_pixel_budget'); return
    if not count:
        for radius in radii: yield radius, []
        return
    width = mask.shape[1]; yy, xx = np.divmod(flat, width)
    index = np.full(mask.size, -1, np.int32); index[flat] = np.arange(count, dtype=np.int32)
    weights = distance.ravel()[flat]; order = np.argsort(-weights, kind='stable')
    parent = np.arange(count, dtype=np.int32); sizes = np.ones(count, np.int32); active = np.zeros(count, bool)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = int(parent[i])
        return i

    cursor = 0
    for radius in radii:
        while cursor < count and weights[order[cursor]] > radius:
            i = int(order[cursor]); cursor += 1; active[i] = True
            x, y, position = int(xx[i]), int(yy[i]), int(flat[i])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy) or not 0 <= x+dx < width or not 0 <= y+dy < mask.shape[0]: continue
                    audit['union_neighbor_checks'] += 1
                    if audit['union_neighbor_checks'] > limits['maximum_union_checks']:
                        audit['limits'].add('sparse_union_budget'); return
                    neighbor = int(index[position+dy*width+dx])
                    if neighbor < 0 or not active[neighbor]: continue
                    a, b = find(i), find(neighbor)
                    if a != b:
                        if sizes[a] < sizes[b]: a, b = b, a
                        parent[b] = a; sizes[a] += sizes[b]
        ids = np.flatnonzero(active)
        audit['component_memberships'] += len(ids)
        if audit['component_memberships'] > limits['maximum_component_memberships']:
            audit['limits'].add('component_membership_budget'); return
        roots = np.array([find(int(i)) for i in ids], np.int32)
        _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
        if not len(counts): yield radius, []; continue
        left = np.full(len(counts), width, np.int32); top = np.full(len(counts), mask.shape[0], np.int32)
        right = np.zeros(len(counts), np.int32); bottom = np.zeros(len(counts), np.int32)
        np.minimum.at(left, inverse, xx[ids]); np.minimum.at(top, inverse, yy[ids])
        np.maximum.at(right, inverse, xx[ids]); np.maximum.at(bottom, inverse, yy[ids])
        permutation = np.argsort(inverse, kind='stable'); ends = np.cumsum(counts); starts = ends-counts
        result = []
        for component, area in enumerate(counts):
            w, h = int(right[component]-left[component]+1), int(bottom[component]-top[component]+1)
            if max(w, h)+2*radius > 192: continue
            selected = ids[permutation[starts[component]:ends[component]]]
            result.append({'points': np.c_[xx[selected], yy[selected]],
                'box': [int(left[component]), int(top[component]), w, h], 'area': int(area)})
        yield radius, sorted(result, key=lambda r: (-r['area'], r['box']))
