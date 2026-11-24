"""Conservative pixel-only evidence for a detached view, not CAD semantics.

A glyph in a repeated table cell cannot authorize frame-based grid withdrawal.
Legacy frame hypotheses receive exactly the same check as new hypotheses.
"""
from collections import Counter
from copy import deepcopy
import cv2
import numpy as np

from . import page_frame_context_v5 as previous
from .grid_role_revision_v1 import sha
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {
    'version': 'frame-view-guard/1.0',
    'maximum_pixels': 4194304, 'maximum_frames': 24,
    'maximum_components': 4096, 'maximum_cells': 8192,
    'maximum_component_samples': 16000000,
    'minimum_view_extent_fraction': 2 * previous.POLICY['minimum_view_extent_fraction'],
    'comparable_cell_area_ratio': 2., 'minimum_repeated_cells': 3,
    'source_semantics_confirmed': False, 'production_admitted': False,
    'physical_geometry_approved': False,
}


def _ruling_component(frame, labels):
    """Vote on observed side centres; never identify a grid through text labels."""
    votes = Counter()
    for side in frame['sides']:
        a, b = np.asarray(side['points_overview'], float)
        u = (b-a)/np.linalg.norm(b-a); n = np.array([-u[1], u[0]])
        width = float(side['stroke_width_overview_px'])
        for t in np.linspace(.1, .9, 17):
            point = a+t*(b-a)
            for offset in (-width/4, 0., width/4):
                x, y = np.rint(point+offset*n).astype(int)
                if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1] and labels[y, x]:
                    votes[int(labels[y, x])] += 1
    if not votes: return None
    identity, count = votes.most_common(1)[0]
    return identity if count > sum(votes.values())/2 else None


def guard(gray, packet):
    checked_gray(gray)
    if packet['source_sha256'] != sha(gray) or packet['size_px'] != list(gray.shape[::-1]):
        raise ValueError('Detached-view guard source binding differs')
    if any(f['frame_semantics_confirmed'] for f in packet['frames']):
        raise ValueError('Semantic frame targets are not inference inputs')
    audit = {'limits': [], 'component_samples': 0, 'cell_maps': 0}
    # Never mutate a frozen detector packet, including nested lists.
    result = deepcopy(packet)
    if not packet['complete']:
        return {**result, 'frames': [], 'view_guard': {'complete': False, 'audit': audit, 'frames': []}}
    if gray.size > POLICY['maximum_pixels'] or len(packet['frames']) > POLICY['maximum_frames']:
        audit['limits'].append('view_guard_pixel_or_frame_budget')
    if not packet['frames']:
        return {**result, 'view_guard': {'complete': not audit['limits'], 'audit': audit, 'frames': []},
                'complete': not audit['limits']}
    if audit['limits']:
        return {**result, 'frames': [], 'complete': False,
                'view_guard': {'complete': False, 'audit': audit, 'frames': []}}
    ink = foreground(gray)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count-1 > POLICY['maximum_components']: audit['limits'].append('view_guard_component_budget')
    cache, component_cache, decisions, accepted = {}, {}, [], []
    for index, frame in enumerate(packet['frames']):
        if audit['limits']: break
        network = _ruling_component(frame, labels)
        candidates = []
        if network is not None and network not in cache:
            # Other detached objects are intentionally free space: their own
            # holes must not be mistaken for cells of the ruling network.
            n, cells, cell_stats, _ = cv2.connectedComponentsWithStats((labels != network).astype(np.uint8), connectivity=4)
            exterior = set(map(int, np.unique(np.r_[cells[0], cells[-1], cells[:, 0], cells[:, -1]])))
            enclosed = [i for i in range(1, n) if i not in exterior]
            cache[network] = (cells, cell_stats, exterior, enclosed)
            audit['cell_maps'] += 1
            if n-1 > POLICY['maximum_cells']: audit['limits'].append('view_guard_cell_budget'); break
        q = np.asarray(frame['quad_overview'], float)
        span = min(np.linalg.norm(q[1]-q[0]), np.linalg.norm(q[3]-q[0]))
        for view in frame['view_components']:
            identity = int(view['component'])
            if not 0 < identity < count: raise ValueError('Stale detached component ID')
            if identity not in component_cache:
                x, y, w, h, _ = map(int, stats[identity])
                yy, xx = np.nonzero(labels[y:y+h, x:x+w] == identity)
                points = np.c_[xx+x, yy+y]
                audit['component_samples'] += len(points)
                component_cache[identity] = points
            points = component_cache[identity]
            if audit['component_samples'] > POLICY['maximum_component_samples']:
                audit['limits'].append('view_guard_sample_budget'); break
            rect = cv2.minAreaRect(points.astype(np.float32))
            extent = float(min(rect[1]))
            proof = {'component': identity, 'minimum_oriented_extent_px': extent,
                     'frame_span_px': float(span), 'relative_extent': extent/max(span, 1.),
                     'ruling_component': network, 'container_cell': None, 'comparable_cells': 0}
            reason = None
            if network is None: reason = 'unresolved_ruling_network'
            else:
                cells, cell_stats, exterior, enclosed = cache[network]
                ids = np.unique(cells[points[:, 1], points[:, 0]])
                if len(ids) != 1 or not ids[0] or ids[0] in exterior:
                    reason = 'no_enclosed_drawing_region'
                else:
                    cell = int(ids[0]); area = int(cell_stats[cell, cv2.CC_STAT_AREA])
                    ratio = POLICY['comparable_cell_area_ratio']
                    peers = [i for i in enclosed if area/ratio <= cell_stats[i, cv2.CC_STAT_AREA] <= area*ratio]
                    proof.update(container_cell=cell, container_area_px=area, comparable_cells=len(peers))
                    if len(peers) >= POLICY['minimum_repeated_cells']: reason = 'repeated_table_cell'
            if reason is None and extent < POLICY['minimum_view_extent_fraction']*span:
                reason = 'insufficient_geometric_extent'
            proof.update(accepted=reason is None, reason=reason)
            candidates.append(proof)
        keep = any(c['accepted'] for c in candidates)
        decisions.append({'frame_index': index, 'retained': keep, 'candidates': candidates})
        if keep: accepted.append(deepcopy(frame))
    complete = packet['complete'] and not audit['limits']
    return {**result, 'frames': accepted if complete else [], 'complete': complete,
            'view_guard': {'version': POLICY['version'], 'complete': complete, 'audit': audit,
                           'frames': decisions, 'source_pixels_changed': 0,
                           'geometry_modified': False, 'semantic_view_confirmed': False}}


def detect(gray):
    return guard(gray, previous.detect(gray))
