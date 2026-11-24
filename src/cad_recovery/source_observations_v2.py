"""Check the observed skeleton inventory and collect source-bound frame strokes."""
from collections import Counter
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .curve_graph_v1 import graph_paths
from .grid_role_revision_v1 import sha
from .geometry_bundle_v1 import validate_geometry
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground


POLICY = {'maximum_pixels': 4_194_304, 'maximum_frames': 24, 'maximum_fragments': 4096,
          'maximum_source_points': 2_000_000, 'maximum_side_checks': 250_000,
          'minimum_line_span_px': 6., 'maximum_line_error_px': 2., 'line_p95_px': 1.25,
          'short_path_edges': 4, 'maximum_excluded_components': 4096, 'axis_angle_deg': 8.}


def edge_counts(paths):
    edges = Counter()
    for path in paths:
        points = [tuple(map(int, point)) for point in path]
        edges.update(tuple(sorted((a, b))) for a, b in zip(points, points[1:]))
    return edges


def excluded_ink(gray, ink):
    raw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    removed = (raw > 0) & (ink == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(removed.astype(np.uint8), connectivity=8)
    complete = count-1 <= POLICY['maximum_excluded_components']
    components = []
    if complete:
        for x, y, w, h, area in stats[1:]:
            components.append({'bbox_xyxy': [int(x), int(y), int(x+w), int(y+h)], 'ink_pixels': int(area),
                               'role': 'unclassified'})
    return {'complete': complete, 'component_count': count-1, 'ink_pixels': int(removed.sum()),
            'mask_sha256': sha(removed), 'components': components,
            'reason': None if complete else 'excluded_component_budget'}


def source_inventory(packet):
    """Check both vertex and edge coverage of the approved skeleton topology."""
    gray = packet['gray']
    if packet.get('input_sha256') != sha(gray):
        raise ValueError('Source inventory belongs to another raster')
    validate_geometry(gray, packet['source'], packet['geometry'])
    ink = foreground(gray)
    skeleton = skeletonize(ink > 0)
    paths, stats = graph_paths(skeleton)
    expected = edge_counts(p['points'] for p in paths)
    observed = edge_counts(p['points_px'] for p in packet['source'])
    mask = np.zeros_like(gray, dtype=bool)
    short = []
    for path in packet['source']:
        points = np.asarray(path['points_px'], int)
        x, y = points.T
        mask[y, x] = True
        if len(points)-1 <= POLICY['short_path_edges']:
            short.append({'source_id': path['id'], 'path_id': path['path_id'], 'points_px': path['points_px'],
                          'edges': len(points)-1})
    missing = int(np.count_nonzero(skeleton & ~mask))
    extra = int(np.count_nonzero(mask & ~skeleton))
    missing_edges = sum((expected-observed).values()) if not stats['search_incomplete'] else None
    extra_edges = sum((observed-expected).values()) if not stats['search_incomplete'] else None
    return {'source_sha256': sha(gray), 'geometry_sha256': object_sha(packet['geometry']),
            'complete': bool(not stats['search_incomplete'] and missing == extra == missing_edges == extra_edges == 0),
            'missing_pixels': missing, 'extra_pixels': extra, 'missing_edges': missing_edges, 'extra_edges': extra_edges,
            'skeleton_search_complete': not stats['search_incomplete'], 'observed_edges': sum(observed.values()),
            'short_paths': short, 'isolated_points': sum(record['edges'] == 0 for record in short),
            'raw_ink_excluded_by_foreground': excluded_ink(gray, ink),
            'geometric_fit_complete': packet['selection']['extraction_complete'],
            'semantic_graph_complete': not packet['graph']['search_incomplete'],
            'legacy_packet_complete': packet['complete']}


def source_fragments(paths, distance, audit):
    """Yield straight subranges of original skeleton walks, with no gap joining."""
    for index, path in enumerate(paths):
        points = np.asarray(path['points'], int)
        audit['source_points'] += len(points)
        if audit['source_points'] > POLICY['maximum_source_points']:
            audit['limits'].append('source_point_budget')
            return
        if len(points) <= POLICY['minimum_line_span_px'] or path['closed']:
            continue
        vertices = cv2.approxPolyDP(points.astype(np.float32), POLICY['line_p95_px'], False).reshape(-1, 2)
        vertex_indices = [0]
        for vertex in vertices[1:-1]:
            remaining = points[vertex_indices[-1]+1:]
            matches = np.flatnonzero(np.all(remaining == vertex, axis=1))
            if not matches.size:
                raise ValueError('Fragment fit was not indexed to its source walk')
            vertex_indices.append(vertex_indices[-1]+int(matches[0])+1)
        vertex_indices.append(len(points)-1)
        for lo, hi in zip(vertex_indices, vertex_indices[1:]):
            q = points[lo:hi+1]
            delta = q[-1]-q[0]
            span = np.linalg.norm(delta)
            if span < POLICY['minimum_line_span_px']:
                continue
            u = delta / span
            errors = abs((q-q[0]) @ np.array([-u[1], u[0]]))
            if errors.max() > POLICY['maximum_line_error_px'] or np.quantile(errors, .95) > POLICY['line_p95_px']:
                continue
            x, y = q.T
            audit['fragments'] += 1
            if audit['fragments'] > POLICY['maximum_fragments']:
                audit['limits'].append('source_fragment_budget')
                return
            yield {'path_number': index, 'source_index_interval': [lo, hi], 'points': q, 'direction': u,
                   'pen_px': max(1., float(2*np.median(distance[y, x])-1)), 'p95_error_px': float(np.quantile(errors, .95))}


def frame_sides(frame):
    if len(frame['sides']) != 4:
        raise ValueError('Four observed frame sides required')
    quad = np.asarray(frame['quad_overview'], float)
    if quad.shape != (4, 2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad.astype(np.float32)):
        raise ValueError('Finite convex frame required')
    for side in frame['sides']:
        start, end = np.asarray(side['points_overview'], float)
        length = float(np.linalg.norm(end-start))
        width = float(side['stroke_width_overview_px'])
        error = float(side['position_uncertainty_overview_px'])
        if not np.isfinite([length, width, error]).all() or min(length, width) <= 0 or error < 0:
            raise ValueError('Invalid frame side measurement')
        direction = (end-start)/length
        yield start, direction, np.array([-direction[1], direction[0]]), length, width/2+error+math.sqrt(.5)


def frame_observations(gray, detector):
    """Expose side and ruling fragments even if no complete GRID group exists."""
    checked_gray(gray)
    if (detector['source_sha256'] != sha(gray) or detector['size_px'] != list(gray.shape[::-1])
            or any(frame.get('frame_semantics_confirmed') is not False for frame in detector['frames'])):
        raise ValueError('Bound automatic frame detections required')
    audit = {'limits': [], 'source_points': 0, 'fragments': 0, 'side_checks': 0}
    output = []
    if not detector['complete']:
        audit['limits'].append('frame_search_incomplete')
    elif gray.size > POLICY['maximum_pixels'] or len(detector['frames']) > POLICY['maximum_frames']:
        audit['limits'].append('frame_observation_budget')
    if detector['frames'] and not audit['limits']:
        ink = foreground(gray)
        distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
        paths, topology = graph_paths(skeletonize(ink > 0))
        audit['skeleton'] = topology
        if topology['search_incomplete']:
            audit['limits'].append('source_skeleton_budget')
        frames = [(np.asarray(f['quad_overview'], np.float32), list(frame_sides(f))) for f in detector['frames']]
        cosine = math.cos(math.radians(POLICY['axis_angle_deg']))
        for fragment in source_fragments(paths, distance, audit) if not audit['limits'] else []:
            q = fragment['points']
            for frame_index, (quad, sides) in enumerate(frames):
                audit['side_checks'] += len(sides)
                if audit['side_checks'] > POLICY['maximum_side_checks']:
                    audit['limits'].append('frame_side_check_budget')
                    break
                matches = []
                for index, (start, unit, normal, length, radius) in enumerate(sides):
                    if abs(fragment['direction'] @ unit) < cosine:
                        continue
                    along = (q-start) @ unit
                    if np.min(along) >= -radius and np.max(along) <= length+radius:
                        offsets = abs((q-start) @ normal)
                        matches.append((float(np.mean(offsets)), index, bool(offsets.max() <= radius)))
                if not matches:
                    continue
                on_side = [entry for entry in matches if entry[2]]
                if not on_side:
                    vectors = np.roll(quad, -1, axis=0) - quad
                    offsets = q[:, None, :] - quad[None, :, :]
                    crosses = vectors[None, :, 0] * offsets[:, :, 1] - vectors[None, :, 1] * offsets[:, :, 0]
                    if not (np.all(crosses >= 0) or np.all(crosses <= 0)):
                        continue
                _, side, is_side = min(on_side or matches)
                output.append({'frame_index': frame_index, 'side_index': side,
                               'observed_kind': 'side_supported' if is_side else 'interior_parallel',
                               'skeleton_path_number': fragment['path_number'],
                               'source_index_interval': fragment['source_index_interval'],
                               'points_px': q.tolist(), 'pen_px': fragment['pen_px'],
                               'p95_error_px': fragment['p95_error_px']})
            if audit['limits']:
                break
    return {'source_sha256': sha(gray), 'detector_sha256': object_sha(detector),
            'complete': not audit['limits'], 'observations': output if not audit['limits'] else [],
            'audit': audit, 'scope': 'observed_source_fragments_only', 'semantic_frame_or_grid_confirmed': False}
