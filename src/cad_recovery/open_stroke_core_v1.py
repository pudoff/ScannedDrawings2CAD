"""Use continuous thick foreground to corroborate observed open contour branches."""
import math

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .cad_object_assembly_v5 import assemble
from .curve_graph_v1 import graph_paths
from .grid_role_revision_v1 import check_authority
from .local_contour_probes_v1 import probes
from .local_contour_profiles_v2 import junction_distance, measure
from .local_contour_witness_v3 import inventory
from .stroke_roles_v1 import foreground

MAX_PIXELS = 4_000_000
MAX_PROFILE_SAMPLES = 2_000_000
MAX_CORE_LEVELS = 8
MAX_COMPARISONS = 250_000
PARALLEL_COSINE = math.cos(math.radians(8))


def measured_lines(packet, blocked, ink, distance, audit):
    geometry = packet['geometry']
    assembled = assemble(packet['gray'], geometry, roles=packet['roles'], protected=blocked)
    objects, search = probes(assembled, geometry)
    if not assembled['search_complete'] or not search['complete']:
        audit['limits'].append('observed_line_search')
        return []
    junctions = junction_distance(geometry, ink.shape)
    work = {'ink': ink, 'distance': distance, 'samples': 0}
    lines = []
    for obj in objects:
        if obj['kind'] != 'line':
            continue
        ends = np.asarray(obj['points_px'], float)
        support = np.asarray(obj['support_points'], int)
        length = float(np.linalg.norm(ends[1] - ends[0]))
        profile = measure(ends, support, packet['gray'], work, junctions, maximum_samples=MAX_PROFILE_SAMPLES)
        if work['samples'] > MAX_PROFILE_SAMPLES:
            audit['limits'].append('profile_budget')
            break
        if profile is None:
            continue
        width, upper = np.quantile(profile['sections_px'], [.25, .95])
        if width < 1 or upper > 2 * width or length < 6 * width:
            continue
        x, y = support.T
        lines.append({'object': obj, 'ends': ends, 'support': support, 'width': float(width),
            'upper': float(upper), 'direction': (ends[1] - ends[0]) / length,
            'clear': bool(np.all(packet['roles'][y, x] == 255) and not blocked[y, x].any()),
            'sources': {m['source_observation_id'] for m in obj['source_members']}})
    audit['profile_samples'] = work['samples']
    return lines


def corner_arms(path, width):
    points = path['points']
    if path['closed'] or len(points) < 12 * width:
        return
    vertices = cv2.approxPolyDP(points.astype(np.float32), max(1.25, .37 * width), False).reshape(-1, 2)
    indices = [0]
    for vertex in vertices[1:-1]:
        index = np.flatnonzero(np.all(points[indices[-1] + 1:] == vertex, axis=1))
        if index.size == 0:
            return
        indices.append(int(index[0]) + indices[-1] + 1)
    indices.append(len(points) - 1)
    arms = []
    for start, end in zip(indices, indices[1:]):
        delta = points[end] - points[start]
        length = float(np.linalg.norm(delta))
        if length >= 6 * width:
            arms.append((start, end, delta / length, length))
    for left, right in zip(arms, arms[1:]):
        a, b, u, length = left
        c, d, v, other_length = right
        walk = points[b:c+1]
        if np.linalg.norm(np.diff(walk, axis=0), axis=1).sum() > 2 * width:
            continue
        angle = math.degrees(math.acos(float(np.clip(abs(u @ v), 0, 1))))
        if angle < 20 or max(length, other_length) < 12 * width:
            continue
        yield (points[a:b+1], u), (points[c:d+1], v), points[a:d+1], angle


def nearby_references(lines, line, route):
    margin = float(np.linalg.norm(np.ptp(route, axis=0))) / 2
    lo, hi = route.min(0) - margin, route.max(0) + margin
    references = []
    used = set(line['sources'])
    for ref in lines:
        center = ref['ends'].mean(0)
        if ref['sources'] & used or ref['upper'] * 1.5 > line['width']:
            continue
        if np.all(center >= lo) and np.all(center <= hi):
            used.update(ref['sources'])
            references.append(ref['object']['id'])
    return references


def matches_arm(line, arm, direction):
    if abs(float(line['direction'] @ direction)) < PARALLEL_COSINE:
        return False
    offsets = line['support'] - arm[0]
    along = offsets @ direction
    normal = np.array([-direction[1], direction[0]])
    # Most original source samples must agree with the observed core arm.
    span = float(np.linalg.norm(arm[-1] - arm[0]))
    overlap = (along >= -line['width']) & (along <= span + line['width'])
    close = np.abs(offsets @ normal) <= max(1.25, .30 * line['width'])
    return bool(np.mean(overlap & close) >= .90)


def arm_radii(arm, distance):
    # Measure the shaft away from endpoint caps and the bend itself.
    margin = max(1, len(arm) // 6)
    x, y = arm[margin:-margin].astype(int).T
    return np.quantile(distance[y, x], [.10, .50, .90])


def analyze(packet, *, authority, protected):
    gray, roles = packet['gray'], packet['roles']
    audit = {'limits': [], 'comparisons': 0, 'core_levels': [], 'witnesses': []}
    mask = np.zeros_like(gray)
    if gray.size > MAX_PIXELS:
        audit['limits'].append('raster_budget')
        return mask, audit
    inventory(packet)
    allowed = check_authority(gray, roles, authority)
    if packet.get('fresh_source_extraction_complete', True) is not True:
        audit['limits'].append('source_extraction_incomplete')
        return mask, audit
    blocked = protected | ((roles != 255) & (roles != 1))
    for head in packet['heads']:
        polygon = np.asarray(head['polygon_px'], float)
        head_mask = np.zeros_like(gray)
        cv2.fillPoly(head_mask, [np.rint(polygon).astype(np.int32)], 255)
        blocked |= head_mask > 0
    ink = foreground(gray)
    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    lines = measured_lines(packet, blocked, ink, distance, audit)
    minimum = min((line['upper'] for line in lines), default=math.inf)
    thick = [line for line in lines if line['clear'] and line['width'] >= 1.5 * minimum]
    levels = sorted({max(2, math.floor(line['width'] / 3)) for line in thick})
    if len(levels) > MAX_CORE_LEVELS:
        audit['limits'].append('core_level_budget')
    selected = {}
    for depth in levels if not audit['limits'] else []:
        paths, stats = graph_paths(skeletonize((distance >= depth) & ~blocked))
        audit['core_levels'].append({'depth_px': depth, 'topology': stats})
        if stats['search_incomplete']:
            audit['limits'].append('core_graph_budget')
            break
        for line in thick:
            if max(2, math.floor(line['width'] / 3)) != depth or line['object']['id'] in selected:
                continue
            width = line['width']
            for path in paths:
                audit['comparisons'] += 1
                if audit['comparisons'] > MAX_COMPARISONS:
                    audit['limits'].append('core_match_budget')
                    break
                if min(path['start_degree'], path['end_degree']) != 1:
                    continue
                for first, second, route, angle in corner_arms(path, width):
                    # A cropped endpoint is not positive evidence of termination.
                    if np.any(route.min(0) <= width) or np.any(route.max(0) >= np.array(gray.shape[::-1]) - 1 - width):
                        continue
                    if not any(matches_arm(line, points, direction) for points, direction in (first, second)):
                        continue
                    radii = np.array([arm_radii(arm, distance) for arm, _ in (first, second)])
                    if np.any(radii[:, 2] > 1.67 * radii[:, 0]) or max(radii[:, 1]) > 1.5 * min(radii[:, 1]):
                        continue
                    refs = nearby_references(lines, line, route)
                    if len(refs) < 3:
                        continue
                    obj = line['object']
                    selected[obj['id']] = line
                    audit['witnesses'].append({'source_object_id': obj['id'], 'source_members': obj['source_members'],
                        'core_depth_px': depth, 'route_px': route.astype(int).tolist(), 'corner_angle_deg': angle,
                        'source_pen_px': width, 'core_arm_radius_quantiles_px': radii.tolist(), 'reference_object_ids': refs})
                    break
                if line['object']['id'] in selected:
                    break
            if audit['limits']:
                break
        if audit['limits']:
            break
    if audit['limits']:
        return mask, audit
    owner = packet['source_owner']
    proposals = packet['graph']['proposals']
    eligible = allowed & (roles == 255) & ~blocked & (owner > 0) & (packet['ambiguous_owner_mask'] == 0) & (ink > 0)
    raster_pixels = 0
    for line in selected.values():
        support = line['support']
        radius = line['width'] / 2 + math.sqrt(.5)
        pad = math.ceil(radius) + 1
        lo = np.maximum(0, support.min(0) - pad)
        hi = np.minimum(gray.shape[::-1], support.max(0) + pad + 1)
        raster_pixels += int(np.prod(hi - lo))
        if raster_pixels > 64_000_000:
            audit['limits'].append('footprint_budget')
            mask.fill(0)
            break
        numbers = set()
        for member in line['object']['source_members']:
            a, b = member['source_index_interval']
            numbers.update(n for n, p in enumerate(proposals, 1) if p['parent_primitive_id'] == member['candidate_id']
                           and max(a, p['source_index_interval'][0]) < min(b, p['source_index_interval'][1]))
        region = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
        seed = np.full(tuple((hi - lo)[::-1]), 255, np.uint8)
        x, y = (support - lo).T
        seed[y, x] = 0
        footprint = cv2.distanceTransform(seed, cv2.DIST_L2, 5) <= radius
        mask[region][eligible[region] & footprint & np.isin(owner[region], list(numbers))] = 255
    return mask, audit
