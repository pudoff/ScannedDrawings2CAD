"""Cross-window joins require overlapping observations and native pixel support."""
import math
from unittest.mock import patch
import numpy as np
from . import native_merge_v1 as previous


def curve_supported(row, page):
    radius = row['radius_px']
    start = math.radians(row.get('start_angle_deg', 0))
    sweep = math.radians(row.get('sweep_angle_deg', 360))
    count = max(72, math.ceil(radius * sweep * 2))
    if count > 200000:
        return False
    angles = np.linspace(start, start + sweep, count)
    points = np.asarray(row['center_px']) + radius * np.column_stack([np.cos(angles), np.sin(angles)])
    # Every sampled location must have source ink within one pixel in each axis.
    near = np.full(count, 255, np.uint8)
    for x, y in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        near = np.minimum(near, page.samples(points + [x, y]))
    return bool(np.all(near < 240))


def union(a, b, page):
    if not set(b['tile_ids']) - set(a['tile_ids']):
        return None
    if a['kind'] == b['kind'] == 'line':
        result = previous._line_union(a, b)  # Positive overlap; never bridges a white gap.
    elif a['kind'] in ('arc', 'circle') and b['kind'] in ('arc', 'circle'):
        if math.dist(a['center_px'], b['center_px']) > 1.5 or abs(a['radius_px'] - b['radius_px']) > 1.5:
            return None
        a0 = a.get('start_angle_deg', 0) % 360
        a1 = a0 + a.get('sweep_angle_deg', 360)
        b0 = b.get('start_angle_deg', 0) % 360
        b1 = b0 + b.get('sweep_angle_deg', 360)
        choices = []
        for shift in [-360, 0, 360]:
            low, high = b0 + shift, b1 + shift
            if min(a1, high) - max(a0, low) > 1e-6:
                choices.append((min(a0, low), max(a1, high)))
        if not choices:
            return None
        low, high = max(choices, key=lambda p: p[1] - p[0])
        result = {**a, 'start_angle_deg': low % 360, 'sweep_angle_deg': min(high - low, 360)}
        if result['sweep_angle_deg'] >= 360 - 1e-7:
            result.update(kind='circle', start_angle_deg=0., sweep_angle_deg=360.)
        else:
            result['kind'] = 'arc'
        if not curve_supported(result, page):
            return None
    else:
        return None
    if result is not None:
        result = {**result, 'tile_ids': sorted(set(a['tile_ids']) | set(b['tile_ids'])),
                  'native_seam_evidence': 'overlapping same-type/source-supported observations; no extrapolated gap'}
    return result


def stitch(rows, page):
    # A dedicated conversion process uses the existing indexed merge; old code is immutable.
    aliases = {r['uid']: r['uid'] for r in rows}
    current = rows
    comparisons = 0
    passes = 0
    for passes in range(1, 9):
        with patch.object(previous, '_geometry_union', lambda a, b: union(a, b, page)):
            objects, links, count = previous.merge(current, 'geometry')
        comparisons += count
        aliases = {origin: links[identity] for origin, identity in aliases.items()}
        if len(objects) == len(current):
            break
        current = [{**r, 'uid': r['id']} for r in objects]
    else:
        raise ValueError('Seam closure exceeds8passes; partial candidates not admitted')
    origins = {}
    for origin, target in aliases.items(): origins.setdefault(target, []).append(origin)
    for row in objects: row['observation_uids'] = sorted(origins[row['id']])
    return objects, dict(aliases=aliases, comparisons=comparisons,
        passes=passes,
        joined_objects=sum(len(r['tile_ids']) > 1 for r in objects),
        unresolved_fragments=sum(r['kind'] == 'arc' for r in objects),
        whole_semantic_objects_accepted=False)
