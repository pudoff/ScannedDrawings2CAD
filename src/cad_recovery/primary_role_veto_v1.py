"""Resolve a weak grid hint using original stroke width on a closed body."""

from collections import defaultdict

import cv2
import numpy as np

from .contour_role_competition_v2 import guard as baseline_guard
from .competitive_grid_witness_v2 import POLICY as GRID_POLICY


WIDTH_RATIO = GRID_POLICY['enclosing_width_ratio']
GRID_HINTS = ({'title_grid'}, {'part_contour', 'title_grid'})


class PenBudgetExceeded(ValueError):
    pass


def measure_pen(points, working_to_source, source_size, read_gray):
    """Measure one isolated stroke across the interior of a straight interval."""
    points = np.asarray(points, dtype=float)
    delta = points[-1] - points[0]
    length = np.linalg.norm(delta)
    if length < 8 or len(points) < 9:
        return None
    normal = np.array([-delta[1], delta[0]]) / length
    if np.max(np.abs((points - points[0]) @ normal)) > 1:
        return None
    scale = float(np.linalg.norm(working_to_source[:2, :2] @ normal))
    if not 3 <= scale <= 64:
        return None
    transformed = points @ working_to_source[:2, :2].T + working_to_source[:2, 2]
    tangent = transformed[-1] - transformed[0]
    perpendicular = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
    start, stop = len(points)//5, len(points) - len(points)//5 - 1
    samples = transformed[np.rint(np.linspace(start, stop, min(24, stop-start+1))).astype(int)]
    radius = round(1.5*scale)
    rays = samples[:, None, :] + np.arange(-radius, radius+1)[None, :, None]*perpendicular
    lo = np.floor(rays.min(axis=(0, 1))).astype(int)
    hi = np.ceil(rays.max(axis=(0, 1))).astype(int) + 1
    if (lo < 0).any() or (hi > source_size).any():
        return None
    image = read_gray([*lo.tolist(), *hi.tolist()])
    offsets = (rays - lo).astype(np.float32)
    pixels = cv2.remap(image, offsets[:, :, 0], offsets[:, :, 1], cv2.INTER_LINEAR).astype(float)
    paper = np.minimum(pixels[:, :3].max(1), pixels[:, -3:].max(1))
    density = np.maximum(paper[:, None] - pixels, 0.)
    contrast = density.max(1)
    strong = density > contrast[:, None]/2
    runs = np.count_nonzero(strong[:, 1:] & ~strong[:, :-1], axis=1)
    valid = (contrast >= 64) & ~strong[:, 0] & ~strong[:, -1] & (runs == 1)
    if valid.sum() < max(5, .75*len(samples)):
        return None
    widths = (density[valid]/contrast[valid, None]).sum(1)
    lower, median, upper = np.quantile(widths, [.25, .5, .75])
    return float(median) if upper <= WIDTH_RATIO*lower else None


def resolve_owners(graph, context, primary, owner_numbers, measure):
    """Require a body boundary, a pen contrast and a connected thick turn."""
    if not all((primary['complete'], context['complete'], context['competition']['complete'])):
        return {}
    items = graph['proposals']
    numbers = {item['id']: pos for pos, item in enumerate(items, 1)}
    tracks = {track['id']: track for track in graph['tracks']}
    protected = set(context['frame_conflict_owner_numbers']) | set(context['other_conflict_owner_numbers'])
    for network in context['competition']['networks']:
        if network['qualified']:
            protected.update(network['source_owner_numbers'])
    weak = set()
    for line in context['context']['lines']:
        hints = set(line['alternatives'])
        if 'title_grid' not in hints:
            continue
        members = {numbers[identity] for identity in tracks[line['id']]['interval_ids']}
        if hints in GRID_HINTS and line['role'] in ('unknown', 'title_grid') and not line['evidence']['head_ids']:
            weak.update(members)
        else:
            protected.update(members)
    eligible = (weak & owner_numbers) - protected
    relaxed = {}
    widths = {}
    for decision in primary['decisions']:
        if not decision['positive_body'] or not decision['holes']:
            continue
        owners = set(decision['source_owner_numbers'])
        if not eligible & owners:
            continue
        # Measure every usable line in the same observed outline. The reference
        # derives from that outline and adapts to the original scan resolution.
        strokes = {}
        by_endpoint = defaultdict(list)
        for number in sorted(owners - protected):
            item = items[number-1]
            if item.get('parent_kind') in ('arc', 'circle'):
                continue
            points = item['points_px']
            if number not in widths:
                widths[number] = measure(points)
            if widths[number] is None:
                continue
            direction = np.asarray(points[-1]) - points[0]
            length = np.linalg.norm(direction)
            if length == 0:
                continue
            strokes[number] = (widths[number], direction/length, points)
            for endpoint in (points[0], points[-1]):
                by_endpoint[tuple(endpoint)].append(number)
        if len(strokes) < 3:
            continue
        reference = float(np.median([value[0] for value in strokes.values()]))
        for number in sorted(eligible & strokes.keys()):
            width, direction, points = strokes[number]
            if width < WIDTH_RATIO*reference:
                continue
            for neighbor in sorted(set(by_endpoint[tuple(points[0])] + by_endpoint[tuple(points[-1])]) - {number}):
                other_width, other_direction, _ = strokes[neighbor]
                if (other_width >= WIDTH_RATIO*reference and
                        max(width, other_width) <= WIDTH_RATIO*min(width, other_width) and
                        abs(direction @ other_direction) <= .5):
                    relaxed[number] = {'boundary': decision['boundary_id'], 'neighbor': neighbor,
                                       'pen_px': width, 'neighbor_pen_px': other_width, 'reference_pen_px': reference}
                    break
    return relaxed


def guard(gray, graph, source_owner, ambiguous_owner, proposed, context, *, primary, measure, positive=None):
    baseline = baseline_guard(gray, graph, source_owner, ambiguous_owner, proposed, context, positive=positive)
    if (positive is not None or not baseline['complete'] or not proposed.any() or
            np.any((proposed > 0) & (primary['mask'] == 0))):
        return baseline
    present = set(map(int, np.unique(source_owner[(proposed > 0) & (baseline['mask'] == 0)])))
    try:
        relaxed = resolve_owners(graph, context, primary, present, measure)
    except PenBudgetExceeded:
        return {**baseline, 'primary_role_resolution': {'reason': 'native_pen_budget'}}
    if not relaxed:
        return baseline
    relaxed_ids = {graph['proposals'][i-1]['id'] for i in relaxed}
    weak_track_ids = {line['id'] for line in context['context']['lines']
                      if set(line['alternatives']) in GRID_HINTS}
    # Restrict only the weak grid track used by the negative guard. Recompute
    # proximity vetoes from all remaining owners and restore only eligible ones.
    tracks = [{**track, 'interval_ids': [i for i in track['interval_ids'] if i not in relaxed_ids]}
              if track['id'] in weak_track_ids else track for track in graph['tracks']]
    amended = baseline_guard(gray, {**graph, 'tracks': tracks}, source_owner, ambiguous_owner, proposed, context)
    restored = (amended['mask'] > 0) & (baseline['mask'] == 0) & np.isin(source_owner, list(relaxed))
    mask = baseline['mask'].copy()
    mask[restored] = 255
    restored_count = int(restored.sum())
    return {**baseline, 'mask': mask, 'primary_role_resolution': {'owners': relaxed,
            'restored_pixels': restored_count, 'baseline_direct_veto_pixels': baseline['direct_veto_pixels'],
            'baseline_contact_veto_pixels': baseline['contact_veto_pixels']},
            'direct_veto_pixels': baseline['direct_veto_pixels'] - restored_count,
            'withdrawn_pixels': int(np.count_nonzero(proposed) - np.count_nonzero(mask))}
