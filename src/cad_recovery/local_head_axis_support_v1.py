"""Source-only local head axes and bounded, independently observed tail paths.

No labels, morphology on a new image, or lowered shaft-length threshold. All
cross-section queries use the existing deterministic sampler and shared cap.
"""
import math

import numpy as np

from .sparse_local_heads_v1 import runs
from .streamed_head_profiles_v1 import finish


POLICY = {'version': 'local-head-axis-support/1.0', 'maximum_turn_deg': 30.,
          'maximum_axis_correction_deg': 20., 'maximum_path_px': 40.,
          'minimum_shaft_extent_px': 6., 'minimum_width_lengths': 2.,
          'minimum_path_continuity': .95, 'minimum_base_support_ratio': 2.5,
          'head_center_fit_relative_error_max': .35,
          'source_sample_budget_shared_with_original': True,
          'native_checks_unchanged': True, 'manual_references_used': False,
          'physical_geometry_approved': False, 'production_admitted': False}


def cross_section(point, direction, seed_width, profiles):
    radius = min(16., math.ceil(max(4., .75*seed_width+2)))
    across = np.arange(-radius, radius+.25, .5)
    entry = {'along': np.array([0.]), 'across': across}
    result = profiles.read_points(point, direction, entry, [0], np.arange(len(across)), 'tail')
    if result is None:
        return None, 'profile_sample_budget'
    ink, inside, coverage = (value[0] for value in result)
    spans = list(runs(ink))
    if not spans:
        return None, 'path_source_gap'
    a, b = min(spans, key=lambda ab: abs((across[ab[0]]+across[ab[1]-1])/2))
    if a == 0 or b == len(across) or not inside[a:b].all():
        return None, 'path_section_clipped'
    lo, hi = max(0, a-2), min(len(across), b+2)
    mass = math.fsum(coverage[lo:hi])*.5
    if mass <= 0 or mass > .8*seed_width:
        return None, 'path_not_thin_relative_to_head_seed'
    center = math.fsum(float(w)*float(x) for w, x in zip(coverage[lo:hi], across[lo:hi]))*.5/mass
    if abs(center) > max(2., mass):
        return None, 'path_center_jump'
    return (float(center), float(mass)), None


def follow_tail(base, unit, seed_width, profiles):
    points = [np.asarray(base, float).copy()]
    widths = []
    direction = -np.asarray(unit, float).copy()
    arc = 0.
    for _ in range(40):
        prediction = points[-1]+direction
        section, reason = cross_section(prediction, direction, seed_width, profiles)
        if section is None:
            if reason == 'profile_sample_budget':
                return None, reason
            break
        center, width = section
        normal = np.array([-direction[1], direction[0]])
        point = prediction+center*normal
        delta = point-points[-1]
        step = float(np.linalg.norm(delta))
        tangent = delta/step
        if float(tangent@direction) < math.cos(math.radians(POLICY['maximum_turn_deg'])):
            break
        if arc+step > POLICY['maximum_path_px']:
            break
        # Re-observe the actual segment after centering, not just its width.
        stations = np.linspace(0., 1., max(3, math.ceil(step/.5)+1))
        probe = points[-1]+stations[:, None]*delta
        if not profiles.charge(len(probe), 'tail'):
            return None, 'profile_sample_budget'
        q = np.rint(probe).astype(int)
        inside = ((q >= 0).all(1) & (q[:, 0] < profiles.mask.shape[1])
                  & (q[:, 1] < profiles.mask.shape[0]))
        ink = np.zeros(len(q), bool)
        ink[inside] = profiles.mask[q[:, 1][inside], q[:, 0][inside]] > 0
        if not inside.all() or np.mean(ink) < POLICY['minimum_path_continuity']:
            break
        if any(np.linalg.norm(point-old) < .5 for old in points[:-3]):
            break
        points.append(point)
        widths.append(width)
        arc += step
        direction = tangent
    if not widths:
        return None, 'no_own_source_path'
    width = float(np.median(widths))
    if arc < max(POLICY['minimum_shaft_extent_px'], POLICY['minimum_width_lengths']*width):
        return None, 'own_source_path_too_short'
    return {'points_px': np.asarray(points).tolist(), 'extent_px': arc,
            'width_px': width, 'widths_px': widths,
            'minimum_segment_continuity': POLICY['minimum_path_continuity'],
            'method': 'source_cross_sections_and_reobserved_centered_segments'}, None


def recover(base, unit, length, width, profiles, text_polygons):
    # Positive source sections propose a head axis, independent of tail bend.
    measured = profiles.measure(base, unit, length, width, 'coarse')
    if measured is None:
        return None, 'profile_sample_budget'
    along, widths, centers, observed = measured
    indices = sorted({int(np.argmin(abs(along-p))) for p in
                      length*np.array([0., .25, .5, .75, 1., 1.5, 2.])})
    support = widths[indices]
    if not any(a >= b-.25 and b >= c-.25 and c > 1 and a-c >= max(.5, .1*a)
               for a, b, c in zip(support, support[1:], support[2:])):
        return None, 'no_local_head_coarse_taper'
    entry = profiles.entry(base, unit, length, width)
    selected = np.flatnonzero((entry['along'] >= 0) & (entry['along'] <= min(192., 2*length)))
    if not profiles.measure_rows(base, unit, width, entry, selected, 'body'):
        return None, 'profile_sample_budget'
    along, widths, centers = (entry[k] for k in ('along', 'widths', 'centers'))
    near = selected[widths[selected] >= max(3., .5*width)]
    if len(near) < 6 or np.ptp(along[near]) < 3:
        return None, 'no_local_head_center_band'
    # A second separated broad shape is not one local head.
    spans = list(runs(np.isin(np.arange(len(along)), near)))
    if len(spans) != 1:
        return None, 'ambiguous_local_head_center_bands'
    a, b = spans[0]
    slope, offset = np.polyfit(along[a:b], centers[a:b], 1)
    error = float(np.median(abs(centers[a:b]-(slope*along[a:b]+offset))))
    if (abs(slope) > math.tan(math.radians(POLICY['maximum_axis_correction_deg']))
            or abs(offset) > max(3., .5*width)
            or error > max(.5, POLICY['head_center_fit_relative_error_max']*width)):
        return None, 'local_head_axis_unstable'
    normal = np.array([-unit[1], unit[0]])
    adjusted = unit+slope*normal
    adjusted /= np.linalg.norm(adjusted)
    origin = base+offset*normal
    path, reason = follow_tail(origin, adjusted, width, profiles)
    if path is None:
        return None, reason
    local_width, extent = path['width_px'], path['extent_px']
    prepared = (origin, adjusted, length, width, (0, 0, local_width, extent))
    head, reason = finish(prepared, profiles, text_polygons)
    if head is not None:
        head['evidence'].update(method=POLICY['version'], shaft_support_path=path,
            local_head_axis={'seed_direction': unit.tolist(), 'direction': adjusted.tolist(),
                             'center_fit_error_px': error, 'source_only': True},
            straight_shaft_extent_claimed=False)
    return head, reason
