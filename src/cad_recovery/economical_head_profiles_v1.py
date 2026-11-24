"""Progressive source profiles: coarse taper, negative shaft, then the head body."""
from collections import OrderedDict
import math

import cv2
import numpy as np

from .sparse_local_heads_v1 import runs, shaft_run
from .local_width_heads_v1 import append_preserving
from .arrow_path_recovery_v1 import native_head_check
from .native_head_bounds_v1 import safe_validator


class Profiles:
    def __init__(self, mask, *, maximum_samples=8000000, eager=False):
        self.mask = mask; self.field = mask.astype(np.float32)/255
        self.maximum_samples = maximum_samples; self.eager = eager
        self.samples = self.hits = self.evictions = 0; self.incomplete = False
        self.cache = OrderedDict(); self.stage_samples = {'cheap': 0, 'coarse': 0, 'tail': 0, 'body': 0}

    def charge(self, count, stage):
        if self.samples+count > self.maximum_samples:
            self.incomplete = True; return False
        self.samples += count; self.stage_samples[stage] += count; return True

    def measure(self, base, unit, length, width, stage):
        key = tuple(np.r_[base, unit, length, width])
        if key in self.cache:
            entry = self.cache[key]; self.cache.move_to_end(key)
        else:
            along = np.arange(-min(40., max(14., length)), min(192., max(8., 2.5*length))+.25, .5)
            across = np.arange(-min(96., width+3), min(96., width+3)+.25, .5)
            entry = {'along': along, 'across': across, 'widths': np.zeros(len(along)),
                'centers': np.full(len(along), np.inf), 'observed': np.zeros(len(along), bool)}
            self.cache[key] = entry
            if len(self.cache) > 4096: self.cache.popitem(last=False); self.evictions += 1
        along, across = entry['along'], entry['across']
        if self.eager or stage == 'body': selected = np.ones(len(along), bool)
        elif stage == 'tail': selected = along <= 0
        else:
            selected = np.zeros(len(along), bool)
            positions = length*np.array([0, .25, .5, .75, 1., 1.5, 2.])
            for position in positions: selected[int(np.argmin(abs(along-position)))] = True
        missing = np.flatnonzero(selected & ~entry['observed'])
        self.hits += int(np.count_nonzero(selected & entry['observed']))
        cost = int(len(missing)*len(across))
        if not self.charge(cost, stage): return None
        if len(missing):
            normal = np.array([-unit[1], unit[0]])
            points = base+along[missing, None, None]*unit+across[None, :, None]*normal
            q = np.rint(points).astype(int)
            inside = (q >= 0).all(2) & (q[:, :, 0] < self.mask.shape[1]) & (q[:, :, 1] < self.mask.shape[0])
            ink = np.zeros(inside.shape, bool); ink[inside] = self.mask[q[:, :, 1][inside], q[:, :, 0][inside]] > 0
            coverage = cv2.remap(self.field, points[:, :, 0].astype(np.float32), points[:, :, 1].astype(np.float32),
                cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            for row, index in enumerate(missing):
                spans = list(runs(ink[row]))
                if not spans: continue
                a, b = min(spans, key=lambda s: abs((across[s[0]]+across[s[1]-1])/2))
                center = (across[a]+across[b-1])/2
                if abs(center) > max(2., width*.25) or not inside[row, a:b].all(): continue
                lo, hi = max(0, a-2), min(len(across), b+2); observed = coverage[row, lo:hi]
                mass = float(observed.sum())
                if mass > 0:
                    entry['widths'][index] = mass*.5
                    entry['centers'][index] = float(observed@across[lo:hi]/mass)
            entry['observed'][missing] = True
        return entry['along'], entry['widths'], entry['centers'], entry['observed']


def prepare(base, unit, length, width, profiles):
    normal = np.array([-unit[1], unit[0]])
    probe = base+np.array([-.5, -2., -4., -6.])[:, None, None]*unit+np.array([-1., 0., 1.])[None, :, None]*normal
    if not profiles.charge(12, 'cheap'): return None, 'profile_sample_budget'
    q = np.rint(probe).astype(int)
    inside = (q >= 0).all(2) & (q[:, :, 0] < profiles.mask.shape[1]) & (q[:, :, 1] < profiles.mask.shape[0])
    ink = np.zeros(inside.shape, bool); ink[inside] = profiles.mask[q[:, :, 1][inside], q[:, :, 0][inside]] > 0
    if np.mean(ink.any(1)) < .75: return None, 'no_near_base_source_run'
    value = profiles.measure(base, unit, length, width, 'coarse')
    if value is None: return None, 'profile_sample_budget'
    along, widths, centers, observed = value
    indices = sorted({int(np.argmin(abs(along-position))) for position in length*np.array([0, .25, .5, .75, 1., 1.5, 2.])})
    support = widths[indices]
    taper_possible = any(a >= b-.25 and b >= c-.25 and c > 1 and a-c >= max(.5, .10*a)
                         for a, b, c in zip(support, support[1:], support[2:]))
    if not taper_possible: return None, 'no_coarse_source_taper'
    value = profiles.measure(base, unit, length, width, 'tail')
    if value is None: return None, 'profile_sample_budget'
    along, widths, centers, _ = value; shaft = shaft_run(along, widths, width)
    if shaft is None: return None, 'no_connected_source_shaft'
    a, b, local_width, extent = shaft
    slope, intercept = np.polyfit(along[a:b], centers[a:b], 1)
    if abs(slope) <= math.tan(math.radians(20)) and abs(intercept) <= max(2., local_width):
        adjusted = unit+slope*normal; adjusted /= np.linalg.norm(adjusted)
        base = base+intercept*normal; unit = adjusted
        value = profiles.measure(base, unit, length, width, 'tail')
        if value is None: return None, 'profile_sample_budget'
        along, widths, centers, _ = value; shaft = shaft_run(along, widths, width)
        if shaft is None: return None, 'refined_axis_has_no_connected_shaft'
    return (base, unit, length, width, shaft), None


def finish(prepared, profiles, text_polygons):
    base, unit, length, width, shaft = prepared
    value = profiles.measure(base, unit, length, width, 'body')
    if value is None: return None, 'profile_sample_budget'
    along, widths, _, _ = value; a, b, local_width, extent = shaft
    band = (along >= 0) & (widths >= max(2., local_width*1.25)) & (widths <= width*1.5)
    candidates = []
    for start, end in runs(band):
        if along[start] > max(3., 2*local_width) or along[end-1]-along[start] < max(3., 1.5*local_width): continue
        peak = start+int(np.argmax(widths[start:end])); lo, hi = along[peak], along[end-1]
        if hi-lo < max(3., 1.5*local_width): continue
        chosen = (along >= lo+.15*(hi-lo)) & (along <= lo+.85*(hi-lo))
        if chosen.sum() < 4: continue
        geometric = np.maximum(0, widths[chosen]-1.)
        taper, intercept = np.polyfit(along[chosen], geometric, 1)
        if taper >= -.05: continue
        expected = taper*along[chosen]+intercept
        error = float(np.median(np.abs(geometric-expected)/np.maximum(local_width, expected)))
        root = float(-intercept/taper); base_width = float(intercept+taper*lo); head_length = root-lo
        # Both operands are integrated source-support widths. Geometry remains
        # in pixel-center coordinates for the unchanged native validator.
        support_ratio = (base_width+1.)/local_width
        if (error > .30 or support_ratio < 2.5 or not 5 <= head_length <= 192
            or root < hi-.5 or root > hi+max(3., 2*local_width) or not 1.2 <= head_length/base_width <= 8): continue
        normal = np.array([-unit[1], unit[0]]); head_base = base+lo*unit
        polygon = np.array([base+root*unit, head_base+base_width/2*normal, head_base-base_width/2*normal])
        head = {'tip_px': polygon[0].tolist(), 'direction': unit.tolist(), 'polygon_px': polygon.tolist(),
            'classification': 'filled_arrow_candidate', 'uncertainty_px': max(.75, local_width/2), 'confidence': .55,
            'evidence': {'method': 'economical-local-heads/1.0', 'path_kind': 'observed_taper_and_own_shaft',
                'local_shaft_width_px': local_width, 'shaft_observed_extent_px': extent,
                'base_to_shaft_ratio': float(support_ratio), 'contrast_domain': 'source_pixel_support_for_both_operands',
                'base_support_width_px': base_width+1., 'base_width_px': base_width, 'head_length_px': head_length,
                'taper_relative_error': error, 'profile_band_px': [float(lo), float(hi)], 'confidence_calibrated': False}}
        valid, check = safe_validator(native_head_check, [])(head, profiles.mask, local_width, text_polygons)
        if not valid: continue
        x, y = np.rint(polygon[0]).astype(int); radius = max(1, math.ceil(local_width))
        left, top = max(0, x-radius), max(0, y-radius)
        right, bottom = min(profiles.mask.shape[1], x+radius+1), min(profiles.mask.shape[0], y+radius+1)
        if right <= left or bottom <= top or not profiles.mask[top:bottom, left:right].any(): continue
        head['evidence'].update(native_head_check=check, source_checked=True); candidates.append(head)
    if not candidates: return None, 'no_source_validated_taper'
    combined, _ = append_preserving([], candidates)
    if len(combined) != 1: return None, 'ambiguous_source_taper'
    return candidates[0], None
