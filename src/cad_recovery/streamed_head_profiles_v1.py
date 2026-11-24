"""Exact source profiles with certified transverse and longitudinal stopping.

The fitter/axis-recovery thresholds remain the frozen v2 rules. A row is complete
only after unseen runs cannot beat its nearest span (or enter the allowed center
band). A body stops only after a measured false band beyond every legal start.
"""
import math
import numpy as np

from .local_head_axis_profiles_v1 import Profiles as OriginalProfiles
from .local_head_axis_profiles_v2 import prepare
from .local_head_axis_profiles_v2 import finish as original_finish
from .sparse_local_heads_v1 import runs


class Profiles(OriginalProfiles):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statistics = {'dense_row_equivalent_samples': 0, 'rows_certified': 0,
            'rows_certified_zero': 0, 'rows_full_width': 0, 'body_prefix_stops': 0,
            'body_positive_rows_unneeded': 0}

    def entry(self, base, unit, length, width):
        key = tuple(np.r_[base, unit, length, width])
        if key not in self.cache:
            along = np.arange(-min(40., max(14., length)), min(192., max(8., 2.5*length))+.25, .5)
            across = np.arange(-min(96., width+3), min(96., width+3)+.25, .5)
            self.cache[key] = {'along': along, 'across': across, 'widths': np.zeros(len(along)),
                'centers': np.full(len(along), np.inf), 'observed': np.zeros(len(along), bool)}
            if len(self.cache) > 4096: self.cache.popitem(last=False); self.evictions += 1
        self.cache.move_to_end(key)
        return self.cache[key]

    def read_points(self, base, unit, entry, rows, columns, stage):
        cost = int(len(rows)*len(columns))
        if not self.charge(cost, stage): return None
        normal = np.array([-unit[1], unit[0]])
        points = (base+entry['along'][rows, None, None]*unit
                  +entry['across'][None, columns, None]*normal)
        q = np.rint(points).astype(int)
        inside = (q >= 0).all(2) & (q[:, :, 0] < self.mask.shape[1]) & (q[:, :, 1] < self.mask.shape[0])
        ink = np.zeros(inside.shape, bool)
        ink[inside] = self.mask[q[:, :, 1][inside], q[:, :, 0][inside]] > 0
        # OpenCV 5 float32 remap changes its final ULPs with SIMD/scalar layout.
        # Use the same float32 map coordinates with explicit float64 bilinear
        # interpolation in both modes, independent of query packing.
        mapped = points.astype(np.float32).astype(np.float64)
        origin = np.floor(mapped).astype(int); fraction = mapped-origin
        coverage = np.zeros(points.shape[:2], np.float64)
        for dx, dy in ((0,0),(1,0),(0,1),(1,1)):
            p = origin+[dx,dy]
            valid = (p >= 0).all(2) & (p[:,:,0] < self.mask.shape[1]) & (p[:,:,1] < self.mask.shape[0])
            wx = fraction[:,:,0] if dx else 1-fraction[:,:,0]
            wy = fraction[:,:,1] if dy else 1-fraction[:,:,1]
            values = np.zeros(coverage.shape, np.float64)
            values[valid] = self.field[p[:,:,1][valid],p[:,:,0][valid]]
            coverage += values*wx*wy
        return ink, inside, coverage

    @staticmethod
    def certificate(ink, across, left, right, tolerance):
        closed = []; lower_bounds = []
        for a, b in runs(ink[left:right]):
            a, b = int(a)+left, int(b)+left
            open_left = a == left and left > 0
            open_right = b == right and right < len(across)
            if not open_left and not open_right:
                closed.append((abs((across[a]+across[b-1])/2), a, b))
            elif open_left and open_right: lower_bounds.append(0.)
            elif open_left:
                upper = (across[left]+across[b-1])/2
                lower_bounds.append(max(0., -upper))
            else:
                lower = (across[a]+across[right-1])/2
                lower_bounds.append(max(0., lower))
        best = min(closed, default=None)
        valid = best is not None and best[0] <= tolerance
        threshold = best[0] if valid else tolerance
        # Strict bounds preserve the original left-to-right tie preference.
        if any(bound <= threshold for bound in lower_bounds): return 'expand', None
        if left > 0 and across[left-1] >= -threshold: return 'expand', None
        if right < len(across) and across[right] <= threshold: return 'expand', None
        return ('valid', best[1:]) if valid else ('zero', None)

    def integrate(self, entry, index, ink, inside, coverage, span, width):
        if span is not None:
            a, b = span; across = entry['across']
            center = (across[a]+across[b-1])/2
            if abs(center) <= max(2., width*.25) and inside[a:b].all():
                lo, hi = max(0, a-2), min(len(across), b+2)
                observed = coverage[lo:hi]; mass = math.fsum(observed)
                if mass > 0:
                    entry['widths'][index] = mass*.5
                    entry['centers'][index] = math.fsum(float(v)*float(x) for v,x in zip(observed,across[lo:hi]))/mass
        entry['observed'][index] = True
        self.statistics['rows_certified'] += 1
        if not entry['widths'][index]: self.statistics['rows_certified_zero'] += 1

    def measure_rows(self, base, unit, width, entry, indices, stage):
        indices = np.asarray(indices, int)
        self.hits += int(np.count_nonzero(entry['observed'][indices]))
        missing = indices[~entry['observed'][indices]]
        if not len(missing): return True
        across = entry['across']; count = len(across)
        self.statistics['dense_row_equivalent_samples'] += int(len(missing)*count)
        if self.eager:
            value = self.read_points(base, unit, entry, missing, np.arange(count), stage)
            if value is None: return False
            ink, inside, coverage = value
            for row, index in enumerate(missing):
                spans = list(runs(ink[row]))
                span = min(spans, key=lambda ab: abs((across[ab[0]]+across[ab[1]-1])/2)) if spans else None
                self.integrate(entry, index, ink[row], inside[row], coverage[row], span, width)
            self.statistics['rows_full_width'] += len(missing)
            return True
        tolerance = max(2., width*.25)
        columns = np.flatnonzero(np.abs(across) <= tolerance+1.)
        left, right = int(columns[0]), int(columns[-1])+1
        value = self.read_points(base, unit, entry, missing, columns, stage)
        if value is None: return False
        initial_ink, initial_inside, initial_coverage = value
        for row, index in enumerate(missing):
            ink = np.zeros(count, bool); inside = np.zeros(count, bool)
            coverage = np.zeros(count, np.float64); seen = np.zeros(count, bool)
            ink[columns] = initial_ink[row]; inside[columns] = initial_inside[row]
            coverage[columns] = initial_coverage[row]; seen[columns] = True
            lo, hi = left, right
            while True:
                status, span = self.certificate(ink, across, lo, hi, tolerance)
                if status != 'expand': break
                spread = hi-lo
                new_lo, new_hi = max(0, lo-spread), min(count, hi+spread)
                extra = np.r_[np.arange(new_lo, lo), np.arange(hi, new_hi)]
                value = self.read_points(base, unit, entry, [index], extra, stage)
                if value is None: return False
                ink[extra], inside[extra], coverage[extra] = (v[0] for v in value)
                seen[extra] = True; lo, hi = new_lo, new_hi
            if span is not None:
                a, b = span; padding = np.arange(max(0, a-2), min(count, b+2))
                extra = padding[~seen[padding]]
                if len(extra):
                    value = self.read_points(base, unit, entry, [index], extra, stage)
                    if value is None: return False
                    ink[extra], inside[extra], coverage[extra] = (v[0] for v in value)
                    seen[extra] = True
            self.integrate(entry, index, ink, inside, coverage, span, width)
            self.statistics['rows_full_width'] += int(seen.all())
        return True

    def measure(self, base, unit, length, width, stage):
        entry = self.entry(base, unit, length, width); along = entry['along']
        if self.eager:
            indices = np.arange(len(along))
        elif stage == 'body' and 'body_shaft_width' in entry:
            local_width = entry['body_shaft_width']; threshold = max(3., 2*local_width)
            positive = np.flatnonzero(along >= 0)
            for start in range(0, len(positive), 16):
                block = positive[start:start+16]
                if not self.measure_rows(base, unit, width, entry, block, stage): return None
                prefix = positive[:start+len(block)]
                band = ((entry['widths'][prefix] >= max(2., local_width*1.25))
                        & (entry['widths'][prefix] <= width*1.5))
                closed = (along[prefix] > threshold) & ~band
                if closed.any():
                    stop = int(prefix[np.flatnonzero(closed)[0]])
                    if entry.get('body_certificate_index') != stop:
                        self.statistics['body_prefix_stops'] += 1
                        self.statistics['body_positive_rows_unneeded'] += int(np.count_nonzero(
                            (along > along[stop]) & ~entry['observed']))
                    entry['body_certificate_index'] = stop
                    break
            return entry['along'], entry['widths'], entry['centers'], entry['observed']
        elif stage == 'body': indices = np.arange(len(along))
        elif stage == 'tail': indices = np.flatnonzero(along <= 0)
        else:
            indices = np.array(sorted({int(np.argmin(abs(along-p))) for p in
                length*np.array([0, .25, .5, .75, 1., 1.5, 2.])}))
        if not self.measure_rows(base, unit, width, entry, indices, stage): return None
        return entry['along'], entry['widths'], entry['centers'], entry['observed']


def finish(prepared, profiles, text_polygons):
    base, unit, length, width, shaft = prepared
    profiles.entry(base, unit, length, width)['body_shaft_width'] = shaft[2]
    return original_finish(prepared, profiles, text_polygons)
