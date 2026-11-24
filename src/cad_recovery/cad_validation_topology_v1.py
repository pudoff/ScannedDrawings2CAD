"""Bounded independent topology checks on native CAD segments.

Analytic signed areas preserve circular sectors. Flattening is used only for
conservative crossing/clearance tests, with its maximum error reported.
"""
import math

from .cad_validation_geometry_v1 import arc_data, finite, segments


class PredicateBudget:
    def __init__(self, maximum=250000):
        self.maximum, self.used = maximum, 0

    def spend(self):
        self.used += 1
        if self.used > self.maximum:
            raise ValueError('Shared topology predicate budget exceeded')


def oriented_segments(contour, geometries):
    result = []
    for edge in contour.edges:
        part = segments(geometries[edge.primitive_id])
        if edge.reversed:
            part = [(b, a, -bulge) for a, b, bulge in reversed(part)]
        result.extend(part)
        if len(result) > 512: raise ValueError('Contour segment budget exceeded')
    return result


def loop_data(parts, tolerance, *, closed=True):
    if not parts: raise ValueError('Empty contour')
    gaps = [math.dist(parts[i][1], parts[i+1][0]) for i in range(len(parts)-1)]
    if closed: gaps.append(math.dist(parts[-1][1], parts[0][0]))
    if max(gaps, default=0.) > tolerance: raise ValueError('Contour has an endpoint gap')
    area, flat, chord_error = 0., [], 0.
    for a, b, bulge in parts:
        if math.dist(a, b) <= tolerance: raise ValueError('Zero-length contour segment')
        area += (a[0]*b[1]-b[0]*a[1])/2
        if bulge:
            center, radius, start, sweep = arc_data(a, b, bulge)
            area += radius*radius*(sweep-math.sin(sweep))/2
            desired = max(1, math.ceil(abs(sweep)/max(1e-5, 2*math.acos(max(-1., 1-min(.001, radius)/radius)))))
            count = min(128, desired)
            chord_error = max(chord_error, radius*(1-math.cos(abs(sweep)/(2*count))))
            flat.extend([(center[0]+radius*math.cos(start+sweep*i/count),
                          center[1]+radius*math.sin(start+sweep*i/count)) for i in range(count)])
        else: flat.append(a)
        if len(flat) > 512: raise ValueError('Flattened contour vertex budget exceeded')
    if not closed: flat.append(parts[-1][1])
    if not all(finite(v) for p in flat for v in p) or not finite(area) or not finite(chord_error):
        raise ValueError('Derived topology is outside finite range')
    if closed and abs(area) <= tolerance*tolerance: raise ValueError('Zero-area closed contour')
    return {'signed_area': area, 'max_gap': max(gaps, default=0.), 'chord_error_bound': chord_error,
            'flat': flat, 'closed': closed}


def _distance(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    denominator = dx*dx+dy*dy
    if denominator <= 1e-24: return math.dist(point, a)
    t = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/denominator))
    return math.dist(point, (a[0]+t*dx, a[1]+t*dy))


def _edge_distance(a, b, c, d):
    cross = lambda x, y, z: (y[0]-x[0])*(z[1]-x[1])-(y[1]-x[1])*(z[0]-x[0])
    p, q, r, s = cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b)
    if p*q < 0 and r*s < 0: return 0.
    return min(_distance(a, c, d), _distance(b, c, d), _distance(c, a, b), _distance(d, a, b))


def _edges(loop):
    flat = loop['flat']
    return list(zip(flat, flat[1:]+(flat[:1] if loop['closed'] else [])))


def check_simple(loop, tolerance, budget):
    edges = _edges(loop)
    for i, (a, b) in enumerate(edges):
        for j in range(i+2, len(edges)):
            if loop['closed'] and i == 0 and j == len(edges)-1: continue
            budget.spend()
            c, d = edges[j]
            # The same curve approximation envelope also applies inside one
            # contour: chords alone can miss an arc crossing a nearby segment.
            if _edge_distance(a, b, c, d) <= tolerance+2*loop['chord_error_bound']:
                raise ValueError('Contour self-intersection or unresolved near-contact')


def _inside(point, loop):
    x, y = point
    inside = False
    for a, b in _edges(loop):
        if (a[1] > y) != (b[1] > y):
            cross_x = a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if x < cross_x: inside = not inside
    return inside


def check_holes(loops, tolerance, budget):
    if not loops or len(loops) > 65: raise ValueError('Invalid HATCH loop count')
    for loop in loops: check_simple(loop, tolerance, budget)
    for i, left in enumerate(loops):
        for j in range(i+1, len(loops)):
            right = loops[j]
            clearance = tolerance+left['chord_error_bound']+right['chord_error_bound']
            for a, b in _edges(left):
                for c, d in _edges(right):
                    budget.spend()
                    if _edge_distance(a, b, c, d) <= clearance:
                        raise ValueError('HATCH boundaries intersect, touch, or exceed resolved curve clearance')
            if i == 0:
                if not _inside(right['flat'][0], left): raise ValueError('HATCH hole lies outside outer boundary')
            elif _inside(right['flat'][0], left) or _inside(left['flat'][0], right):
                raise ValueError('Nested HATCH holes are unsupported')
    net = abs(loops[0]['signed_area'])-sum(abs(loop['signed_area']) for loop in loops[1:])
    if not finite(net) or net <= 0: raise ValueError('Invalid HATCH net area')
    return net


def loop_vertices(parts, *, counterclockwise):
    area = loop_data(parts, 1e-7)['signed_area']
    if (area > 0) != counterclockwise:
        parts = [(b, a, -bulge) for a, b, bulge in reversed(parts)]
    return [(*a, bulge) for a, _, bulge in parts]


def cyclic_equal(expected, actual, tolerance):
    if len(expected) != len(actual): return False
    if not expected: return False
    for offset, vertex in enumerate(actual):
        if math.dist(expected[0][:2], vertex[:2]) > tolerance: continue
        if all(math.dist(a[:2], actual[(i+offset) % len(actual)][:2]) <= tolerance
               and abs(a[2]-actual[(i+offset) % len(actual)][2]) <= 1e-10
               for i, a in enumerate(expected)):
            return True
    return False
