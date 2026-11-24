"""Bounded contour validation preserving exact arc/bulge geometry for DXF.

Curves are flattened only for exclusion tests. A conservative chord-error
clearance check rejects near contacts that this sampling cannot resolve.
"""
import math

from .graph_v2 import ArcV2, CircleV2, LineV2, PolylineV2

TOLERANCE = 1e-8
MAX_FLAT_VERTICES = 2048
MAX_PREDICATES = 250_000


def _finite(*values):
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite derived topology geometry cannot be verified')


class TopologyBudget:
    """One shared work limit for all boundaries of an export, including holes."""
    def __init__(self, limit=MAX_PREDICATES):
        if type(limit) is not int or not 1 <= limit <= MAX_PREDICATES:
            raise ValueError('Invalid topology work budget')
        self.remaining = limit
        self.used = 0

    def charge(self):
        if self.remaining == 0:
            raise ValueError('Topology predicate budget exceeded; boundary remains unverified')
        self.remaining -= 1
        self.used += 1


def _point(center, radius, angle):
    a = math.radians(angle)
    result = (center[0]+radius*math.cos(a), center[1]+radius*math.sin(a))
    _finite(*result)
    return result


def geometry_segments(geometry):
    if isinstance(geometry, LineV2):
        return [(geometry.start, geometry.end, 0.)]
    if isinstance(geometry, ArcV2):
        sweep = (geometry.end_angle_deg-geometry.start_angle_deg) % 360
        return [(_point(geometry.center, geometry.radius, geometry.start_angle_deg),
                 _point(geometry.center, geometry.radius, geometry.end_angle_deg), math.tan(math.radians(sweep)/4))]
    if isinstance(geometry, CircleV2):
        a = (geometry.center[0]+geometry.radius, geometry.center[1])
        b = (geometry.center[0]-geometry.radius, geometry.center[1])
        return [(a, b, 1.), (b, a, 1.)]
    if isinstance(geometry, PolylineV2):
        ends = geometry.vertices[1:]+(geometry.vertices[:1] if geometry.closed else [])
        return [(a.point, b.point, a.bulge) for a, b in zip(geometry.vertices, ends)]
    raise ValueError('Unsupported contour primitive')


def contour_segments(contour, geometries):
    segments = []
    for edge in contour.edges:
        part = geometry_segments(geometries[edge.primitive_id])
        if edge.reversed:
            part = [(b, a, -bulge) for a, b, bulge in reversed(part)]
        segments.extend(part)
    if not segments:
        raise ValueError('Empty contour')
    if any(math.dist(a[1], b[0]) > TOLERANCE for a, b in zip(segments, segments[1:])):
        raise ValueError('Contour edges have an open gap; endpoints are never merged')
    is_closed = math.dist(segments[-1][1], segments[0][0]) <= TOLERANCE
    if is_closed != contour.closed:
        raise ValueError('Declared contour closure differs from its geometry')
    return segments


def bulge_arc(a, b, bulge):
    chord = math.dist(a, b)
    _finite(*a, *b, bulge, chord)
    if not 1e-12 <= abs(bulge) <= 1e6 or chord <= TOLERANCE:
        raise ValueError('A bulge arc requires a nonzero chord and bulge')
    factor = (1/bulge-bulge)/4
    dx, dy = b[0]-a[0], b[1]-a[1]
    center = ((a[0]+b[0])/2-dy*factor, (a[1]+b[1])/2+dx*factor)
    radius = chord*(abs(bulge)+1/abs(bulge))/4
    angle = math.atan2(a[1]-center[1], a[0]-center[0])
    sweep = 4*math.atan(bulge)
    _finite(*center, radius, angle, sweep)
    return center, radius, angle, sweep


def flatten_segments(segments):
    result, max_error = [], 0.
    for a, b, bulge in segments:
        _finite(*a, *b, bulge)
        result.append(a)
        if bulge:
            center, radius, start, sweep = bulge_arc(a, b, bulge)
            count = max(2, math.ceil(abs(sweep)/math.radians(5)))
            if len(result)+count > MAX_FLAT_VERTICES:
                raise ValueError('Contour flattening budget exceeded')
            max_error = max(max_error, radius*(1-math.cos(abs(sweep)/count/2)))
            result.extend((center[0]+radius*math.cos(start+sweep*i/count),
                           center[1]+radius*math.sin(start+sweep*i/count)) for i in range(1, count))
        if len(result) > MAX_FLAT_VERTICES:
            raise ValueError('Contour vertex budget exceeded')
    _finite(max_error, *[value for point in result for value in point])
    return result, max_error


def signed_area(segments):
    area = 0.
    for a, b, bulge in segments:
        area += (a[0]*b[1]-b[0]*a[1])/2
        if bulge:
            _, radius, _, sweep = bulge_arc(a, b, bulge)
            area += radius*radius*(sweep-math.sin(sweep))/2
        _finite(area)
    return area


def _cross(a, b, c):
    value = (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    _finite(value)
    return value


def _distance(point, a, b):
    delta = (b[0]-a[0], b[1]-a[1])
    length2 = delta[0]**2+delta[1]**2
    _finite(*delta, length2)
    if length2 <= TOLERANCE**2:
        return math.dist(point, a)
    t = max(0., min(1., sum((point[i]-a[i])*delta[i] for i in (0, 1))/length2))
    return math.dist(point, (a[0]+t*delta[0], a[1]+t*delta[1]))


def segment_distance(a, b, c, d):
    orientations = [_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)]
    if orientations[0]*orientations[1] < 0 and orientations[2]*orientations[3] < 0:
        return 0.
    return min(_distance(a, c, d), _distance(b, c, d), _distance(c, a, b), _distance(d, a, b))


def _inside(point, polygon):
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:]+polygon[:1]):
        if (a[1] > y) != (b[1] > y):
            cross_x = a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])
            if cross_x > x:
                inside = not inside
    return inside


def validate_contour(contour, geometries, *, budget=None):
    budget = budget if budget is not None else TopologyBudget()
    segments = contour_segments(contour, geometries)
    flat, error = flatten_segments(segments)
    if not contour.closed:
        # Open edges are retained but cannot become a HATCH boundary.
        return {'id': contour.id, 'closed': False, 'segments': segments, 'flat': [*flat, segments[-1][1]],
                'signed_area': None, 'chord_error_bound': error, 'status': 'open_contour_preserved'}
    area = signed_area(segments)
    if abs(area) <= TOLERANCE:
        raise ValueError('Closed contour has zero area')
    edges = list(zip(flat, flat[1:]+flat[:1]))
    for i, (a, b) in enumerate(edges):
        for j in range(i+1, len(edges)):
            if j == i+1 or (i == 0 and j == len(edges)-1):
                continue
            budget.charge()
            c, d = edges[j]
            if segment_distance(a, b, c, d) <= 2*error+TOLERANCE:
                raise ValueError('Contour self-intersects or its curve clearance is unresolved')
    return {'id': contour.id, 'closed': True, 'segments': segments, 'flat': flat,
            'signed_area': area, 'chord_error_bound': error, 'status': 'simple_closed_contour_checked'}


def validate_hatch_boundaries(outer, holes, geometries, *, budget=None):
    if len(holes) > 64:
        raise ValueError('HATCH hole budget exceeded')
    budget = budget if budget is not None else TopologyBudget()
    checked = [validate_contour(c, geometries, budget=budget) for c in [outer, *holes]]
    if any(not item['closed'] for item in checked):
        raise ValueError('HATCH cannot use an open boundary')
    for i, left in enumerate(checked):
        for right in checked[i+1:]:
            clearance = left['chord_error_bound']+right['chord_error_bound']+TOLERANCE
            for a, b in zip(left['flat'], left['flat'][1:]+left['flat'][:1]):
                for c, d in zip(right['flat'], right['flat'][1:]+right['flat'][:1]):
                    budget.charge()
                    if segment_distance(a, b, c, d) <= clearance:
                        raise ValueError('HATCH boundaries intersect, touch or have unresolved curve clearance')
    for index, hole in enumerate(checked[1:], 1):
        if not _inside(hole['flat'][0], checked[0]['flat']):
            raise ValueError('HATCH hole is outside its outer boundary')
        for other in checked[1:index]:
            if _inside(hole['flat'][0], other['flat']) or _inside(other['flat'][0], hole['flat']):
                raise ValueError('Nested holes require an explicit island hierarchy; refusing implicit filling')
    area = abs(checked[0]['signed_area'])-sum(abs(h['signed_area']) for h in checked[1:])
    _finite(area)
    if area <= TOLERANCE:
        raise ValueError('HATCH net area is not positive')
    return checked, area


def polyline_boundary(checked, *, clockwise=False):
    segments = checked['segments']
    if (checked['signed_area'] < 0) != clockwise:
        segments = [(b, a, -bulge) for a, b, bulge in reversed(segments)]
    return [(a[0], a[1], bulge) for a, _, bulge in segments]
