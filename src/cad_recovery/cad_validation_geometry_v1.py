"""Independent graph-derived geometric expectations for CAD validation.

This module does not import the exporter, its measurement helper, or its native
snapshot. Coordinates below are the declared graph-to-model contract.
"""
import math

from .tolerances import dimension_tolerances

MAX_ABS_NATIVE = 1e30


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and abs(value) <= MAX_ABS_NATIVE


def transform(view, point):
    x, y = point
    if view.physical_scale_confirmed:
        sx, sy = view.model_scale_xy
        angle = math.radians(view.model_rotation_deg)
        c, s = math.cos(angle), math.sin(angle)
        result = (c*sx*x-s*sy*y+view.model_origin[0], s*sx*x+c*sy*y+view.model_origin[1])
    else:
        a = view.view_to_sheet
        result = (a[0][0]*x+a[0][1]*y+a[0][2], a[1][0]*x+a[1][1]*y+a[1][2])
    if not all(finite(v) for v in result): raise ValueError('Graph transform exceeds finite comparison range')
    return result


def sheet_point(view, point):
    x, y = point
    a = view.view_to_sheet
    return (a[0][0]*x+a[0][1]*y+a[0][2], a[1][0]*x+a[1][1]*y+a[1][2])


def vector(view, point):
    a, b = transform(view, (0, 0)), transform(view, point)
    return (b[0]-a[0], b[1]-a[1])


def similarity_scale(view):
    x, y = vector(view, (1, 0)), vector(view, (0, 1))
    a, b = math.hypot(*x), math.hypot(*y)
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or abs(x[0]*y[0]+x[1]*y[1]) > a*b*1e-9 or x[0]*y[1]-x[1]*y[0] <= 0:
        raise ValueError('Expected native curve requires an orientation-preserving similarity')
    return a


def angle(view, degrees):
    direction = vector(view, (math.cos(math.radians(degrees)), math.sin(math.radians(degrees))))
    return math.degrees(math.atan2(direction[1], direction[0])) % 360


def geometry_expected(geometry, view):
    kind = geometry.kind
    if kind == 'line':
        return {'type': 'LINE', 'start': transform(view, geometry.start), 'end': transform(view, geometry.end)}
    if kind in ('circle', 'arc'):
        result = {'type': kind.upper(), 'center': transform(view, geometry.center), 'radius': geometry.radius*similarity_scale(view)}
        if kind == 'arc': result.update(start_angle=angle(view, geometry.start_angle_deg), end_angle=angle(view, geometry.end_angle_deg))
        return result
    if kind == 'polyline':
        if any(v.bulge for v in geometry.vertices): similarity_scale(view)
        return {'type': 'LWPOLYLINE', 'closed': geometry.closed,
                'vertices': [(*transform(view, v.point), v.bulge) for v in geometry.vertices]}
    raise ValueError('Unsupported graph primitive')


def raw_geometry(entity):
    kind = entity.dxftype()
    def point(name):
        p = getattr(entity.dxf, name)
        if not all(finite(float(v)) for v in p) or abs(p.z) > 1e-8:
            raise ValueError('Nonfinite or nonplanar native point')
        return (float(p.x), float(p.y))
    if kind == 'LINE': return {'type': kind, 'start': point('start'), 'end': point('end')}
    if kind in ('ARC', 'CIRCLE'):
        result = {'type': kind, 'center': point('center'), 'radius': float(entity.dxf.radius)}
        if not finite(result['radius']) or result['radius'] <= 0: raise ValueError('Invalid native radius')
        if kind == 'ARC':
            angles = [float(entity.dxf.start_angle), float(entity.dxf.end_angle)]
            if not all(finite(a) for a in angles): raise ValueError('Invalid native arc angles')
            result.update(start_angle=angles[0] % 360, end_angle=angles[1] % 360)
        return result
    if kind == 'LWPOLYLINE':
        if len(entity) > 10000: raise ValueError('Native polyline vertex budget exceeded')
        points = [tuple(float(v) for v in p) for p in entity.get_points('xyseb')]
        if any(not all(finite(v) for v in p) or p[2] != 0 or p[3] != 0 or abs(p[4]) > 1e6 for p in points):
            raise ValueError('Native polyline contains invalid coordinates, bulges or unexpected width')
        if abs(entity.dxf.elevation) > 1e-8 or entity.dxf.const_width != 0:
            raise ValueError('Native polyline has unexpected elevation or width')
        return {'type': kind, 'closed': entity.closed, 'vertices': [(p[0], p[1], p[4]) for p in points]}
    raise ValueError('Unsupported native geometric type')


def angular_difference(a, b):
    return abs((a-b+180) % 360-180)


def compare_geometry(expected, actual, tolerance):
    errors, point_error, scalar_error = [], 0., 0.
    if expected['type'] != actual['type']: return ['type'], None, None
    for key in ('start', 'end', 'center'):
        if key in expected:
            distance = math.dist(expected[key], actual[key])
            point_error = max(point_error, distance)
            if distance > tolerance: errors.append(key)
    if 'radius' in expected:
        scalar_error = abs(expected['radius']-actual['radius'])
        if scalar_error > tolerance: errors.append('radius')
    for key in ('start_angle', 'end_angle'):
        if key in expected and angular_difference(expected[key], actual[key]) > 1e-7:
            errors.append(key)
    if 'vertices' in expected:
        if len(expected['vertices']) != len(actual['vertices']): errors.append('vertex_count')
        else:
            for index, (a, b) in enumerate(zip(expected['vertices'], actual['vertices'])):
                distance = math.dist(a[:2], b[:2])
                point_error = max(point_error, distance)
                if distance > tolerance: errors.append('vertex_'+str(index))
                if abs(a[2]-b[2]) > 1e-10: errors.append('bulge_'+str(index))
        if expected['closed'] != actual['closed']: errors.append('closed')
    return errors, point_error, scalar_error


def arc_data(a, b, bulge):
    if not 1e-12 <= abs(bulge) <= 1e6: raise ValueError('Unsupported native bulge range')
    dx, dy = b[0]-a[0], b[1]-a[1]
    chord = math.hypot(dx, dy)
    if chord <= 1e-12: raise ValueError('Zero-length curved segment')
    factor = (1/bulge-bulge)/4
    center = ((a[0]+b[0])/2-dy*factor, (a[1]+b[1])/2+dx*factor)
    radius = chord*(abs(bulge)+1/abs(bulge))/4
    start = math.atan2(a[1]-center[1], a[0]-center[0])
    sweep = 4*math.atan(bulge)
    if not all(finite(v) for v in (*center, radius, start, sweep)): raise ValueError('Unresolvable native arc')
    return center, radius, start, sweep


def segments(geometry):
    kind = geometry['type']
    if kind == 'LINE': return [(geometry['start'], geometry['end'], 0.)]
    if kind in ('CIRCLE', 'ARC'):
        x, y = geometry['center']; radius = geometry['radius']
        if kind == 'CIRCLE': return [((x+radius, y), (x-radius, y), 1.), ((x-radius, y), (x+radius, y), 1.)]
        start, end = geometry['start_angle'], geometry['end_angle']
        point = lambda a: (x+radius*math.cos(math.radians(a)), y+radius*math.sin(math.radians(a)))
        sweep = (end-start) % 360
        if sweep <= 1e-9: raise ValueError('Zero or full-circle ARC is ambiguous')
        return [(point(start), point(end), math.tan(math.radians(sweep)/4))]
    points = geometry['vertices']
    ends = points[1:]+(points[:1] if geometry['closed'] else [])
    return [(a[:2], b[:2], a[2]) for a, b in zip(points, ends)]


def point_on_geometry(point, geometry, tolerance, *, center_allowed=True):
    if geometry['type'] in ('ARC', 'CIRCLE') and center_allowed and math.dist(point, geometry['center']) <= tolerance:
        return True
    for a, b, bulge in segments(geometry):
        if not bulge:
            dx, dy = b[0]-a[0], b[1]-a[1]
            denominator = dx*dx+dy*dy
            if denominator <= 1e-24: continue
            parameter = min(1., max(0., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/denominator))
            if math.dist(point, (a[0]+parameter*dx, a[1]+parameter*dy)) <= tolerance: return True
        else:
            center, radius, start, sweep = arc_data(a, b, bulge)
            if abs(math.dist(point, center)-radius) > tolerance: continue
            bearing = math.atan2(point[1]-center[1], point[0]-center[0])
            travelled = (bearing-start) % (2*math.pi) if sweep > 0 else (start-bearing) % (2*math.pi)
            if travelled <= abs(sweep)+tolerance/radius or travelled >= 2*math.pi-tolerance/radius: return True
    return False


def dimension_expectation(dimension, view):
    points = [transform(view, p) for p in dimension.points]
    kind = dimension.kind
    if kind == 'angular':
        center, a, b = points
        measurement = math.degrees((math.atan2(b[1]-center[1], b[0]-center[0])-math.atan2(a[1]-center[1], a[0]-center[0])) % (2*math.pi))
        definitions = {'defpoint4': center, 'defpoint2': a, 'defpoint3': b}
        native_kind = 5
    elif kind in ('radius', 'diameter'):
        a, b = points
        measurement = math.dist(a, b)*(2 if kind == 'diameter' else 1)
        definitions = {'defpoint': a, 'defpoint4': b} if kind == 'radius' else {
            'defpoint': b, 'defpoint4': (2*a[0]-b[0], 2*a[1]-b[1])}
        native_kind = 4 if kind == 'radius' else 3
    else:
        a, b = points
        direction = math.atan2(b[1]-a[1], b[0]-a[0]) if kind == 'aligned' else math.radians(angle(view, dimension.angle_deg))
        measurement = math.dist(a, b) if kind == 'aligned' else abs((b[0]-a[0])*math.cos(direction)+(b[1]-a[1])*math.sin(direction))
        definitions = {'defpoint2': a, 'defpoint3': b}
        native_kind = 1 if kind == 'aligned' else 0
    if not finite(measurement) or measurement <= 1e-12: raise ValueError('Degenerate expected dimension')
    return {'points': points, 'measurement': measurement, 'definitions': definitions, 'native_kind': native_kind}


def computed_text(dimension, measured, *, include_tolerance=True):
    def number(value): return format(value, '.6f').rstrip('0').rstrip('.') or '0'
    def plain(value):
        for code, symbol in (('c', 'Ø'), ('d', '°'), ('p', '±')):
            value = value.replace('%%'+code, symbol).replace('%%'+code.upper(), symbol)
        return value
    prefix, suffix = plain(dimension.prefix), plain(dimension.suffix)
    symbol = {'radius': 'R', 'diameter': 'Ø', 'longitudinal_diameter': 'Ø'}.get(dimension.kind, '')
    if symbol and not (prefix.startswith(symbol) or dimension.kind == 'longitudinal_diameter' and prefix.startswith('⌀')): prefix = symbol+prefix
    unit = '°' if dimension.kind == 'angular' else ''
    result = prefix+number(measured)+unit+suffix
    deviations = dimension_tolerances(dimension) if include_tolerance else None
    if deviations is not None:
        upper, lower = deviations.upper, deviations.lower
        if upper == -lower:
            result += '±'+number(upper)+unit
        else:
            def signed(value):
                return ('+' if value > 0 else '-' if value < 0 else ' ')+number(abs(value))+unit
            result += signed(upper)+'^'+signed(lower)
    return result


def validate_attachments(dimension, points, references, tolerance):
    """All declared supports must participate in the requested measurement kind."""
    if not references: raise ValueError('Dimension has no resolved native references')
    if dimension.kind in ('radius', 'diameter'):
        center, contact = points
        for geometry in references:
            if geometry['type'] not in ('CIRCLE', 'ARC') or math.dist(center, geometry['center']) > tolerance or not point_on_geometry(contact, geometry, tolerance, center_allowed=False):
                raise ValueError('Radial dimension requires the actual center and finite circular contact on every reference')
        return
    if dimension.kind == 'angular':
        if len(references) != 2 or any(g['type'] != 'LINE' for g in references):
            raise ValueError('Angular dimension requires exactly two LINE references')
        center, a, b = points
        if not all(point_on_geometry(center, g, tolerance, center_allowed=False) for g in references):
            raise ValueError('Angular vertex is not on both finite referenced lines')
        if not any(point_on_geometry(a, references[i], tolerance, center_allowed=False) and
                   point_on_geometry(b, references[1-i], tolerance, center_allowed=False) for i in (0, 1)):
            raise ValueError('Angular ray endpoints do not bind to distinct finite arms')
        return
    if dimension.kind == 'longitudinal_diameter' and any(g['type'] not in ('LINE', 'LWPOLYLINE') or
            any(part[2] for part in segments(g)) for g in references):
        raise ValueError('Longitudinal diameter requires straight profile references')
    covers = [[point_on_geometry(p, g, tolerance, center_allowed=dimension.kind in ('linear', 'aligned'))
               for g in references] for p in points]
    if not all(any(row) for row in covers) or not all(any(row[i] for row in covers) for i in range(len(references))):
        raise ValueError('Every measured point and every declared reference must participate in the attachment')


def geometry_extrema(geometry):
    result = []
    for a, b, bulge in segments(geometry):
        result.extend((a, b))
        if bulge:
            center, radius, start, sweep = arc_data(a, b, bulge)
            for bearing in (0., math.pi/2, math.pi, 3*math.pi/2):
                travelled = (bearing-start) % (2*math.pi) if sweep > 0 else (start-bearing) % (2*math.pi)
                if travelled <= abs(sweep)+1e-12:
                    result.append((center[0]+radius*math.cos(bearing), center[1]+radius*math.sin(bearing)))
    return result
