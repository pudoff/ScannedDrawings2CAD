"""Exact native-CAD incidence candidates without selecting a measured contour.

An incidence is not an associative DIMENSION link, a verified semantic role or
a pixel label. Several exact candidates are retained; nearby geometry is never
substituted for missing evidence. Geometry inside INSERTs is not flattened.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np

CAD_TOLERANCE = 1e-6
MAX_MODELSPACE_ENTITIES = 200_000
MAX_GEOMETRY_PRIMITIVES = 20_000
MAX_DIMENSIONS = 5_000
MAX_INCIDENCE_COMPARISONS = 2_000_000
SUPPORTED_KINDS = {'linear': ('defpoint2', 'defpoint3'),
                   'aligned': ('defpoint2', 'defpoint3'),
                   'diameter': ('defpoint', 'defpoint4'),
                   'radius': ('defpoint', 'defpoint4')}
NATIVE_KINDS = {0: 'linear', 1: 'aligned', 3: 'diameter', 4: 'radius'}


def _xyz(value):
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return point if point.shape == (3,) and np.isfinite(point).all() else None


def _planar(entity):
    vector = _xyz(entity.dxf.get('extrusion', (0, 0, 1)))
    return vector is not None and np.allclose(vector, (0, 0, 1), rtol=0, atol=1e-12)


def _inventory(doc):
    primitives, skipped, counts = [], Counter(), Counter()
    complete = True
    for index, entity in enumerate(doc.modelspace()):
        if index >= MAX_MODELSPACE_ENTITIES:
            complete = False
            break
        kind = entity.dxftype()
        counts[kind] += 1
        if kind not in ('LINE', 'LWPOLYLINE', 'CIRCLE', 'ARC'):
            continue
        if not _planar(entity):
            skipped['non_default_extrusion'] += 1
            continue
        common = {'geometry_handle': entity.dxf.handle, 'geometry_type': kind,
                  'geometry_layer': entity.dxf.layer, 'owner_block': '*Model_Space',
                  'semantic_role': 'unreviewed', 'source_instance_path': []}
        new = []
        if kind == 'LINE':
            a, b = _xyz(entity.dxf.start), _xyz(entity.dxf.end)
            if a is None or b is None or abs(a[2]) > CAD_TOLERANCE or abs(b[2]) > CAD_TOLERANCE:
                skipped['nonfinite_or_nonplanar_geometry'] += 1
                continue
            if np.linalg.norm(b-a) <= CAD_TOLERANCE:
                skipped['zero_or_tiny_line'] += 1
                continue
            new.append((common, 'line', (a[:2], b[:2])))
        elif kind == 'LWPOLYLINE':
            # Each vertex is a primitive; reject before copying an arbitrarily
            # large vertex array. The edge count is checked below as well.
            if len(entity) > MAX_GEOMETRY_PRIMITIVES-len(primitives):
                complete = False
                break
            elevation = entity.dxf.get('elevation', 0.)
            points = np.asarray(list(entity.get_points('xyb')), dtype=float)
            if (not math.isfinite(elevation) or abs(elevation) > CAD_TOLERANCE
                    or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()):
                skipped['nonfinite_or_nonplanar_geometry'] += 1
                continue
            for vertex, point in enumerate(points):
                new.append(({**common, 'vertex_index': vertex}, 'vertex', point[:2]))
            for edge in range(len(points) if entity.closed else max(0, len(points)-1)):
                if points[edge, 2] != 0:
                    skipped['bulged_polyline_edge'] += 1
                    continue
                a, b = points[edge, :2], points[(edge+1) % len(points), :2]
                if np.linalg.norm(b-a) <= CAD_TOLERANCE:
                    skipped['zero_or_tiny_polyline_edge'] += 1
                    continue
                new.append(({**common, 'edge_index': edge}, 'edge', (a, b)))
        else:
            center, radius = _xyz(entity.dxf.center), entity.dxf.radius
            if (center is None or abs(center[2]) > CAD_TOLERANCE or not math.isfinite(radius)
                    or radius <= CAD_TOLERANCE):
                skipped['nonfinite_nonplanar_or_degenerate_curve'] += 1
                continue
            start, span = None, None
            if kind == 'ARC':
                start, end = entity.dxf.start_angle, entity.dxf.end_angle
                if not math.isfinite(start) or not math.isfinite(end):
                    skipped['nonfinite_arc_angles'] += 1
                    continue
                span = (end-start) % 360
                if span < 1e-9:
                    skipped['degenerate_arc_span'] += 1
                    continue
            new.append((common, 'curve', (center[:2], float(radius), start, span)))
        if len(primitives)+len(new) > MAX_GEOMETRY_PRIMITIVES:
            complete = False
            break
        primitives.extend(new)
    return primitives, {'complete': complete, 'modelspace_entity_counts': dict(counts),
                        'primitive_count': len(primitives), 'skipped_geometry': dict(skipped),
                        'modelspace_insert_count': counts['INSERT'],
                        'insert_contents_expanded': False,
                        'dimension_graphics_blocks_considered': False,
                        'layer_names_used_as_semantic_roles': False}


def _point_candidates(point, primitives, role):
    results = []
    p = point[:2]
    for common, primitive, data in primitives:
        relation, distance, detail = None, None, {}
        if primitive in ('line', 'edge') and role == 'generic_point':
            a, b = data
            delta = b-a
            t = float((p-a) @ delta/(delta @ delta))
            finite = a+np.clip(t, 0, 1)*delta
            distance = float(np.linalg.norm(p-finite))
            da, db = float(np.linalg.norm(p-a)), float(np.linalg.norm(p-b))
            if primitive == 'line' and min(da, db) <= CAD_TOLERANCE:
                relation = 'endpoint'
                detail['endpoint'] = 'start' if da <= db else 'end'
                distance = min(da, db)
            elif min(da, db) > CAD_TOLERANCE and distance <= CAD_TOLERANCE:
                relation = 'segment' if primitive == 'line' else 'straight_polyline_edge'
            detail.update(closest_finite_xy=finite.tolist(), unclamped_segment_parameter=t)
        elif primitive == 'vertex' and role == 'generic_point':
            distance = float(np.linalg.norm(p-data))
            relation = 'vertex'
            detail['vertex_xy'] = data.tolist()
        elif primitive == 'curve':
            center, radius, start, span = data
            center_distance = float(np.linalg.norm(p-center))
            if role in ('generic_point', 'curve_center') and center_distance <= CAD_TOLERANCE:
                relation, distance = 'center', center_distance
            elif role in ('generic_point', 'curve_point') and abs(center_distance-radius) <= CAD_TOLERANCE:
                relation, distance = ('circle_point' if start is None else 'arc_point'), abs(center_distance-radius)
                if start is not None:
                    angle = math.degrees(math.atan2(p[1]-center[1], p[0]-center[0])) % 360
                    relative = (angle-start) % 360
                    if relative > span:
                        # A fixed angular slack grows without bound in CAD
                        # units as the radius grows. Use the actual distance
                        # to a finite arc endpoint for an out-of-span point.
                        ends = [center+radius*np.array([math.cos(math.radians(a)),
                                                       math.sin(math.radians(a))])
                                for a in (start, start+span)]
                        distance = min(float(np.linalg.norm(p-end)) for end in ends)
                        if distance > CAD_TOLERANCE:
                            continue
                    detail.update(angle_deg=angle, start_angle_deg=start, sweep_deg=span)
            detail.update(center_xy=center.tolist(), radius_cad=radius)
        if relation is not None and distance is not None and distance <= CAD_TOLERANCE:
            results.append({**common, 'relation': relation, 'distance_cad': distance, **detail,
                            'annotation_approved': False, 'training_eligible': False})
    # Identity includes vertex/edge to preserve two geometrically distinct
    # polyline locations while preventing repeated records for the same feature.
    unique = {}
    for item in results:
        identity = (item['geometry_handle'], item['relation'], item.get('vertex_index'),
                    item.get('edge_index'), item.get('endpoint'))
        unique[identity] = item
    return list(unique.values())


def collect_attachment_candidates(doc, dimensions: list[dict]) -> dict:
    """Collect exact planar modelspace incidence for every supplied dimension.

    Input records are P08.6 CAD-semantic observations. Native handle/type/points
    are checked before use. An entire query is refused if its comparison budget
    cannot cover all geometry; a truncated inventory is never called exact.
    """
    if not isinstance(dimensions, list) or len(dimensions) > MAX_DIMENSIONS:
        raise ValueError('Dimension inventory exceeds the supported limit')
    if any(not isinstance(item, dict) or not isinstance(item.get('handle'), str)
           or not isinstance(item.get('insert_path', []), list) for item in dimensions):
        raise ValueError('Invalid semantic dimension record')
    identities = [(d['handle'], tuple(str(v) for v in d.get('insert_path', []))) for d in dimensions]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate dimension-instance identity')
    primitives, inventory = _inventory(doc)
    estimates = sum(len(SUPPORTED_KINDS.get(d.get('kind'), ())) for d in dimensions if not d.get('insert_path'))*len(primitives)
    budget_ok = inventory['complete'] and estimates <= MAX_INCIDENCE_COMPARISONS
    records = []
    for dimension in dimensions:
        kind, handle = dimension.get('kind'), dimension['handle']
        source_points = dimension.get('definition_points_modelspace', {})
        names = SUPPORTED_KINDS.get(kind, ())
        status = 'incidence_candidates_only'
        if dimension.get('insert_path'):
            status = 'instance_review_required'
        elif dimension.get('entity_type') != 'DIMENSION' or kind not in SUPPORTED_KINDS:
            status = 'unsupported_dimension_kind'
        elif not budget_ok:
            status = 'geometry_budget_exceeded'
        entity = doc.entitydb.get(handle)
        if status == 'incidence_candidates_only' and (entity is None or entity.dxftype() != 'DIMENSION'
                or entity.dxf.owner != doc.modelspace().block_record_handle or NATIVE_KINDS.get(entity.dimtype) != kind):
            status = 'source_definition_mismatch'
        if status == 'incidence_candidates_only' and not _planar(entity):
            status = 'nonplanar_dimension'
        points = {}
        for name in (names or list(source_points)):
            raw = source_points.get(name)
            point = _xyz(raw)
            point_status = status if status != 'incidence_candidates_only' else None
            candidates = []
            if point_status is None:
                native = _xyz(entity.dxf.get(name))
                if (point is None or native is None or not np.allclose(point, native, rtol=0, atol=CAD_TOLERANCE)):
                    point_status = 'source_definition_mismatch'
                elif abs(point[2]) > CAD_TOLERANCE:
                    point_status = 'nonplanar_definition_point'
                else:
                    role = ('curve_center' if kind == 'radius' and name == 'defpoint' else
                            'curve_point' if kind in ('radius','diameter') else 'generic_point')
                    candidates = _point_candidates(point, primitives, role)
                    point_status = 'no_exact_candidate' if not candidates else 'exact_candidates' if len(candidates) == 1 else 'ambiguous_candidates'
            points[name] = {'cad_xyz': None if point is None else point.tolist(), 'status': point_status,
                            'candidates': candidates, 'candidate_count': len(candidates),
                            'geometry_role_verified': False, 'selected_candidate': None,
                            'annotation_approved': False, 'training_eligible': False}
        records.append({'handle': handle, 'insert_path': dimension.get('insert_path', []),
                        'entity_type': dimension.get('entity_type'), 'kind': kind, 'status': status,
                        'points': points, 'units': 'native CAD units; no physical conversion',
                        'annotation_approved': False, 'training_eligible': False})
    return {'schema_version': '1.0', 'method': 'exact-modelspace-incidence-v1',
            'dimensions': records, 'geometry_inventory': inventory,
            'summary': {'dimension_count': len(records),
                        'dimension_statuses': dict(Counter(r['status'] for r in records)),
                        'point_statuses': dict(Counter(p['status'] for r in records for p in r['points'].values()))},
            'limits': {'cad_tolerance': CAD_TOLERANCE, 'max_geometry_primitives': MAX_GEOMETRY_PRIMITIVES,
                       'max_modelspace_entities': MAX_MODELSPACE_ENTITIES, 'max_dimensions': MAX_DIMENSIONS,
                       'max_incidence_comparisons': MAX_INCIDENCE_COMPARISONS,
                       'estimated_incidence_comparisons': estimates},
            'annotation_approved': False, 'training_eligible': False,
            'limitations': ['Exact incidence is not proof of measured contour ownership or CAD associativity.',
                           'Frames, axes and unrelated modelspace entities retain unverified roles.',
                           'No nearest geometry, infinite-line projection or implicit unit assignment is used.',
                           'INSERT contents and dimension graphics blocks are not treated as modelspace contour geometry.']}
