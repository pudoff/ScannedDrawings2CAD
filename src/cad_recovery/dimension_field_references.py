"""Link required DIMENSION points to already reviewed raster intervals.

Coordinate coincidence links an existing observation only. It never chooses a
nearest point, refits an alignment, assigns a contour role or approves accuracy.
The caller must pin the native record, comparison report and source files.
"""
from copy import deepcopy
import math
import re

NATIVE_MATCH_TOLERANCE = 1e-6
MAX_REFERENCE_POINTS = 10000
ANCHOR_ROLES = {
    'linear': {'defpoint2': 'first_extension_origin', 'defpoint3': 'second_extension_origin'},
    'aligned': {'defpoint2': 'first_extension_origin', 'defpoint3': 'second_extension_origin'},
    'diameter': {'defpoint': 'diameter_endpoint', 'defpoint4': 'opposite_diameter_endpoint'},
    'radius': {'defpoint': 'center', 'defpoint4': 'point_on_circle'},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
             'Expected a canonical source SHA-256')
    return value


def _numbers(value, size):
    _require(isinstance(value, (list, tuple)) and len(value) == size
             and all(type(item) in (int, float) and math.isfinite(item) for item in value),
             'Expected finite numeric coordinates without boolean values')
    return [float(item) for item in value]


def _text(value):
    _require(isinstance(value, str) and 0 < len(value) <= 256
             and value.strip() == value and not any(ord(c) < 32 for c in value),
             'Expected a nonempty bounded identifier')
    return value


def _point_records(report):
    _require(isinstance(report, dict) and report.get('schema_version') == '1.0'
             and report.get('status') == 'complete', 'Expected a complete raster comparison report')
    _require(report.get('alignment_refitted') is False and report.get('references_modified') is False
             and report.get('training_eligible') is False and report.get('annotation_approved') is False,
             'Raster comparison cannot be refitted, modified or approved by this link')
    points = report.get('points')
    _require(isinstance(points, list) and len(points) <= MAX_REFERENCE_POINTS,
             'Invalid raster reference inventory')
    result, seen = [], set()
    for point in points:
        _require(isinstance(point, dict), 'Invalid raster reference record')
        identifier = _text(point.get('point_id'))
        _require(identifier not in seen, 'Duplicate raster reference identity')
        seen.add(identifier)
        _sha(point.get('source_sha256'))
        _sha(point.get('target_sha256'))
        _require(point.get('alignment_refitted') is False and point.get('training_eligible') is False
                 and point.get('annotation_approved') is False, 'Reference point cannot approve a field')
        cad = _numbers(point.get('cad_xyz'), 3)
        projected = _numbers(point.get('projected_source_xy'), 2)
        center = _numbers(point.get('center_source_xy'), 2)
        bounds = _numbers(point.get('interval_source_xyxy'), 4)
        _require(abs(cad[2]) <= NATIVE_MATCH_TOLERANCE, 'Reference point must be planar')
        _require(bounds[0] < bounds[2] and bounds[1] < bounds[3]
                 and bounds[0] <= center[0] <= bounds[2] and bounds[1] <= center[1] <= bounds[3],
                 'Invalid authored reference interval')
        inside = bounds[0] <= projected[0] <= bounds[2] and bounds[1] <= projected[1] <= bounds[3]
        expected_status = 'inside_reference_interval' if inside else 'outside_reference_interval'
        comparison = point.get('comparison')
        _require(isinstance(comparison, dict) and comparison.get('status') == expected_status
                 and comparison.get('ground_truth_error_px') is None
                 and comparison.get('annotation_approved') is False,
                 'Reference comparison disagrees with its unapproved interval evidence')
        result.append((point, cad, expected_status))
    if 'summary' in report:
        _require(isinstance(report['summary'], dict), 'Invalid reference summary')
        if 'point_count' in report['summary']:
            _require(type(report['summary']['point_count']) is int
                     and report['summary']['point_count'] == len(points), 'Reference point count differs')
    return result


def match_reference_anchors(native_dimension, source_sha, target_sha, point_report):
    """Return per-anchor observation links in original PNG coordinates.

Only direct modelspace DIMENSIONs and the explicit standard anchor roles below
are supported. Two links do not imply a dimension's geometric role, physical
nominal, complete correspondence or accurate endpoints have been approved.
"""
    _sha(source_sha)
    _sha(target_sha)
    _require(isinstance(native_dimension, dict) and native_dimension.get('entity_type') == 'DIMENSION',
             'Expected a native DIMENSION record')
    handle = _text(native_dimension.get('handle'))
    owner = _text(native_dimension.get('owner_block'))
    instance = native_dimension.get('insert_path')
    _require(isinstance(instance, list) and len(instance) <= 32, 'Expected an explicit bounded instance path')
    for item in instance:
        _text(item)
    references = _point_records(point_report)
    kind = _text(native_dimension.get('kind'))
    role_map = ANCHOR_ROLES.get(kind, {})
    roles = native_dimension.get('definition_point_roles')
    coordinates = native_dimension.get('definition_points_modelspace')
    _require(isinstance(roles, dict) and isinstance(coordinates, dict), 'Missing native definition point evidence')
    for field, role in role_map.items():
        _require(roles.get(field) == role, 'Native definition point role differs from its dimension kind')
    supported_instance = not instance and owner == '*Model_Space'
    points = {}
    for field, role in role_map.items():
        coordinate = _numbers(coordinates[field], 3) if field in coordinates else None
        matches = []
        status = 'not_linked'
        if coordinate is None:
            status = 'not_available'
        elif not supported_instance:
            status = 'unsupported_dimension_instance'
        elif abs(coordinate[2]) > NATIVE_MATCH_TOLERANCE:
            status = 'nonplanar_not_linked'
        else:
            for reference, cad, interval_status in references:
                if reference['source_sha256'] != source_sha or reference['target_sha256'] != target_sha:
                    continue
                distance = math.dist(coordinate, cad)
                if distance > NATIVE_MATCH_TOLERANCE:
                    continue
                matches.append({'point_id': reference['point_id'],
                    'source_sha256': source_sha, 'target_sha256': target_sha,
                    'reference_cad_xyz': list(cad), 'native_identity_distance': distance,
                    'projected_source_xy': deepcopy(reference['projected_source_xy']),
                    'center_source_xy': deepcopy(reference['center_source_xy']),
                    'interval_source_xyxy': deepcopy(reference['interval_source_xyxy']),
                    'interval_status': interval_status,
                    'reference_evidence': deepcopy(reference.get('reference_evidence')),
                    'annotation_approved': False, 'raster_accuracy_approved': False})
            if len(matches) > 1:
                status = 'ambiguous_reference_links'
            elif matches:
                status = ('agrees_with_authored_interval' if matches[0]['interval_status'] == 'inside_reference_interval'
                          else 'outside_authored_interval')
        points[field] = {'role': role, 'cad_xyz': coordinate, 'status': status,
                         'reference_matches': matches, 'annotation_approved': False,
                         'geometry_role_approved': False, 'raster_accuracy_approved': False}
    linked = sum(point['status'] in ('agrees_with_authored_interval', 'outside_authored_interval')
                 for point in points.values())
    return {'schema_version': '1.0', 'identity': {'handle': handle, 'owner_block': owner,
                                                'insert_path': list(instance)},
        'status': ('unsupported_anchor_roles' if not role_map else 'unsupported_dimension_instance'
                   if not supported_instance else 'explicit_reference_links_reviewed'),
        'points': points, 'required_fields': len(role_map), 'linked_fields': linked,
        'all_required_fields_linked': bool(role_map) and linked == len(role_map),
        'native_match_tolerance': NATIVE_MATCH_TOLERANCE,
        'native_match_tolerance_meaning': 'Native coordinate identity only, not raster or physical accuracy.',
        'alignment_refitted': False, 'references_modified': False,
        'physical_nominal_approved': False, 'geometry_role_approved': False,
        'raster_accuracy_approved': False, 'annotation_approved': False, 'training_eligible': False,
        'limitations': ['Authored interval agreement is an observation, not a ground-truth error bound.',
                       'A native coordinate link does not assign a contour role or verify the measured geometry.',
                       'Unlinked, unsupported or ambiguous anchors remain unresolved independently of other fields.']}
