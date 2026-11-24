"""Validate semantic intervals on each emitted primitive's own source edges."""

from collections import Counter
from copy import deepcopy

import cv2
import numpy as np

from .cad_object_assembly_v1 import compact
from .cad_object_assembly_v5 import _member_ledger, _members
from .cad_primitive_fit_v2 import checked, sector
from .part_admission_v1 import object_sha


POLICY = {'version': 'primitive-role-guard/1.0', 'minimum_fraction': .8,
          'minimum_boundary_edges': 3, 'unknown_gap': 'measured_width_plus_two_pixel_diagonals',
          'maximum_objects': 16384, 'unknown_or_conflicting_support': 'editable_draft'}
KNOWN = set(range(1, 10))


def edge_inventory(objects):
    return Counter((m['candidate_id'], i) for obj in objects for m in obj['source_members']
                   for i in range(*m['source_index_interval']))


def point_inventory(objects):
    return Counter((m['candidate_id'], m['source_index_interval'][0])
                   for obj in objects for m in obj['source_members']
                   if m['source_index_interval'][0] == m['source_index_interval'][1])


def source_points(obj, lookup):
    points = []
    for member in obj['source_members']:
        candidate = lookup[member['candidate_id']]
        first = candidate['source_index_interval'][0]
        low, high = member['source_index_interval']
        points.extend(candidate.get('support_points', candidate.get('points_px'))[low-first:high-first+1])
    return np.unique(np.asarray(points, int), axis=0)


def runs(values):
    if not len(values):
        return []
    cuts = np.r_[0, np.flatnonzero(values[1:] != values[:-1])+1, len(values)]
    return [(int(a), int(b), int(values[a])) for a, b in zip(cuts, cuts[1:])]


def _close_unknown(labels, points, gap, closed):
    """Fill only short unlabelled intervals bracketed by the same observed role."""
    result = labels.copy()
    length = len(labels)
    if length < 2:
        return result
    # Circular source paths repeat their first sample. Rotate at a known sample
    # so a gap crossing that arbitrary seam is treated like every other gap.
    unique = length-1 if closed else length
    offset = int(np.flatnonzero(labels[:unique] != 255)[0]) if closed and np.any(labels[:unique] != 255) else 0
    index = (np.arange(unique)+offset) % unique if closed else np.arange(unique)
    if closed:
        index = np.r_[index, index[0]]
    ordered = labels[index].copy()
    along = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points[index], axis=0), axis=1))]
    for low, high, value in runs(ordered):
        if value == 255 and low > 0 and high < len(ordered) and ordered[low-1] == ordered[high]:
            if along[high]-along[low-1] <= gap:
                ordered[low:high] = ordered[low-1]
    result[index] = ordered
    if closed:
        result[-1] = result[0]
    return result


def _model_parameters(obj, points):
    """Locate source samples on the fixed parent geometry, in walk order."""
    if obj['kind'] == 'line':
        ends = np.asarray(obj['points_px'], float)
        delta = ends[1]-ends[0]
        denominator = float(delta @ delta)
        if denominator <= 0:
            return None
        positions = (points-ends[0]) @ delta / denominator
        direction = float(positions[-1]-positions[0])
        if abs(direction) < 1e-10:
            return None
        if direction < 0:
            ends = ends[::-1]
            positions = 1.-positions
        return {'positions': positions, 'ends': ends}
    if obj['kind'] not in ('arc', 'circle'):
        return None
    center = np.asarray(obj['center_px'], float)
    angles = np.unwrap(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))
    delta = float(angles[-1]-angles[0])
    if abs(delta) < 1e-10:
        return None
    direction = 1. if delta > 0 else -1.
    if obj['kind'] == 'circle':
        if not np.array_equal(points[0], points[-1]):
            return None
        start, span = float(angles[0]), 2*np.pi
    else:
        span = np.radians(obj['sweep_angle_deg'])
        if not 0 < span < 2*np.pi:
            return None
        start = np.radians(obj['start_angle_deg']) + (span if direction < 0 else 0.)
        start += 2*np.pi*round((angles[0]-start)/(2*np.pi))
    delta = direction*span
    return {'positions': (angles-start)/delta, 'start': start, 'delta': delta}


def _model_piece(obj, points, support, parameters=None, interval=None):
    if obj['kind'] == 'line':
        low, high = interval
        ends = parameters['ends']
        projected = ends[0] + np.outer([low, high], ends[1]-ends[0])
        # Exact external model endpoints may have been adjusted to meet an ARC.
        # Reprojecting the raw raster endpoints would reopen that CAD junction.
        if low == 0.:
            projected[0] = ends[0]
        if high == 1.:
            projected[1] = ends[1]
        model = {'kind': 'line', 'points_px': projected.tolist(),
                 'fit_max_px': obj.get('fit_max_px', 0.), 'fit_p95_px': obj.get('fit_p95_px', 0.)}
    elif obj['kind'] in ('arc', 'circle'):
        model = sector(points, np.asarray(obj['center_px']), obj['radius_px'])
        if model is None:
            return None
        angles = parameters['start'] + np.asarray(interval)*parameters['delta']
        model.update(start_angle_deg=float(np.degrees(min(angles)) % 360),
                     sweep_angle_deg=float(np.degrees(abs(angles[1]-angles[0]))))
        # Splitting changes the angular interval only. Retain the evidence for
        # the parent's circle parameters instead of claiming a new local fit.
        for key in ('circle_family_support', 'circle_family_source_sectors',
                    'parameter_sensitivity_native_px', 'parameter_evidence', 'locally_stable'):
            if key in obj:
                model[key] = deepcopy(obj[key])
    elif obj['kind'] == 'polyline':
        model = {'kind': 'polyline', 'points_px': compact(points), 'closed': False,
                 'fit_max_px': 0., 'fit_p95_px': 0.}
    else:
        return None
    return checked(model, points, support)


def _split(obj, labels, ledger, support):
    points = np.asarray(obj['support_points'], int)
    # A boundary edge with disagreeing endpoint roles stays explicitly unknown.
    edges = np.where(labels[:-1] == labels[1:], labels[:-1], 255)
    spans = runs(edges)
    confirmed = {r for a, b, r in spans if r in KNOWN and b-a >= POLICY['minimum_boundary_edges']}
    if not confirmed or (len(spans) == 1):
        return None
    parameters = _model_parameters(obj, points)
    if obj['kind'] in ('line', 'arc', 'circle'):
        if parameters is None:
            return None
        positions = parameters['positions'].copy()
        positions[0], positions[-1] = 0., 1.
        cuts = [a for a, _, _ in spans] + [spans[-1][1]]
        boundary_positions = positions[cuts]
        # Ordered intervals must partition precisely the original model extent.
        # A raster wiggle cannot introduce overlaps, reversed or missing pieces.
        if (not np.isfinite(boundary_positions).all()
                or np.any(np.diff(boundary_positions) <= 1e-10)):
            return None
    else:
        positions = None
    output = []
    for low, high, role in spans:
        part = points[low:high+1]
        interval = None if positions is None else (positions[low], positions[high])
        model = _model_piece(obj, part, support, parameters, interval)
        if model is None:
            return None
        part_edges = {tuple(sorted((tuple(a), tuple(b)))) for a, b in zip(part, part[1:])}
        links = [link for link in obj.get('source_junction_links', [])
                 if all(tuple(sorted((tuple(a), tuple(b)))) in part_edges
                        for a, b in zip(link['points_px'], link['points_px'][1:]))]
        item = {'points': part, 'path_ids': obj['path_ids'], 'source_junction_links': links}
        try:
            members = _members(item, ledger)
        except ValueError:
            return None
        assigned = role if role in confirmed and high-low >= POLICY['minimum_boundary_edges'] else 255
        identity = 'cad-role-segment/'+object_sha({'parent': obj['id'], 'members': members, 'geometry': model})[:24]
        output.append({**model, 'id': identity, 'source_members': members,
                       'path_ids': sorted({m['path_id'] for m in members}),
                       'support_points': part.tolist(), 'role': assigned,
                       'source_junction_links': links,
                       'status': 'draft_requires_review', 'semantic_ownership_confirmed': False,
                       'physical_geometry_approved': False,
                       'semantic_role_guard': {'decision': 'confirmed_role_boundary',
                           'parent_object_id': obj['id'], 'previous_role': obj['role'],
                           'source_support_interval': [low, high], 'assigned_role': assigned}})
    # Hidden junction edges cannot disappear or get duplicated during a split.
    if edge_inventory(output) != edge_inventory([obj]):
        return None
    return output


def apply(assembly, gray, geometry, roles):
    """Return a guarded assembly without modifying observations or geometry inputs."""
    if roles.shape != gray.shape or roles.dtype != np.uint8:
        raise ValueError('Aligned uint8 semantic map required')
    if not np.isin(roles, list(range(10))+[255]).all():
        raise ValueError('Invalid semantic role')
    result = deepcopy(assembly)
    lookup = {c['id']: c for c in geometry}
    ledger = _member_ledger(geometry)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    radius = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    support = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    objects = []
    for obj in result['objects']:
        own = source_points(obj, lookup)
        observed = roles[own[:, 1], own[:, 0]]
        counts = Counter(map(int, observed))
        points = np.asarray(obj['support_points'], int)
        measured_width = max(1., float(np.median(radius[own[:, 1], own[:, 0]])*2-1))
        # Two endpoint samples each have one raster-cell diagonal uncertainty.
        # This is a sampling bound, not a tolerance for bridging missing ink.
        maximum_gap = measured_width + 2*np.sqrt(2.)
        labels = roles[points[:, 1], points[:, 0]].copy()
        labels[~np.isin(labels, list(KNOWN))] = 255
        closed = len(points) > 1 and np.array_equal(points[0], points[-1])
        labels = _close_unknown(labels, points, maximum_gap, closed)
        proof = {'previous_role': obj['role'], 'source_points': len(own),
                 'observed_role_counts': {str(k): v for k, v in sorted(counts.items())},
                 'maximum_unknown_gap_px': maximum_gap, 'assigned_role': 255}
        if 'axis_pattern_evidence' in obj or 'axis_review' in obj:
            # These carry a separately checked dash-pattern or reviewed contract.
            # Automatic patterns remain AXES_PATTERN_DRAFT, never AXES_CV.
            proof.update(decision='separate_axis_contract', assigned_role=obj['role'])
        else:
            known = set(counts) & KNOWN
            role = next(iter(known)) if len(known) == 1 else 255
            fraction = counts.get(role, 0)/len(own) if role != 255 else 0.
            # Bound unsupported tails and internal holes by the measured width.
            along = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
            unknown_spans = [float(along[min(b, len(points)-1)]-along[max(0, a-1)])
                             for a, b, value in runs(labels) if value == 255]
            covered_fraction = float(np.mean(labels == role)) if role != 255 else 0.
            sufficient = (role != 255 and covered_fraction >= POLICY['minimum_fraction']
                          and max(unknown_spans, default=0.) <= maximum_gap)
            pieces = _split(obj, labels, ledger, support) if not sufficient and len(points) > 1 else None
            if pieces is not None:
                for piece in pieces:
                    piece['semantic_role_guard'].update(parent_evidence=proof)
                objects.extend(pieces)
                continue
            obj['role'] = role if sufficient else 255
            proof.update(decision='own_support_confirmed' if sufficient else 'insufficient_or_conflicting_support',
                         assigned_role=obj['role'], observed_fraction=fraction, covered_fraction=covered_fraction,
                         largest_unknown_span_px=max(unknown_spans, default=0.))
        obj['semantic_role_guard'] = proof
        objects.append(obj)
    if len(objects) > POLICY['maximum_objects']:
        raise ValueError('Semantic split object budget exceeded')
    if edge_inventory(objects) != edge_inventory(assembly['objects']):
        raise ValueError('Semantic guard changed source edge partition')
    if point_inventory(objects) != point_inventory(assembly['objects']):
        raise ValueError('Semantic guard changed isolated source points')
    result.update(objects=objects, object_count=len(objects),
                  kind_counts=dict(Counter(o['kind'] for o in objects)),
                  multi_candidate_objects=sum(len({m['candidate_id'] for m in o['source_members']}) > 1 for o in objects),
                  primitive_role_guard={'policy': POLICY, 'source_edge_partition_exact': True,
                      'decisions': dict(Counter(o['semantic_role_guard']['decision'] for o in objects)),
                      'changed_role_objects': sum(o['role'] != o['semantic_role_guard']['previous_role'] for o in objects),
                      'input_objects': len(assembly['objects']), 'output_objects': len(objects)})
    return result
