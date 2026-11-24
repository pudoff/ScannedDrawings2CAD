"""Separate observed transverse stroke footprints while retaining shared ink."""
import math

import cv2
import numpy as np

from .grid_role_revision_v1 import sha
from .local_contour_profiles_v1 import junction_distance, measure
from .native_frame_v1 import side_masks
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground


POLICY = {'maximum_tracks': 512, 'maximum_pairs': 250_000, 'maximum_fit_pixels': 64_000_000,
          'maximum_profile_samples': 2_000_000, 'minimum_width_ratio': 1.67, 'minimum_crossing_angle_deg': 20,
          'minimum_run_in_widths': 12., 'minimum_independent_crossings': 2}


def track_records(gray, graph, work, audit, *, profile_measure=measure):
    items = {item['id']: item for item in graph['proposals']}
    if len(items) != len(graph['proposals']):
        raise ValueError('Source interval identifiers must be unique')
    junctions = junction_distance([{'id': i, 'support_points': p['points_px']} for i, p in items.items()], gray.shape)
    for track in graph['tracks']:
        ends = np.asarray(track['points'], float)
        length = float(np.linalg.norm(ends[1]-ends[0]))
        if length < 12 or not track['interval_ids']:
            continue
        members = [items[i] for i in track['interval_ids']]
        q = np.concatenate([np.asarray(item['points_px'], int) for item in members])
        if (q.min(0) < 0).any() or (q.max(0) >= gray.shape[::-1]).any():
            raise ValueError('Track source support outside raster')
        profile = profile_measure(ends, q, gray, work, junctions, maximum_samples=POLICY['maximum_profile_samples'])
        if work['samples'] > POLICY['maximum_profile_samples']:
            audit['limits'].append('profile_budget')
            return
        if profile is None:
            continue
        low, high = map(float, np.quantile(profile['sections_px'], [.25, .95]))
        if high > 2*low:
            continue
        unit = (ends[1]-ends[0])/length
        yield {'id': track['id'], 'ends': ends, 'unit': unit, 'normal': np.array([-unit[1], unit[0]]),
               'length': length, 'pen_px': low, 'upper_pen_px': high, 'members': track['interval_ids'],
               'has_material_hint': any(p['role'] == 'part_contour' for p in members), 'profile': profile}


def crossing(first, second):
    u, v = first['unit'], second['unit']
    if abs(float(u @ v)) > math.cos(math.radians(POLICY['minimum_crossing_angle_deg'])):
        return None
    a, b = first['ends'][0], second['ends'][0]
    i, j = np.linalg.solve(np.column_stack((u, -v)), b-a)
    # Need observed shaft on both sides and a substantial transverse stroke.
    if not (2*second['upper_pen_px'] <= i <= first['length']-2*second['upper_pen_px']
            and 2*first['upper_pen_px'] <= j <= second['length']-2*first['upper_pen_px']):
        return None
    return a+i*u


def visible_fit(ink, record, audit):
    audit['fit_pixels'] += ink.size
    if audit['fit_pixels'] > POLICY['maximum_fit_pixels']:
        audit['limits'].append('fit_budget')
        return None
    fit, reason = side_masks(ink, record['ends'], record['upper_pen_px'], math.sqrt(.5), audit)
    if reason and reason.endswith('_budget'):
        audit['limits'].append(reason)
    if fit is not None and fit['sections'] < .55*np.ptp(record['ends'], axis=0).max():
        return None
    return fit


def contact_fit(ink, record, point, audit):
    fit = visible_fit(ink, record, audit)
    if fit is not None or audit['limits']:
        return fit
    # Distant hatching can invalidate a single digital-edge model for the
    # entire rib. Independently fit the observed interval around this joint.
    radius = max(18., 4*record['upper_pen_px'])
    a = record['ends'][0]
    u = record['unit']
    center = float((point-a) @ u)
    measured = np.clip(center+np.array([-radius, radius]), 0, record['length'])
    return visible_fit(ink, {**record, 'ends': a+measured[:, None]*u}, audit)


def band(gray, record, radius):
    y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
    dx, dy = x-record['ends'][0, 0], y-record['ends'][0, 1]
    u, n = record['unit'], record['normal']
    along = dx*u[0]+dy*u[1]
    return (abs(dx*n[0]+dy*n[1]) <= radius) & (along >= -radius) & (along <= record['length']+radius)


def resolve_axis_contacts(gray, graph, proposed, raw_negative, baseline, protected):
    """Withdraw only proposals on a thin through stroke outside exact PART bands.

    All arrays are in identical native pixel coordinates. Raw-negative evidence
    can narrow the operation but cannot certify a source direction by itself.
    """
    checked_gray(gray)
    if any(a.shape != gray.shape for a in (proposed, raw_negative, baseline, protected)):
        raise ValueError('Aligned original source, proposals and protection required')
    if any(a.dtype != bool for a in (proposed, raw_negative, protected)) or baseline.dtype != np.uint8:
        raise ValueError('Boolean masks and uint8 baseline roles required')
    audit = {'limits': [], 'pairs': 0, 'fit_pixels': 0, 'source_samples': 0, 'witnesses': []}
    eligible = proposed & raw_negative & (baseline == 255) & ~protected
    withdrawn = np.zeros_like(eligible)
    shared = np.zeros_like(eligible)
    if not eligible.any():
        return {'withdrawn': withdrawn, 'shared': shared, 'audit': audit, 'complete': True}
    if graph.get('search_incomplete') or len(graph['tracks']) > POLICY['maximum_tracks']:
        audit['limits'].append('source_graph_incomplete')
        return {'withdrawn': withdrawn, 'shared': shared, 'audit': audit, 'complete': False}
    ink = foreground(gray)
    work = {'ink': ink, 'distance': cv2.distanceTransform(ink, cv2.DIST_L2, 5), 'samples': 0}
    records = list(track_records(gray, graph, work, audit))
    boolean_ink = ink > 0
    for track in records if not audit['limits'] else []:
        if track['length'] < POLICY['minimum_run_in_widths']*track['upper_pen_px']:
            continue
        possible = eligible & band(gray, track, track['upper_pen_px']/2+math.sqrt(.5))
        if not possible.any():
            continue
        peers = []
        for other in records:
            audit['pairs'] += 1
            if audit['pairs'] > POLICY['maximum_pairs']:
                audit['limits'].append('track_pair_budget')
                break
            if (other['pen_px'] < POLICY['minimum_width_ratio']*track['upper_pen_px']
                    or not other['has_material_hint'] or not set(other['members']).isdisjoint(track['members'])):
                continue
            point = crossing(track, other)
            if point is not None:
                peers.append((other, point))
        if len(peers) < POLICY['minimum_independent_crossings'] or audit['limits']:
            continue
        thin = visible_fit(boolean_ink, track, audit)
        if thin is None:
            continue
        exact = possible & thin['safe']
        if not exact.any():
            continue
        thick_possible = np.zeros_like(eligible)
        thick_safe = np.zeros_like(eligible)
        local_contact = np.zeros_like(eligible)
        observed_peers = []
        for other, point in peers:
            fit = contact_fit(boolean_ink, other, point, audit)
            if fit is None:
                break
            thick_possible |= fit['possible']
            thick_safe |= fit['safe']
            # This restricts the change to the joint neighbourhood. The exact
            # masks, not the radius of the neighbourhood, decide overlap.
            radius = 2*(other['upper_pen_px']+track['upper_pen_px'])+math.sqrt(.5)
            y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
            local_contact |= (x-point[0])**2+(y-point[1])**2 <= radius**2
            observed_peers.append({'source_ids': other['members'], 'point_px': point.tolist(),
                                   'source_sections': fit['sections'], 'pen_px': other['pen_px']})
        if len(observed_peers) != len(peers) or audit['limits']:
            continue
        if not any(set(first['members']).isdisjoint(second['members'])
                   and np.linalg.norm(a-b) > first['upper_pen_px']+second['upper_pen_px']
                   for i, (first, a) in enumerate(peers) for second, b in peers[i+1:]):
            continue
        both = exact & thick_safe & local_contact
        only_thin = exact & ~thick_possible & local_contact
        shared |= both
        withdrawn |= only_thin
        if only_thin.any() or both.any():
            audit['witnesses'].append({'thin_track_id': track['id'], 'thin_source_ids': track['members'],
                'thin_direction': track['unit'].tolist(), 'thin_profile': track['profile'], 'contacts': observed_peers,
                'exclusive_thin_pixels': int(only_thin.sum()), 'shared_pixels': int(both.sum())})
    # If any hypothesis supports genuine overlap, retain it across all candidates.
    withdrawn &= ~shared
    if audit['limits']:
        withdrawn.fill(False)
        shared.fill(False)
    audit['profile_samples'] = work['samples']
    return {'withdrawn': withdrawn, 'shared': shared, 'audit': audit, 'complete': not audit['limits'],
            'source_sha256': sha(gray), 'graph_sha256': object_sha(graph)}


def observed_annotation_overlaps(packet, material, evidence):
    """Record dual support from two measured paths even at one source pixel."""
    gray = packet['gray']
    graph = packet['graph']
    owner = packet['source_owner']
    positive = evidence['positive']
    contacts = evidence['contacts']
    role = evidence['guard']['stages']['role']
    if (positive['source_sha256'] != sha(gray) or positive['source_graph_sha256'] != object_sha(graph)
            or positive['source_owner_sha256'] != sha(owner) or positive['source_heads_sha256'] != object_sha(packet['heads'])
            or contacts['input_mask_sha256'] != sha(material) or not role['complete']):
        raise ValueError('Bound material and annotation profile observations required')
    measured_tracks = role['contact_resolution']['track_profiles']
    points = {index: np.asarray(p['points_px'], int) for index, p in enumerate(graph['proposals'], 1)}
    count = 0
    complete = True
    overlaps = []
    shared = np.zeros_like(gray, bool)
    for track in measured_tracks:
        profile = track['profile']
        if profile is None:
            continue
        numbers = set(track['owner_numbers'])
        supports = np.concatenate([points[n] for n in numbers])
        x, y = supports.T
        seed = np.full_like(gray, 255)
        seed[y, x] = 0
        # Measure discrete source-pixel distance, keeping the source footprint.
        radius = track['clearance_radius_px']
        nearby = (cv2.distanceTransform(seed, cv2.DIST_L2, 5) <= radius) & (material > 0)
        ends = np.asarray(profile['endpoints_px'], float)
        u = (ends[1]-ends[0])/np.linalg.norm(ends[1]-ends[0])
        for obj in positive['objects']:
            count += 1
            if count > POLICY['maximum_pairs']:
                complete = False
                break
            if obj.get('witness_only') or obj['id'] in contacts['rejected_object_ids'] or not numbers.isdisjoint(obj['source_owner_numbers']):
                continue
            p = np.asarray(obj['points_px'], float)
            length = float(np.linalg.norm(p[1]-p[0]))
            if length == 0:
                continue
            v = (p[1]-p[0])/length
            if abs(u @ v) > math.cos(math.radians(POLICY['minimum_crossing_angle_deg'])):
                continue
            selected = nearby & np.isin(owner, obj['source_owner_numbers'])
            if not selected.any():
                continue
            yy, xx = np.nonzero(selected)
            displacement = np.c_[xx, yy]-p[0]
            material_radius = obj['observed_pen_px']/2+math.sqrt(.5)
            along = displacement @ v
            bound = ((along >= -material_radius) & (along <= length+material_radius)
                     & (abs(displacement @ np.array([-v[1], v[0]])) <= material_radius))
            yy, xx = yy[bound], xx[bound]
            if not len(xx):
                continue
            shared[yy, xx] = True
            # Retain the source IDs for both observations. The existing owner
            # map continues to record the same physical source sample.
            overlaps.append({'material_object_id': obj['id'], 'material_owner_numbers': obj['source_owner_numbers'],
                'material_direction': v.tolist(), 'annotation_track_id': track['track_id'],
                'annotation_owner_numbers': sorted(numbers), 'annotation_direction': u.tolist(),
                'points_px': np.c_[xx, yy].tolist(), 'semantic_role_assigned': False})
        if not complete:
            break
    if not complete:
        shared.fill(False)
    return {'complete': complete, 'source_sha256': sha(gray), 'shared': shared,
            'observations': overlaps if complete else [], 'checks': count}
