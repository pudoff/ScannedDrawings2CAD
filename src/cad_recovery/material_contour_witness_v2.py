"""Second material witness: pen-scale network ambiguity and the legacy seed extent, never a CAD fitter.

Only measured original ink with an unambiguous immutable source owner may be
proposed. This module does not infer physical size or accept semantic roles.
"""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from .cad_object_assembly_v5 import assemble
from .grid_role_revision_v1 import check_authority, sha
from .local_contour_profiles_v2 import measure, junction_distance
from .local_contour_probes_v1 import probes
from .local_contour_witness_v2 import inventory
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray, foreground
from .stroke_curve_control_v1 import distance_to_polyline

POLICY = {
    'version': 'material-contour-witness/2.0', 'maximum_pixels': 4000000,
    'maximum_objects': 4096, 'maximum_lines': 512, 'maximum_pair_checks': 250000,
    'maximum_profile_samples': 2000000, 'maximum_local_raster_pixels': 64000000,
    'maximum_component_distance_samples': 2000000,
    'minimum_reference_length_in_pens': 6., 'minimum_branch_length_in_pens': 6.,
    'minimum_seed_length_in_pens': 12., 'minimum_pen_ratio': 1.5,
    'maximum_pair_pen_ratio': 1.67, 'maximum_profile_ratio': 2.,
    'minimum_thin_reference_sources': 3, 'minimum_orthogonal_angle_deg': 70.,
    'minimum_curve_sweep_deg': 90., 'minimum_radius_in_pens': 12.,
    'ambiguous_rectangular_component_withheld': True,
    'direction_tolerance_deg': 8., 'minimum_return_length_in_pens': 6.,
    'source_owner_remapped': False, 'geometry_modified': False,
    'production_admitted': False, 'physical_geometry_approved': False,
}


def line_profile(obj, points, gray, work, junctions):
    ends = np.asarray(obj['points_px'], float)
    previous = measure(ends, points, gray, work, junctions, maximum_samples=POLICY['maximum_profile_samples'])
    if work['samples'] > POLICY['maximum_profile_samples']:
        return None
    if previous is not None:
        low, high = np.quantile(previous['sections_px'], [.25, .95])
        if low >= 1. and high <= POLICY['maximum_profile_ratio'] * low:
            return previous
    delta = ends[1]-ends[0]; length = float(np.linalg.norm(delta))
    if not length:
        return None
    normal = np.array([-delta[1], delta[0]]) / length
    q = np.asarray(points, int)
    widths, sites, fractions = [], [], []
    used = set()
    # Periodic hatch junctions can cover all nine old sampling sites. Inspect
    # more observed sites without reducing the required count or accepting a
    # junction section. Its own DT radius controls that site's exclusion.
    for fraction in np.linspace(.1, .9, 33):
        site = q[np.argmin(np.linalg.norm(q-(ends[0]+fraction*delta), axis=1))]
        key = tuple(site)
        if key in used:
            continue
        used.add(key)
        radius = max(1., float(work['distance'][site[1], site[0]]))
        if junctions[site[1], site[0]] <= 2*radius+math.sqrt(.5):
            continue
        extent = 2*radius+math.sqrt(.5)
        positions = np.arange(-extent, extent+.25, .25)
        sampled = np.rint(site+positions[:, None]*normal).astype(int)
        work['samples'] += len(sampled)
        if work['samples'] > POLICY['maximum_profile_samples']:
            return None
        if (sampled < 0).any() or (sampled[:, 0] >= gray.shape[1]).any() or (sampled[:, 1] >= gray.shape[0]).any():
            continue
        values = work['ink'][sampled[:, 1], sampled[:, 0]] > 0
        origin = int(np.argmin(abs(positions)))
        if not values[origin]:
            continue
        lo = hi = origin
        while lo > 0 and values[lo-1]:
            lo -= 1
        while hi+1 < len(values) and values[hi+1]:
            hi += 1
        if lo == 0 or hi == len(values)-1:
            continue
        widths.append(float((hi-lo+1)*.25)); sites.append(site.tolist()); fractions.append(float(fraction))
    if len(widths) < 5 or max(fractions)-min(fractions) < .4:
        return None
    return {'pen_px': float(np.quantile(widths, .25)), 'sections_px': widths, 'source_centres_px': sites,
            'site_fractions': fractions, 'normal': normal.tolist(), 'minimum_valid_sites': 5,
            'width_measurement': 'distributed_source_sections_outside_local_junction_radius'}


def curve_profile(obj, points, gray, work, junctions):
    """Sample radial sections through actual source centres, outside junctions."""
    center = np.asarray(obj['center_px'], float)
    radius = float(obj['radius_px'])
    if center.shape != (2,) or not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0:
        raise ValueError('Finite circular model required')
    q = np.asarray(points, int)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    if not lengths[-1]:
        return None
    local_radius = max(1., float(np.quantile(work['distance'][q[:, 1], q[:, 0]], .95)))
    extent = 2 * local_radius + math.sqrt(.5)
    positions = np.arange(-extent, extent + .25, .25)
    sections, sites, normals = [], [], []
    for fraction in np.linspace(.1, .9, 13):
        site = q[int(np.argmin(abs(lengths - fraction * lengths[-1])))]
        if junctions[site[1], site[0]] <= 2 * local_radius + math.sqrt(.5):
            continue
        normal = site - center
        norm = float(np.linalg.norm(normal))
        if not norm:
            continue
        normal /= norm
        sampled = np.rint(site + positions[:, None] * normal).astype(int)
        work['samples'] += len(sampled)
        if work['samples'] > POLICY['maximum_profile_samples']:
            return None
        if (sampled < 0).any() or (sampled[:, 0] >= gray.shape[1]).any() or (sampled[:, 1] >= gray.shape[0]).any():
            continue
        values = work['ink'][sampled[:, 1], sampled[:, 0]] > 0
        origin = int(np.argmin(abs(positions)))
        if not values[origin]:
            continue
        lo = hi = origin
        while lo > 0 and values[lo - 1]:
            lo -= 1
        while hi + 1 < len(values) and values[hi + 1]:
            hi += 1
        if lo == 0 or hi == len(values) - 1:
            continue
        sections.append(float((hi - lo + 1) * .25))
        sites.append(site.tolist()); normals.append(normal.tolist())
    if len(sections) < 7 or len({tuple(s) for s in sites}) < 7:
        return None
    return {'pen_px': float(np.quantile(sections, .25)), 'sections_px': sections,
            'source_centres_px': sites, 'normals': normals,
            'width_measurement': 'radial_sections_at_observed_source_centres', 'minimum_valid_sites': 7}


def connected(ink, blocked, a, b, pen, audit):
    """Require both endpoints in one unprotected local ink component."""
    pad = math.ceil(pen + math.sqrt(.5))
    lo = np.maximum(0, np.minimum(a, b) - pad)
    hi = np.minimum(ink.shape[::-1], np.maximum(a, b) + pad + 1)
    audit['local_raster_pixels'] += int(np.prod(hi - lo))
    if audit['local_raster_pixels'] > POLICY['maximum_local_raster_pixels']:
        audit['limits'].append('contact_raster_budget')
        return None
    local = ((ink[lo[1]:hi[1], lo[0]:hi[0]] > 0) & ~blocked[lo[1]:hi[1], lo[0]:hi[0]]).astype(np.uint8)
    _, labels = cv2.connectedComponents(local, connectivity=8)
    x, y = a - lo, b - lo
    label = int(labels[x[1], x[0]])
    if label and label == int(labels[y[1], y[0]]):
        return {'points_px': [a.tolist(), b.tolist()], 'source_connected': True,
                'bbox_px': [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]}
    return None


def rectangular_component(label, pen, labels, stats, audit, *, hatched=False):
    """A plain rectangular wire/network cannot prove material rather than a grid.

    This is a conservative ambiguity refusal, not a semantic FRAME label.
    Interior holes or multiple rectangular cells do not resolve that ambiguity.
    """
    x, y, width, height, _ = map(int, stats[label])
    result = {'whole': False, 'holes': []}
    if min(width, height) < POLICY['minimum_seed_length_in_pens'] * pen:
        return result
    radius = max(1, math.ceil(pen))
    audit['local_raster_pixels'] += (width+2*radius)*(height+2*radius)*(1 if hatched or pen < 3 else 2)
    if audit['local_raster_pixels'] > POLICY['maximum_local_raster_pixels']:
        audit['limits'].append('component_raster_budget')
        return {'whole': True, 'holes': []}
    mask = np.where(labels[y:y+height, x:x+width] == label, 255, 0).astype(np.uint8)
    # A scan gap may open a frame cell. Closing is an analysis-only ambiguity
    # probe at the measured pen scale, never source reconstruction or PART ink.
    mask = cv2.copyMakeBorder(mask, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((2*radius+1, 2*radius+1), np.uint8)
    variants = [cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)]
    if not hatched and pen >= 3:
        size = max(3, int(math.floor(pen)) | 1)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
        variants.append(cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0))
    contours, hierarchy = [], []
    for variant in variants:
        found, tree = cv2.findContours(variant, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        offset = len(contours)
        contours.extend(found)
        if tree is not None:
            hierarchy.extend([int(h[3])+offset if h[3] >= 0 else -1 for h in tree[0]])
    if len(contours) > POLICY['maximum_objects']:
        audit['limits'].append('component_contour_budget')
        return {'whole': True, 'holes': []}
    for i, contour in enumerate(contours):
        vertices = cv2.approxPolyDP(contour, pen+math.sqrt(.5), True).reshape(-1, 2)
        if len(vertices) != 4 or not cv2.isContourConvex(vertices):
            continue
        edges = np.roll(vertices, -1, axis=0)-vertices
        lengths = np.linalg.norm(edges, axis=1)
        if min(lengths) < POLICY['minimum_seed_length_in_pens']*pen:
            continue
        units = edges/lengths[:, None]
        if np.max(abs(np.sum(units*np.roll(units, 1, axis=0), axis=1))) > math.sin(math.radians(POLICY['direction_tolerance_deg'])):
            continue
        if hierarchy[i] < 0:
            result['whole'] = True
        else:
            result['holes'].append((vertices+[x-radius, y-radius]).tolist())
    return result


def material_hatch_components(lines, ink, blocked, labels, audit):
    """Three independent transverse source strokes joined to two thicker rails.

    Free-ended thin marks attached to a table cannot supply the second rail.
    This is local positive context, not a HATCH CAD object or a page-level role.
    """
    groups = {}; cosine = math.cos(math.radians(POLICY['direction_tolerance_deg']))
    for thin in lines:
        ends = thin['source_ends']; choices = [[], []]
        for thick in lines:
            audit['pair_checks'] += 1
            if audit['pair_checks'] > POLICY['maximum_pair_checks']:
                audit['limits'].append('hatch_pair_budget'); return set(), []
            if (thin['source_ids'] & thick['source_ids'] or thick['pen'] < POLICY['minimum_pen_ratio']*thin['upper_pen']
                    or abs(float(thin['u']@thick['u'])) > math.cos(math.radians(20))):
                continue
            for side, point in enumerate(ends):
                delta = thick['ends'][1]-thick['ends'][0]
                t = float(np.clip((point-thick['ends'][0])@delta/(delta@delta), 0, 1))
                projected = thick['ends'][0]+t*delta
                near = thick['q'][np.argmin(np.linalg.norm(thick['q']-projected, axis=1))]
                if np.linalg.norm(point-near) > thick['pen']+math.sqrt(.5):
                    continue
                contact = connected(ink, blocked, point, near, thick['pen'], audit)
                if contact:
                    choices[side].append(thick['id'])
        for a in choices[0]:
            for b in choices[1]:
                if a == b:
                    continue
                key = tuple(sorted((a, b)))
                groups.setdefault(key, []).append(thin)
    accepted, evidence = set(), []
    for rails, candidates in sorted(groups.items()):
        for seed in candidates:
            parallel = [r for r in candidates if abs(float(r['u']@seed['u'])) >= cosine]
            chosen, used = [], set()
            for r in sorted(parallel, key=lambda r:r['id']):
                if r['source_ids'] & used:
                    continue
                chosen.append(r); used |= r['source_ids']
            if len(chosen) < POLICY['minimum_thin_reference_sources']:
                continue
            current = set()
            for r in chosen:
                current |= set(map(int, labels[r['q'][:, 1], r['q'][:, 0]]))-{0}
            accepted |= current
            evidence.append({'rail_object_ids': list(rails), 'thin_object_ids': [r['id'] for r in chosen],
                             'component_ids': sorted(current), 'both_source_contacts_required': True})
            break
    return accepted, evidence


def analyze(packet, *, authority, protected=None):
    gray, roles = packet['gray'], packet['roles']
    checked_gray(gray)
    if gray.size > POLICY['maximum_pixels'] or packet.get('input_sha256') != sha(gray):
        raise ValueError('Bounded, source-bound raster required')
    source_inventory = inventory(packet)
    owner, ambiguous = packet['source_owner'], packet['ambiguous_owner_mask']
    if (roles.dtype != np.uint8 or roles.shape != gray.shape or owner.shape != gray.shape or owner.dtype.kind not in 'ui'
            or ambiguous.shape != gray.shape or ambiguous.dtype != np.uint8
            or owner.min(initial=0) < 0 or owner.max(initial=0) > len(packet['graph']['proposals'])
            or np.any((ambiguous > 0) & (owner != 0)) or not np.isin(roles, [*range(10), 255]).all()):
        raise ValueError('Aligned valid role/source-owner arrays required')
    protected = np.zeros(gray.shape, bool) if protected is None else protected
    if protected.shape != gray.shape or protected.dtype != bool:
        raise ValueError('Aligned boolean protection required')
    allowed = check_authority(gray, roles, authority)
    arrays = (gray, roles, owner, ambiguous, protected)
    snapshots = [sha(a) for a in arrays]
    before = object_sha([packet['geometry'], packet['graph'], packet['heads']])
    blocked = protected.copy() | (roles == 6)
    for head in packet['heads']:
        p = np.asarray(head.get('polygon_px'), float)
        if p.shape != (3, 2) or not np.isfinite(p).all():
            raise ValueError('Finite observed arrow polygon required')
        mask = np.zeros_like(gray); cv2.fillPoly(mask, [np.rint(p).astype(np.int32)], 255)
        blocked |= mask > 0
    assembly = assemble(gray, packet['geometry'], roles=roles, protected=blocked)
    objects, probe_audit = probes(assembly, packet['geometry'], maximum_segments=POLICY['maximum_objects'])
    audit = {'limits': [], 'rejections': {}, 'profile_samples': 0, 'pair_checks': 0, 'local_raster_pixels': 0,
             'component_distance_samples': 0,
             'source_probe_search': probe_audit}
    if packet.get('fresh_source_extraction_complete', True) is not True:
        audit['limits'].append('source_extraction_incomplete')
    if not assembly['search_complete'] or not probe_audit['complete'] or len(objects) > POLICY['maximum_objects']:
        audit['limits'].append('source_object_search')
    ink = foreground(gray)
    work = {'ink': ink, 'distance': cv2.distanceTransform(ink, cv2.DIST_L2, 5), 'samples': 0}
    junctions = junction_distance(packet['geometry'], gray.shape)
    records = []
    rejected = Counter()
    for obj in objects if not audit['limits'] else []:
        if obj['kind'] not in ('line', 'arc', 'circle'):
            continue
        q = np.asarray(obj['support_points'], int)
        length = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
        profile = (line_profile(obj, q, gray, work, junctions) if obj['kind'] == 'line'
                   else curve_profile(obj, q, gray, work, junctions))
        if work['samples'] > POLICY['maximum_profile_samples']:
            audit['limits'].append('profile_budget'); break
        if profile is None:
            rejected['profile_unresolved'] += 1; continue
        low, high = map(float, np.quantile(profile['sections_px'], [.25, .95]))
        if low < 1. or high > POLICY['maximum_profile_ratio'] * low or length < POLICY['minimum_reference_length_in_pens'] * low:
            rejected['unstable_or_short_profile'] += 1; continue
        row = {'id': obj['id'], 'object': obj, 'q': q, 'length': length, 'pen': low, 'upper_pen': high,
               'profile': profile, 'known': set(map(int, roles[q[:, 1], q[:, 0]])) - {255},
               'source_ids': {m['source_observation_id'] for m in obj['source_members']}}
        if obj['kind'] == 'line':
            ends = np.asarray(obj['points_px'], float)
            norm = float(np.linalg.norm(ends[1] - ends[0]))
            if not norm:
                continue
            row.update(ends=ends, u=(ends[1] - ends[0]) / norm,
                       source_ends=q[np.argmin(np.linalg.norm(q[:, None] - ends[None, :], axis=2), axis=0)])
        records.append(row)
    lines = [r for r in records if r['object']['kind'] == 'line']
    if len(lines) > POLICY['maximum_lines']:
        audit['limits'].append('direction_budget')
    witnesses, selected = [], set()
    _, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    hatch_labels, hatch_evidence = material_hatch_components(lines, ink, blocked, component_labels, audit) if not audit['limits'] else (set(), [])
    ambiguous_components = {}
    for row in lines if not audit['limits'] else []:
        labels = set(map(int, component_labels[row['q'][:, 1], row['q'][:, 0]])) - {0}
        row['rectangular_ambiguity'] = False
        for label in labels:
            key = (label, row['pen'])
            if key not in ambiguous_components:
                ambiguous_components[key] = rectangular_component(label, row['pen'], component_labels, component_stats, audit, hatched=label in hatch_labels)
            context = ambiguous_components[key]
            row['rectangular_ambiguity'] |= context['whole']
            for vertices in context['holes']:
                v = np.asarray(vertices, float)
                clearance = row['pen']+math.sqrt(.5)
                if (row['q'].max(0) < v.min(0)-clearance).any() or (row['q'].min(0) > v.max(0)+clearance).any():
                    continue
                audit['component_distance_samples'] += len(row['q'])
                if audit['component_distance_samples'] > POLICY['maximum_component_distance_samples']:
                    audit['limits'].append('component_distance_budget'); break
                distances = distance_to_polyline(row['q'], np.r_[v, v[:1]])
                if np.count_nonzero(distances <= clearance) >= POLICY['minimum_branch_length_in_pens']*row['pen']:
                    row['rectangular_ambiguity'] = True
                    break
        if row['rectangular_ambiguity']:
            rejected['ambiguous_rectangular_component'] += 1
        if audit['limits']:
            break

    def references(branches):
        points = np.concatenate([r['q'] for r in branches])
        margin = max(r['length'] for r in branches) / 2
        lo, hi = points.min(0) - margin, points.max(0) + margin
        used = set().union(*(r['source_ids'] for r in branches))
        refs = []
        for r in sorted(lines, key=lambda r: r['id']):
            audit['pair_checks'] += 1
            if audit['pair_checks'] > POLICY['maximum_pair_checks']:
                audit['limits'].append('pair_budget'); break
            center = r['ends'].mean(0)
            if (r['source_ids'] & used or (center < lo).any() or (center > hi).any()
                    or r['upper_pen'] * POLICY['minimum_pen_ratio'] > min(b['pen'] for b in branches)):
                continue
            refs.append(r); used |= r['source_ids']
        return refs, [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]

    for r in records if not audit['limits'] else []:
        obj = r['object']
        if r['known'] or obj['kind'] not in ('arc', 'circle'):
            continue
        if (obj['radius_px'] < POLICY['minimum_radius_in_pens'] * r['upper_pen']
                or obj.get('sweep_angle_deg', 360.) < POLICY['minimum_curve_sweep_deg']
                or r['length'] < POLICY['minimum_seed_length_in_pens'] * r['pen']):
            rejected['short_or_tight_curve'] += 1; continue
        refs, box = references([r])
        if len(refs) < POLICY['minimum_thin_reference_sources']:
            rejected['insufficient_local_thin_evidence'] += 1; continue
        selected.add(r['id'])
        witnesses.append({'kind': 'curved_boundary', 'object_ids': [r['id']], 'reference_object_ids': [x['id'] for x in refs],
                          'profiles': [r['profile']], 'context_bbox_px': box})

    branches = ([r for r in lines if not r['known'] and not r['rectangular_ambiguity']
                 and r['length'] >= POLICY['minimum_branch_length_in_pens'] * r['pen']]
                if not audit['limits'] else [])
    for i, a in enumerate(branches) if not audit['limits'] else []:
        for b in branches[i + 1:]:
            audit['pair_checks'] += 1
            if audit['pair_checks'] > POLICY['maximum_pair_checks']:
                audit['limits'].append('pair_budget'); break
            if max(a['length']/a['pen'], b['length']/b['pen']) < POLICY['minimum_seed_length_in_pens']:
                continue
            angle = math.degrees(math.acos(float(np.clip(abs(a['u'] @ b['u']), 0, 1))))
            if angle < POLICY['minimum_orthogonal_angle_deg'] or max(a['pen'], b['pen']) / min(a['pen'], b['pen']) > POLICY['maximum_pair_pen_ratio']:
                continue
            distances = np.linalg.norm(a['source_ends'][:, None] - b['source_ends'][None, :], axis=2)
            ai, bi = np.unravel_index(int(np.argmin(distances)), (2, 2))
            p, q = a['source_ends'][ai], b['source_ends'][bi]
            if distances[ai, bi] > a['pen'] + b['pen'] + 2 * math.sqrt(.5):
                continue
            contact = connected(ink, blocked, p, q, max(a['pen'], b['pen']), audit)
            if contact is None:
                rejected['disconnected_corner'] += 1; continue
            center = (p + q) / 2
            opposites = []
            cosine = math.cos(math.radians(POLICY['direction_tolerance_deg']))
            for branch, near in ((a, ai), (b, bi)):
                direction = branch['ends'][1-near] - branch['ends'][near]
                direction /= np.linalg.norm(direction)
                found = False
                for other in lines:
                    audit['pair_checks'] += 1
                    if audit['pair_checks'] > POLICY['maximum_pair_checks']:
                        audit['limits'].append('pair_budget'); break
                    if other['id'] in (a['id'], b['id']):
                        continue
                    close = int(np.argmin(np.linalg.norm(other['source_ends'] - center, axis=1)))
                    if (np.linalg.norm(other['source_ends'][close]-center) > 4 * branch['pen'] + 2 * math.sqrt(.5)
                            or other['length'] < POLICY['minimum_return_length_in_pens'] * other['pen']
                            or not 1/POLICY['maximum_pair_pen_ratio'] <= other['pen']/branch['pen'] <= POLICY['maximum_pair_pen_ratio']):
                        continue
                    vector = other['ends'][1-close] - other['ends'][close]
                    if float(vector @ direction / np.linalg.norm(vector)) <= -cosine:
                        found = True; break
                opposites.append(found)
            if all(opposites):
                rejected['two_thick_through_paths'] += 1; continue
            refs, box = references([a, b])
            if len(refs) < POLICY['minimum_thin_reference_sources']:
                rejected['insufficient_local_thin_evidence'] += 1; continue
            selected |= {a['id'], b['id']}
            witnesses.append({'kind': 'orthogonal_boundary', 'object_ids': [a['id'], b['id']], 'angle_deg': angle,
                              'contact': contact, 'profiles': [a['profile'], b['profile']],
                              'reference_object_ids': [x['id'] for x in refs], 'context_bbox_px': box})
        if audit['limits']:
            break

    proposal = np.zeros_like(gray); object_evidence = []
    eligible = allowed & (roles == 255) & ~blocked & (owner > 0) & (ambiguous == 0) & (ink > 0)
    for row in records if not audit['limits'] else []:
        if row['id'] not in selected:
            continue
        obj, q = row['object'], row['q']
        numbers = set()
        for member in obj['source_members']:
            start, end = member['source_index_interval']
            numbers |= {n for n, interval in enumerate(packet['graph']['proposals'], 1)
                        if interval['parent_primitive_id'] == member['candidate_id']
                        and max(start, interval['source_index_interval'][0]) < min(end, interval['source_index_interval'][1])}
        radius = row['pen']/2 + math.sqrt(.5)
        pad = math.ceil(radius) + 1
        lo, hi = np.maximum(0, q.min(0)-pad), np.minimum(gray.shape[::-1], q.max(0)+pad+1)
        audit['local_raster_pixels'] += int(np.prod(hi-lo))
        if audit['local_raster_pixels'] > POLICY['maximum_local_raster_pixels']:
            audit['limits'].append('proposal_raster_budget'); break
        region = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
        seed = np.zeros(tuple((hi-lo)[::-1]), np.uint8)
        seed[q[:, 1]-lo[1], q[:, 0]-lo[0]] = 255
        distance = cv2.distanceTransform(255-seed, cv2.DIST_L2, 5)
        accepted = eligible[region] & np.isin(owner[region], list(numbers)) & (distance <= radius)
        proposal[region][accepted] = 255
        object_evidence.append({'id': row['id'], 'kind': obj['kind'], 'source_members': obj['source_members'],
                                'source_owner_numbers': sorted(numbers), 'observed_pen_px': row['pen'],
                                'proposed_pixels': int(accepted.sum())})
    if audit['limits']:
        proposal[:] = 0; witnesses = []; object_evidence = []
    audit.update(profile_samples=work['samples'], measured_line_count=len(lines), measured_object_count=len(records), rejections=dict(rejected))
    if snapshots != [sha(a) for a in arrays] or before != object_sha([packet['geometry'], packet['graph'], packet['heads']]):
        raise ValueError('Material witnesses changed source observations')
    return {'mask': proposal, 'complete': not audit['limits'], 'source_inventory': source_inventory,
            'material_hatch_context': hatch_evidence,
            'objects': object_evidence, 'witnesses': witnesses, 'audit': audit,
            'source_sha256': sha(gray), 'source_graph_sha256': object_sha(packet['graph']),
            'source_owner_sha256': sha(owner), 'raw_positive_pixels': int(np.count_nonzero(proposal)),
            'source_immutable': True, 'rectangular_context_scope': 'pen_scaled_analysis_closing_for_ambiguity_only',
            'policy': deepcopy(POLICY)}
