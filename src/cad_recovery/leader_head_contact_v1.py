"""Propose source-backed leader contacts without overwriting accepted roles.

A verified head observation supplements the old taper witness. A one-head stroke is still a
leader-role proposal: missing text or another undetected head cannot establish
the semantic type of the entire annotation.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import leader_contact_evidence_v1 as contact
from . import dimension_association_v3 as association
from .grid_role_revision_v1 import sha as pixel_sha
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import verify as verify_head
from .stroke_roles_v1 import checked_gray, foreground


POLICY = {'version': 'leader-head-contact/1.0', 'maximum_heads': 256,
          'maximum_pixels': 4_194_304,
          'maximum_tracks': 512, 'maximum_source_points': 500_000,
          'maximum_mask_pixels': 64_000_000, 'maximum_head_checks': 4096,
          'direction_degrees': 8., 'minimum_taper_head_coverage': .75,
          'leader_role': 4, 'unknown_role': 255,
          'semantic_role_confirmed': False, 'production_admitted': False}


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _mask(value, shape, label):
    if value is None:
        return np.zeros(shape, bool)
    value = np.asarray(value)
    if value.shape != shape or value.dtype not in (np.bool_, np.uint8):
        raise ValueError(f'Aligned boolean or uint8 {label} required')
    return value > 0


def _graph(graph, shape):
    proposals = {p['id']: p for p in graph['proposals']}
    tracks = {p['id']: p for p in graph['tracks']}
    if len(proposals) != len(graph['proposals']) or len(tracks) != len(graph['tracks']):
        raise ValueError('Unique graph observation and track IDs required')
    total = 0
    for proposal in proposals.values():
        points = np.asarray(proposal['points_px'], float)
        width = float(proposal.get('width_px', 1.))
        if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 1
                or not np.isfinite(points).all() or (points < 0).any()
                or (points >= np.array(shape[::-1])).any()):
            raise ValueError('Graph source support must lie on the source raster')
        if not math.isfinite(width) or not 0 < width <= max(shape):
            raise ValueError('Bounded finite positive observed width required')
        total += len(points)
    lines = []
    for track in tracks.values():
        points = np.asarray(track['points'], float)
        if (points.shape != (2, 2) or not np.isfinite(points).all()
                or np.linalg.norm(points[1]-points[0]) <= 0
                or (points < 0).any() or (points >= np.array(shape[::-1])).any()
                or not track['interval_ids'] or any(i not in proposals for i in track['interval_ids'])):
            raise ValueError('Bounded track with actual source interval IDs required')
        widths = [float(proposals[i].get('width_px', 1.)) for i in track['interval_ids']]
        if not np.isfinite(widths).all() or min(widths) <= 0:
            raise ValueError('Finite positive observed stroke widths required')
        lines.append({'id': track['id'], 'kind': 'line', 'points_px': points.tolist(),
                      'uncertainty_px': .5, 'width_px': float(np.median(widths)),
                      'source_members': deepcopy(track['interval_ids'])})
    return proposals, tracks, lines, total


def _band(shape, a, b, radius):
    a, b = np.asarray(a), np.asarray(b)
    delta = b-a
    length = float(np.linalg.norm(delta))
    if length <= 0:
        return np.zeros(shape, bool)
    unit = delta/length
    y, x = np.ogrid[:shape[0], :shape[1]]
    dx, dy = x-a[0], y-a[1]
    along = dx*unit[0]+dy*unit[1]
    return ((along >= 0) & (along <= length)
            & (abs(dx*unit[1]-dy*unit[0]) <= radius))


def _other_source_mask(shape, proposals, members):
    mask = np.zeros(shape, np.uint8)
    for identity, proposal in proposals.items():
        if identity in members:
            continue
        points = np.rint(proposal['points_px']).astype(np.int32)
        width = max(1, math.ceil(float(proposal.get('width_px', 1.))))
        if len(points) > 1:
            cv2.polylines(mask, [points], False, 255, width)
        else:
            cv2.circle(mask, tuple(points[0]), max(0, width//2), 255, -1)
    return mask > 0


def _matches_witness(head, witness):
    if witness['shaft_track_id'] not in head['evidence'].get('geometry_ids', []):
        return False
    toward = np.asarray(witness['direction_toward_taper'])
    if np.dot(head['direction'], toward) < math.cos(math.radians(POLICY['direction_degrees'])):
        return False
    # The freshly observed taper must belong to this head envelope, rather than a
    # different head anywhere on the same long carrier.
    origin = np.asarray(witness['shaft_points_px'][0])
    sections = np.asarray(witness['taper']['source_cross_sections'])
    centers = origin-sections[:, :1]*toward
    polygon = np.asarray(head['polygon_px'], np.float32)
    hits = [cv2.pointPolygonTest(polygon, tuple(point), True) >= -witness['pen_px']
            for point in centers]
    return bool(np.mean(hits) >= POLICY['minimum_taper_head_coverage'])


def analyze(source_image, *, graph, heads, roles, shared=None, protected=None):
    source_image = Path(source_image)
    source_hash = _digest(source_image)
    with Image.open(source_image) as image:
        if image.width*image.height > POLICY['maximum_pixels']:
            raise ValueError('Leader contact raster exceeds bounded image size')
        if 'A' in image.getbands() or 'transparency' in image.info:
            canvas = Image.new('RGBA', image.size, 'white')
            canvas.alpha_composite(image.convert('RGBA'))
            gray = np.array(canvas.convert('L'))
        else:
            gray = np.array(image.convert('L'))
    checked_gray(gray)
    if (heads.get('source_sha256') != source_hash
            or heads.get('processed_size_px') != list(gray.shape[::-1])):
        raise ValueError('Verified heads belong to another source raster or frame')
    if not isinstance(roles, np.ndarray) or roles.dtype != np.uint8 or roles.shape != gray.shape:
        raise ValueError('Aligned uint8 input role map required')
    if not np.isin(roles, [*range(10), 255]).all():
        raise ValueError('Unknown role code')
    shared_mask = _mask(shared, gray.shape, 'shared-contact mask')
    protected_mask = _mask(protected, gray.shape, 'protected mask')
    proposals, tracks, lines, point_count = _graph(graph, gray.shape)
    report = {'schema_version': POLICY['version'], 'source_sha256': source_hash,
              'source_pixel_sha256': pixel_sha(gray), 'processed_size_px': list(gray.shape[::-1]),
              'graph_sha256': object_sha(graph), 'heads_sha256': object_sha(heads),
              'records': [], 'limits': [], 'mask_pixels': 0, 'head_checks': 0,
              'semantic_role_confirmed': False, 'production_admitted': False,
              'CAD_promotions': 0, 'policy': deepcopy(POLICY)}
    eligible = np.zeros_like(gray, bool)
    counts = np.zeros_like(gray, np.uint16)
    verified = heads.get('verified_candidates', [])
    common = {'source_sha256': source_hash, 'processed_size_px': list(gray.shape[::-1])}
    association._validate(source_image, {**common, 'candidates': lines},
                          {**common, 'lines': [], 'notations': []}, {**common, 'candidates': verified})
    if graph.get('search_incomplete') or heads.get('search_complete') is not True:
        report['limits'].append('upstream_observation_incomplete')
    if len(tracks) > POLICY['maximum_tracks'] or point_count > POLICY['maximum_source_points']:
        report['limits'].append('source_graph_budget')
    if len(verified) > POLICY['maximum_heads']:
        report['limits'].append('head_budget')
    witnesses = contact.observe(gray, graph) if not report['limits'] else None
    if witnesses is not None and not witnesses['complete']:
        report['limits'].append('taper_witness_incomplete')
    report['fresh_taper_witnesses'] = witnesses
    ink = foreground(gray) > 0
    for witness in witnesses['witnesses'] if witnesses and not report['limits'] else []:
        row = {'shaft_track_id': witness['shaft_track_id'], 'status': 'unresolved',
               'candidate_head_ids': [], 'semantic_role_confirmed': False}
        matches = []
        for head in verified:
            report['head_checks'] += 1
            if report['head_checks'] > POLICY['maximum_head_checks']:
                report['limits'].append('head_check_budget')
                break
            if _matches_witness(head, witness):
                native, reason = verify_head(head, gray, lines, [], witness['pen_px'])
                if native is not None and witness['shaft_track_id'] in native['evidence']['geometry_ids']:
                    matches.append(native)
        row['candidate_head_ids'] = [h['id'] for h in matches]
        if report['limits']:
            break
        if len(matches) != 1:
            row['reason'] = 'no_verified_head_for_taper' if not matches else 'ambiguous_head_observations'
            report['records'].append(row)
            continue
        head = matches[0]
        continuations = witness['observed_continuations']
        if not continuations:
            row['reason'] = 'no_observed_downstream_continuation'
            report['records'].append(row)
            continue
        track_ids = {witness['shaft_track_id'], *(c['track_id'] for c in continuations)}
        if any(other['id'] != head['id'] and track_ids.intersection(other['evidence'].get('geometry_ids', []))
               and np.dot(other['direction'], head['direction']) < -.9 for other in verified):
            row['reason'] = 'opposed_head_dimension_context'
            report['records'].append(row)
            continue
        source_members = set(witness['source_members'])
        for continuation in continuations:
            source_members.update(continuation['source_members'])
        report['mask_pixels'] += int(gray.size)*3
        if report['mask_pixels'] > POLICY['maximum_mask_pixels']:
            report['limits'].append('mask_budget')
            break
        a, b = np.asarray(witness['shaft_points_px'])
        unit = (b-a)/np.linalg.norm(b-a)
        end = max([b, *(np.asarray(p) for c in continuations for p in c['points_px'])],
                  key=lambda point: float((point-a)@unit))
        candidate = _band(gray.shape, a, end, witness['pen_px']/2+math.sqrt(.5)) & ink
        competing = _other_source_mask(gray.shape, proposals, source_members)
        exclusion = shared_mask | protected_mask | (roles != 255) | competing
        proposal = candidate & ~exclusion
        counts += proposal
        eligible |= proposal
        row.update(status='eligible_role_proposal', reason='verified_head_observation_and_connected_source_intervals',
                   head_id=head['id'], head_observation_proof=deepcopy(head['evidence']),
                   head_base_position_confirmed=head['evidence'].get('base_position_precisely_confirmed') is True,
                   triangle_geometry_approved=head['evidence'].get('triangle_geometry_approved') is True,
                   source_members=sorted(source_members), track_ids=sorted(track_ids),
                   shaft_and_continuation_range_px=[a.tolist(), end.tolist()],
                   corridor_radius_px=witness['pen_px']/2+math.sqrt(.5),
                   candidate_ink_pixels=int(candidate.sum()), eligible_pixels=int(proposal.sum()),
                   excluded_shared_or_other_source_pixels=int((candidate & (shared_mask | competing)).sum()),
                   unchanged_known_or_protected_pixels=int((candidate & ((roles != 255) | protected_mask)).sum()))
        report['records'].append(row)
    ambiguous = counts > 1
    eligible &= ~ambiguous
    if report['limits']:
        eligible[:] = False
    if _digest(source_image) != source_hash:
        raise ValueError('Source changed during leader contact observation')
    report.update(complete=not report['limits'], eligible_pixels=int(eligible.sum()),
                  ambiguous_proposal_pixels=int(ambiguous.sum()), input_roles_preserved=True,
                  shared_contacts_preserved=True, protected_pixels_preserved=True,
                  head_taper_and_shaft_observed=any(r.get('head_id') for r in report['records']),
                  head_base_position_confirmed=bool(report['records']) and all(
                      r.get('head_base_position_confirmed', False) for r in report['records']))
    proposed = roles.copy()
    proposed[eligible] = POLICY['leader_role']
    return {'eligible_mask': eligible, 'proposed_roles': proposed, 'report': report}
