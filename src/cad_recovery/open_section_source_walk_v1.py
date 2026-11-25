"""Local open section evidence; never close paths or create CAD geometry."""
from copy import deepcopy
import math

import cv2
import numpy as np
from PIL import Image

from .grid_role_revision_v1 import check_authority, sha
from .part_admission_v1 import object_sha
from .part_contact_v1 import bind_source
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'open-section-source-walk/1.0', 'maximum_pixels': 4194304,
          'maximum_intervals': 4096, 'maximum_lines': 512,
          'maximum_source_points': 2000000, 'maximum_checks': 250000,
          'maximum_raster_work': 64000000, 'minimum_hatches': 4,
          'parallel_tolerance_deg': 8., 'minimum_crossing_angle_deg': 20.,
          'minimum_pen_ratio': 1.5, 'maximum_width_radius_ratio': 2.,
          'width_core_fraction': .5, 'line_support': 'connected_observed_source_walk',
          'maximum_spacing_ratio': 1.67, 'source_pixels_erased': 0,
          'geometry_modified': False, 'production_admitted': False,
          'physical_geometry_approved': False}
PIXEL_RADIUS = math.sqrt(.5)


def checked_original(gray, original, proof):
    """Only whole-page AREA/BOX provenance; crops cannot borrow page authority."""
    checked_gray(original)
    if proof is None or proof.get('GT_used') is not False:
        raise ValueError('Checked full original provenance required')
    if proof['input_sha256'] != sha(gray) or proof['source_sha256'] != sha(original):
        raise ValueError('Original source pixels differ')
    matrix = np.asarray(proof['matrix'], float)
    sx, sy = original.shape[1]/gray.shape[1], original.shape[0]/gray.shape[0]
    expected = np.array([[sx, 0., (sx-1)/2], [0., sy, (sy-1)/2], [0., 0., 1.]])
    if matrix.shape != (3, 3) or not np.array_equal(matrix, expected):
        raise ValueError('Whole original resize matrix required; crop not supported')
    if proof['resize_method'] == 'CV_AREA':
        resized = cv2.resize(original, gray.shape[::-1], interpolation=cv2.INTER_AREA)
    elif proof['resize_method'] == 'PIL_BOX':
        resized = np.array(Image.fromarray(original).resize(gray.shape[::-1], Image.Resampling.BOX))
    else:
        raise ValueError('Exact AREA/BOX original provenance required')
    if not np.array_equal(gray, resized):
        raise ValueError('Original source resize is not exact')
    canonical = bind_source(gray, original, matrix, proof.get('source_pin', {}), resize_method=proof['resize_method'])
    if proof != canonical:
        raise ValueError('Rebound original source provenance')


def _observed_lines(gray, graph, ink, dt, blocked, audit):
    proposals = graph['proposals']
    lookup = {p['id']: n for n, p in enumerate(proposals, 1)}
    if len(lookup) != len(proposals):
        raise ValueError('Unique current source intervals required')
    points = {}
    for n, p in enumerate(proposals, 1):
        q = np.asarray(p['points_px'], float)
        if q.ndim != 2 or q.shape[1] != 2 or not len(q) or not np.isfinite(q).all():
            raise ValueError('Finite current source points required')
        if not np.array_equal(q, np.rint(q)):
            raise ValueError('Source points must be observed pixel centres')
        q = q.astype(int)
        if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any():
            raise ValueError('Source points outside current raster')
        audit['source_points'] += len(q)
        if audit['source_points'] > POLICY['maximum_source_points']:
            audit['limits'].append('source_point_budget'); return []
        points[n] = q
    records = list(graph.get('tracks', []))
    tracked = {i for t in records for i in t['interval_ids']}
    records += [{'id': p['id'], 'points': [p['points_px'][0], p['points_px'][-1]],
                 'interval_ids': [p['id']]} for p in proposals
                if p.get('linear_context') and p['id'] not in tracked]
    if len(records) > POLICY['maximum_lines']:
        audit['limits'].append('line_budget'); return []
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Unique observed line IDs required')
    lines = []
    for r in records:
        if not r['interval_ids'] or not set(r['interval_ids']) <= lookup.keys():
            raise ValueError('Track refers to foreign source intervals')
        ids = sorted({lookup[i] for i in r['interval_ids']})
        ends = np.asarray(r['points'], float)
        if ends.shape != (2, 2) or not np.isfinite(ends).all():
            raise ValueError('Finite observed track endpoints required')
        q = np.concatenate([points[n] for n in ids])
        vector = ends[1]-ends[0]; length = float(np.linalg.norm(vector))
        if length <= 0 or len(q) < 3: continue
        direction = vector/length; normal = np.array([-direction[1], direction[0]])
        values = dt[q[:, 1], q[:, 0]]
        if np.any(values <= 0): continue
        relative = q-ends[0]; along = relative@direction
        trim = (1-POLICY['width_core_fraction'])/2
        core = values[(along >= trim*length) & (along <= (1-trim)*length)]
        if len(core) < 3: continue
        low, high = np.quantile(core, [.25, .95])
        if high > POLICY['maximum_width_radius_ratio']*low: continue
        pen = max(1., float(2*low-1)); radius = float(low)
        if any(np.linalg.norm(q-endpoint, axis=1).min() > PIXEL_RADIUS for endpoint in ends):
            continue
        if (np.max(abs(relative@normal)) > radius+PIXEL_RADIUS or
                along.min() < -radius-PIXEL_RADIUS or along.max() > length+radius+PIXEL_RADIUS):
            continue
        # The chord is checked against actual source ink; tracks cannot bridge
        # an unobserved break merely because their endpoints are collinear.
        rounded = np.rint(ends).astype(int)
        if ((rounded < 0).any() or (rounded[:, 0] >= gray.shape[1]).any() or
                (rounded[:, 1] >= gray.shape[0]).any()): continue
        lo, hi = rounded.min(0), rounded.max(0)+1
        audit['raster_work'] += int(np.prod(hi-lo))
        if audit['raster_work'] > POLICY['maximum_raster_work']:
            audit['limits'].append('raster_budget'); return []
        chord = np.zeros(tuple((hi-lo)[::-1]), np.uint8)
        cv2.line(chord, tuple(rounded[0]-lo), tuple(rounded[1]-lo), 255, 1, cv2.LINE_8)
        yy, xx = np.nonzero(chord); samples = np.c_[xx+lo[0], yy+lo[1]]
        # A cropped digital line can have a different Bresenham phase from
        # the chord between its observed endpoints. Validate the actual owned
        # source walk rather than demanding that every ideal chord pixel is ink.
        qlo, qhi = q.min(0), q.max(0)+1
        audit['raster_work'] += int(np.prod(qhi-qlo))
        if audit['raster_work'] > POLICY['maximum_raster_work']:
            audit['limits'].append('source_walk_raster_budget'); return []
        walk = np.zeros(tuple((qhi-qlo)[::-1]), np.uint8)
        walk[q[:, 1]-qlo[1], q[:, 0]-qlo[0]] = 1
        if cv2.connectedComponents(walk, connectivity=8)[0] != 2: continue
        mismatches = int(np.count_nonzero(ink[samples[:, 1], samples[:, 0]] == 0))
        audit['digital_chord_phase_mismatch_pixels'] = audit.get('digital_chord_phase_mismatch_pixels', 0)+mismatches
        if np.any(blocked[q[:, 1], q[:, 0]]): continue
        # Every declared member must actually be a straight source observation.
        if not all(proposals[n-1].get('linear_context') for n in ids): continue
        lines.append({'id': r['id'], 'owners': ids, 'ends': ends,
                      'u': direction, 'n': normal, 'length': length,
                      'pen': pen, 'radius': radius})
    return lines


def _termination(boundary, hatch, endpoint, ink, *, check_continuation=True):
    """One-sided observed endpoint with no source continuation beyond boundary."""
    point = hatch['ends'][endpoint]; other = hatch['ends'][1-endpoint]
    delta = point-boundary['ends'][0]; t = float(delta@boundary['u'])
    clearance = boundary['radius']+hatch['radius']+PIXEL_RADIUS
    if not clearance < t < boundary['length']-clearance: return None
    if abs(float(delta@boundary['n'])) > clearance: return None
    side_distance = float((other-boundary['ends'][0])@boundary['n'])
    if abs(side_distance) <= 2*clearance: return None
    contact = {'t': t, 'side': 1 if side_distance > 0 else -1}
    if not check_continuation: return contact
    inward = (other-point)/hatch['length']
    sine = abs(float(inward@boundary['n']))
    # Measure continuation past the entire boundary/hatch footprint, accounting
    # for crossing angle. No fixed pixel search tolerance is introduced.
    start = 2*clearance/sine
    ray = point-np.arange(math.ceil(start), math.ceil(start+3*hatch['pen'])+1)[:, None]*inward
    q = np.rint(ray).astype(int)
    if ((q < 0).any() or (q[:, 0] >= ink.shape[1]).any() or (q[:, 1] >= ink.shape[0]).any()): return None
    if np.any(ink[q[:, 1], q[:, 0]]): return None
    return contact


def analyze(gray, graph, raw_roles, source_owner, ambiguous_owner, heads, *,
            authority, original_gray=None, source_binding=None, page_negative=None):
    """Return unknown-only local proposals and traceable source witnesses."""
    checked_gray(gray)
    if (raw_roles.shape != gray.shape or raw_roles.dtype != np.uint8 or
            not np.isin(raw_roles, list(range(10))+[255]).all() or
            source_owner.shape != gray.shape or source_owner.dtype.kind not in 'ui' or
            source_owner.max(initial=0) > len(graph['proposals']) or source_owner.min(initial=0) < 0 or
            ambiguous_owner.shape != gray.shape or ambiguous_owner.dtype != np.uint8):
        raise ValueError('Aligned current roles/source owner domain required')
    if page_negative is None: page_negative = np.zeros_like(gray)
    if page_negative.shape != gray.shape or page_negative.dtype != np.uint8:
        raise ValueError('Aligned source page-negative mask required')
    allowed = check_authority(gray, raw_roles, authority)
    before = object_sha([graph, heads, source_binding])
    arrays = [gray, raw_roles, source_owner, ambiguous_owner, page_negative]
    if original_gray is not None: arrays.append(original_gray)
    snapshots = [sha(a) for a in arrays]
    audit = {'limits': [], 'source_points': 0, 'checks': 0, 'raster_work': 0}
    witnesses = []; mask = np.zeros_like(gray); reason = None
    if original_gray is None or source_binding is None: reason = 'full_original_unavailable'
    else: checked_original(gray, original_gray, source_binding)
    if len(graph['proposals']) > POLICY['maximum_intervals']: audit['limits'].append('interval_budget')
    if graph.get('search_incomplete') or graph.get('boundaries', {}).get('search_incomplete'):
        audit['limits'].append('incomplete_source_graph')
    blocked = np.isin(raw_roles, [2, 3, 4, 5, 6, 7, 8]) | (page_negative > 0) | ~allowed
    for head in heads:
        polygon = np.asarray(head['polygon_px'], float)
        if polygon.shape != (3, 2) or not np.isfinite(polygon).all():
            raise ValueError('Finite observed arrow head required')
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
    blocked |= mask > 0; mask[:] = 0
    frame_ids = {i for b in graph.get('boundaries', {}).get('boundaries', [])
                 if b.get('frame_hypothesis') for i in b['owners']}
    ink = foreground(gray); dt = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    lines = _observed_lines(gray, graph, ink, dt, blocked, audit) if not reason and not audit['limits'] else []
    cosine = math.cos(math.radians(POLICY['parallel_tolerance_deg']))
    sine = math.sin(math.radians(POLICY['minimum_crossing_angle_deg']))
    for i, a in enumerate(lines):
        for b in lines[i+1:]:
            audit['checks'] += 1
            if audit['checks'] > POLICY['maximum_checks']: audit['limits'].append('pair_budget'); break
            if abs(float(a['u']@b['u'])) > math.cos(math.radians(20)): continue
            ratio = a['pen']/b['pen']
            if not 1/1.67 <= ratio <= 1.67: continue
            if set(a['owners']) & set(b['owners']): continue
            owners = a['owners']+b['owners']
            if any(graph['proposals'][n-1]['id'] in frame_ids for n in owners): continue
            separation = np.linalg.norm(a['ends'][:, None]-b['ends'][None, :], axis=2)
            if separation.min() > a['radius']+b['radius']+PIXEL_RADIUS: continue
            hits = []; unsafe_termination = False
            for h in lines:
                audit['checks'] += 1
                if audit['checks'] > POLICY['maximum_checks']: audit['limits'].append('pair_budget'); break
                if set(h['owners']) & set(owners): continue
                if min(a['pen'], b['pen']) < POLICY['minimum_pen_ratio']*h['pen']: continue
                if min(abs(float(h['u']@a['n'])), abs(float(h['u']@b['n']))) < sine: continue
                for end in (0, 1):
                    ta = _termination(a, h, end, ink, check_continuation=False)
                    tb = _termination(b, h, 1-end, ink, check_continuation=False)
                    if ta and tb:
                        if _termination(a, h, end, ink) and _termination(b, h, 1-end, ink):
                            hits.append({'h': h, 'a': ta, 'b': tb})
                        else: unsafe_termination = True
                        break
            if audit['limits']: break
            # All supporting hatch strokes must agree, not a selected convenient
            # subset of a two-sided or nonparallel family.
            if unsafe_termination or len(hits) < POLICY['minimum_hatches']: continue
            if any(abs(float(hits[0]['h']['u']@hit['h']['u'])) < cosine for hit in hits): continue
            if any(len({hit[key]['side'] for hit in hits}) != 1 for key in ('a', 'b')): continue
            hnormal = hits[0]['h']['n']
            offsets = sorted(float(hit['h']['ends'].mean(0)@hnormal) for hit in hits)
            gaps = np.diff(offsets)
            if gaps.min() <= max(hit['h']['pen'] for hit in hits) or gaps.max()/gaps.min() > POLICY['maximum_spacing_ratio']:
                continue
            crossed = False
            for line, key in ((a, 'a'), (b, 'b')):
                lo, hi = min(hit[key]['t'] for hit in hits), max(hit[key]['t'] for hit in hits)
                for h in lines:
                    audit['checks'] += 1
                    if audit['checks'] > POLICY['maximum_checks']: audit['limits'].append('pair_budget'); break
                    if (set(h['owners']) & set(owners) or
                            abs(float(h['u']@hits[0]['h']['u'])) < cosine or
                            min(a['pen'], b['pen']) < POLICY['minimum_pen_ratio']*h['pen']): continue
                    signed = (h['ends']-line['ends'][0])@line['n']
                    clearance = line['radius']+h['radius']+PIXEL_RADIUS
                    if signed.min() < -clearance and signed.max() > clearance:
                        fraction = -signed[0]/(signed[1]-signed[0])
                        cross = h['ends'][0]+fraction*(h['ends'][1]-h['ends'][0])
                        if lo <= float((cross-line['ends'][0])@line['u']) <= hi: crossed = True
                if audit['limits']: break
            if audit['limits']: break
            if crossed: continue
            witness = {'boundary_track_ids': [a['id'], b['id']], 'source_owner_numbers': owners,
                       'hatch_track_ids': [hit['h']['id'] for hit in hits],
                       'boundary_pens_px': [a['pen'], b['pen']],
                       'hatch_pens_px': [hit['h']['pen'] for hit in hits],
                       'hatch_offsets_px': offsets, 'sides': [hits[0][k]['side'] for k in ('a', 'b')],
                       'local_spans_px': [], 'proposed_pixels': 0}
            for line, key in ((a, 'a'), (b, 'b')):
                lo, hi = min(hit[key]['t'] for hit in hits), max(hit[key]['t'] for hit in hits)
                witness['local_spans_px'].append([lo, hi])
                # No full-page temporary raster per pair. Only pixels owned by
                # this observed side and inside the supported local interval.
                yy, xx = np.nonzero(np.isin(source_owner, line['owners']))
                audit['raster_work'] += gray.size
                if audit['raster_work'] > POLICY['maximum_raster_work']: audit['limits'].append('raster_budget'); break
                xy = np.c_[xx, yy]; relative = xy-line['ends'][0]
                along = relative@line['u']; across = abs(relative@line['n'])
                valid = ((along > lo) & (along < hi) & (across <= line['radius']+PIXEL_RADIUS) &
                         (raw_roles[yy, xx] == 255) & ~blocked[yy, xx] &
                         (ink[yy, xx] > 0) & (ambiguous_owner[yy, xx] == 0))
                # Preserve each thin/thick contact footprint, including pixels
                # whose single-valued ownership happens to favour the boundary.
                for hit in hits:
                    h = hit['h']; normal_distance = abs((xy-h['ends'][0])@h['n'])
                    valid &= normal_distance > h['radius']+PIXEL_RADIUS
                mask[yy[valid], xx[valid]] = 255
                witness['proposed_pixels'] += int(valid.sum())
            witnesses.append(witness)
        if audit['limits']: break
    complete = reason is None and not audit['limits']
    if not complete: mask[:] = 0
    immutable = before == object_sha([graph, heads, source_binding]) and snapshots == [sha(a) for a in arrays]
    if not immutable: raise ValueError('Open-section witness mutated source observations')
    return {'mask': mask, 'witnesses': witnesses, 'audit': audit, 'complete': complete,
            'reason': reason, 'source_immutable': immutable, 'policy': deepcopy(POLICY),
            'source_sha256': sha(gray), 'graph_sha256': object_sha(graph),
            'source_owner_sha256': sha(source_owner), 'previous_PART_used_for_positive_evidence': False,
            'dimension_chain_used_for_positive_evidence': False, 'path_closed': False,
            'geometry_modified': False, 'CAD_semantics_approved': False}
