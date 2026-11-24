"""Independent context checks on added external-dimension hypotheses only."""
from collections import defaultdict
import heapq
import math
import numpy as np
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'open-reference-context/1.0', 'maximum_source_points': 2000000,
          'maximum_component_edge_visits': 500000, 'maximum_parallel_pair_checks': 250000,
          'maximum_frame_pair_checks': 250000,
          'minimum_wide_track_length_fraction': .8, 'parallel_angle_deg': 8.,
          'minimum_repeat_offsets': 4, 'maximum_gap_CV': .25,
          'minimum_span_overlap': .85, 'maximum_span_length_ratio': 1.25,
          'changes_pixel_roles': False, 'physical_geometry_approved': False, 'production_admitted': False}


class Context:
    def __init__(self, gray, previous):
        self.gray = gray; self.previous = previous; self.ink = foreground(gray)
        self.graph = defaultdict(list); self.lookup = {}; self.length = {}; self.valid = set()
        self.audit = {'source_points': 0, 'component_edge_visits': 0, 'parallel_pair_checks': 0, 'frame_pair_checks': 0, 'limits': []}
        if len(previous['proposals']) > 4096 or len(previous['tracks']) > 1024: raise ValueError('Bounded source graph required')
        for item in previous['proposals']:
            if item['id'] in self.lookup: raise ValueError('Unique source IDs required')
            p = np.asarray(item['points_px'], float); self.audit['source_points'] += len(p)
            if self.audit['source_points'] > POLICY['maximum_source_points']: self.limit('source_points'); break
            if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.isfinite(p).all(): raise ValueError('Finite source path required')
            if not math.isfinite(item['width_px']) or item['width_px'] <= 0: raise ValueError('Finite positive source pen required')
            q = np.rint(p).astype(int)
            if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any(): raise ValueError('Source outside image')
            self.lookup[item['id']] = item
            extent = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()); self.length[item['id']] = extent
            if len(p) < 2 or not np.all(self.ink[q[:, 1], q[:, 0]]) or np.max(np.abs(np.diff(q, axis=0))) > 1: continue
            self.valid.add(item['id'])
            a, b = tuple(p[0]), tuple(p[-1])
            if a == b: continue
            self.graph[a].append((b, item['id'], extent)); self.graph[b].append((a, item['id'], extent))

    def limit(self, reason):
        if reason not in self.audit['limits']: self.audit['limits'].append(reason)

    def component(self, chain):
        refs = chain['local_reference_evidence']['references']
        threshold = .65*min(r['body_width'] for r in refs)
        forbidden = set(chain['shaft_interval_ids']) | {i for group in chain['extension_interval_ids'] for i in group}
        queue = [(0., tuple(r['foot'])) for r in refs]; heapq.heapify(queue); best = {p: d for d, p in queue}
        visited = set()
        while queue and not self.audit['limits']:
            narrow, node = heapq.heappop(queue)
            if narrow != best[node]: continue
            for other, key, extent in self.graph[node]:
                self.audit['component_edge_visits'] += 1
                if self.audit['component_edge_visits'] > POLICY['maximum_component_edge_visits']:
                    self.limit('component_edges'); break
                if key in forbidden: continue
                value = 0. if self.lookup[key]['width_px'] >= threshold else narrow+extent
                if value > 2*threshold: continue
                visited.add(key)
                if value < best.get(other, math.inf):
                    best[other] = value; heapq.heappush(queue, (value, other))
        return visited, threshold

    def sidedness(self, chain, component, threshold):
        tracks = {t['id']: t for t in self.previous['tracks']}; shaft = tracks[chain['shaft_track']]
        p = np.asarray(shaft['points'], float); axis = (p[1]-p[0])/np.linalg.norm(p[1]-p[0]); normal = np.array([-axis[1], axis[0]])
        feet = np.asarray(chain['feet_px']); signed = (feet-p[0]) @ normal
        if signed[0]*signed[1] <= 0: return {'confirmed': False, 'reason': 'references_on_different_physical_sides'}
        sign = 1. if signed.mean() > 0 else -1.
        body_width = min(r['body_width'] for r in chain['local_reference_evidence']['references'])
        margin = body_width+shaft['width']+2**.5/2
        minimum = math.inf; witness = None
        for key in sorted(component):
            item = self.lookup[key]
            if item['width_px'] < threshold: continue
            points = np.asarray(item['points_px']); distances = (points-p[0]) @ normal*sign
            index = int(np.argmin(distances)); value = float(distances[index])
            if value < minimum: minimum = value; witness = {'source_interval_id': key, 'point_px': points[index].tolist()}
        return {'confirmed': minimum >= -margin, 'reason': None if minimum >= -margin else 'wide_component_crosses_dimension_shaft',
                'minimum_signed_distance_px': minimum if math.isfinite(minimum) else None,
                'observed_stroke_margin_px': margin, 'reference_side_sign': sign, 'witness': witness}

    def wide_tracks(self, component, threshold):
        wide = []
        for track in self.previous['tracks']:
            total = sum(self.length.get(key, 0.) for key in track['interval_ids'])
            support = sum(self.length.get(key, 0.) for key in track['interval_ids']
                          if key in self.valid and (component is None or key in component) and self.lookup[key]['width_px'] >= threshold)
            if total > 0 and support/total >= .8:
                p = np.asarray(track['points'], float); length = float(np.linalg.norm(p[1]-p[0]))
                if length > 0: wide.append((track, p, (p[1]-p[0])/length, length))
        return wide

    def clipped_frame(self, chain, component, threshold, side):
        tracks = {t['id']: t for t in self.previous['tracks']}; shaft = tracks[chain['shaft_track']]
        p = np.asarray(shaft['points']); axis = (p[1]-p[0])/np.linalg.norm(p[1]-p[0]); normal = np.array([-axis[1], axis[0]])
        sign = side.get('reference_side_sign', 1.); margin = side.get('observed_stroke_margin_px', 0.)
        body_width = min(r['body_width'] for r in chain['local_reference_evidence']['references'])
        tolerance = 1.5*body_width+2**.5/2
        def observed_ends(track):
            points = np.asarray([self.lookup[key]['points_px'][end] for key in track['interval_ids'] if key in self.valid for end in (0, -1)])
            along = points @ axis
            return points[[int(np.argmin(along)), int(np.argmax(along))]]
        def at_edges(points):
            distance = np.minimum.reduce([points[:, 0], points[:, 1], self.gray.shape[1]-1-points[:, 0], self.gray.shape[0]-1-points[:, 1]])
            return bool(np.all(distance <= tolerance))
        candidates = self.wide_tracks(None, threshold)
        for local, _, local_axis, _ in self.wide_tracks(component, threshold):
            if abs(local_axis @ axis) < math.cos(math.radians(8)): continue
            local_ends = observed_ends(local)
            if not at_edges(local_ends): continue
            for other, _, other_axis, _ in candidates:
                self.audit['frame_pair_checks'] += 1
                if self.audit['frame_pair_checks'] > POLICY['maximum_frame_pair_checks']:
                    self.limit('frame_pairs'); return None
                if other['id'] == local['id'] or abs(other_axis @ axis) < math.cos(math.radians(8)): continue
                ends = observed_ends(other)
                if not at_edges(ends) or not np.all((ends-p[0]) @ normal*sign < -margin): continue
                low, high = sorted((ends-p[0]) @ axis)
                overlap = max(0., min(shaft['length'], high)-max(0., low))
                if overlap < .8*shaft['length']: continue
                return {'reference_track': local['id'], 'opposite_track': other['id'],
                        'reference_observed_ends_px': local_ends.tolist(), 'opposite_observed_ends_px': ends.tolist(),
                        'opposite_source_interval_ids': list(other['interval_ids']),
                        'border_tolerance_from_pen_px': tolerance, 'shaft_span_overlap_px': overlap}
        return None

    def periodic(self, component, threshold):
        wide = self.wide_tracks(component, threshold)
        groups = []
        for seed, p, axis, length in wide:
            normal = np.array([-axis[1], axis[0]]); a = sorted(p @ axis); peers = []
            for track, q, direction, extent in wide:
                self.audit['parallel_pair_checks'] += 1
                if self.audit['parallel_pair_checks'] > POLICY['maximum_parallel_pair_checks']:
                    self.limit('parallel_pairs'); return groups
                if abs(axis @ direction) < math.cos(math.radians(8)): continue
                if max(length, extent)/min(length, extent) > 1.25: continue
                b = sorted(q @ axis); overlap = max(0., min(a[1], b[1])-max(a[0], b[0]))
                if overlap/min(length, extent) < .85: continue
                peers.append({'track_id': track['id'], 'offset': float(q.mean(0) @ normal), 'width': track['width'],
                              'span': b, 'source_interval_ids': list(track['interval_ids'])})
            peers.sort(key=lambda t: (t['offset'], t['track_id'])); distinct = []
            median_pen = float(np.median([r['width'] for r in peers])) if peers else 0.
            for peer in peers:
                if not distinct or peer['offset']-distinct[-1]['offset'] > .5*median_pen: distinct.append(peer)
            if len(distinct) < 4: continue
            gaps = np.diff([t['offset'] for t in distinct]); cv = float(gaps.std()/gaps.mean())
            if gaps.min() < median_pen or cv > .25: continue
            groups.append({'seed_track': seed['id'], 'members': distinct, 'offset_count': len(distinct),
                           'gaps_px': gaps.tolist(), 'gap_CV': cv, 'median_pen_px': median_pen})
            break
        return groups


def filter_context(gray, previous, inherited, basis_packet):
    checked_gray(gray)
    if basis_packet['pixel_roles'].shape != gray.shape or not np.array_equal(basis_packet['pixel_roles'], inherited['pixel_roles']):
        raise ValueError('Aligned unchanged roles required')
    if len(basis_packet['open_reference_chains']) > 4096: raise ValueError('Bounded candidate chains required')
    if not basis_packet['open_reference_chains']:
        return {**basis_packet, 'dimension_chains': list(inherited['dimension_chains']),
                'reference_context_audit': {'source_points': 0, 'component_edge_visits': 0, 'parallel_pair_checks': 0, 'frame_pair_checks': 0, 'limits': []},
                'reference_context_decisions': [], 'basis_candidate_count': 0}
    context = Context(gray, previous); accepted = []; decisions = []
    for index, chain in enumerate(basis_packet['open_reference_chains']):
        if context.audit['limits']: break
        component, threshold = context.component(chain)
        side = context.sidedness(chain, component, threshold)
        clipped = context.clipped_frame(chain, component, threshold, side) if side['confirmed'] else None
        repeated = context.periodic(component, threshold) if side['confirmed'] else []
        reasons = []
        if not component or side.get('minimum_signed_distance_px') is None: reasons.append('insufficient_wide_reference_context')
        if not side['confirmed']: reasons.append(side['reason'])
        if clipped: reasons.append('clipped_opposing_frame_strokes')
        if repeated: reasons.append('equal_span_periodic_wide_component')
        admitted = not reasons and not context.audit['limits']
        if admitted: accepted.append(chain)
        decisions.append({'candidate_index': index, 'head_ids': list(chain['head_ids']), 'shaft_track': chain['shaft_track'],
            'accepted': admitted, 'reasons': reasons, 'wide_pen_threshold_px': threshold,
            'component_interval_ids': sorted(component), 'sidedness': side, 'clipped_frame': clipped, 'periodic_groups': repeated})
    incomplete = basis_packet.get('search_incomplete', False) or bool(context.audit['limits'])
    if incomplete: accepted = []
    return {**basis_packet, 'dimension_chains': list(inherited['dimension_chains'])+accepted,
            'open_reference_chains': accepted, 'basis_candidate_count': len(basis_packet['open_reference_chains']),
            'reference_context_audit': context.audit, 'reference_context_decisions': decisions,
            'search_incomplete': incomplete, 'pixel_roles_unchanged': True,
            'production_admitted': False, 'physical_geometry_approved': False, 'source_pixels_erased': 0}
