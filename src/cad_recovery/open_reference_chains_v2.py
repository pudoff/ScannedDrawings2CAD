"""Versioned basis correction: physical side and canonical derived track representations."""
from collections import defaultdict
from copy import deepcopy
import heapq
import math
import numpy as np

from .interval_roles_v1 import ink_segment
from .stroke_roles_v1 import checked_gray, foreground, line_distance
from .source_chain_routes_v1 import Routes

POLICY = {'version': 'open-reference-chains/2.0', 'maximum_source_points': 2000000,
          'maximum_edge_visits': 500000, 'maximum_pair_checks': 250000,
          'minimum_body_to_extension_width': 1.75, 'body_component_width_fraction': .65,
          'reference_direction_deg': 20., 'extension_direction_deg': 35.,
          'maximum_extension_length_in_spans': 2., 'maximum_body_route_in_spans': 6.,
          'source_pixels_erased': 0, 'changes_pixel_roles': False,
          'physical_geometry_approved': False, 'production_admitted': False}


class LocalReferences:
    def __init__(self, gray, previous, heads):
        self.ink = foreground(gray); self.previous = previous
        self.items = {p['id']: p for p in previous['proposals']}; self.graph = defaultdict(list)
        self.audit = {'source_points': 0, 'edge_visits': 0, 'pair_checks': 0, 'limits': [],
                      'head_pair_checks': len(previous['tracks'])*len(heads),
                      'invalid_source_intervals': 0, 'ambiguous_shafts': 0}
        self.head_contacts = contacts(previous['tracks'], heads, self.ink)
        headed = {i for t in previous['tracks'] if self.head_contacts[t['id']] for i in t['interval_ids']}
        self.blocked = set()
        for b in previous.get('boundaries', {}).get('boundaries', []):
            if b['frame_hypothesis']: self.blocked.update(b['owners'])
        for bundle in previous.get('hatch_bundles', []):
            if not headed.intersection(bundle['interval_ids']): self.blocked.update(bundle['interval_ids'])
        for item in previous['proposals']:
            points = np.asarray(item['points_px'], float); self.audit['source_points'] += len(points)
            if self.audit['source_points'] > POLICY['maximum_source_points']: self.limit('source_points'); break
            if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
                raise ValueError('Finite nonempty source intervals required')
            if not math.isfinite(item['width_px']) or item['width_px'] <= 0: raise ValueError('Finite positive stroke width required')
            q = np.rint(points).astype(int)
            if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any():
                raise ValueError('Source interval outside image')
            if not np.all(self.ink[q[:, 1], q[:, 0]]) or (len(q) > 1 and np.max(np.abs(np.diff(q, axis=0))) > 1):
                self.audit['invalid_source_intervals'] += 1; continue
            if len(points) < 2: continue
            a, b = tuple(points[0]), tuple(points[-1]); length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if a != b:
                self.graph[a].append((b, item['id'], length)); self.graph[b].append((a, item['id'], length))
        self.routes = Routes(previous['proposals'], self.ink)

    def limit(self, reason):
        if reason not in self.audit['limits']: self.audit['limits'].append(reason)

    def visit(self):
        self.audit['edge_visits'] += 1
        if self.audit['edge_visits'] > POLICY['maximum_edge_visits']: self.limit('edge_visits'); return False
        return True

    def support(self, start, shaft_axis, sign, width, forbidden):
        """An observed thick branch pointing into the measured span; no virtual edge."""
        queue = [(0., start, (), None, 0.)]; seen = {start: 0.}
        normal = np.array([-shaft_axis[1], shaft_axis[0]])
        while queue and not self.audit['limits']:
            length, node, route, anchor, body_width = heapq.heappop(queue)
            if length != seen[node]: continue
            progress = float((np.asarray(node)-start) @ shaft_axis) * sign
            if anchor is not None and progress >= max(4*width, 2*body_width):
                before = route[:anchor[1]]; reference = route[anchor[1]:]
                return {'foot': list(anchor[0]), 'connectors': list(before), 'reference_ids': list(reference),
                        'body_width': body_width, 'support_length_px': length}
            for other, key, extent in self.graph[node]:
                if not self.visit(): break
                if key in forbidden or key in self.blocked or key in route: continue
                item = self.items[key]; pen = float(item['width_px']); p = np.asarray(item['points_px'])
                delta = np.asarray(other)-node; norm = np.linalg.norm(delta)
                aligned = norm > 0 and delta @ shaft_axis * sign / norm >= math.cos(math.radians(20))
                substantial = aligned and pen >= 1.75*width
                if not substantial and extent > 2*max(width, pen): continue
                corridor = max(2*width, body_width, pen)
                if np.max(np.abs((p-np.asarray(start)) @ normal)) > corridor: continue
                if float((np.asarray(other)-start) @ shaft_axis)*sign < -width: continue
                maximum = 16*max(width, body_width, pen)
                if length+extent > maximum and not substantial: continue
                next_anchor = anchor; next_width = body_width
                if anchor is None and substantial:
                    next_anchor = (node, len(route)); next_width = pen
                elif anchor is not None and substantial:
                    next_width = min(body_width, pen)
                distance = length+extent
                if distance < seen.get(other, math.inf):
                    seen[other] = distance
                    heapq.heappush(queue, (distance, other, route+(key,), next_anchor, next_width))
        return None

    def thin_forward(self, node, axis, width, forbidden):
        queue = [(0., node)]; seen = {node}; start = np.asarray(node)
        while queue and not self.audit['limits']:
            length, point = heapq.heappop(queue)
            if (np.asarray(point)-start) @ axis >= 3*width: return True
            for other, key, extent in self.graph[point]:
                if not self.visit(): break
                if key in forbidden or key in self.blocked or other in seen: continue
                item = self.items[key]; pen = item['width_px']; delta = np.asarray(other)-point
                if not .65*width <= pen <= 1.5*width: continue
                if delta @ axis <= 0: continue
                cross = axis[0]*delta[1]-axis[1]*delta[0]
                if abs(cross) > max(2*width, .5*np.linalg.norm(delta)): continue
                if delta @ axis >= 3*width and abs(cross) <= 2*width: return True
                if length+extent > 6*width: continue
                seen.add(other); heapq.heappush(queue, (length+extent, other))
        return False

    def feet(self, ext, shaft, hit, sign, forbidden):
        axis = np.asarray(ext['u'], float)*sign; normal = np.array([-axis[1], axis[0]])
        origin = np.asarray(ext['points'][0], float); width = float(ext['width'])
        span = float(shaft['length']); shaft_axis = np.asarray(shaft['u'], float)
        center = np.mean(shaft['points'], axis=0)
        inward = 1 if (center-hit) @ shaft_axis >= 0 else -1
        starts = {tuple(self.items[i]['points_px'][end]) for i in ext['interval_ids'] for end in (0, -1)
                  if np.linalg.norm(np.asarray(self.items[i]['points_px'][end])-hit) <= 4*max(width, shaft['width'])}
        queue = [(float(np.linalg.norm(np.asarray(p)-hit)), p, ()) for p in starts]
        heapq.heapify(queue); seen = {p: d for d, p, _ in queue}; found = []
        while queue and not self.audit['limits']:
            length, node, route = heapq.heappop(queue)
            if length != seen[node]: continue
            distance = float((np.asarray(node)-hit) @ axis)
            if distance >= 3*max(width, shaft['width']):
                support = self.support(node, shaft_axis, inward, width, forbidden | set(route))
                if support and not self.thin_forward(node, axis, width, forbidden | set(route) | set(support['reference_ids'])):
                    found.append({**support, 'extension_route_ids': list(route)+support['connectors'],
                                  'extension_track': ext['id'], 'extension_interval_ids': ext['interval_ids'],
                                  'extension_width': width, 'side': sign, 'distance_from_head_px': distance})
                    continue
            for other, key, extent in self.graph[node]:
                if not self.visit(): break
                if key in forbidden or key in self.blocked or key in route: continue
                item = self.items[key]; p = np.asarray(item['points_px']); pen = item['width_px']
                delta = np.asarray(other)-node; norm = np.linalg.norm(delta)
                if (np.asarray(other)-hit) @ axis < -width: continue
                if delta @ axis < -width: continue
                if extent > 4*max(width, pen) and norm > 0 and delta @ axis/norm < math.cos(math.radians(35)): continue
                if np.max(np.abs((p-origin) @ normal)) > max(2*width, .75*pen)+2**.5/2: continue
                distance = length+extent
                if distance > 2*span or distance >= seen.get(other, math.inf): continue
                seen[other] = distance; heapq.heappush(queue, (distance, other, route+(key,)))
        return found

    def component(self, a, b, forbidden, span):
        start, goal = tuple(a['foot']), tuple(b['foot']); threshold = .65*min(a['body_width'], b['body_width'])
        queue = [(0., start, ())]; seen = {start: 0.}
        while queue and not self.audit['limits']:
            length, node, route = heapq.heappop(queue)
            if length != seen[node]: continue
            if node == goal: return {'interval_ids': list(route), 'source_length_px': length, 'minimum_body_pen_px': threshold}
            for other, key, extent in self.graph[node]:
                if not self.visit(): break
                if key in forbidden or key in self.blocked: continue
                pen = self.items[key]['width_px']
                if pen < threshold and extent > 2*threshold: continue
                distance = length+extent
                if distance > 6*span or distance >= seen.get(other, math.inf): continue
                seen[other] = distance; heapq.heappush(queue, (distance, other, route+(key,)))
        return None


def contacts(tracks, heads, ink):
    result = defaultdict(list)
    for track in tracks:
        points = np.asarray(track['points']); axis = np.asarray(track['u'])
        for head in heads:
            tip = np.asarray(head['tip_px']); base = np.asarray(head['polygon_px'][1:]).mean(0)
            direction = np.asarray(head['direction']); direction = direction/np.linalg.norm(direction)
            if abs(axis @ direction) < math.cos(math.radians(20)): continue
            if line_distance(base, points)[0] > 1.5*track['width']+head.get('uncertainty_px', 0): continue
            if np.min((points-tip) @ direction) >= -.5*np.linalg.norm(base-tip): continue
            along = (base-points[0]) @ axis; nearest = points[0]+np.clip(along, 0, track['length'])*axis
            if ink_segment(ink, base, nearest): result[track['id']].append(head)
    return result


def canonical_features(previous, heads):
    """Orient derived axes and iteration order; original source observations stay intact."""
    result = deepcopy(previous)
    def endpoints(item):
        p = item['points_px']
        return (*min(tuple(p[0]), tuple(p[-1])), *max(tuple(p[0]), tuple(p[-1])), float(item['width_px']), item['id'])
    result['proposals'].sort(key=endpoints)
    ranks = {p['id']: i for i, p in enumerate(result['proposals'])}
    for track in result['tracks']:
        p = np.asarray(track['points'], float)
        if p.shape != (2, 2) or not np.isfinite(p).all(): raise ValueError('Finite two-endpoint track required')
        if tuple(p[1]) < tuple(p[0]): p = p[::-1]
        length = float(np.linalg.norm(p[1]-p[0]))
        if length <= 0 or not math.isfinite(track['length']) or abs(track['length']-length) > 1e-6*max(1., length):
            raise ValueError('Consistent nonzero source track required')
        track['points'] = p.tolist(); track['u'] = ((p[1]-p[0])/length).tolist()
        track['interval_ids'].sort(key=lambda key: ranks[key])
    result['tracks'].sort(key=lambda t: (*t['points'][0], *t['points'][1], t['width'], t['id']))
    ordered_heads = sorted(heads, key=lambda h: (*h['tip_px'], *h['direction'], h['id']))
    return result, ordered_heads


def extend(gray, previous, inherited, heads):
    checked_gray(gray)
    if len(previous['proposals']) > 4096 or len(previous['tracks']) > 1024 or len(heads) > 512:
        raise ValueError('Bounded source objects required')
    if inherited['pixel_roles'].shape != gray.shape or inherited['pixel_roles'].dtype != np.uint8:
        raise ValueError('Aligned uint8 role map required')
    if len({p['id'] for p in previous['proposals']}) != len(previous['proposals']) or len({h['id'] for h in heads}) != len(heads):
        raise ValueError('Unique source/head IDs required')
    for head in heads:
        polygon = np.asarray(head['polygon_px']); direction = np.asarray(head['direction']); tip = np.asarray(head['tip_px'])
        if polygon.shape != (3, 2) or direction.shape != (2,) or tip.shape != (2,) or not np.isfinite([*polygon.ravel(), *direction, *tip]).all():
            raise ValueError('Finite directed triangular observation required')
        if abs(np.linalg.norm(direction)-1) > 1e-6: raise ValueError('Unit observed head direction required')
    previous, heads = canonical_features(previous, heads)
    search = LocalReferences(gray, previous, heads)
    attached = search.head_contacts; additions = []; rejected = []
    for shaft in previous['tracks']:
        if search.audit['limits']: break
        hh = attached[shaft['id']]
        if len(hh) != 2 or np.asarray(hh[0]['direction']) @ np.asarray(hh[1]['direction']) > -math.cos(math.radians(20)): continue
        if any(c['shaft_track'] == shaft['id'] for c in inherited['dimension_chains']): continue
        points = np.asarray(shaft['points']); axis = np.asarray(shaft['u']); origin = points[0]
        projection = [0., shaft['length'], *[float((np.asarray(h['tip_px'])-origin) @ axis) for h in hh]]
        span = origin+np.array([min(projection), max(projection)])[:, None]*axis
        options = [[], []]; forbidden = set(shaft['interval_ids'])
        for ext in previous['tracks']:
            search.audit['pair_checks'] += 1
            if search.audit['pair_checks'] > POLICY['maximum_pair_checks']: search.limit('pair_checks'); break
            if ext['id'] == shaft['id'] or abs(np.asarray(ext['u']) @ axis) > .25: continue
            tolerance = max(shaft['width'], ext['width'])+max(h.get('uncertainty_px', 0) for h in hh)
            hit, proof = search.routes.intersection(span, ext, tolerance, shaft, hh)
            if hit is None: continue
            nearest = int(np.argmin([np.linalg.norm(hit-h['tip_px']) for h in hh]))
            if np.linalg.norm(hit-hh[nearest]['tip_px']) > 3*shaft['width']+tolerance: continue
            for sign in (-1, 1):
                for foot in search.feet(ext, shaft, hit, sign, forbidden):
                    options[nearest].append({**foot, 'head_id': hh[nearest]['id'], 'junction': proof, 'hit_px': hit.tolist()})
        # Equivalent anchors within the observed stroke footprint form one local choice.
        selected = []
        for choices in options:
            clusters = []
            for foot in sorted(choices, key=lambda f: (f['distance_from_head_px'], tuple(f['foot']))):
                cluster = next((c for c in clusters if np.linalg.norm(np.asarray(c[0]['foot'])-foot['foot']) <=
                                foot['extension_width']+.5*max(foot['body_width'], c[0]['body_width'])+2**.5/2), None)
                if cluster is None: clusters.append([foot])
                else: cluster.append(foot)
            if len(clusters) != 1: break
            cluster = clusters[0]
            tracks = {t['id']: t for t in previous['tracks']}
            chosen = min(cluster, key=lambda f: (line_distance(np.asarray(f['foot']), np.asarray(tracks[f['extension_track']]['points']))[0], tuple(f['foot'])))
            selected.append(chosen)
        if len(selected) != 2:
            if any(options): search.audit['ambiguous_shafts'] += 1
            rejected.append({'shaft_track': shaft['id'], 'reason': 'missing_or_ambiguous_local_reference', 'options': [len(v) for v in options]})
            continue
        a, b = selected
        if a['extension_track'] == b['extension_track']: continue
        normal = np.array([-axis[1], axis[0]])
        physical_sides = [float((np.asarray(f['foot'])-f['hit_px']) @ normal) for f in selected]
        if physical_sides[0]*physical_sides[1] <= 0: continue
        if np.linalg.norm(np.asarray(a['hit_px'])-b['hit_px']) < .6*shaft['length']: continue
        excluded = forbidden | set(a['extension_route_ids']) | set(b['extension_route_ids'])
        common = search.component(a, b, excluded, shaft['length'])
        if common is None:
            rejected.append({'shaft_track': shaft['id'], 'reason': 'reference_components_not_connected_without_dimension'}); continue
        additions.append({'shaft_track': shaft['id'], 'shaft_interval_ids': list(shaft['interval_ids']),
            'extension_interval_ids': [list(dict.fromkeys(f['extension_interval_ids']+f['extension_route_ids'])) for f in selected],
            'head_ids': [h['id'] for h in hh], 'visible_heads': 2, 'feet_px': [f['foot'] for f in selected],
            'boundary_id': None, 'status': 'geometric_hypothesis_only', 'evidence': 'two_source_connected_open_references',
            'local_reference_evidence': {'references': selected, 'body_route': common, 'physical_geometry_approved': False}})
    incomplete = bool(search.audit['limits']) or search.routes.incomplete or previous.get('search_incomplete', False)
    if incomplete: additions = []
    return {**inherited, 'dimension_chains': list(inherited['dimension_chains'])+additions,
            'open_reference_chains': additions, 'open_reference_audit': search.audit,
            'open_reference_junction_audit': search.routes.audit, 'open_reference_rejections': rejected,
            'search_incomplete': inherited.get('search_incomplete', False) or incomplete,
            'pixel_roles_unchanged': True, 'source_pixels_erased': 0,
            'production_admitted': False, 'physical_geometry_approved': False}
