"""Independent chain evidence before any hatch-based head exclusion."""
from collections import defaultdict
import math

import numpy as np

from .interval_roles_v1 import BoundaryQueries, finite_hit, ink_segment
from .stroke_roles_v1 import line_distance


def verify(intervals, raw_tracks, boundaries, heads, ink):
    tracks = [{**t, 'points': np.asarray(t['points'], float), 'u': np.asarray(t['u'], float)} for t in raw_tracks]
    lookup = {p['id']: p for p in intervals}
    contacts = defaultdict(list)
    query = BoundaryQueries()
    for track in tracks:
        for head in heads:
            tip = np.asarray(head['tip_px'], float)
            base = np.asarray(head['polygon_px'][1:], float).mean(axis=0)
            direction = np.asarray(head['direction'], float).copy()
            direction /= np.linalg.norm(direction)
            if abs(track['u']@direction) < math.cos(math.radians(20)):
                continue
            if line_distance(base, track['points'])[0] > 1.5*track['width']+head.get('uncertainty_px', 0):
                continue
            if np.min((track['points']-tip)@direction) >= -.5*np.linalg.norm(base-tip):
                continue
            along = (base-track['points'][0])@track['u']
            near = track['points'][0]+np.clip(along, 0, track['length'])*track['u']
            if ink_segment(ink, base, near):
                contacts[track['id']].append(head)
    headed = {i for t in tracks if contacts[t['id']] for i in t['interval_ids']}
    eligible = [b for b in boundaries if not b['frame_hypothesis'] and not set(b['owners']) & headed]
    chains, leaders, signatures, checks = [], [], set(), 0
    for shaft in tracks:
        attached = contacts[shaft['id']]
        if not attached:
            continue
        origin, axis = shaft['points'][0], shaft['u']
        projections = [0., shaft['length'], *[float((np.asarray(h['tip_px'])-origin)@axis) for h in attached]]
        span = origin+np.array([min(projections), max(projections)])[:, None]*axis
        options = []
        for ext in tracks:
            if ext is shaft or abs(ext['u']@axis) > .25:
                continue
            checks += 1
            if checks > 250000:
                return {'chains': [], 'leaders': [], 'eligible_boundaries': eligible, 'contacts': {}, 'incomplete': True}
            tolerance = max(shaft['width'], ext['width'])+max(h.get('uncertainty_px', 0) for h in attached)
            hit = finite_hit(span, ext['points'], tolerance, ink)
            if hit is None or min(np.linalg.norm(hit-p) for p in span) > 3*shaft['width']+tolerance:
                continue
            for boundary in eligible:
                outside = query.distance(shaft['points'].mean(axis=0), boundary)
                if outside is None or outside >= -shaft['width']:
                    continue
                x, y, r, b = boundary['box']
                feet = []
                for identifier in ext['interval_ids']:
                    for endpoint in (lookup[identifier]['points_px'][0], lookup[identifier]['points_px'][-1]):
                        p = np.asarray(endpoint)
                        if not (x-tolerance <= p[0] <= r+tolerance and y-tolerance <= p[1] <= b+tolerance):
                            continue
                        distance = query.distance(p, boundary)
                        if distance is not None and abs(distance) <= tolerance and np.linalg.norm(p-hit) >= 3*ext['width']:
                            feet.append(p)
                if not feet:
                    continue
                foot = min(feet, key=lambda p: np.linalg.norm(p-hit))
                direction = (foot-hit)/np.linalg.norm(foot-hit)
                selected = []
                for identifier in ext['interval_ids']:
                    point = np.asarray(lookup[identifier]['points_px']).mean(axis=0)
                    distance = query.distance(point, boundary)
                    along = float((point-hit)@direction)
                    if distance is not None and distance < -.25*ext['width'] and -4*ext['width'] <= along <= np.linalg.norm(foot-hit):
                        selected.append(identifier)
                if selected:
                    options.append((ext, boundary, hit, foot, selected))
        found_chain = False
        for i, a in enumerate(options):
            for b in options[i+1:]:
                if a[0]['id'] == b[0]['id'] or a[1]['id'] != b[1]['id'] or np.linalg.norm(a[2]-b[2]) < .6*shaft['length']:
                    continue
                signature = (shaft['id'], tuple(sorted((a[0]['id'], b[0]['id']))), tuple(a[1]['owners']))
                if signature in signatures:
                    continue
                signatures.add(signature)
                found_chain = True
                chains.append({'shaft_track': shaft['id'], 'shaft_interval_ids': shaft['interval_ids'],
                    'extension_interval_ids': [a[4], b[4]], 'boundary_id': a[1]['id'],
                    'head_ids': [h['id'] for h in attached], 'visible_heads': len(attached),
                    'feet_px': [a[3].tolist(), b[3].tolist()], 'status': 'geometric_hypothesis_only',
                    'evidence': 'two_observed_extensions_to_same_non_shaft_boundary'})
        if found_chain:
            continue
        for head in attached:
            tip = np.asarray(head['tip_px'])
            back = shaft['points'][np.argmax(np.linalg.norm(shaft['points']-tip, axis=1))]
            for boundary in eligible:
                near, far = query.distance(tip, boundary), query.distance(back, boundary)
                if near is not None and far is not None and abs(near) <= 2*shaft['width'] and far < -2*shaft['width']:
                    leaders.append({'interval_ids': shaft['interval_ids'], 'head_ids': [head['id']],
                                    'boundary_id': boundary['id'], 'status': 'geometric_hypothesis_only'})
    return {'chains': chains if not query.incomplete else [], 'leaders': leaders if not query.incomplete else [],
        'eligible_boundaries': eligible, 'contacts': {k: [h['id'] for h in v] for k, v in contacts.items()},
        'boundary_edge_tests': min(query.work, 2000000), 'incomplete': query.incomplete}
