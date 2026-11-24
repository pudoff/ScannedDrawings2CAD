"""Jointly place the two ends of an existing LINE on its adjacent ARC models."""
from collections import defaultdict
from copy import deepcopy
import math

import numpy as np

from .cad_object_assembly_v5 import _align_endpoints as original_align
from .cad_primitive_fit_v2 import checked, sector, POLICY as LIMITS
from .stroke_curve_control_v1 import distance_to_polyline


def refine_failed(items, support, audit):
    previous = deepcopy(items); original_align(previous, support)
    failed = [i for i, item in enumerate(items) if item['model'] and item['model']['kind'] == 'line' and not previous[i]['model']]
    endings = defaultdict(list)
    for i, item in enumerate(items):
        if len(item['points']) > 1:
            endings[tuple(item['points'][0])].append((i, 0)); endings[tuple(item['points'][-1])].append((i, -1))
    changes = []; audit['joint_line_candidates'] = len(failed); audit['joint_lines_accepted'] = 0
    for li in failed:
        item = items[li]; neighbors = []; choices = []
        for endpoint in (0, -1):
            node = item['points'][endpoint]; incident = endings[tuple(node)]
            others = [(i, e) for i, e in incident if i != li]
            if len(incident) != 2 or len(others) != 1:
                break
            ci, ce = others[0]; arc_item = items[ci]; model = arc_item['model']
            if (not model or model['kind'] != 'arc' or not model.get('locally_stable', True)
                    or len(item['roles'] | arc_item['roles']) > 1
                    or not set(item['path_ids']) & set(arc_item['path_ids'])):
                break
            center = np.asarray(model['center_px']); radius = model['radius_px']
            theta = math.atan2(node[1]-center[1], node[0]-center[0])
            delta = 2*math.asin(min(1., LIMITS['max_px']/(2*radius)))
            angles = theta+np.linspace(-delta, delta, LIMITS['maximum_refine_positions'])
            q = center+radius*np.c_[np.cos(angles), np.sin(angles)]
            q = q[np.linalg.norm(q-node, axis=1) <= LIMITS['max_px']]
            neighbors.append((ci, ce)); choices.append(q)
        if len(neighbors) != 2 or neighbors[0][0] == neighbors[1][0]:
            continue
        candidates = []
        for q0 in choices[0]:
            for q1 in choices[1]:
                ends = np.array([q0, q1]); error = distance_to_polyline(item['points'], ends)
                if error.max() <= LIMITS['max_px'] and np.percentile(error, 95) <= LIMITS['p95_px']:
                    candidates.append((float(error@error), ends, error))
        candidates.sort(key=lambda row: row[0])
        for _, ends, error in candidates:
            if audit['tests'] >= 8192:
                audit['search_complete'] = False; break
            audit['tests'] += 1
            line = dict(**item['model']); line.update(points_px=ends.tolist(), fit_max_px=float(error.max()),
                fit_p95_px=float(np.percentile(error, 95)))
            if not checked(line, item['points'], support):
                continue
            models = []
            for q, (ci, ce) in zip(ends, neighbors):
                model = items[ci]['model']; p = items[ci]['points'].astype(float).copy(); p[ce] = q
                arc = sector(p, np.asarray(model['center_px']), model['radius_px'])
                arc = {**model, **arc} if arc else None
                models.append(checked(arc, items[ci]['points'], support) if arc else None)
            if not all(models):
                continue
            item['model'] = line
            for (ci, ce), model in zip(neighbors, models):
                items[ci]['model'] = model
            audit['joint_lines_accepted'] += 1
            changes.append(dict(method='joint_LINE_two_ARC_endpoints', line_item=li,
                arc_items=[ci for ci, _ in neighbors], model_nodes=ends.tolist(),
                maximum_source_node_shift_px=float(np.linalg.norm(ends-item['points'][[0, -1]], axis=1).max())))
            break
    return changes
