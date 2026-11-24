"""Withhold a new contour when its own positive witness has incompatible pen.

This does not infer AXIS from width and never changes accepted baseline roles.
"""
from copy import deepcopy
import math
import cv2
import numpy as np

from .contour_page_competition_v1 import _binary
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha
from .primary_contour_v3 import POLICY as PRIMARY_POLICY
from .stroke_roles_v1 import checked_gray, foreground

POLICY = {'version': 'primary-stroke-guard/1.0', 'maximum_pixels': 4194304,
    'maximum_intervals': 4096, 'maximum_source_points': 2000000,
    'maximum_pen_ratio': PRIMARY_POLICY['maximum_hole_pen_ratio'],
    'maximum_decisions': PRIMARY_POLICY['maximum_boundaries'],
    'positive_role_created': False, 'geometry_modified': False,
    'semantic_AXIS_assigned': False, 'physical_geometry_approved': False}


def analyze(gray, graph, source_owner, ambiguous_owner, proposed, *, primary,
            primary_mask, binding, baseline_roles, protected):
    checked_gray(gray); items = graph['proposals']
    if gray.size > POLICY['maximum_pixels'] or len(items) > POLICY['maximum_intervals']:
        raise ValueError('Bounded stroke-guard source required')
    for value, name in ((proposed, 'proposal'), (primary_mask, 'original primary'),
                        (ambiguous_owner, 'ambiguity'), (protected, 'review protection')):
        _binary(value, gray.shape, name)
    if (baseline_roles.dtype != np.uint8 or baseline_roles.shape != gray.shape
            or not np.isin(baseline_roles, list(range(10))+[255]).all()):
        raise ValueError('Aligned baseline roles required')
    if (source_owner.shape != gray.shape or source_owner.dtype.kind not in 'ui'
            or source_owner.max(initial=0) > len(items)
            or np.any((proposed > 0) & ((source_owner == 0) | (ambiguous_owner > 0)))):
        raise ValueError('Unambiguous current source-owner proposals required')
    if np.any((proposed > 0) & (primary_mask == 0)):
        raise ValueError('Negative guard may only revise original primary pixels')
    if (binding['source_sha256'] != sha(gray) or binding['graph_sha256'] != object_sha(graph)
            or binding['primary_sha256'] != object_sha(primary) or binding['primary_mask_sha256'] != sha(primary_mask)
            or primary['policy'] != PRIMARY_POLICY or primary.get('source_immutable') is not True):
        raise ValueError('Frozen source-bound primary witness required')
    facts = primary['decisions']
    if len(facts) > POLICY['maximum_decisions'] or len({f['boundary_id'] for f in facts}) != len(facts):
        raise ValueError('Bounded unique primary witnesses required')
    selected = {f['boundary_id'] for f in facts if f['positive_body']}
    selected |= {h for f in facts if f['positive_body'] for h in f['holes']}
    if not selected <= {f['boundary_id'] for f in facts}: raise ValueError('Missing primary hole witness')
    pens = {}
    for fact in facts:
        if fact['boundary_id'] not in selected: continue
        pen = float(fact['observed_pen_px'])
        if not math.isfinite(pen) or pen <= 0: raise ValueError('Finite primary pen required')
        for number in fact['source_owner_numbers']:
            if type(number) is not int or not 1 <= number <= len(items): raise ValueError('Foreign primary source owner')
            pens.setdefault(number, []).append(pen)
    before = object_sha([graph, primary, binding])
    arrays = (gray, source_owner, ambiguous_owner, proposed, primary_mask, baseline_roles, protected)
    hashes = [sha(a) for a in arrays]
    distance = cv2.distanceTransform(foreground(gray), cv2.DIST_L2, 5)
    eligible = (proposed > 0) & (baseline_roles == 255) & (protected == 0)
    audit = {'source_points': 0, 'limits': [], 'discarded_decisions': 0}
    if graph.get('search_incomplete') or not primary['complete']: audit['limits'].append('incomplete_primary_or_graph')
    veto = np.zeros_like(gray); decisions = []
    for number in sorted(map(int, np.unique(source_owner[eligible]))) if not audit['limits'] else []:
        if number not in pens: continue
        points = np.asarray(items[number-1]['points_px'], float)
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
                or (points < 0).any() or (points[:, 0] >= gray.shape[1]).any() or (points[:, 1] >= gray.shape[0]).any()):
            raise ValueError('Finite source points inside input required')
        audit['source_points'] += len(points)
        if audit['source_points'] > POLICY['maximum_source_points']:
            audit['limits'].append('source_point_budget'); break
        q = np.rint(points).astype(int)
        if (q < 0).any() or (q[:, 0] >= gray.shape[1]).any() or (q[:, 1] >= gray.shape[0]).any():
            raise ValueError('Rounded source samples outside input')
        samples = distance[q[:, 1], q[:, 0]]; samples = samples[samples > 0]
        if not len(samples): continue
        observed = max(1., float(2*np.quantile(samples, .25)-1))
        reference = min(pens[number])
        if observed < reference/POLICY['maximum_pen_ratio']:
            selected_pixels = eligible & (source_owner == number); veto[selected_pixels] = 255
            decisions.append({'owner_number': number, 'interval_id': items[number-1]['id'],
                'observed_pen_px': observed, 'minimum_supporting_pen_px': reference,
                'withdrawn_pixels': int(selected_pixels.sum())})
    complete = not audit['limits']; output = proposed.copy()
    if complete: output[veto > 0] = 0
    else:
        output[eligible] = 0; veto[:] = 0; audit['discarded_decisions'] = len(decisions); decisions = []
    if before != object_sha([graph, primary, binding]) or hashes != [sha(a) for a in arrays]:
        raise ValueError('Stroke guard mutated source evidence')
    return {'mask': output, 'veto_mask': veto, 'complete': complete, 'source_immutable': True,
        'width_veto_pixels': int(np.count_nonzero(veto)),
        'abstained_pixels': int(np.count_nonzero(proposed)-np.count_nonzero(output)) if not complete else 0,
        'decisions': decisions, 'audit': audit, 'policy': deepcopy(POLICY),
        'positive_role_created': False, 'geometry_modified': False,
        'semantic_AXIS_assigned': False, 'physical_geometry_approved': False}
