"""Confirm an UNKNOWN CAD LINE from whole-page, source-checked frame sides.

No geometry or raster role is changed. A partial original stamp crop is not
enough evidence: this contract uses the registered whole page, its existing
bounded detector and the object's own original source samples.
"""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from .grid_role_revision_v1 import sha
from .page_frame_context_v3 import detect, POLICY as FRAME_POLICY
from .part_admission_v1 import object_sha
from .primitive_role_guard_v1 import source_points, runs
from .stroke_roles_v1 import checked_gray

POLICY = dict(version='whole-page-frame-semantics/1.0', maximum_pixels=4_000_000,
    minimum_fraction=FRAME_POLICY['minimum_source_support'],
    maximum_angle_deg=FRAME_POLICY['maximum_axis_angle_deg'],
    unknown_gap='observed_side_width_plus_two_working_pixel_diagonals',
    geometry_changed=False, role_map_changed=False, physical_geometry_approved=False)


def context(registered_gray, working_gray, unpadded_size):
    checked_gray(registered_gray); checked_gray(working_gray)
    if max(registered_gray.size, working_gray.size) > POLICY['maximum_pixels']:
        raise ValueError('Frame semantics requires bounded whole-page views')
    w, h = unpadded_size
    if any(type(n) is not int or n <= 0 for n in (w, h)):
        raise ValueError('Positive integer unpadded size required')
    small = cv2.resize(registered_gray, (w, h), interpolation=cv2.INTER_AREA)
    small = np.pad(small, [(0, h % 2), (0, w % 2)], constant_values=255)
    if not np.array_equal(small, working_gray):
        raise ValueError('Frame context is not registered to this working image')
    sx, sy = w/registered_gray.shape[1], h/registered_gray.shape[0]
    if max(sx, sy)/min(sx, sy) > 1.02:
        raise ValueError('Frame semantics requires near-uniform registration')
    matrix = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
    observation = detect(registered_gray)
    return dict(policy=deepcopy(POLICY), context_sha256=sha(registered_gray),
        working_sha256=sha(working_gray), context_size_px=list(registered_gray.shape[::-1]),
        working_size_px=list(working_gray.shape[::-1]), context_to_working=matrix.tolist(),
        detector=observation, independent_semantic_accuracy_claimed=False)


def apply(assembly, gray, geometry, roles, evidence, *, protected=None):
    before = object_sha([assembly, geometry, evidence]); hashes = (sha(gray), sha(roles))
    if sha(gray) != evidence['working_sha256'] or list(gray.shape[::-1]) != evidence['working_size_px']:
        raise ValueError('Stale frame semantics context')
    if roles.shape != gray.shape or roles.dtype != np.uint8:
        raise ValueError('Aligned role map required')
    if protected is None:
        protected = np.zeros(gray.shape, bool)
    if protected.shape != gray.shape:
        raise ValueError('Aligned review protection required')
    result = deepcopy(assembly); decisions = []; lookup = {row['id']: row for row in geometry}
    matrix = np.asarray(evidence['context_to_working'], float)
    sides = []
    detector = evidence['detector']
    if detector['complete']:
        for index, frame in enumerate(detector['frames']):
            for number, side in enumerate(frame['sides']):
                p = np.asarray(side['points_overview'], float) @ matrix[:2, :2].T+matrix[:2, 2]
                axis = p[1]-p[0]; length = float(np.linalg.norm(axis)); axis /= length
                normal = np.array([-axis[1], axis[0]])
                scale = float(np.linalg.norm(matrix[:2, :2].T@normal))
                width = side['stroke_width_overview_px']*scale
                # Pixel-centre registration adds one working-cell sampling bound.
                tolerance = (width/2+side['position_uncertainty_overview_px']*scale+math.sqrt(.5))
                sides.append((index, number, p[0], axis, length, tolerance, width))
    for obj in result['objects']:
        if obj['kind'] != 'line' or obj['role'] != 255 or 'axis_pattern_evidence' in obj or 'axis_review' in obj:
            continue
        points = source_points(obj, lookup)
        x, y = points.T
        if np.any(protected[y, x]) or np.any(np.isin(roles[y, x], [1, 2, 3, 4, 5, 6, 8, 9])):
            continue
        ends = np.asarray(obj['points_px'], float); axis = ends[1]-ends[0]
        if np.linalg.norm(axis) <= 0:
            continue
        axis /= np.linalg.norm(axis)
        ordered = np.asarray(obj['support_points'], float)
        if len(ordered) < 2:
            continue
        along_walk = np.r_[0., np.cumsum(np.linalg.norm(np.diff(ordered, axis=0), axis=1))]
        matches = []
        for index, number, start, u, length, tolerance, width in sides:
            if abs(axis@u) < math.cos(math.radians(POLICY['maximum_angle_deg'])):
                continue
            def supported(p):
                offset = p-start; along = offset@u
                return ((np.abs(offset@np.array([-u[1], u[0]])) <= tolerance)
                        & (along >= -tolerance) & (along <= length+tolerance))
            own = supported(points); walk = supported(ordered)
            unsupported = [float(along_walk[min(b, len(ordered)-1)]-along_walk[max(0, a-1)])
                           for a, b, value in runs(walk.astype(np.uint8)) if value == 0]
            maximum_gap = width+2*math.sqrt(2.)
            if (float(np.mean(own)) >= POLICY['minimum_fraction']
                    and float(np.mean(walk)) >= POLICY['minimum_fraction']
                    and max(unsupported, default=0.) <= maximum_gap):
                matches.append(dict(frame_index=index, side_index=number,
                    source_fraction=float(np.mean(own)), walk_fraction=float(np.mean(walk)),
                    largest_unsupported_span_px=max(unsupported, default=0.),
                    maximum_unsupported_span_px=maximum_gap, sampling_bound_px=tolerance))
        if not matches:
            continue
        proof = obj['semantic_role_guard']
        obj['role'] = 7
        proof.update(previous_role_map_decision=proof['decision'], decision='whole_page_frame_side_confirmed',
                     assigned_role=7, whole_page_frame_evidence=matches,
                     frame_context_sha256=evidence['context_sha256'])
        decisions.append(dict(object_id=obj['id'], assigned_role=7, evidence=matches))
    result['primitive_role_guard'].update(
        decisions=dict(Counter(o['semantic_role_guard']['decision'] for o in result['objects'])),
        changed_role_objects=sum(o['role'] != o['semantic_role_guard']['previous_role'] for o in result['objects']))
    result['whole_page_frame_semantics'] = dict(policy=deepcopy(POLICY), decisions=decisions,
        detector_complete=detector['complete'], observed_frames=len(detector['frames']),
        working_sha256=evidence['working_sha256'], context_sha256=evidence['context_sha256'],
        role_map_changed=False, geometry_changed=False)
    if before != object_sha([assembly, geometry, evidence]) or hashes != (sha(gray), sha(roles)):
        raise ValueError('Frame semantics mutated its inputs')
    return result
