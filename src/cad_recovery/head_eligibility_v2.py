"""Explain source counterevidence without editing or certifying observations.

This sidecar may exclude an observation from a diagnostic binding experiment.
Unproved cases remain reviewable; no decision approves semantic CAD geometry.
"""
import cv2
import numpy as np

from .family_catalog_v2 import require

POLICY = {'version': 'head-eligibility/2.0', 'maximum_source_pixels': 25000000,
          'maximum_candidates': 512, 'maximum_roi_side_px': 192,
          'maximum_combined_morphology_pixels': 40000000,
          'minimum_large_cavity_area_px': 16, 'minimum_cavity_overlap_px': 2,
          'minimum_cavity_interior_fraction': .08,
          'maximum_digital_convex_defect_px': 2**.5,
          'minimum_compact_triangle_interior_fill': .98,
          'maximum_compact_component_backward_extent_px': 3.,
          'source_erasure_allowed': False, 'observations_immutable': True,
          'manual_references_used': False, 'production_admitted': False,
          'physical_geometry_approved': False}


def source_width(head):
    evidence = head.get('evidence', {})
    width = evidence.get('local_shaft_width_px')
    if width is None:
        width = evidence.get('native_shaft', {}).get('width_px')
    if width is None:
        legacy = evidence.get('legacy_profile_evidence', {})
        scale = evidence.get('normalized_scale', 1.)
        if legacy.get('shaft_width_px') is not None and scale > 0:
            width = legacy['shaft_width_px']/scale
    return float(width) if width is not None and np.isfinite(width) and width > 0 else 2.


def inspect_head(mask, head, audit):
    polygon = np.asarray(head['polygon_px'], float)
    unit = np.asarray(head['direction'], float)
    require(polygon.shape == (3, 2) and unit.shape == (2,) and np.isfinite(polygon).all()
            and np.isfinite(unit).all() and np.abs(polygon).max() <= 1e7
            and abs(np.linalg.norm(unit)-1.) < .01, 'Invalid head geometry')
    base = polygon[1:].mean(0)
    length = float(np.linalg.norm(polygon[0]-base))
    breadth = float(np.linalg.norm(polygon[1]-polygon[2]))
    margin = max(12., length+breadth)
    lo = np.maximum(0, np.floor(polygon.min(0)-margin)).astype(int)
    hi = np.minimum(mask.shape[::-1], np.ceil(polygon.max(0)+margin+1)).astype(int)
    if (hi <= lo).any() or (hi-lo > POLICY['maximum_roi_side_px']).any():
        return 'review', 'local_topology_window_unavailable', {}
    cost = int(np.prod(hi-lo))*5
    if audit['inherited_morphology_pixels']+audit['topology_pass_pixels']+cost > POLICY['maximum_combined_morphology_pixels']:
        audit['limits'].add('eligibility_topology_pixel_budget')
        return 'review', 'eligibility_budget_unchecked', {}
    audit['topology_pass_pixels'] += cost
    local = mask[lo[1]:hi[1], lo[0]:hi[0]]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
    _, background, bg_stats, _ = cv2.connectedComponentsWithStats((local == 0).astype(np.uint8), connectivity=8)
    triangle = np.zeros_like(local)
    cv2.fillConvexPoly(triangle, np.rint(polygon-lo).astype(np.int32), 255)
    interior = cv2.erode(triangle, np.ones((3, 3), np.uint8)) > 0
    total = int(interior.sum())
    details = {'roi_xyxy': [*lo.tolist(), *hi.tolist()], 'interior_pixels': total,
               'topology_pass_pixels': cost, 'counterevidence': []}
    # Only background components NOT reaching this ROI's border prove closure.
    borders = set(np.r_[background[0], background[-1], background[:, 0], background[:, -1]].tolist())
    width = source_width(head)
    cavity_minimum = max(POLICY['minimum_large_cavity_area_px'], 4*width*width)
    cavities = []
    if total >= 6:
        ids, overlap = np.unique(background[interior], return_counts=True)
        for identifier, pixels in zip(ids, overlap):
            identifier, pixels = int(identifier), int(pixels)
            area = int(bg_stats[identifier, cv2.CC_STAT_AREA])
            if (identifier == 0 or identifier in borders or area < cavity_minimum
                    or area < total or pixels < POLICY['minimum_cavity_overlap_px']):
                continue
            if pixels/total >= POLICY['minimum_cavity_interior_fraction']:
                cavities.append({'area_px': area, 'interior_overlap_px': pixels,
                                 'interior_fraction': pixels/total})
    if cavities:
        details['counterevidence'].append({'kind': 'enclosed_source_cavity_inside_filled_head', 'cavities': cavities})
    # A compact, complete, nearly convex triangle with no backward protrusion
    # cannot provide an independently observed tail. Border-clipped components
    # never support this conclusion.
    if total >= 6:
        ids, pixels = np.unique(labels[interior & (local > 0)], return_counts=True)
        if len(ids):
            identifier = int(ids[int(np.argmax(pixels))])
            component = (labels == identifier).astype(np.uint8)*255
            complete = not (component[0].any() or component[-1].any()
                            or component[:, 0].any() or component[:, -1].any())
            if complete:
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(contours, key=cv2.contourArea)
                area = float(cv2.contourArea(contour))
                hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
                points = contour.reshape(-1, 2)+lo
                backward = float(max(0., -np.min((points-base)@unit)))
                hull = cv2.convexHull(contour, returnPoints=False)
                try:
                    defects = cv2.convexityDefects(contour, hull) if len(contour) >= 4 and len(hull) >= 3 else None
                    defect = float(np.asarray(defects).reshape(-1,4)[:,3].max()/256.) if defects is not None else 0.
                    enclosing_area, enclosing = cv2.minEnclosingTriangle(contour)
                except cv2.error:
                    details['component'] = {'complete': True, 'simple_contour': False}
                    return ('ineligible', details['counterevidence'][0]['kind'], details) if details['counterevidence'] else ('review', 'component_contour_unavailable', details)
                interior_fill = None; triangular_pixels = 0
                if enclosing is not None and enclosing_area > 0:
                    triangular = np.zeros_like(component)
                    cv2.fillConvexPoly(triangular, np.rint(enclosing.reshape(3,2)).astype(np.int32), 255)
                    inner = cv2.erode(triangular, np.ones((3,3),np.uint8)) > 0
                    triangular_pixels = int(inner.sum())
                    if triangular_pixels >= 6:
                        interior_fill = float(np.mean(component[inner] > 0))
                details['component'] = {'complete': True,
                    'solidity_for_diagnosis_only': area/hull_area if hull_area else 0.,
                    'backward_extent_px': backward, 'maximum_convex_defect_px': defect,
                    'enclosing_triangle_interior_pixels': triangular_pixels,
                    'enclosing_triangle_interior_fill': interior_fill}
                # A one-pixel staircase can change area ratios substantially.
                # Test the discrete boundary and eroded interior instead.
                if (defect <= POLICY['maximum_digital_convex_defect_px']
                        and interior_fill is not None and interior_fill >= POLICY['minimum_compact_triangle_interior_fill']
                        and backward <= POLICY['maximum_compact_component_backward_extent_px']):
                    details['counterevidence'].append({'kind': 'compact_filled_triangle_without_independent_tail'})
    if details['counterevidence']:
        return 'ineligible', details['counterevidence'][0]['kind'], details
    return 'review', 'no_decisive_source_counterevidence', details


def apply(gray, observations, *, inherited_morphology_pixels=0):
    require(isinstance(gray, np.ndarray) and gray.ndim == 2 and gray.dtype == np.uint8
            and 0 < gray.size <= POLICY['maximum_source_pixels'], 'Expected bounded grayscale source')
    require(len(observations) <= POLICY['maximum_candidates'], 'Too many head observations')
    require(0 <= inherited_morphology_pixels <= POLICY['maximum_combined_morphology_pixels'], 'Invalid inherited morphology cost')
    identities = [head['id'] for head in observations]
    require(len(set(identities)) == len(identities), 'Duplicate observation IDs')
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    audit = {'threshold_preparation_pixels': int(gray.size),
             'inherited_morphology_pixels': int(inherited_morphology_pixels),
             'topology_pass_pixels': 0, 'observations_total': len(observations),
             'observations_inspected': 0, 'limits': set()}
    decisions = []
    for head in observations:
        status, reason, evidence = inspect_head(mask, head, audit)
        audit['observations_inspected'] += 1
        decisions.append({'candidate_id': head['id'], 'status': status, 'reason': reason,
                          'evidence': evidence, 'observation_retained': True,
                          'eligible_for_CAD': False, 'diagnostic_retained': status != 'ineligible'})
    # A partially checked source receives no new exclusions.
    if audit['limits']:
        for decision in decisions:
            decision.update(status='review', reason='eligibility_budget_unchecked', diagnostic_retained=True)
    audit['limits'] = sorted(audit['limits'])
    audit['combined_morphology_pixels'] = inherited_morphology_pixels+audit['topology_pass_pixels']
    return {'schema': POLICY['version'], 'policy': dict(POLICY), 'decisions': decisions,
            'diagnostic_candidate_ids': [d['candidate_id'] for d in decisions if d['diagnostic_retained']],
            'analysis': audit, 'search_incomplete': bool(audit['limits']),
            'source_modified': False, 'observations_modified': False,
            'automatic_CAD_promotions': 0, 'production_admitted': False, 'physical_geometry_approved': False}
