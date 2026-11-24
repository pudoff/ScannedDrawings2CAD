"""Part proposals abstain where a dimension interval has competing ink support."""
from copy import deepcopy

import cv2
import numpy as np

from .boundary_role_overlay_v1 import analyze as previous_analyze
from .stroke_roles_v1 import foreground

POLICY = {'version': 'boundary-role-overlay/2.0', 'maximum_competitor_pairs': 2000000,
          'maximum_competitor_source_points': 2000000, 'pixel_footprint_radius_px': 2**.5/2,
          'maximum_combined_raster_domain_pixels': 40000000, 'source_erasure_allowed': False,
          'production_admitted': False, 'semantic_ownership_confirmed': False, 'physical_geometry_approved': False}


def analyze(gray, previous, inherited_roles, local_result, *, heads=(), diagnostic_heads=()):
    result = previous_analyze(gray, previous, inherited_roles, local_result, heads=heads, diagnostic_heads=diagnostic_heads)
    output = result['pixel_roles']; assignments = result['pixel_interval_owner']
    ink = foreground(gray); distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
    intervals = {p['id']: p for p in previous['proposals']}
    chains = local_result['hybrid']['dimension_chains']
    cache = {}; used_points = pairs = removed = 0; incomplete = False
    combined_domain = result['raster_work_pixels']+gray.size
    if combined_domain > 40000000: incomplete = True
    for number, row in enumerate(result['interval_admission'], 1):
        if not row['added_pixels'] or incomplete: continue
        ids = sorted({key for w in row['witnesses'] for index in [w['chain_index']]
                      for group in [chains[index]['shaft_interval_ids'], *chains[index]['extension_interval_ids']] for key in group})
        key = tuple(ids)
        if key not in cache:
            points = []; radii = []; core_radii = []
            for identifier in ids:
                p = np.asarray(intervals[identifier]['points_px'], float)
                used_points += len(p)
                if used_points > 2000000: incomplete = True; break
                q = np.rint(p).astype(int); measured = distance[q[:, 1], q[:, 0]]
                valid = measured > 0
                if valid.sum() < 3: incomplete = True; break
                core_radius = float(np.quantile(measured[valid], .25))
                core_radii.append(core_radius)
                # Junctions inflate DT: use the observed thin body as the upper
                # bound, plus the raster pixel's circumradius (scale follows input).
                points.append(p[valid]); radii.append(np.minimum(measured[valid], core_radius)+2**.5/2)
            if incomplete: break
            cache[key] = (np.concatenate(points), np.concatenate(radii), max(core_radii))
        source, radius, thin_radius = cache[key]
        y, x = np.nonzero(assignments == number); candidates = np.column_stack((x, y))
        maximum = float(radius.max()); lo = candidates.min(0)-maximum; hi = candidates.max(0)+maximum
        selected = ((source >= lo) & (source <= hi)).all(1); source_here, radius_here = source[selected], radius[selected]
        cost = len(candidates)*len(source_here)
        if pairs+cost > 2000000: incomplete = True; break
        pairs += cost; conflict = np.zeros(len(candidates), bool)
        for start in range(0, len(candidates), 64):
            p = candidates[start:start+64]
            squared = ((p[:, None, :]-source_here[None, :, :])**2).sum(2)
            conflict[start:start+64] = (squared <= radius_here[None, :]**2).any(1)
        # A source interval can end at the first intersection while the thin
        # stroke continues into a thick curved body. Nearest-source ownership
        # cannot resolve that contact. Abstain locally using both observed
        # stroke radii and the pixel footprint, without extending or relabeling it.
        feet = np.unique(np.asarray([foot for w in row['witnesses'] for foot in w['feet_px']], float), axis=0)
        contact_radius = float(row['body_radius_q95_px'])+thin_radius+2**.5/2
        cost = len(candidates)*len(feet)
        if pairs+cost > 2000000: incomplete = True; break
        pairs += cost
        conflict |= (((candidates[:, None, :]-feet[None, :, :])**2).sum(2) <= contact_radius**2).any(1)
        output[y[conflict], x[conflict]] = 255; assignments[y[conflict], x[conflict]] = 0
        count = int(conflict.sum()); row['added_pixels'] -= count; removed += count
        row['competitor_guard'] = {'source_interval_ids': ids, 'removed_pixels': count, 'contact_radius_px': contact_radius,
                                   'reason': 'competing_dimension_source_support_plus_pixel_footprint'}
    if incomplete:
        removed = result['added_pixels']; output[:] = inherited_roles; assignments[:] = 0
        for row in result['interval_admission']: row['added_pixels'] = 0
    result.update(schema=POLICY['version'], competitor_policy=deepcopy(POLICY),
                  competitor_guard={'source_points': used_points, 'pair_checks': pairs, 'removed_pixels': removed,
                                    'combined_raster_domain_pixels': combined_domain, 'incomplete': incomplete},
                  unguarded_added_pixels=result['added_pixels'],
                  added_pixels=int(np.count_nonzero(output != inherited_roles)),
                  search_incomplete=result['search_incomplete'] or incomplete,
                  additions_subset_of_v1=True)
    return result
