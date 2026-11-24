"""Source-validated short-head profiles over sparse distance superlevel regions."""
from collections import Counter
from copy import deepcopy
import math

import cv2
import numpy as np

from .arrow_path_recovery_v1 import native_head_check
from .native_head_bounds_v1 import safe_validator
from .local_width_heads_v1 import append_preserving
from .sparse_head_regions_v1 import regions
from .family_catalog_v2 import require

POLICY = {'version': 'sparse-local-heads/1.0', 'radii_px': [7, 5, 4, 3, 2, 1],
    'maximum_pixels': 25000000, 'maximum_active_pixels': 2000000, 'maximum_union_checks': 8000000,
    'maximum_component_memberships': 8000000, 'maximum_morphology_pixels': 40000000,
    'maximum_components_per_level': 256, 'maximum_proposals': 512, 'maximum_candidates': 512,
    'maximum_profile_samples': 8000000, 'maximum_profile_cache_entries': 4096,
    'profile_step_px': .5, 'minimum_shaft_length_px': 6., 'minimum_base_to_shaft_ratio': 2.5,
    'minimum_shaft_continuity': .95, 'maximum_taper_relative_error': .30,
    'head_length_range_px': [5, 192], 'head_aspect_range': [1.2, 8.],
    'source_erasure_allowed': False, 'DT_head_proposals_used': False, 'LINE_fitting_required': False,
    'manual_references_used': False, 'confidence_calibrated': False,
    'production_admitted': False, 'physical_geometry_approved': False}


class Profiles:
    def __init__(self, mask):
        self.mask = mask; self.field = mask.astype(np.float32)/255
        self.samples = 0; self.hits = 0; self.cache = {}; self.incomplete = False

    def measure(self, base, unit, length, width):
        key = tuple(np.r_[base, unit, length, width])
        if key in self.cache:
            self.hits += 1; return self.cache[key]
        along = np.arange(-min(40., max(14., length)), min(192., max(8., 2.5*length))+.25, .5)
        across = np.arange(-min(96., width+3), min(96., width+3)+.25, .5)
        cost = int(len(along)*len(across))
        if self.samples+cost > POLICY['maximum_profile_samples'] or len(self.cache) >= POLICY['maximum_profile_cache_entries']:
            self.incomplete = True; return None
        self.samples += cost
        normal = np.array([-unit[1], unit[0]])
        point = base+along[:, None, None]*unit+across[None, :, None]*normal
        pixel = np.rint(point).astype(int)
        inside = (pixel >= 0).all(2) & (pixel[:, :, 0] < self.mask.shape[1]) & (pixel[:, :, 1] < self.mask.shape[0])
        ink = np.zeros(inside.shape, bool); ink[inside] = self.mask[pixel[:, :, 1][inside], pixel[:, :, 0][inside]] > 0
        coverage = cv2.remap(self.field, point[:, :, 0].astype(np.float32), point[:, :, 1].astype(np.float32),
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        widths = np.zeros(len(along)); centers = np.full(len(along), np.inf)
        for i, row in enumerate(ink):
            changes = np.diff(np.r_[False, row, False].astype(int))
            spans = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
            if not spans: continue
            a, b = min(spans, key=lambda s: abs((across[s[0]]+across[s[1]-1])/2))
            center = (across[a]+across[b-1])/2
            if abs(center) > max(2., width*.25) or not inside[i, a:b].all(): continue
            lo, hi = max(0, a-2), min(len(across), b+2)
            observed = coverage[i, lo:hi]
            mass = float(observed.sum())
            if mass <= 0: continue
            widths[i], centers[i] = mass*.5, float(observed@across[lo:hi]/mass)
        value = along, widths, centers
        self.cache[key] = value
        return value


def runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))


def shaft_run(along, widths, seed_width):
    options = []
    # A reconstructed seed can be narrower than the original head. This is a
    # proposal band only; final observed base/shaft contrast remains >= 2.5.
    for a, b in runs((along < 0) & (widths > 0) & (widths <= seed_width*.8)):
        width = float(np.median(widths[a:b])); extent = float(along[b-1]-along[a]+.5)
        if extent < max(6., 2*width) or along[b-1] < -max(3., 2*width): continue
        bridge = (along >= along[b-1]) & (along <= 0)
        if np.mean(widths[bridge] > 0) >= .95: options.append((a, b, width, extent))
    return max(options, key=lambda r: r[3]) if options else None


def validate(base, unit, length, width, profiles, text_polygons):
    # Cheap source evidence rejects isolated glyph/blob axes before an expensive
    # profile. This is only a proposal gate; final continuity/contrast is stricter.
    normal = np.array([-unit[1], unit[0]])
    probe = base+np.array([-.5, -2., -4., -6.])[:, None, None]*unit+np.array([-1., 0., 1.])[None, :, None]*normal
    if profiles.samples+12 > POLICY['maximum_profile_samples']:
        profiles.incomplete = True; return None, 'profile_sample_budget'
    profiles.samples += 12
    q = np.rint(probe).astype(int)
    inside = (q >= 0).all(2) & (q[:, :, 0] < profiles.mask.shape[1]) & (q[:, :, 1] < profiles.mask.shape[0])
    observed = np.zeros(inside.shape, bool)
    observed[inside] = profiles.mask[q[:, :, 1][inside], q[:, :, 0][inside]] > 0
    if np.mean(observed.any(1)) < .75: return None, 'no_near_base_source_run'
    measured = profiles.measure(base, unit, length, width)
    if measured is None: return None, 'profile_sample_budget'
    along, widths, centers = measured; shaft = shaft_run(along, widths, width)
    if shaft is None: return None, 'no_connected_source_shaft'
    a, b, local_width, extent = shaft
    # A local observed centerline may refine the seed axis, independent of page LINE fitting.
    slope, intercept = np.polyfit(along[a:b], centers[a:b], 1)
    if abs(slope) <= math.tan(math.radians(20)) and abs(intercept) <= max(2., local_width):
        normal = np.array([-unit[1], unit[0]])
        adjusted = unit+slope*normal; adjusted /= np.linalg.norm(adjusted)
        base = base+intercept*normal; unit = adjusted
        measured = profiles.measure(base, unit, length, width)
        if measured is None: return None, 'profile_sample_budget'
        along, widths, centers = measured; shaft = shaft_run(along, widths, width)
        if shaft is None: return None, 'refined_axis_has_no_connected_shaft'
        a, b, local_width, extent = shaft
    band = (along >= 0) & (widths >= max(2., local_width*1.25)) & (widths <= width*1.5)
    candidates = []
    for start, end in runs(band):
        if along[start] > max(3., 2*local_width) or along[end-1]-along[start] < max(3., 1.5*local_width): continue
        peak = start+int(np.argmax(widths[start:end])); lo, hi = along[peak], along[end-1]
        if hi-lo < max(3., 1.5*local_width): continue
        chosen = (along >= lo+.15*(hi-lo)) & (along <= lo+.85*(hi-lo))
        if chosen.sum() < 4: continue
        # Integrated raster support includes the one-pixel footprint; geometry uses pixel centers.
        geometric_width = np.maximum(0, widths[chosen]-1.)
        taper, intercept = np.polyfit(along[chosen], geometric_width, 1)
        if taper >= -.05: continue
        expected = taper*along[chosen]+intercept
        error = float(np.median(np.abs(geometric_width-expected)/np.maximum(local_width, expected)))
        root = float(-intercept/taper); base_width = float(intercept+taper*lo); head_length = root-lo
        if (error > .30 or base_width/local_width < 2.5 or not 5 <= head_length <= 192
            or root < hi-.5 or root > hi+max(3., 2*local_width)
            or not 1.2 <= head_length/base_width <= 8): continue
        normal = np.array([-unit[1], unit[0]])
        head_base = base+lo*unit
        polygon = np.array([base+root*unit, head_base+base_width/2*normal, head_base-base_width/2*normal])
        head = {'tip_px': polygon[0].tolist(), 'direction': unit.tolist(), 'polygon_px': polygon.tolist(),
            'classification': 'filled_arrow_candidate', 'uncertainty_px': max(.75, local_width/2), 'confidence': .55,
            'evidence': {'method': POLICY['version'], 'path_kind': 'observed_taper_and_own_shaft',
                'local_shaft_width_px': local_width, 'shaft_observed_extent_px': extent,
                'shaft_extent_includes_sample_footprint': True, 'base_to_shaft_ratio': float(base_width/local_width),
                'taper_relative_error': error, 'head_length_px': head_length, 'base_width_px': base_width,
                'taper_width_excludes_pixel_footprint': True,
                'profile_band_px': [float(lo), float(hi)], 'confidence_calibrated': False}}
        valid, check = safe_validator(native_head_check, [])(head, profiles.mask, local_width, text_polygons)
        if not valid: continue
        x, y = np.rint(polygon[0]).astype(int); radius = max(1, math.ceil(local_width))
        left, top = max(0, x-radius), max(0, y-radius)
        right, bottom = min(profiles.mask.shape[1], x+radius+1), min(profiles.mask.shape[0], y+radius+1)
        if right <= left or bottom <= top or not profiles.mask[top:bottom, left:right].any(): continue
        head['evidence'].update(native_head_check=check, source_checked=True)
        candidates.append(head)
    if not candidates: return None, 'no_source_validated_taper'
    # Multiple nonoverlapping taper tips remain ambiguous; do not average them.
    combined, _ = append_preserving([], candidates)
    if len(combined) != 1: return None, 'ambiguous_source_taper'
    return candidates[0], None


def axes(contour):
    p = np.asarray(contour).reshape(-1, 2).astype(float)
    if len(p) < 3: return []
    _, vectors = np.linalg.eigh(np.cov(p.T)); axis = vectors[:, -1]
    directions = [axis, -axis]
    _, raw = cv2.minEnclosingTriangle(contour)
    if raw is not None:
        triangle = raw.reshape(3, 2)
        for i in range(3):
            d = triangle[i]-triangle[[j for j in range(3) if j != i]].mean(0)
            if np.linalg.norm(d) > 0: directions.append(d/np.linalg.norm(d))
    result = []
    for unit in directions:
        if any(unit@other[1] > math.cos(math.radians(4)) for other in result): continue
        normal = np.array([-unit[1], unit[0]]); projection = p@unit; across = p@normal
        extent, width = float(np.ptp(projection)), float(np.ptp(across))
        if extent < 3 or width < 2 or not .8 <= extent/width <= 10: continue
        base = projection.min()*unit+np.median(across)*normal
        result.append((base, unit.copy(), extent, width))
    return result


def search(gray, *, baseline=(), text_polygons=(), dense_reference=False):
    require(isinstance(gray, np.ndarray) and gray.ndim == 2 and gray.dtype == np.uint8
            and 0 < gray.size <= POLICY['maximum_pixels'], 'Expected bounded uint8 grayscale')
    require(len(baseline) <= 512 and len(text_polygons) <= 256, 'Bounded heads/text required')
    for p in text_polygons:
        q = np.asarray(p, float)
        require(q.ndim == 2 and q.shape[1] == 2 and 3 <= len(q) <= 4096
                and np.isfinite(q).all() and np.abs(q).max() <= 1e7, 'Invalid text polygon')
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    audit = {'distance_preparation_pixels': 0, 'sparse_active_pixels': 0, 'dense_threshold_pixels': 0,
        'union_neighbor_checks': 0, 'component_memberships': 0, 'morphology_pixels': 0, 'components_examined': 0,
        'directional_proposals': 0, 'duplicate_supported_regions': 0, 'limits': set()}
    profiles = Profiles(mask); accepted = []; reasons = Counter()
    for radius, components in regions(mask, POLICY['radii_px'], audit, POLICY, dense=dense_reference):
        if len(components) > POLICY['maximum_components_per_level']: audit['limits'].add('component_budget')
        for region in components[:POLICY['maximum_components_per_level']]:
            if audit['components_examined'] >= POLICY['maximum_proposals']:
                audit['limits'].add('proposal_budget'); break
            audit['components_examined'] += 1
            x, y, w, h = region['box']; left, top = max(0, x-radius), max(0, y-radius)
            right, bottom = min(mask.shape[1], x+w+radius), min(mask.shape[0], y+h+radius)
            cost = (right-left)*(bottom-top)
            if audit['morphology_pixels']+cost > POLICY['maximum_morphology_pixels']:
                audit['limits'].add('morphology_pixel_budget'); break
            audit['morphology_pixels'] += cost
            seed = np.zeros((bottom-top, right-left), np.uint8); points = region['points']-[left, top]
            seed[points[:, 1], points[:, 0]] = 255
            opened = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
            opened &= mask[top:bottom, left:right]
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 4: continue
                contour = contour+np.array([left, top], np.int32)
                # Reuse already observed heads, rather than profiling their body at every radius.
                covered = False
                for head in list(baseline)+accepted:
                    polygon = np.asarray(head['polygon_px'], np.float32)
                    if all(cv2.pointPolygonTest(polygon, tuple(map(float, p)), False) >= 0 for p in contour.reshape(-1, 2)):
                        covered = True; break
                if covered:
                    audit['duplicate_supported_regions'] += 1; continue
                for base, unit, length, width in axes(contour):
                    audit['directional_proposals'] += 1
                    head, reason = validate(base, unit, length, width, profiles, text_polygons)
                    if reason: reasons[reason] += 1
                    else:
                        head['evidence']['source_level_radius_px'] = radius; accepted.append(head)
                    if profiles.incomplete: break
                if profiles.incomplete: break
            if profiles.incomplete or 'morphology_pixel_budget' in audit['limits']: break
        if profiles.incomplete or 'proposal_budget' in audit['limits'] or 'morphology_pixel_budget' in audit['limits']: break
    if profiles.incomplete: audit['limits'].add('profile_sample_budget')
    combined, added = append_preserving(baseline, accepted)
    if len(combined) > 512:
        audit['limits'].add('candidate_budget'); combined = combined[:512]; added = combined[len(baseline):]
    # Retain original observation IDs exactly, use a separate namespace for new heads.
    old_ids = {h['id'] for h in baseline}
    for i, head in enumerate(added, 1):
        identity = 'sparse-head-'+str(i).zfill(5)
        while identity in old_ids: identity += '-new'
        head['id'] = identity; old_ids.add(identity)
    for i, head in enumerate(added): combined[len(baseline)+i] = deepcopy(head)
    audit.update(profile_samples=profiles.samples, profile_cache_hits=profiles.hits, profile_cache_entries=len(profiles.cache),
        rejections=dict(reasons), limits=sorted(audit['limits']), raw_accepted=len(accepted), added=len(added))
    return {'schema': POLICY['version'], 'policy': deepcopy(POLICY), 'candidates': combined, 'added': added,
        'analysis': audit, 'search_incomplete': bool(audit['limits']), 'dense_reference': dense_reference,
        'production_admitted': False, 'physical_geometry_approved': False, 'source_modified': False, 'CAD_promotions': 0}
