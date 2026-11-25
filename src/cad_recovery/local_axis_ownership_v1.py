"""Reject a new opposite taper only when its source witness belongs to another head.

This is an additional observation guard. It never erases source pixels, edits a
baseline observation, or certifies CAD semantics. Unproved/cyclic ownership is
retained for review, and all rejected observations remain in proposals.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from .arrow_path_recovery_v1 import native_head_check
from .native_head_bounds_v1 import safe_validator
from .family_catalog_v2 import require
from .sparse_local_heads_v1 import runs

POLICY = {'version': 'local-axis-ownership/1.0', 'maximum_pixels': 25000000,
    'maximum_source_samples': 131072, 'maximum_owner_validation_pixels': 1000000,
    'maximum_pair_checks': 32768, 'maximum_containment_points': 2000000,
    'sample_step_px': .5, 'maximum_witness_extent_px': 40.,
    'minimum_observed_witness_px': 6., 'minimum_owned_fraction': .95,
    'maximum_opposition_error_deg': 20., 'polygon_footprint_margin_px': .5,
    'baseline_immutable': True, 'source_erasure_allowed': False,
    'confidence_calibrated': False, 'production_admitted': False,
    'physical_geometry_approved': False}


def _geometry(head):
    polygon = np.asarray(head['polygon_px'], float); unit = np.asarray(head['direction'], float)
    require(polygon.shape == (3, 2) and unit.shape == (2,) and np.isfinite(polygon).all()
        and np.isfinite(unit).all() and abs(np.linalg.norm(unit)-1) < .01, 'Invalid head geometry')
    return polygon, unit


def _straight_witness(mask, head, audit):
    evidence = head.get('evidence', {})
    width = evidence.get('local_shaft_width_px'); extent = evidence.get('shaft_observed_extent_px')
    band = evidence.get('profile_band_px')
    if width is None or extent is None or band is None:
        return None
    width, extent = float(width), float(extent)
    require(np.isfinite([width, extent, *band]).all() and width > 0 and extent > 0,
            'Invalid claimed shaft support')
    if extent > POLICY['maximum_witness_extent_px']: return None
    polygon, unit = _geometry(head); base = polygon[1:].mean(0)
    origin = base-float(band[0])*unit
    step = POLICY['sample_step_px']; count = int(math.floor(extent/step))
    if audit['source_samples']+count > POLICY['maximum_source_samples']:
        audit['limits'].add('ownership_source_sample_budget'); return None
    audit['source_samples'] += count
    along = -(np.arange(count)+.5)*step
    points = origin+along[:, None]*unit; q = np.rint(points).astype(int)
    inside = (q >= 0).all(1) & (q[:, 0] < mask.shape[1]) & (q[:, 1] < mask.shape[0])
    ink = np.zeros(count, bool); ink[inside] = mask[q[:, 1][inside], q[:, 0][inside]] > 0
    spans = [(b-a, a, b) for a, b in runs(ink)]
    if not spans: return None
    _, a, b = max(spans); a, b = int(a), int(b)
    if (b-a)*step < max(POLICY['minimum_observed_witness_px'], 2*width): return None
    return {'points': points[a:b], 'base': base, 'origin': origin,
        'extent_px': (b-a)*step, 'sample_count': b-a,
        'along_interval_px': [float(along[b-1]-step/2), float(along[a]+step/2)],
        'method': 'independently_reobserved_centerline_from_profile_origin_not_original_profile_replay'}


def _witness(mask, head, audit):
    path = head.get('evidence', {}).get('shaft_support_path')
    if path is None: return _straight_witness(mask, head, audit)
    points = np.asarray(path['points_px'], float)
    require(points.ndim == 2 and points.shape[1] == 2 and 2 <= len(points) <= 41
        and np.isfinite(points).all(), 'Invalid source support path')
    width = float(path['width_px']); segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    extent = float(segments.sum())
    require(np.isfinite(width) and width > 0 and (segments > 0).all()
        and extent <= POLICY['maximum_witness_extent_px']+1e-6, 'Invalid path extent')
    samples = np.concatenate([a+np.linspace(0,1,max(3,math.ceil(d/.5)+1))[:,None]*(b-a)
        for a,b,d in zip(points[:-1],points[1:],segments)])
    if audit['source_samples']+len(samples) > POLICY['maximum_source_samples']:
        audit['limits'].add('ownership_source_sample_budget'); return None
    audit['source_samples'] += len(samples)
    q=np.rint(samples).astype(int)
    inside=(q>=0).all(1)&(q[:,0]<mask.shape[1])&(q[:,1]<mask.shape[0])
    ink=np.zeros(len(q),bool);ink[inside]=mask[q[:,1][inside],q[:,0][inside]]>0
    if (not inside.all() or np.mean(ink)<.95
        or extent<max(POLICY['minimum_observed_witness_px'],2*width)):return None
    polygon,_=_geometry(head)
    return {'points':samples,'base':polygon[1:].mean(0),'origin':points[0],
        'extent_px':extent,'sample_count':len(samples),
        'method':'independent_source_reobservation_of_curved_support_path'}


def _resolve(relations, baseline_ids):
    """Resolve a dependency DAG conservatively; retain unsupported cycles."""
    independent = set(baseline_ids) | {h for h, owners in relations.items() if not owners}
    unresolved = {h for h, owners in relations.items() if owners}; excluded = {}
    while unresolved:
        rejected = {h: sorted((o for o in relations[h] if o['owner_id'] in independent),
                             key=lambda x: x['owner_id']) for h in unresolved}
        rejected = {h: value for h, value in rejected.items() if value}
        if not rejected: break
        excluded.update(rejected); unresolved -= rejected.keys()
        released = {h for h in unresolved if all(o['owner_id'] in excluded for o in relations[h])}
        independent |= released; unresolved -= released
    return excluded, unresolved


def apply(gray, packet, *, baseline=()):
    require(isinstance(gray, np.ndarray) and gray.ndim == 2 and gray.dtype == np.uint8
            and 0 < gray.size <= POLICY['maximum_pixels'], 'Expected bounded grayscale source')
    require(len(baseline) <= 512 and len(packet['added']) <= 512, 'Too many head observations')
    require(packet['candidates'][:len(baseline)] == list(baseline), 'Baseline prefix mismatch')
    heads = list(baseline)+packet['added']; identities = [h['id'] for h in heads]
    require(len(set(identities)) == len(identities), 'Ambiguous observation IDs')
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    audit = {'threshold_preparation_pixels': int(gray.size), 'source_samples': 0,
        'owner_validation_pixels': 0, 'pair_checks': 0, 'containment_points': 0, 'limits': set()}
    baseline_ids = {h['id'] for h in baseline}; validations = {}; relations = {}; witnesses = {}
    for candidate in packet['added']:
        identity = candidate['id']; relations[identity] = []
        witness = _witness(mask, candidate, audit); witnesses[identity] = witness
        if witness is None: continue
        _, direction = _geometry(candidate)
        for owner in heads:
            if identity == owner['id']: continue
            if audit['pair_checks'] >= POLICY['maximum_pair_checks']:
                audit['limits'].add('ownership_pair_budget'); break
            audit['pair_checks'] += 1
            evidence = owner.get('evidence', {})
            if not evidence.get('source_checked') or not evidence.get('local_shaft_width_px'): continue
            polygon, owner_direction = _geometry(owner)
            dot = float(direction@owner_direction)
            if dot > -math.cos(math.radians(POLICY['maximum_opposition_error_deg'])): continue
            margin = POLICY['polygon_footprint_margin_px']
            if ((witness['origin'] < polygon.min(0)-margin).any()
                or (witness['origin'] > polygon.max(0)+margin).any()): continue
            if owner['id'] not in validations:
                lo = np.floor(polygon.min(0)); hi = np.ceil(polygon.max(0))+1
                cost = int(np.prod(hi-lo))+32
                if audit['owner_validation_pixels']+cost > POLICY['maximum_owner_validation_pixels']:
                    audit['limits'].add('ownership_owner_validation_budget'); break
                audit['owner_validation_pixels'] += cost
                valid, check = safe_validator(native_head_check, [])(owner, mask,
                    float(evidence['local_shaft_width_px']), ())
                validations[owner['id']] = {'available': bool(valid), 'native_check': check}
            if not validations[owner['id']]['available']: continue
            cost = len(witness['points'])+1
            if audit['containment_points']+cost > POLICY['maximum_containment_points']:
                audit['limits'].add('ownership_containment_budget'); break
            audit['containment_points'] += cost
            contour = polygon.astype(np.float32)
            base_distance = cv2.pointPolygonTest(contour, tuple(witness['base']), True)
            if base_distance < -margin: continue
            owned = np.array([cv2.pointPolygonTest(contour, tuple(p), True) >= -margin
                              for p in witness['points']])
            fraction = float(owned.mean())
            if fraction >= POLICY['minimum_owned_fraction']:
                relations[identity].append({'owner_id': owner['id'], 'direction_dot': dot,
                    'owned_fraction': fraction, 'owned_samples': int(owned.sum()),
                    'candidate_base_signed_distance_px': float(base_distance),
                    'owner_native_check': deepcopy(validations[owner['id']]['native_check'])})
        if audit['limits']: break
    # Resolve independently of input order. A rejected new hypothesis cannot
    # take ownership of another observation; mutually dependent cycles remain.
    excluded, unresolved = _resolve(relations, baseline_ids)
    decisions = []
    for head in packet['added']:
        identity = head['id']; witness = witnesses.get(identity)
        if identity in excluded: reason = 'source_shaft_witness_owned_by_opposite_head'
        elif identity in unresolved: reason = 'mutual_ownership_unresolved_retained'
        elif identity not in relations: reason = 'ownership_budget_unchecked_retained'
        elif witness is None: reason = 'no_reobserved_witness_retained'
        else: reason = 'no_reused_opposite_head_support_proved'
        record = {'candidate_id': identity, 'excluded': identity in excluded,
            'reason': reason, 'observation_retained': True, 'relations': relations.get(identity, [])}
        if witness is not None:
            record['witness'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                 for k, v in witness.items()}
        decisions.append(record)
    # On incomplete guard evaluation make no new exclusions. The incomplete flag
    # prevents admission; untouched observations remain reviewable.
    if audit['limits']:
        for decision in decisions:
            decision['excluded'] = False; decision['reason'] = 'ownership_budget_unchecked_retained'
        excluded = {}
    retained = [h for h in packet['added'] if h['id'] not in excluded]
    result = deepcopy(packet); result['schema'] = POLICY['version']
    result['ownership_policy'] = deepcopy(POLICY)
    result['candidates'] = deepcopy(list(baseline)+retained); result['added'] = deepcopy(retained)
    result['ownership_decisions'] = decisions
    result['ownership_exclusions'] = [d for d in decisions if d['excluded']]
    result['source_search_incomplete'] = packet['search_incomplete']
    result['ownership_incomplete'] = bool(audit['limits'])
    audit['limits'] = sorted(audit['limits']); result['analysis']['ownership'] = audit
    result['analysis']['accepted_additions'] = len(retained)
    result['analysis']['limits'] = sorted(set(packet['analysis']['limits']) | set(audit['limits']))
    result['search_incomplete'] = packet['search_incomplete'] or bool(audit['limits'])
    result['analysis']['coverage_complete'] = packet['analysis']['coverage_complete'] and not audit['limits']
    return result


def search(gray, *, baseline=(), text_polygons=(), eager_reference=False):
    """Fresh public pipeline; apply() also supports SHA-pinned detector packets."""
    from .local_head_context_v2 import search as source_search
    packet = source_search(gray, baseline=baseline, text_polygons=text_polygons,
                           eager_reference=eager_reference)
    return apply(gray, packet, baseline=baseline)
