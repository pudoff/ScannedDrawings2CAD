"""Compare a finite ink-model family and retain common observed source wings.

This is a local, conditional rendering experiment.  Neither a compatible
renderer nor its common pixels identify a whole semantic CAD arrowhead.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

from . import native_head_shape_v1 as shape_observer
from .native_dimension_transfer_v1 import _sha
from .part_admission_v1 import object_sha


POLICY = {'version': 'native-head-ink/1.0', 'maximum_roi_pixels': 4_000_000,
    'maximum_crop_side': 512, 'maximum_crop_pixels': 262_144,
    'maximum_models': 256, 'maximum_rendered_pixels': 32_000_000,
    'maximum_heads': 2, 'maximum_rle_runs': 100_000,
    'maximum_tolerance_radius_px': 16, 'maximum_evidence_violations': 4,
    'maximum_heldout_violations': 4, 'flank_phases_px': [-.25, 0., .25],
    'default_split_stripe_width_px': 4., 'semantic_exclusive_head_confirmed': False,
    'physical_geometry_approved': False, 'whole_triangle_ownership_confirmed': False,
    'head_uncertainty_reduced': False, 'roles_changed': 0, 'cad_modification': False}


class _Limit(ValueError):
    pass


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _rle(mask, origin=(0, 0)):
    rows = []
    for y, bits in enumerate(mask):
        changes = np.diff(np.r_[False, bits, False].astype(np.int8))
        starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        if len(rows)+len(starts) > POLICY['maximum_rle_runs']:
            raise _Limit('ink_consensus_rle_budget')
        rows.extend([[y+int(origin[1]), int(a)+int(origin[0]), int(b)+int(origin[0])]
                     for a, b in zip(starts, stops)])
    return rows


def _decode(rows, box):
    left, top, right, bottom = box
    mask = np.zeros((bottom-top, right-left), bool)
    for y, a, b in rows:
        if top <= y < bottom and max(a, left) < min(b, right):
            mask[y-top, max(a, left)-left:min(b, right)-left] = True
    return mask


def _score(prediction, ink, mask):
    return {'tp': int(np.count_nonzero(mask & prediction & ink)),
        'fp': int(np.count_nonzero(mask & prediction & ~ink)),
        'fn': int(np.count_nonzero(mask & ~prediction & ink)),
        'tn': int(np.count_nonzero(mask & ~prediction & ~ink)),
        'pixels': int(np.count_nonzero(mask))}


def _render(polygon, radius, x, y):
    if polygon is None:
        return np.zeros(np.broadcast_shapes(x.shape, y.shape), bool)
    vertices = np.asarray(polygon, float)
    products = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        delta = b-a
        products.append(delta[0]*(y-a[1])-delta[1]*(x-a[0]))
    products = np.asarray(products)
    mask = np.all(products >= -1e-9, axis=0) | np.all(products <= 1e-9, axis=0)
    if radius > 0:
        for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
            delta = b-a
            t = np.clip(((x-a[0])*delta[0]+(y-a[1])*delta[1])/float(delta@delta), 0., 1.)
            mask |= (x-a[0]-t*delta[0])**2+(y-a[1]-t*delta[1])**2 <= radius**2+1e-9
    return mask


def _models(head):
    result = [{'id': 'M0', 'family': 'no_head', 'radius_px': 0., 'polygon_px': None,
               'logical_apex_px': None, 'base_observation_px': None}]
    if not head.get('shape_pattern_observed'):
        return result
    p = head['profile']; reference = np.asarray(p['reference_tip_px'], float)
    axis, normal = np.asarray(p['axis_unit']), np.asarray(p['normal_unit'])
    left, right = head['sides']['left'], head['sides']['right']
    width, uncertainty = float(head['shaft_width_px']), float(head['axis_uncertainty_px'])
    radii = sorted({0., max(0., width/2-uncertainty), width/2, width/2+uncertainty})
    lo, hi = map(float, head['base']['interval_along_axis_px'])
    phases = POLICY['flank_phases_px']
    if 1+len(radii)*3*len(phases)**2 > POLICY['maximum_models']:
        raise _Limit('ink_model_count_budget')
    for radius in radii:
        for base in (lo, (lo+hi)/2, hi):
            for dl in phases:
                for dr in phases:
                    al, ar = float(left['slope']), float(right['slope'])
                    bl = float(left['intercept'])+float(dl)+radius*math.sqrt(1+al*al)
                    br = float(right['intercept'])+float(dr)-radius*math.sqrt(1+ar*ar)
                    if ar-al <= 1e-8:
                        continue
                    apex_s = (bl-br)/(ar-al)
                    apex_u = al*apex_s+bl
                    base_s = base-radius
                    ul, ur = al*base_s+bl, ar*base_s+br
                    if base_s <= apex_s or ur <= ul:
                        continue
                    polygon = [reference+axis*apex_s+normal*apex_u,
                        reference+axis*base_s+normal*ul, reference+axis*base_s+normal*ur]
                    result.append({'id': 'M'+str(len(result)),
                        'family': 'filled_triangle' if radius == 0 else 'round_stroked_filled_triangle',
                        'radius_px': float(radius), 'polygon_px': [v.tolist() for v in polygon],
                        'logical_apex_px': polygon[0].tolist(), 'base_observation_px': float(base),
                        'left_flank_phase_px': float(dl), 'right_flank_phase_px': float(dr)})
    return result


def _fresh_shape(gray, head):
    if head.get('source_window_sha256') != _sha(gray) or head.get('source_size_px') != list(gray.shape[::-1]):
        raise ValueError('Current native ink shape window differs')
    if not head.get('complete'):
        return
    profile = head.get('profile')
    if not profile:
        raise ValueError('Complete source shape sampling profile required')
    fresh = shape_observer.measure(gray, tip_px=profile['reference_tip_px'],
        toward_base=profile['axis_unit'], shaft_width_px=head['shaft_width_px'],
        axis_uncertainty_px=head['axis_uncertainty_px'], search_length_px=head['search_length_px'],
        backward_length_px=head['backward_length_px'], half_width_px=head['half_width_px'],
        head_id=head.get('head_id'), tip_uncertainty_px=head.get('preserved_tip_bound_px'))['report']
    if any(key not in head for key in fresh) or object_sha(fresh) != object_sha({key: head[key] for key in fresh}):
        raise ValueError('Persisted source shape differs from fresh native sampling')


def _mask(value, shape, label):
    if value is None:
        return None
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != bool:
        raise ValueError('Aligned boolean '+label+' required')
    return value


def measure(gray, shape_head_report, *, nuisance_possible=None, visible_mask=None,
            evidence_mask=None, heldout_mask=None):
    """Compare every fixed model; logical contact never filters pixel models."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError('Native uint8 source window required')
    supplied = [_mask(value, gray.shape, label) for value, label in (
        (nuisance_possible, 'nuisance mask'), (visible_mask, 'visible mask'),
        (evidence_mask, 'evidence mask'), (heldout_mask, 'reserved mask'))]
    if (supplied[2] is None) != (supplied[3] is None):
        raise ValueError('Both evidence and reserved masks must be supplied together')
    before = _sha(gray), object_sha(shape_head_report), [None if a is None else _sha(a) for a in supplied]
    head = shape_head_report
    report = {'schema_version': POLICY['version'], 'policy': deepcopy(POLICY),
        'head_id': head.get('head_id'), 'source_window_sha256': _sha(gray),
        'shape_head_sha256': object_sha(head), 'source_size_px': list(gray.shape[::-1]),
        'complete': False, 'limits': [], 'models': [], 'image_compatible_ids': [],
        'joint_compatible_ids': [], 'image_compatible_count': 0, 'joint_compatible_count': 0,
        'null_model_compatible': False, 'model_consensus_pixels': 0, 'consensus_rle': [],
        'semantic_exclusive_head_confirmed': False, 'physical_geometry_approved': False,
        'whole_triangle_ownership_confirmed': False, 'logical_identifiability_confirmed': False,
        'head_uncertainty_reduced': False, 'roles_changed': 0, 'cad_modification': False,
        'finite_grid_consensus_not_continuous_parameter_proof': True,
        'validation_is_local_conditional_on_existing_shape_proposal': True,
        'independent_acceptance_claimed': False, 'winner_selected': False,
        'input_mask_sha256': {label: None if mask is None else _sha(mask) for label, mask in
            zip(('nuisance_possible', 'visible', 'evidence', 'heldout'), supplied)},
        'partial_observation_rolled_back': False}
    consensus_global = None
    try:
        if gray.size > POLICY['maximum_roi_pixels']:
            raise _Limit('ink_source_roi_budget')
        _fresh_shape(gray, shape_head_report)
        consensus_global = np.zeros(gray.shape, bool)
        if not head.get('complete'):
            raise _Limit('preceding_shape_sampling_incomplete')
        profile = head['profile']; ref = np.asarray(profile['reference_tip_px'])
        axis, normal = np.asarray(profile['axis_unit']), np.asarray(profile['normal_unit'])
        corners = np.asarray([ref+axis*s+normal*u for s in (min(profile['station_positions_px']),
            max(profile['station_positions_px'])) for u in (min(profile['transverse_offsets_px']), max(profile['transverse_offsets_px']))])
        low = np.maximum(0, np.floor(corners.min(0)).astype(int))
        high = np.minimum(gray.shape[::-1], np.ceil(corners.max(0)).astype(int)+1)
        box = [int(low[0]), int(low[1]), int(high[0]), int(high[1])]
        crop_shape = (box[3]-box[1], box[2]-box[0])
        if min(crop_shape) <= 0 or max(crop_shape) > POLICY['maximum_crop_side'] or math.prod(crop_shape) > POLICY['maximum_crop_pixels']:
            raise _Limit('ink_render_crop_budget')
        models = _models(head)
        if len(models) > POLICY['maximum_models']:
            raise _Limit('ink_model_count_budget')
        if len(models)*math.prod(crop_shape) > POLICY['maximum_rendered_pixels']:
            raise _Limit('ink_render_pixel_budget')
        ys, xs = slice(box[1], box[3]), slice(box[0], box[2])
        y, x = np.ogrid[box[1]:box[3], box[0]:box[2]]
        s = (x-ref[0])*axis[0]+(y-ref[1])*axis[1]
        u = (x-ref[0])*normal[0]+(y-ref[1])*normal[1]
        otsu_threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        ink = binary[ys, xs] > 0
        nuisance = np.zeros(crop_shape, bool) if supplied[0] is None else supplied[0][ys, xs].copy()
        visible = np.ones(crop_shape, bool) if supplied[1] is None else supplied[1][ys, xs].copy()
        if supplied[2] is None:
            stripe = np.floor(s/POLICY['default_split_stripe_width_px']).astype(int) % 2 == 0
            if head.get('base'):
                end = float(head['base']['center_along_axis_px'])
                reserve_base = (abs(s-end) <= max(4., float(head['shaft_width_px']))) & (abs(u) > float(head['shaft_width_px'])/2)
            else:
                reserve_base = np.zeros(crop_shape, bool)
            evidence, heldout = visible & stripe & ~reserve_base, visible & (~stripe | reserve_base)
        else:
            evidence, heldout = supplied[2][ys, xs] & visible, supplied[3][ys, xs] & visible
            overlap = np.any(evidence & heldout)
            outside_visible = supplied[1] is not None and np.any((supplied[2] | supplied[3]) & ~supplied[1])
            if overlap or outside_visible:
                raise ValueError('Evidence and reserved masks must be disjoint within visibility')
        evidence &= ~nuisance
        heldout &= ~nuisance
        maximum_residual = max((float(side['max_residual_px']) for side in (head.get('sides') or {}).values()), default=0.)
        tolerance = math.ceil(maximum_residual+math.sqrt(.5)+max(abs(float(p)) for p in POLICY['flank_phases_px']))
        if tolerance > POLICY['maximum_tolerance_radius_px']:
            raise _Limit('ink_boundary_tolerance_budget')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
        predictions = [_render(model['polygon_px'], float(model['radius_px']), x, y) for model in models]
        candidate_union = np.zeros(crop_shape, bool)
        for prediction in predictions:
            candidate_union |= prediction
        domain_radius = tolerance+1
        domain_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2*domain_radius+1, 2*domain_radius+1))
        domain = (cv2.dilate(candidate_union.astype(np.uint8), domain_kernel,
            borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0) if len(models) > 1 else np.ones(crop_shape, bool)
        original_visibility = visible.copy()
        out_of_scope_ink = ink & original_visibility & ~domain
        visible &= domain
        evidence &= domain
        heldout &= domain
        intersection = np.ones(crop_shape, bool)
        certain_intersection = np.ones(crop_shape, bool)
        for model, prediction in zip(models, predictions):
            certain = cv2.erode(prediction.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            possible = cv2.dilate(prediction.astype(np.uint8), kernel,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            stats = {}
            for label, mask in (('fit', evidence), ('heldout', heldout)):
                stats[label] = _score(prediction, ink, mask)
                fp = int(np.count_nonzero(mask & certain & ~ink))
                fn = int(np.count_nonzero(mask & ink & ~possible))
                stats[label+'_tolerance'] = {'white_inside_certain_head': fp,
                    'ink_outside_possible_head': fn, 'violations': fp+fn}
            image_ok = (stats['fit_tolerance']['violations'] <= POLICY['maximum_evidence_violations']
                and stats['heldout_tolerance']['violations'] <= POLICY['maximum_heldout_violations'])
            bound = head.get('preserved_tip_bound_px')
            distance = None if model['logical_apex_px'] is None else float(np.linalg.norm(np.asarray(model['logical_apex_px'])-ref))
            contact_ok = distance is not None and bound is not None and distance <= float(bound)
            record = {**model, **stats, 'prediction_sha256': _sha(prediction),
                'prediction_ink_pixels': int(prediction.sum()), 'image_compatible': bool(image_ok),
                'contact_compatible': bool(contact_ok), 'logical_apex_reference_distance_px': distance,
                'preserved_tip_bound_px': bound, 'contact_prior_filters_pixel_models': False}
            report['models'].append(record)
            if image_ok:
                intersection &= prediction
                certain_intersection &= certain
                report['image_compatible_ids'].append(model['id'])
                if contact_ok:
                    report['joint_compatible_ids'].append(model['id'])
        if not report['image_compatible_ids']:
            intersection[:] = False
            certain_intersection[:] = False
        raw_consensus = intersection & ink & visible & ~nuisance
        consensus = certain_intersection & ink & visible & ~nuisance
        consensus_global[ys, xs] = consensus
        raw_global = np.zeros(gray.shape, bool)
        raw_global[ys, xs] = raw_consensus
        kernel_y, kernel_x = np.nonzero(kernel)
        domain_y, domain_x = np.nonzero(domain_kernel)
        origin = (box[0], box[1])
        report.update(complete=True, rendering_box_px=box, evaluation_box_px=list(box), prediction_dtype='bool',
            source_otsu_threshold=float(otsu_threshold), tolerance_radius_px=int(tolerance),
            morphology_kernel_offsets_px=np.column_stack((kernel_x-tolerance, kernel_y-tolerance)).tolist(),
            morphology_kernel_sha256=_sha(kernel), erosion_border_value=False, dilation_border_value=False,
            candidate_union_sha256=_sha(candidate_union), evaluation_domain_sha256=_sha(domain),
            evaluation_domain_rle=_rle(domain, origin), original_visibility_rle=_rle(original_visibility, origin),
            domain_margin_radius_px=domain_radius,
            domain_margin_kernel_offsets_px=np.column_stack((domain_x-domain_radius, domain_y-domain_radius)).tolist(),
            domain_scope='union_all_head_candidates_plus_fixed_background_band_before_residuals',
            domain_null_only_fallback=len(models) == 1,
            out_of_scope_source_ink_pixels=int(out_of_scope_ink.sum()),
            out_of_scope_source_ink_rle=_rle(out_of_scope_ink, origin),
            out_of_scope_ink_is_not_nuisance_or_assigned_ownership=True,
            visible_rle=_rle(visible, origin), evidence_rle=_rle(evidence, origin),
            heldout_rle=_rle(heldout, origin), nuisance_possible_rle=_rle(nuisance, origin),
            visible_mask_sha256=_sha(visible), evidence_mask_sha256=_sha(evidence),
            heldout_mask_sha256=_sha(heldout), nuisance_possible_sha256=_sha(nuisance),
            image_compatible_count=len(report['image_compatible_ids']),
            joint_compatible_count=len(report['joint_compatible_ids']),
            null_model_compatible='M0' in report['image_compatible_ids'],
            model_consensus_pixels=int(consensus.sum()), consensus_rle=_rle(consensus, origin),
            raw_prediction_consensus_pixels=int(raw_consensus.sum()),
            raw_prediction_consensus_rle=_rle(raw_consensus, origin),
            raw_prediction_consensus_sha256=_sha(raw_global),
            consensus_scope='intersection_of_eroded_certain_heads_for_all_pixel_compatible_models_not_only_contact_compatible',
            consensus_sha256=_sha(consensus_global), no_image_compatible_model=not bool(report['image_compatible_ids']),
            consensus_mask_hash_scope='full_source_window_bool_bytes',
            rendered_pixel_work=len(models)*math.prod(crop_shape),
            scoring_scope='fixed_visible_cells_outside_possible_nuisance; reserved_crosscheck_is_not_independent_acceptance')
    except _Limit as exc:
        if consensus_global is None:
            # A zero broadcast view keeps oversized-input rejection bounded.
            consensus_global = np.broadcast_to(np.array(False), gray.shape)
        else:
            consensus_global[:] = False
        report.update(complete=False, limits=[str(exc)], models=[], image_compatible_ids=[],
            joint_compatible_ids=[], image_compatible_count=0, joint_compatible_count=0,
            null_model_compatible=False, model_consensus_pixels=0, consensus_rle=[],
            partial_observation_rolled_back=True)
    if before != (_sha(gray), object_sha(shape_head_report), [None if a is None else _sha(a) for a in supplied]):
        raise ValueError('Ink model comparison changed its source evidence')
    return {'report': _plain(report), 'complete': bool(report['complete']), 'consensus_mask': consensus_global}


def _bound_shape(shape, member, gray):
    if (shape.get('schema_version') != shape_observer.POLICY['version']
            or member.get('schema_version') != 'native-head-membership/1.0'
            or shape.get('source_box_px') != member.get('source_box_px')
            or shape.get('working_to_source') != member.get('working_to_source')):
        raise ValueError('Current shape and membership frames differ')
    box = np.asarray(member['source_box_px'], float)
    if box.shape != (4,) or tuple((box[2:]-box[:2])[::-1]) != gray.shape:
        raise ValueError('Ink comparison source window dimensions differ')
    for key in ('working_gray_sha256', 'roles_sha256', 'hard_protection_sha256',
                'geometry_sha256', 'source_heads_sha256', 'admitted_heads_sha256',
                'chain_sha256', 'context_sha256', 'head_refinement_sha256'):
        if not member.get(key) or shape.get(key) != member[key]:
            raise ValueError('Shape and membership provenance differs: '+key)
    for row in (shape, member, *shape.get('heads', [])):
        if (row.get('semantic_exclusive_head_confirmed') is not False
                or row.get('physical_geometry_approved') is not False
                or row.get('cad_modification') is not False
                or row.get('head_uncertainty_reduced') is not False
                or type(row.get('roles_changed')) is not int or row['roles_changed'] != 0):
            raise ValueError('Source ink model cannot inherit semantic or geometric authority')
    if not shape.get('complete'):
        return
    if ([h.get('head_id') for h in shape.get('heads', [])] != member.get('qualified_head_ids')
            or shape.get('head_guard16_or_existing_bound_preserved') is not True):
        raise ValueError('Current qualified head pair and unchanged protections required')
    for head in shape['heads']:
        claims = [c for c in member['claimants'] if c['kind'] == 'head_possible' and c['id'] == head['head_id']]
        if len(claims) != 1 or len(claims[0].get('carrier_ids', [])) != 1:
            raise ValueError('Unique head and carrier membership required')
        claim = claims[0]
        bounds = claim['shape_bounds']
        current = [b for b in bounds if b['basis'] == 'current_admitted']
        carriers = [c for c in member['claimants'] if c['kind'] == 'measured_carrier'
                    and c['id'] == claim['carrier_ids'][0]]
        if len(current) != 1 or len(carriers) != 1:
            raise ValueError('Unique current observation bounds and carrier required')
        carrier = carriers[0]
        if (head.get('inherited_protection_bound_px') != max(b['protection_uncertainty_px'] for b in bounds)
                or head.get('preserved_tip_bound_px') != current[0]['own_observation_bound_px']
                or head.get('old_and_current_head_guards_unchanged') is not True
                or head.get('shaft_width_px') != carrier['width_px']
                or head.get('axis_uncertainty_px') != carrier['uncertainty_px']
                or head.get('measured_carrier_points_px') != carrier['measured_points_px']):
            raise ValueError('Current shape width, carrier or protection bounds differ')


def observe(gray, *, shape_report, membership):
    """Bind the finite experiment to freshly observed V22 source evidence."""
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('Native uint8 source window required')
    shape = shape_report.get('report', shape_report)
    member = membership.get('report', membership)
    base = {key: value for key, value in member.items() if key != 'head_shape_observation'}
    if (shape.get('source_window_sha256') != _sha(gray)
            or member.get('source_window_sha256') != _sha(gray)
            or shape.get('membership_sha256') != object_sha(base)
            or object_sha(member.get('head_shape_observation')) != object_sha(shape)
            or member.get('complete') is not True
            or shape.get('source_context_id') != member.get('source_context_id')):
        raise ValueError('Current source shape and membership binding required')
    _bound_shape(shape, member, gray)
    before = _sha(gray), object_sha(shape), object_sha(member)
    report = {'schema_version': POLICY['version'], 'policy': deepcopy(POLICY),
        'source_context_id': member['source_context_id'], 'source_box_px': list(member['source_box_px']),
        'source_window_sha256': _sha(gray), 'membership_sha256': object_sha(member),
        'shape_sha256': object_sha(shape), 'complete': bool(shape.get('complete')),
        'limits': [], 'heads': [], 'model_consensus_pixels': 0,
        'semantic_exclusive_head_confirmed': False, 'physical_geometry_approved': False,
        'whole_triangle_ownership_confirmed': False, 'logical_identifiability_confirmed': False,
        'head_uncertainty_reduced': False, 'roles_changed': 0, 'cad_modification': False,
        'existing_membership_rle_unchanged': True, 'head_guards_unchanged': True}
    outputs = {}
    try:
        if gray.size > POLICY['maximum_roi_pixels']:
            raise _Limit('ink_source_roi_budget')
        if len(shape.get('heads', [])) > POLICY['maximum_heads']:
            raise _Limit('ink_head_count_budget')
        if not shape.get('complete'):
            raise _Limit('preceding_shape_sampling_incomplete')
        possible = np.zeros(gray.shape, np.uint64)
        for y, a, b, bits, _, _ in member['classes_rle']:
            possible[y, a:b] = np.uint64(bits)
        all_bits = sum(c['bit'] for c in member['claimants'])
        for head in shape['heads']:
            own = [c['bit'] for c in member['claimants'] if c['kind'] == 'head_possible' and c['id'] == head['head_id']]
            if len(own) != 1:
                raise ValueError('Unique native head membership bit required')
            nuisance = (possible & np.uint64(all_bits ^ own[0])) != 0
            # Opposite visible wedges remain possible participants even when
            # no independent head identity was produced by the detector.
            profile = head.get('profile')
            if profile:
                ref = np.asarray(profile['reference_tip_px']); axis = np.asarray(profile['axis_unit'])
                normal = np.asarray(profile['normal_unit']); y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
                t = -((x-ref[0])*axis[0]+(y-ref[1])*axis[1])
                u = (x-ref[0])*normal[0]+(y-ref[1])*normal[1]
                guard = float(head.get('inherited_protection_bound_px', 0.))
                for taper in head.get('opposing_tapers', []):
                    left, right = taper['sides']['left'], taper['sides']['right']
                    margin = max(float(left['max_residual_px']), float(right['max_residual_px']))+math.sqrt(.5)+float(head['axis_uncertainty_px'])
                    nuisance |= ((t >= float(taper['apex_along_axis_px'])-guard)
                        & (u >= float(left['slope'])*t+float(left['intercept'])-margin)
                        & (u <= float(right['slope'])*t+float(right['intercept'])+margin))
            result = measure(gray, head, nuisance_possible=nuisance)
            report['heads'].append(result['report'])
            if result['complete']:
                outputs[head['head_id']] = result['consensus_mask']
        if any(not row['complete'] for row in report['heads']):
            raise _Limit('head_ink_partial_comparison_incomplete')
        report['model_consensus_pixels'] = sum(row['model_consensus_pixels'] for row in report['heads'])
    except _Limit as exc:
        report.update(complete=False, limits=[str(exc)], heads=[], model_consensus_pixels=0,
                      partial_observation_rolled_back=True)
        outputs = {}
    if before != (_sha(gray), object_sha(shape), object_sha(member)):
        raise ValueError('Native head ink observation changed its source inputs')
    return {'report': _plain(report), 'complete': bool(report['complete']), 'consensus_masks': outputs}
