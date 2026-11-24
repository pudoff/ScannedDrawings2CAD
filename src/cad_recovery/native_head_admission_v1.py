"""Admit consistent source-rechecked tip revisions without hiding other heads.

The public entry point always observes current pixels. Persisted contexts are
diagnostics, not caller-supplied authorization for a tip or semantic revision.
"""
from copy import deepcopy
from itertools import combinations
import math

import numpy as np

from .native_dimension_context_v2 import inspect
from .native_head_bridge_v1 import build as base_bridge
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix, transform


PIXEL_RADIUS = math.sqrt(.5)


def _choices(context, identity):
    required, observed, choices, owner_failures = set(), set(), [], []
    for row in context['contexts']:
        if identity not in row['head_ids']:
            continue
        index = row['head_ids'].index(identity)
        seeds = row.get('extension_seed_ids', [[], []])[index]
        required.update(seeds)
        strokes = row.get('stroke_observation', {})
        if (not strokes.get('complete') or 'extension_recheck_budget' in row['reasons']):
            continue
        lines = {line['id']: line for line in strokes['lines']}
        bindings = [b for b in row.get('head_extension_bindings', []) if b['head_id'] == identity]
        if not bindings:
            continue
        observed.update(set(seeds).intersection(lines))
        for option in bindings[0]['extension_options']:
            if not option['source_shape_rechecked']:
                if option['recheck_reason'] in {
                    'multiple_refitted_source_shaft_owners', 'refitted_source_shaft_owner_changed',
                    'ambiguous_carrier_owners', 'ambiguous_parallel_carriers', 'no_observed_carrier_owner'}:
                    owner_failures.append({'context_id': row['id'], 'extension_id': option['extension_id'],
                        'reason': option['recheck_reason'], 'old_tip_consistent': option['old_tip_consistent']})
                continue
            if option['extension_id'] not in lines:
                raise ValueError('Rechecked head lacks its measured extension')
            choices.append((row, option, lines[option['extension_id']]))
    return required, observed, choices, owner_failures


def _revision(head, choices):
    """Require agreement across every successful local hypothesis, never rank."""
    proposals = []
    for row, option, extension in choices:
        candidate = option['candidate']
        owners = candidate['evidence']['geometry_ids']
        if len(owners) != 1 or owners != head['evidence']['geometry_ids']:
            return None, 'refined_owner_changed_or_ambiguous'
        origin = np.asarray(row['source_box_px'][:2], float)
        tip = np.asarray(candidate['tip_px']) + origin
        proposals.append((tip, option['extension_id'], owners))
    first = proposals[0]
    spread = max((float(np.linalg.norm(a[0]-b[0])) for a,b in combinations(proposals, 2)), default=0.)
    if spread > PIXEL_RADIUS or any(eid != first[1] or owners != first[2]
                                  for _, eid, owners in proposals[1:]):
        return None, 'inconsistent_source_tip_revisions'
    # Selection only makes duplicate agreeing observations deterministic; no
    # score, distance-to-seed or convenient pair chooses among alternatives.
    row, option, extension = sorted(choices, key=lambda c: c[0]['id'])[0]
    candidate = deepcopy(option['candidate'])
    origin = np.asarray(row['source_box_px'][:2], float)
    candidate['tip_px'] = (np.asarray(candidate['tip_px'])+origin).tolist()
    candidate['polygon_px'] = (np.asarray(candidate['polygon_px'])+origin).tolist()
    bounds = []
    for observation, _, measured in choices:
        unit = np.diff(np.asarray(measured['points_px']), axis=0)[0]
        unit /= np.linalg.norm(unit)
        sine = abs(float(np.linalg.det(np.column_stack((unit, head['direction'])))))
        if sine < .98:
            return None, 'unstable_head_extension_intersection'
        bounds.append((float(measured['uncertainty_px'])/sine, observation['id'],
                       float(measured['uncertainty_px']), sine))
    contribution, bound_context, measured_bound, sine = max(bounds)
    components = {'inherited_head_bound_px': float(head['uncertainty_px']),
        'measured_extension_bound_px': measured_bound, 'bound_context_id': bound_context,
        'intersection_sine': sine, 'pixel_radius_px': PIXEL_RADIUS,
        'agreement_spread_px': spread,
        'independently_calibrated': False,
        'scope': 'conditional_on_rechecked_head_axis_and_measured_extension'}
    bound = components['inherited_head_bound_px'] + contribution + PIXEL_RADIUS + spread
    components['propagated_bound_px'] = bound
    candidate['uncertainty_px'] = bound
    candidate['confidence'] = head['confidence']
    candidate['source_ref'] = deepcopy(head['source_ref'])
    candidate['evidence']['tip_revision_not_admitted_to_detector'] = False
    candidate['evidence']['tip_uncertainty_independently_calibrated'] = False
    return {'candidate': candidate, 'row': row, 'extension': extension,
        'bound': components, 'context_ids': sorted({c[0]['id'] for c in choices}),
        'agreement_spread_px': spread}, None


def _project(old, revision, inverse, source_head, context):
    row = deepcopy(old)
    candidate = revision['candidate']
    polygon = transform(candidate['polygon_px'], inverse)
    tip = transform(candidate['tip_px'], inverse)
    delta = polygon[0]-polygon[1:].mean(0)
    normal = np.array([-candidate['direction'][1], candidate['direction'][0]])
    scale = float(np.linalg.svd(inverse[:2, :2], compute_uv=False).max())
    proof = {'source_arrow_id': source_head['id'], 'original_tip_px': source_head['tip_px'],
        'revised_tip_px': candidate['tip_px'], 'context_ids': revision['context_ids'],
        'extension_id': revision['extension']['id'], 'uncertainty': revision['bound'],
        'agreement_spread_px': revision['agreement_spread_px'],
        'current_source_context_sha256': object_sha(context),
        'source_box_px': revision['row']['source_box_px'],
        'source_window_sha256': revision['row']['source_window_sha256'],
        'current_local_recheck': deepcopy(revision['row']['head_extension_bindings']),
        'measurement_coordinate_space': 'original_image_pixels',
        'original_owner_longitudinal_intervals_preserved': True,
        'semantic_dimension_approved': False, 'physical_geometry_approved': False}
    row.update(tip_px=tip.tolist(), polygon_px=polygon.tolist(), direction=(delta/np.linalg.norm(delta)).tolist(),
        uncertainty_px=candidate['uncertainty_px']*scale)
    row['evidence'].update(shaft_width_px=candidate['evidence']['shaft_width_px']*float(np.linalg.norm(inverse[:2,:2]@normal)),
        base_width_px=float(np.linalg.norm(polygon[1]-polygon[2])),
        arrow_length_px=float(np.linalg.norm(delta)), head_length_px=float(np.linalg.norm(delta)),
        tip_admission=proof)
    # Keep the immutable detector observation explicitly labelled as a prior;
    # consumers get the current coordinates from the outer projected head.
    row['evidence']['original_head']['is_pre_revision_observation'] = True
    return row


def _admit(bridge, context, source_heads, matrix):
    result = deepcopy(bridge)
    inverse = np.linalg.inv(checked_matrix(matrix))
    source_by_id = {h['id']: h for h in source_heads['verified_candidates']}
    events, withheld = [], []
    global_limit = set(context['limits'])-{'native_context_window_budget'}
    for index, old in enumerate(result['arrows']['candidates']):
        identity = old['id'].removeprefix('native/')
        head = source_by_id[identity]
        required, observed, choices, owner_failures = _choices(context, identity)
        event = {'source_arrow_id': identity, 'status': 'unchanged', 'reasons': [],
            'required_extension_seed_ids': sorted(required), 'observed_extension_seed_ids': sorted(observed),
            'local_seed_domain_complete': required <= observed}
        events.append(event)
        if global_limit:
            event.update(status='withheld', reasons=['source_context_search_incomplete'])
            withheld.append(identity)
            continue
        if owner_failures:
            # A shape-supported intersection whose source shaft no longer has
            # the original unique owner must not silently reuse the coarse
            # bridge's earlier ownership. Keep its head in the veto domain.
            event.update(status='withheld', reasons=['current_refitted_source_ownership_unresolved'],
                ownership_failures=owner_failures)
            withheld.append(identity)
            continue
        if not choices:
            event['reasons'].append('no_verified_revision')
            continue
        revision, reason = _revision(head, choices)
        if revision is None:
            event.update(status='withheld', reasons=[reason])
            withheld.append(identity)
            continue
        if not required <= observed:
            if any(not option['old_tip_consistent'] for _, option, _ in choices):
                event.update(status='withheld', reasons=['inconsistent_prior_tip_with_unobserved_extension_alternative'])
                withheld.append(identity)
            else:
                event['reasons'].append('unobserved_extension_alternative')
            continue
        revised = _project(old, revision, inverse, head, context)
        tip = np.asarray(revised['tip_px'])
        if (tip < 0).any() or (tip >= bridge['report']['working_size_px']).any():
            event.update(status='withheld', reasons=['revised_tip_outside_working_frame'])
            withheld.append(identity)
            continue
        result['arrows']['candidates'][index] = revised
        for span in result['support']['spans']:
            if span['arrow_id'] == old['id']:
                span.update(points_px=[revised['tip_px'], np.asarray(revised['polygon_px'])[1:].mean(0).tolist()],
                    uncertainty_px=revised['uncertainty_px'], tip_admission=deepcopy(revised['evidence']['tip_admission']))
        for support in result['support']['events']:
            if support['arrow_id'] == old['id']:
                support.update(tip_px=revised['tip_px'], base_px=np.asarray(revised['polygon_px'])[1:].mean(0).tolist(),
                    tip_admission=deepcopy(revised['evidence']['tip_admission']))
        event.update(status='refined', revised_tip_original_px=revision['candidate']['tip_px'],
            shift_original_px=float(np.linalg.norm(np.asarray(revision['candidate']['tip_px'])-head['tip_px'])),
            uncertainty=revision['bound'], context_ids=revision['context_ids'])
    complete = bridge['support']['complete'] and not withheld
    if not complete:
        result['support']['spans'] = []
    result['support']['complete'] = complete
    result['arrows']['analysis']['qualification_complete'] = complete
    admission = {'schema_version': 'native-head-admission/1.0', 'context_sha256': object_sha(context),
        'source_context_complete': context['complete'], 'head_domain_complete': complete,
        'admission_complete': complete, 'events': events, 'withheld_source_head_ids': withheld,
        'refined_heads': sum(e['status']=='refined' for e in events),
        'source_head_ids_preserved': [h['id'] for h in result['arrows']['candidates']] == [h['id'] for h in bridge['arrows']['candidates']],
        'confidence_recalibrated': False, 'physical_geometry_approved': False,
        'complete_dimension_acceptance': False, 'roles_revised_by_admission': 0}
    result['report'].update(tip_admission=admission, complete=complete,
        qualification_complete=bridge['report']['qualification_complete'] and not withheld,
        qualification_withheld_source_head_ids=sorted(set(bridge['report']['qualification_withheld_source_head_ids']) | set(withheld)),
        published_support_heads=len(result['support']['spans']))
    result['support']['tip_admission'] = admission
    return result


def build(original_source, working_image, *, geometry, ocr, roles, source_heads, working_to_source):
    bridge = base_bridge(original_source, working_image, geometry=geometry, ocr=ocr,
        source_heads=source_heads, working_to_source=working_to_source)
    context = inspect(original_source, working_image, geometry=geometry, ocr=ocr, roles=roles,
        source_heads=source_heads, working_to_source=working_to_source)
    if context['native_bridge'] != bridge['report']:
        raise ValueError('Native head evidence changed while observing its context')
    result = _admit(bridge, context, source_heads, working_to_source)
    result['context'] = context
    return result
