"""Observe coupled dimension geometry in source pixels, retaining contact gaps.

A geometric chain is not authorization to change a CV role or a CAD interval.
In particular, projected working seeds are never presented as native ink.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import dimension_association_v3 as shared
from .joint_dimension_roles_v1 import _source_intervals
from .native_arc_extension_contact_v1 import observe as observe_contact
from .native_dimension_context_v1 import _line, _span
from .native_head_admission_v2 import build
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix, digest, source_windows, transform


POLICY = {'version': 'native-dimension-chain/1.0', 'maximum_links': 32,
    'maximum_heads': 512, 'maximum_roi_pixels': 4_000_000,
    'maximum_total_roi_pixels': 32_000_000, 'maximum_interval_checks': 250_000,
    'maximum_inventory_points': 100_000, 'maximum_width_samples': 2_000_000,
    'minimum_direction_dot': .98,
    'minimum_visible_fraction': .95, 'roles_changed': False,
    'interval_ownership_extended': False, 'semantic_dimension_approved': False,
    'physical_geometry_approved': False}


class _Limit(Exception):
    pass


def _span_ok(proof):
    return proof['fraction'] >= POLICY['minimum_visible_fraction'] and proof['maximum_gap_px'] <= proof['allowed_gap_px']


def _width_profile(ink, a, b, width, excluded, work):
    """Reject a long unexplained thick/absent run, even with visible centres."""
    length = float(np.linalg.norm(b-a)); unit=(b-a)/length
    normal=np.array([-unit[1],unit[0]])
    count=max(2,math.ceil(length)+1)
    offsets=np.arange(-math.ceil(3*width+2),math.ceil(3*width+2)+.25,.5)
    work['width_samples'] += count*len(offsets)
    if work['width_samples']>POLICY['maximum_width_samples']:
        raise _Limit('source_width_profile_budget')
    sites=np.linspace(a,b,count); q=np.rint(sites).astype(int)
    inside=((q>=0).all(1)&(q[:,0]<ink.shape[1])&(q[:,1]<ink.shape[0]))
    exempt=np.zeros(count,bool); exempt[inside]=excluded[q[inside,1],q[inside,0]]
    bad=np.zeros(count,bool); measured=[]
    center=int(np.argmin(abs(offsets)))
    for index,site in enumerate(sites):
        if exempt[index]: continue
        pixels=np.rint(site+offsets[:,None]*normal).astype(int)
        if ((pixels<0).any() or (pixels[:,0]>=ink.shape[1]).any() or (pixels[:,1]>=ink.shape[0]).any()):
            bad[index]=True; continue
        values=ink[pixels[:,1],pixels[:,0]]>0
        if not values[center]:
            bad[index]=True; continue
        lo=hi=center
        while lo>0 and values[lo-1]: lo-=1
        while hi+1<len(values) and values[hi+1]: hi+=1
        observed=float((hi-lo+1)*.5)
        measured.append(observed)
        bad[index]=lo==0 or hi==len(values)-1 or abs(observed-width)>2*math.sqrt(.5)
    changes=np.diff(np.r_[False,bad,False].astype(np.int8))
    gaps=np.flatnonzero(changes==-1)-np.flatnonzero(changes==1)
    maximum=float(gaps.max(initial=0)*length/(count-1))
    available=int((~exempt).sum()); bound=float(2*width+2*math.sqrt(.5))
    matching=float(1-bad.sum()/available) if available else 0.
    return {'accepted':bool(available>=24 and available>=count*.5 and maximum<=bound
                             and matching>=POLICY['minimum_visible_fraction']),
        'sample_count':count,'unexcluded_sections':available,'excluded_sections':int(exempt.sum()),
        'unexplained_sections':int(bad.sum()),'maximum_unexplained_run_px':maximum,
        'matching_width_fraction':matching,'minimum_matching_width_fraction':POLICY['minimum_visible_fraction'],
        'allowed_unexplained_run_px':bound,'model_width_px':float(width),
        'observed_width_range_px':[float(min(measured)),float(max(measured))] if measured else None,
        'excluded_mask_sha256':hashlib.sha256(excluded.tobytes()).hexdigest(),
        'role_semantics_inferred':False}


def _choose(context, heads, decisions, origin):
    options, reasons = [], []
    for identity in context['head_ids']:
        event = decisions.get(identity)
        if identity not in heads or not event or event['status'] == 'withheld':
            return [], ['head_not_admitted']
        if not event['local_seed_domain_complete']:
            return [], ['head_extension_seed_domain_incomplete']
        groups = [b for b in context.get('head_extension_bindings', []) if b['head_id'] == identity]
        candidates = [o for b in groups for o in b['extension_options'] if o['source_shape_rechecked']]
        if len(candidates) != 1:
            return [], ['unique_rechecked_extension_missing']
        option = candidates[0]
        if np.linalg.norm(np.asarray(option['intersection_px'])+origin-heads[identity]['tip']) > math.sqrt(.5):
            return [], ['extension_tip_revision_not_admitted']
        if len(option.get('arc_anchors', [])) != 1:
            return [], ['unique_source_arc_anchor_missing']
        if not option['arc_observation']['complete']:
            return [], ['source_arc_observation_incomplete']
        options.append(option)
    return options, reasons


def _interval_inventory(geometry, roles, matrix, origin, extension, anchor, tip, contact, work):
    """Inventory coarse interval collisions, without assigning native owners."""
    typed = {c['id']: c for c in geometry['candidates']}
    a, b = np.asarray(anchor['anchor_px']), np.asarray(tip)
    length = float(np.linalg.norm(b-a)); unit = (b-a)/length
    normal = np.array([-unit[1], unit[0]])
    scale = float(np.linalg.svd(matrix[:2, :2], compute_uv=False).max())
    projection_radius = math.sqrt(.5)*scale
    output = []
    for interval in geometry.get('ownership_intervals', geometry['candidates']):
        work['interval_checks'] += 1
        if work['interval_checks'] > POLICY['maximum_interval_checks']:
            raise _Limit('source_interval_inventory_budget')
        points = np.asarray(interval.get('support_points', interval.get('points_px', [])), float)
        if not len(points):
            continue
        native = transform(points, matrix)-origin
        delta = native-a; along = delta@unit; across = abs(delta@normal)
        candidate = typed.get(interval['id'])
        model_bound = float(candidate.get('uncertainty_px', 1.))*scale if candidate else scale
        tolerance = projection_radius+model_bound+extension['width_px']/2+extension['uncertainty_px']
        near = (along >= -tolerance) & (along <= length+tolerance) & (across <= tolerance)
        if not near.any():
            continue
        selected = points[near]; projected = native[near]
        q = np.rint(selected).astype(int)
        if ((q<0).any() or (q[:,0]>=roles.shape[1]).any() or (q[:,1]>=roles.shape[0]).any()):
            raise ValueError('Rounded working support lies outside its role raster')
        work['inventory_points'] += len(q)
        if work['inventory_points'] > POLICY['maximum_inventory_points']:
            raise _Limit('source_interval_point_inventory_budget')
        source_q = np.rint(projected).astype(int)
        inside = ((source_q >= 0).all(1) & (source_q[:, 0] < contact['shared'].shape[1])
                  & (source_q[:, 1] < contact['shared'].shape[0]))
        # These counts describe locations of seeds, not a resampling transfer.
        shared_count = int(contact['shared'][source_q[inside, 1], source_q[inside, 0]].sum())
        curve_count = int(contact['curve_possible'][source_q[inside, 1], source_q[inside, 0]].sum())
        values, counts = np.unique(roles[q[:, 1], q[:, 0]], return_counts=True)
        output.append({'id': interval['id'], 'kind': candidate['kind'] if candidate else 'untyped',
            'selected_working_points': selected.tolist(), 'projected_seed_locations_px': projected.tolist(),
            'working_role_counts': {str(int(k)): int(v) for k, v in zip(values, counts)},
            'projected_seed_shared_pixel_hits': shared_count, 'projected_seed_possible_arc_hits': curve_count,
            'source_members': deepcopy(candidate.get('source_members', [])) if candidate else None,
            'is_original_extension_interval': interval['id'] == extension['id'],
            'is_anchor_curve_interval': interval['id'] in anchor['source_curve_ids'],
            'projection_radius_px': projection_radius, 'seed_model_bound_px': model_bound,
            'projected_seeds_are_native_ink_observations': False,
            'exclusive_native_interval_ownership_confirmed': False})
    return output


def _observe_link(context, native_heads, all_heads, decisions, geometry, roles, matrix, gray, work):
    origin = np.asarray(context['source_box_px'][:2], float)
    row = {'id': context['id'], 'head_ids': list(context['head_ids']), 'status': 'withheld',
        'reasons': [], 'source_box_px': context['source_box_px'],
        'source_window_sha256': hashlib.sha256(gray.tobytes()).hexdigest(),
        'source_context_id': context['id'], 'contacts': [], 'nominal_value': None,
        'reading_used_for_geometry': False, 'semantic_dimension_approved': False,
        'complete_dimension_acceptance': False, 'physical_geometry_approved': False,
        'source_interval_ownership_complete': False, 'roles_changed': 0}
    if row['source_window_sha256'] != context['source_window_sha256']:
        raise ValueError('Current original contact window differs from its context')
    options, reasons = _choose(context, native_heads, decisions, origin)
    if reasons:
        row['reasons'] = reasons
        return row
    first, second = [native_heads[i] for i in context['head_ids']]
    tips = [h['tip']-origin for h in (first, second)]
    delta = tips[1]-tips[0]; span = float(np.linalg.norm(delta))
    if span <= 0:
        row['reasons'] = ['heads_share_one_transverse_junction']; return row
    axis = delta/span
    stroke = max(first['width'], second['width'])
    tolerance = stroke+max(first['uncertainty'], second['uncertainty'])+math.sqrt(.5)
    row['head_span_px'] = span
    row['head_tolerance_px'] = tolerance
    if span <= max(8*stroke, 4*tolerance):
        row['reasons'] = ['source_heads_not_independently_separated']; return row
    if (first['direction']@second['direction'] > -POLICY['minimum_direction_dot']
            or any(abs(h['direction']@axis) < POLICY['minimum_direction_dot'] for h in (first,second))):
        row['reasons'] = ['source_heads_not_opposed_on_one_axis']; return row
    intervening = [h['id'] for h in all_heads if h['native_id'] not in context['head_ids']
        and 2*tolerance < (h['tip']-first['tip'])@axis < span-2*tolerance
        and abs((h['tip']-first['tip'])@np.array([-axis[1],axis[0]])) <= tolerance
        and abs(h['direction']@axis) >= POLICY['minimum_direction_dot']]
    if intervening:
        row.update(reasons=['intervening_source_or_working_head'], intervening_head_ids=intervening); return row
    measured = {line['id']: line for line in context['stroke_observation']['lines']}
    owners = sorted({o for h in (first,second) for o in h['owner_ids']})
    if not set(owners) <= measured.keys():
        row['reasons'] = ['measured_source_carrier_owner_missing']; return row
    carrier_proofs = []
    for identity in owners:
        line = measured[identity]; ends, unit, _ = _line(line['points_px'])
        bound = line['width_px']/2+line['uncertainty_px']+math.sqrt(.5)
        offsets = abs((np.asarray(tips)-ends[0])@np.array([-unit[1],unit[0]]))
        if abs(unit@axis) < POLICY['minimum_direction_dot'] or max(offsets) > bound:
            row['reasons'] = ['refitted_source_carriers_disagree']; return row
        carrier_proofs.append({'id':identity, 'measured_points_px': line['points_px'],
            'width_px':line['width_px'], 'uncertainty_px':line['uncertainty_px'],
            'tip_offsets_px':offsets.tolist(), 'source_profile':deepcopy(line['source_profile']),
            'observed_extent_is_not_original_ownership_extent':True})
    ink = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    distance = cv2.distanceTransform(255-ink,cv2.DIST_L2,5)
    head_mask=np.zeros_like(ink,np.uint8)
    for option,h in zip(options,(first,second)):
        footprint=np.zeros_like(ink,np.uint8)
        cv2.fillConvexPoly(footprint,np.rint(option['candidate']['polygon_px']).astype(np.int32),1)
        radius=math.ceil(h['uncertainty'])
        footprint=cv2.dilate(footprint,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
        head_mask |= footprint
    carrier = _span(ink,*tips,stroke/2+math.sqrt(.5),distance=distance)
    row.update(carrier_pixel_support=carrier, carrier_observations=carrier_proofs,
        carrier_prior_conflicts=deepcopy(context.get('carrier_prior_conflicts', [])))
    if not _span_ok(carrier):
        row['reasons'] = ['source_carrier_ink_missing']; return row
    row['carrier_width_profile']=_width_profile(ink,*tips,stroke,head_mask>0,work)
    if not row['carrier_width_profile']['accepted']:
        row['reasons']=['source_carrier_width_inconsistent']; return row
    anchors = [o['arc_anchors'][0] for o in options]
    anchor_points = [np.asarray(a['anchor_px']) for a in anchors]
    sides = [float(np.linalg.det(np.column_stack((p-tip,axis)))) for p,tip in zip(anchor_points,tips)]
    if sides[0]*sides[1] <= 0 or options[0]['extension_id'] == options[1]['extension_id']:
        row['reasons'] = ['source_anchors_do_not_form_same_side_distinct_extensions']; return row
    for option, anchor, tip in zip(options, anchors, tips):
        extension = measured[option['extension_id']]
        _, unit, _ = _line(extension['points_px'])
        if abs(unit@axis) > .2:
            row['reasons'] = ['source_extension_not_transverse']; return row
        contact = observe_contact(gray, extension, anchor, tip.tolist())
        item = {'head_id': first['native_id'] if not row['contacts'] else second['native_id'],
            'extension_id':extension['id'], 'anchor':deepcopy(anchor), 'tip_px':tip.tolist(),
            'contact':contact['report'], 'through_crossings':deepcopy(option.get('crossings',[])),
            'semantic_part_confirmed':False, 'original_curve_geometry_preserved':True}
        row['contacts'].append(item)
        if not contact['complete']:
            raise _Limit('source_contact_observation_budget')
        if not contact['report']['accepted']:
            row['reasons'] = ['source_arc_extension_contact_unconfirmed']; return row
        item['extension_width_profile']=_width_profile(ink,np.asarray(anchor['anchor_px']),tip,
            extension['width_px'],contact['curve_possible']|(head_mask>0),work)
        if not item['extension_width_profile']['accepted']:
            row['reasons']=['source_extension_width_inconsistent']; return row
        item['interval_inventory'] = _interval_inventory(geometry,roles,matrix,origin,
            extension,anchor,tip,contact,work)
        item['projection_is_not_native_ownership'] = True
    row.update(status='source_linear_chain_observed', reasons=[], anchors_px=[p.tolist() for p in anchor_points],
        geometry_chain_confirmed=True,
        role_release_holds=['native_anchor_part_semantics_not_confirmed',
            'contact_interval_ownership_and_projection_not_admitted',
            'competing_CV_roles_require_explicit_revision_authority'],
        shared_contacts_are_preserved=True)
    return row


def inspect(original_source, working_image, *, geometry, ocr, roles, source_heads,
            working_to_source, arrows=None):
    """Rebuild all source evidence; serialized certificates are never inputs."""
    matrix = checked_matrix(working_to_source)
    admission = build(original_source,working_image,geometry=geometry,ocr=ocr,roles=roles,
        source_heads=source_heads,working_to_source=matrix)
    context = admission['context']
    with Image.open(working_image) as image:
        working_gray = np.array(image.convert('L'))
    intervals, _, failure = _source_intervals(geometry,working_gray)
    if intervals is None:
        raise ValueError('Complete bounded input ownership inventory required: '+failure)
    for interval in intervals:
        q=np.rint(np.asarray(interval['points_px'],float)).astype(int)
        if ((q<0).any() or (q[:,0]>=roles.shape[1]).any() or (q[:,1]>=roles.shape[0]).any()):
            raise ValueError('Rounded working support lies outside its role raster')
    inputs = object_sha([geometry,ocr,source_heads,arrows])
    role_sha = hashlib.sha256(roles.tobytes()).hexdigest()
    decisions = {e['source_arrow_id']: e for e in admission['report']['tip_admission']['events']}
    heads = []
    for candidate in admission['arrows']['candidates']:
        direction = matrix[:2,:2]@np.asarray(candidate['direction']); direction /= np.linalg.norm(direction)
        scale = float(np.linalg.svd(matrix[:2,:2],compute_uv=False).max())
        normal = np.array([-candidate['direction'][1],candidate['direction'][0]])
        heads.append({'id':candidate['id'], 'native_id':candidate['id'].removeprefix('native/'),
            'tip':transform(candidate['tip_px'],matrix), 'direction':direction,
            'width':float(candidate['evidence']['shaft_width_px'])*float(np.linalg.norm(matrix[:2,:2]@normal)),
            'uncertainty':float(candidate['uncertainty_px'])*scale,
            'owner_ids':candidate['evidence']['geometry_ids']})
    native_heads = {h['native_id']:h for h in heads}
    if arrows is not None:
        shared._validate(working_image,geometry,ocr,arrows)
        for candidate in arrows['candidates']:
            direction = matrix[:2,:2]@np.asarray(candidate['direction']); direction /= np.linalg.norm(direction)
            heads.append({'id':candidate['id'],'native_id':None,
                'tip':transform(candidate['tip_px'],matrix),'direction':direction})
    work = {'interval_checks':0, 'inventory_points':0, 'windows':0, 'roi_pixels':0,'width_samples':0}
    links, limits = [], []
    if len(heads)>POLICY['maximum_heads'] or len(context['contexts'])>POLICY['maximum_links']:
        limits.append('source_chain_domain_budget')
    eligible = [c for c in context['contexts'] if c['status']=='source_context_observed'
        and c.get('stroke_observation',{}).get('complete')]
    boxes = list(dict.fromkeys(tuple(c['source_box_px']) for c in eligible))
    pixels = sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes)
    if pixels>POLICY['maximum_total_roi_pixels'] or any((b[2]-b[0])*(b[3]-b[1])>POLICY['maximum_roi_pixels'] for b in boxes):
        limits.append('source_chain_roi_budget')
    domain_complete = admission['support']['complete'] and (arrows is None or not arrows.get('analysis',{}).get('search_limited',False))
    if not domain_complete:
        limits.append('source_head_domain_incomplete')
    windows = source_windows(original_source,context['source_sha256'],source_heads['processed_size_px'],boxes) if boxes and not limits else []
    by_box = dict(zip(boxes,windows))
    if not limits:
        work.update(windows=len(boxes),roi_pixels=pixels)
        try:
            for c in context['contexts']:
                if c not in eligible:
                    links.append({'id':c['id'],'head_ids':c['head_ids'],'status':'withheld',
                        'reasons':deepcopy(c['reasons']) or ['source_context_not_observed']})
                    continue
                links.append(_observe_link(c,native_heads,heads,decisions,geometry,roles,matrix,
                    by_box[tuple(c['source_box_px'])],work))
        except _Limit as exc:
            limits.append(str(exc))
    consumers = {}
    for row in links:
        if row['status']=='source_linear_chain_observed':
            identities = row['head_ids']+[p['id'] for p in row['carrier_observations']]+[c['extension_id'] for c in row['contacts']]
            for identity in identities:
                consumers.setdefault(identity,set()).add(row['id'])
    conflict_ids = {identity for identity, rows in consumers.items() if len(rows)>1}
    for row in links:
        if row['status']!='source_linear_chain_observed':
            continue
        ids = row['head_ids']+[p['id'] for p in row['carrier_observations']]+[c['extension_id'] for c in row['contacts']]
        if limits or conflict_ids.intersection(ids):
            row.update(status='withheld',geometry_chain_confirmed=False,
                reasons=['source_chain_search_incomplete' if limits else 'competing_source_chains'])
    if (digest(original_source)!=context['source_sha256'] or digest(working_image)!=context['working_source_sha256']
            or role_sha!=hashlib.sha256(roles.tobytes()).hexdigest()
            or inputs!=object_sha([geometry,ocr,source_heads,arrows])):
        raise ValueError('Source-chain inputs changed during observation')
    report = {key:deepcopy(context[key]) for key in ('source_sha256','working_source_sha256',
        'geometry_sha256','ocr_sha256','source_heads_sha256','roles_sha256','working_to_source')}
    report.update(schema_version=POLICY['version'],policy=deepcopy(POLICY),
        context_sha256=object_sha(context),admission_sha256=object_sha(admission['report']['tip_admission']),
        working_arrows_sha256=object_sha(arrows) if arrows is not None else None,
        legacy_head_domain_checked=arrows is not None, head_domain_complete=domain_complete,
        source_context_complete=context['complete'], complete=not limits and context['complete'],
        local_observation_complete=not limits, links=links,limits=limits,work=work,
        summary=dict(Counter(r['status'] for r in links)), roles_changed=0,geometry_changed=False,
        semantic_dimension_approved=False,physical_geometry_approved=False,production_changed=False,
        coordinate_space='source_window_pixels_with_explicit_box_original_px',
        source_interval_ownership_complete=False,reference_used_for_inference=False)
    return {'report':report,'admission':admission}
