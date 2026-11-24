"""Corroborate an existing PART prior on two freshly observed ARC flanks.

This is not a new material classifier: unknown curves never become PART.
Only distinct, visible input PART pixels with exclusive native ink count.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np

from .native_arc_extension_contact_v1 import observe as contact_observe
from .native_dimension_transfer_v1 import _footprint, _supported_matrix, _possible_working
from .native_dimension_transfer_v1 import _Limit as TransferLimit
from .native_dimension_context_v1 import _native_geometry
from .native_dimension_strokes_v1 import observe as observe_strokes
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix,local_lines


POLICY={'version':'native-arc-part-prior/1.0','maximum_roi_pixels':4_000_000,
    'maximum_working_pixels':4_000_000,'maximum_intervals':4096,
    'maximum_source_points':2_000_000,'maximum_cell_pixels':2_000_000,
    'maximum_competitor_raster_pixels':40_000_000,'minimum_points_per_flank':3,
    'maximum_native_competitor_checks':32,'maximum_native_competitor_samples':2_000_000,
    'maximum_anchor_points':10_000,'maximum_competitor_point_checks':250_000,
    'existing_PART_required':True,'new_PART_classification':False,
    'whole_source_ink_footprint_required':True,'roles_changed':False,
    'physical_geometry_approved':False}


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class _Limit(Exception):
    pass


def _points(item,shape):
    points=np.asarray(item.get('support_points',item.get('points_px',[])),float)
    if points.size==0:return np.empty((0,2))
    if points.ndim!=2 or points.shape[1]!=2 or not np.isfinite(points).all():
        raise ValueError('Finite working support points required')
    q=np.rint(points).astype(int)
    if ((q<0).any() or (q[:,0]>=shape[1]).any() or (q[:,1]>=shape[0]).any()):
        raise ValueError('Rounded working support outside role raster')
    return points


def _angles(points,center,radial):
    delta=points-center
    tangent=np.array([-radial[1],radial[0]])
    return np.degrees(np.arctan2(delta@tangent,delta@radial))


def _fresh_competitor(gray,typed,measured,matrix,box,identity,work):
    """Rebuild the named straight competitor; metadata alone grants no release."""
    if identity not in measured or identity not in typed or typed[identity]['kind']!='line':
        return None
    work['native_competitor_checks']+=1
    if work['native_competitor_checks']>POLICY['maximum_native_competitor_checks']:
        raise _Limit('native_competitor_count_budget')
    native=_native_geometry({'candidates':[typed[identity]]},matrix)
    local=local_lines(native,box)
    fresh=observe_strokes(gray,local)
    work['native_competitor_samples']+=fresh['audit']['source_support_samples']
    if not fresh['complete'] or work['native_competitor_samples']>POLICY['maximum_native_competitor_samples']:
        raise _Limit('native_competitor_observation_budget')
    if len(fresh['lines'])!=1 or fresh['lines'][0]!=measured[identity]:
        return None
    return fresh['lines'][0]


def _anchor(gray,working_ink,geometry,roles,contact,context,matrix,origin,protection,work):
    anchor=contact['anchor'];evidence=anchor['evidence']
    ids=set(anchor['source_curve_ids'])
    typed={c['id']:c for c in geometry['candidates']}
    row={'curve_id':anchor['curve_id'],'source_curve_ids':sorted(ids),'accepted':False,
        'reasons':[],'flanks':[],'working_support_ids':[],
        'support_kind':'existing_working_PART_corroborated_by_current_native_ARC',
        'new_PART_classification':False}
    if not ids or not ids<=typed.keys():
        row['reasons']=['source_curve_inventory_missing'];return row
    center=np.asarray(evidence['curve_center_px'],float)
    point=np.asarray(anchor['anchor_px'],float)
    radius=float(evidence['curve_radius_px']);width=float(evidence['measured_curve_width_px'])
    radial=point-center;radial/=np.linalg.norm(radial)
    sections=np.asarray(evidence['source_section_points_px'],float)
    theta=_angles(sections,center,radial)
    extent=float(evidence['flank_extent_deg'])
    ranges=[]
    for side in (-1,1):
        values=theta[theta*side>0]
        if len(values)<8:
            row['reasons']=['two_observed_flanks_required'];return row
        lo=max(-extent,float(values.min()));hi=min(extent,float(values.max()))
        if lo>=hi:
            row['reasons']=['observed_flank_domain_missing'];return row
        ranges.append((lo,hi))
    row['observed_flank_angle_ranges_deg']=[list(r) for r in ranges]
    measured={line['id']:line for line in context['stroke_observation']['lines']}
    if contact['extension_id'] not in measured:
        row['reasons']=['measured_extension_missing'];return row
    fresh=contact_observe(gray,measured[contact['extension_id']],anchor,contact['tip_px'])
    if not fresh['complete']:
        raise _Limit('native_contact_budget')
    if not fresh['report']['accepted']:
        row['reasons']=['current_arc_contact_not_observed'];return row
    row['contact_sha256']=object_sha(fresh['report'])
    # The broad uncertainty band is unchanged in contact_observe. A narrow
    # measured band below verifies ink, never replaces that protection.
    row['preserved_arc_uncertainty_px']=fresh['report']['preserved_arc_localization_uncertainty_px']
    blocked=fresh['shared']|fresh['extension_exclusive']
    for binding in context.get('head_extension_bindings',[]):
        for option in binding['extension_options']:
            if not option.get('source_shape_rechecked'):continue
            polygon=np.asarray(option['candidate']['polygon_px'],float)
            mask=np.zeros(gray.shape,np.uint8)
            cv2.fillConvexPoly(mask,np.rint(polygon).astype(np.int32),1)
            blocked|=mask>0
    intervals=geometry.get('ownership_intervals',geometry['candidates'])
    competitors=np.zeros(roles.shape,bool)
    candidates={}
    for identity in sorted(ids):
        for p in _points(typed[identity],roles.shape):
            q=tuple(map(int,np.rint(p).astype(int)))
            candidates.setdefault(q,set()).add(identity)
            if len(candidates)>POLICY['maximum_anchor_points']:
                raise _Limit('anchor_point_budget')
    competitors_at_pixel={q:[] for q in candidates}
    for interval in intervals:
        points=_points(interval,roles.shape)
        if not len(points) or interval['id'] in ids:continue
        work['competitor_point_checks']+=len(candidates)
        if work['competitor_point_checks']>POLICY['maximum_competitor_point_checks']:
            raise _Limit('competitor_point_budget')
        uncertainty=float(typed.get(interval['id'],{}).get('uncertainty_px',1.))
        box,mask=_possible_working(points,uncertainty,roles.shape,work)
        if work['competitor_raster_pixels']>POLICY['maximum_competitor_raster_pixels']:
            raise _Limit('competitor_support_budget')
        left,top,right,bottom=box
        competitors[top:bottom,left:right]|=mask
        for q in candidates:
            if left<=q[0]<right and top<=q[1]<bottom and mask[q[1]-top,q[0]-left]:
                competitors_at_pixel[q].append(interval['id'])
    ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]>0
    bound=width/2+float(evidence['fit_max_residual_px'])+math.sqrt(.5)
    accepted=[[],[]];rejected=Counter();conflicts=[];ownership_holds=[];resolved_competitors=[]
    competitor_cache={}
    extension_point=np.asarray(anchor['anchor_px'],float)
    extension_delta=np.asarray(contact['tip_px'],float)-extension_point
    extension_length=float(np.linalg.norm(extension_delta))
    extension_unit=extension_delta/extension_length
    for q,owners in sorted(candidates.items()):
        native=np.asarray(q)@matrix[:2,:2].T+matrix[:2,2]-origin
        angle=float(_angles(native[None,:],center,radial)[0])
        sides=[i for i,(lo,hi) in enumerate(ranges) if lo<=angle<=hi]
        if not sides:continue
        # A known competing role in this observed ARC domain blocks the prior,
        # even if an exclusion would otherwise hide that conflict.
        value=int(roles[q[1],q[0]])
        if value in range(2,10):
            conflicts.append({'working_pixel':list(q),'role':value});continue
        if value!=1:rejected['not_input_PART']+=1;continue
        if not working_ink[q[1],q[0]]:
            rejected['working_ink_missing']+=1;continue
        box=_footprint(q,matrix,origin,gray.shape)
        if box is None:
            rejected['source_cell_clipped']+=1;continue
        left,top,right,bottom=box
        work['cell_pixels']+=(right-left)*(bottom-top)
        if work['cell_pixels']>POLICY['maximum_cell_pixels']:
            raise _Limit('source_cell_budget')
        ys,xs=np.nonzero(ink[top:bottom,left:right])
        if not len(xs):
            rejected['native_ink_missing']+=1;continue
        xs=xs+left;ys=ys+top
        p=np.column_stack((xs,ys)).astype(float)
        possible=list(competitors_at_pixel[q])
        if contact['extension_id'] in possible:
            # Replace only this extension's coarse possible footprint with its
            # current fully reobserved native body. Outside the checked span,
            # the original possible owner is still retained.
            corners=np.array([[left-.5,top-.5],[right-.5,top-.5],
                [right-.5,bottom-.5],[left-.5,bottom-.5]])
            along=(corners-extension_point)@extension_unit
            if (along.min()>=0 and along.max()<=extension_length
                    and not (fresh['shared']|fresh['extension_exclusive'])[ys,xs].any()):
                possible.remove(contact['extension_id'])
                resolved_competitors.append({'working_pixel':list(q),'interval_id':contact['extension_id'],
                    'current_native_contact_sha256':object_sha(fresh['report']),
                    'native_extension_band_radius_px':fresh['report']['extension_band_radius_px'],
                    'entire_cell_inside_observed_longitudinal_span':True,
                    'native_ink_intersects_observed_extension':False})
        for identity in list(possible):
            if identity not in competitor_cache:
                competitor_cache[identity]=_fresh_competitor(gray,typed,measured,matrix,
                    context['source_box_px'],identity,work)
            current=competitor_cache[identity]
            if current is None:continue
            endpoints=np.asarray(current['points_px'],float)
            unit=endpoints[1]-endpoints[0];unit/=np.linalg.norm(unit)
            normal=np.array([-unit[1],unit[0]])
            band=float(current['width_px']/2+current['uncertainty_px']+math.sqrt(.5))
            offsets=abs((p-endpoints[0])@normal)
            if np.all(offsets>band):
                possible.remove(identity)
                resolved_competitors.append({'working_pixel':list(q),'interval_id':identity,
                    'current_native_line_sha256':object_sha(current),'native_line_band_radius_px':band,
                    'minimum_native_ink_distance_px':float(offsets.min()),
                    'current_source_profile_recomputed':True,
                    'longitudinal_endpoint_not_used_for_exclusion':True})
        if protection[q[1],q[0]] or possible:
            ownership_holds.append({'working_pixel':list(q),'working_protected':bool(protection[q[1],q[0]]),
                'possible_other_interval_ids':possible})
            rejected['working_protection_or_other_interval']+=1;continue
        observed_angles=_angles(p,center,radial)
        lo,hi=ranges[sides[0]]
        if blocked[ys,xs].any():
            rejected['source_head_or_extension_contact']+=1;continue
        if not np.all((observed_angles>=lo)&(observed_angles<=hi)):
            rejected['source_cell_outside_observed_flank']+=1;continue
        if np.any(abs(np.linalg.norm(p-center,axis=1)-radius)>bound):
            rejected['native_ink_outside_measured_arc']+=1;continue
        accepted[sides[0]].append({'working_pixel':list(q),'candidate_ids':sorted(owners),
            'source_cell_px':list(box),'source_ink_pixels':len(xs),
            'source_ink_sha256':object_sha(p.tolist())})
    row.update(flanks=[{'side':side,'distinct_working_PART_pixels':len(points),'supports':points}
        for side,points in zip((-1,1),accepted)],rejections=dict(rejected),
        competing_known_roles=conflicts,measured_arc_ink_band_px=bound,
        ownership_holds=ownership_holds,
        native_resolved_competitors=resolved_competitors,
        measured_band_does_not_replace_model_uncertainty=True)
    row['accepted']=not conflicts and all(len(p)>=POLICY['minimum_points_per_flank'] for p in accepted)
    row['reasons']=(['known_competing_role_on_observed_arc'] if conflicts else
        [] if row['accepted'] else ['insufficient_distinct_PART_support_on_both_observed_flanks'])
    return row


def observe(gray,working_gray,geometry,roles,chain_row,context_row,matrix,*,protection=None):
    """Caller supplies current freshly inspected rows, not saved certificates."""
    matrix=checked_matrix(matrix)
    if (gray.ndim!=2 or gray.dtype!=np.uint8 or working_gray.ndim!=2 or working_gray.dtype!=np.uint8
            or roles.shape!=working_gray.shape or roles.dtype!=np.uint8
            or not np.isin(roles,[*range(10),255]).all()):
        raise ValueError('Aligned grayscale and uint8 role rasters required')
    if protection is not None and (protection.shape!=roles.shape or protection.dtype!=np.uint8):
        raise ValueError('Aligned uint8 working protection required')
    arrays=[gray,working_gray,roles]+([] if protection is None else [protection])
    hashes=[_sha(a) for a in arrays]
    immutable=object_sha([geometry,chain_row,context_row])
    report={'schema_version':POLICY['version'],'policy':deepcopy(POLICY),'accepted':False,
        'complete':True,'anchors':[],'reasons':[],'source_window_sha256':hashes[0],
        'working_gray_sha256':hashes[1],'roles_sha256':hashes[2],
        'protection_sha256':None if protection is None else hashes[3],
        'geometry_sha256':object_sha(geometry),'chain_sha256':object_sha(chain_row),
        'context_sha256':object_sha(context_row),'working_to_source':matrix.tolist(),
        'roles_changed':0,'new_PART_classification':False,'physical_geometry_approved':False}
    work={'cell_pixels':0,'competitor_raster_pixels':0,'native_competitor_checks':0,
        'native_competitor_samples':0,'competitor_point_checks':0}
    if gray.size>POLICY['maximum_roi_pixels'] or roles.size>POLICY['maximum_working_pixels']:
        report.update(complete=False,reasons=['raster_budget'])
    elif protection is None:
        report['reasons']=['working_protection_required']
    elif not _supported_matrix(matrix):
        report['reasons']=['unsupported_source_cell_transform']
    elif chain_row.get('status')!='source_linear_chain_observed':
        report['reasons']=['current_chain_not_observed']
    else:
        if (chain_row['id']!=context_row['id'] or chain_row['source_box_px']!=context_row['source_box_px']
                or chain_row['source_window_sha256']!=hashes[0] or context_row['source_window_sha256']!=hashes[0]):
            raise ValueError('ARC PART prior belongs to another current source window')
        box=chain_row['source_box_px']
        if gray.shape!=(box[3]-box[1],box[2]-box[0]):
            raise ValueError('Source window shape differs from its original box')
        intervals=geometry.get('ownership_intervals',geometry['candidates'])
        if (len(intervals)>POLICY['maximum_intervals'] or
                sum(len(i.get('support_points',i.get('points_px',[]))) for i in intervals)>POLICY['maximum_source_points']):
            report.update(complete=False,reasons=['source_inventory_budget'])
        elif len(chain_row['contacts'])!=2:
            report['reasons']=['two_current_contacts_required']
        else:
            working_ink=cv2.threshold(working_gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]>0
            try:
                for contact in chain_row['contacts']:
                    report['anchors'].append(_anchor(gray,working_ink,geometry,roles,contact,context_row,matrix,
                        np.asarray(chain_row['source_box_px'][:2]),protection,work))
                report['accepted']=all(a['accepted'] for a in report['anchors'])
                if not report['accepted']:report['reasons']=['existing_PART_prior_not_corroborated_for_both_anchors']
            except (_Limit,TransferLimit) as exc:
                report.update(accepted=False,complete=False,reasons=[str(exc)])
    report['work']=work
    if hashes!=[_sha(a) for a in arrays] or immutable!=object_sha([geometry,chain_row,context_row]):
        raise ValueError('ARC prior observation mutated inputs')
    return {'report':report,'accepted':report['accepted'],'complete':report['complete']}
