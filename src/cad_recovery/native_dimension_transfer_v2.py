"""Bind dimension bodies on all measured source axes without tilting to tips.

The frozen V1 binding and input guards remain unchanged. This experimental
wrapper changes only source target construction during its synchronous call.
"""
import math
from unittest.mock import patch

import cv2
import numpy as np

from . import native_dimension_transfer_v1 as previous
from .native_dimension_transfer_v1 import (
    _Limit, _exclusive_sections, _footprint, _possible_working, _sha,
    _strip, _supported_matrix, contact_observe, _line, _width_profile,
)


POLICY = {'version':'native-dimension-transfer/2.0', 'maximum_roi_pixels':4_000_000,
    'maximum_working_pixels':4_000_000, 'maximum_intervals':4096,
    'maximum_source_points':2_000_000, 'maximum_cell_pixels':2_000_000,
    'maximum_source_mask_pixels':32_000_000, 'maximum_competitor_raster_pixels':40_000_000,
    'maximum_width_samples':2_000_000, 'minimum_source_scale':1.,
    'maximum_carrier_axis_pairs':250_000,
    'whole_source_ink_footprint_required':True, 'known_roles_immutable':True,
    'semantic_dimension_approved':False, 'physical_geometry_approved':False,
    'carrier_body_axis':'intersection_of_all_observed_source_carriers',
    'carrier_tip_use':'longitudinal_bounds_only', 'carrier_band_expanded':False}


def _carrier_models(measured, owners, tips):
    """Use every current source axis, with bounded and continuous coverage.

    Tips bound the length. They do not define a new transverse centreline.
    Distinct owner axes must agree over that finite length within their existing
    localization uncertainties. Their measured spans must jointly cover it.
    """
    tips=np.asarray(tips,float)
    if tips.shape!=(2,2) or not np.isfinite(tips).all() or not owners:
        raise ValueError('Two finite source tips and measured carrier owners required')
    if len(owners)*(len(owners)-1)//2>POLICY['maximum_carrier_axis_pairs']:
        raise _Limit('carrier_axis_agreement_budget')
    models=[]
    for identity in sorted(owners):
        row=measured[identity]
        points=np.asarray(row['points_px'],float)
        width=float(row['width_px']);uncertainty=float(row['uncertainty_px'])
        if (points.shape!=(2,2) or not np.isfinite(points).all()
                or not math.isfinite(width) or width<=0
                or not math.isfinite(uncertainty) or uncertainty<0):
            raise ValueError('Finite observed native carrier measurements required')
        delta=points[1]-points[0];length=float(np.linalg.norm(delta))
        if length<=0:raise ValueError('Nondegenerate source carrier required')
        unit=delta/length
        projected=points[0]+((tips-points[0])@unit)[:,None]*unit
        if np.dot(projected[1]-projected[0],tips[1]-tips[0])<=0:
            return None,'measured_carrier_transverse_to_tip_span'
        models.append({'id':identity,'a':projected[0],'b':projected[1],
            'observed_points':points,'width':width,'uncertainty':uncertainty})
    for index,first in enumerate(models):
        for second in models[index+1:]:
            distance=max(float(np.linalg.norm(first[k]-second[k])) for k in ('a','b'))
            if distance>first['uncertainty']+second['uncertainty']+1e-9:
                return None,'measured_carrier_axes_disagree'
    reference=models[0];origin=reference['a'];axis=reference['b']-origin
    length=float(np.linalg.norm(axis));axis/=length
    intervals=sorted(tuple(sorted(map(float,(model['observed_points']-origin)@axis))) for model in models)
    covered=0.
    for start,end in intervals:
        if end<0:continue
        if start>covered+1e-9:
            return None,'measured_carrier_coverage_gap'
        covered=max(covered,end)
        if covered>=length-1e-9:break
    if covered<length-1e-9:
        return None,'measured_carrier_does_not_bound_tip_span'
    return models,None


def _source_target_masks(gray, row, context, work):
    """Recheck current contact/width evidence and derive finite observed bodies."""
    measured={x['id']:x for x in context['stroke_observation']['lines']}
    heads=np.zeros(gray.shape,np.uint8)
    for identity in row['head_ids']:
        options=[o for b in context['head_extension_bindings'] if b['head_id']==identity
                 for o in b['extension_options'] if o['source_shape_rechecked']]
        if len(options)!=1:
            return None,'unique_current_head_extension_missing'
        polygon=np.asarray(options[0]['candidate']['polygon_px'],float)
        if polygon.shape!=(3,2) or not np.isfinite(polygon).all():
            raise ValueError('Current source triangle required')
        cv2.fillConvexPoly(heads,np.rint(polygon).astype(np.int32),1)
    radius=math.ceil(float(row['head_tolerance_px']))
    if radius<0 or radius>max(gray.shape):
        raise ValueError('Invalid source head protection radius')
    heads=cv2.dilate(heads,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
    ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    curves=np.zeros(gray.shape,bool);contacts=[]
    for contact in row['contacts']:
        identity=contact['extension_id']
        if identity not in measured:
            return None,'measured_extension_missing'
        fresh=contact_observe(gray,measured[identity],contact['anchor'],contact['tip_px'])
        if not fresh['complete']:
            raise _Limit('contact_observation_budget')
        if not fresh['report']['accepted']:
            return None,'current_source_contact_not_observed'
        curves |= fresh['curve_possible']
        contacts.append((contact,measured[identity]))
    if len(contacts)!=2:
        return None,'two_contacts_required'
    targets=[]
    owners={x['id'] for x in row['carrier_observations']}
    if not owners or not owners<=measured.keys():
        return None,'measured_carrier_missing'
    tips=[np.asarray(x['tip_px'],float) for x in row['contacts']]
    carriers,failure=_carrier_models(measured,owners,tips)
    if failure:return None,failure
    targets.append({'role':2,'ids':owners,'a':carriers[0]['a'],'b':carriers[0]['b'],
        'width':carriers[0]['width'],'uncertainty':carriers[0]['uncertainty'],
        'carrier_models':carriers})
    for contact,line in contacts:
        targets.append({'role':3,'ids':{contact['extension_id']},
            'a':np.asarray(contact['anchor']['anchor_px'],float),'b':np.asarray(contact['tip_px'],float),
            'width':float(line['width_px']),'uncertainty':float(line['uncertainty_px'])})
    all_target_ids=set().union(*(t['ids'] for t in targets))
    observed_competitors=np.zeros(gray.shape,bool)
    for identity,line in measured.items():
        if identity in all_target_ids:
            continue
        ends,_,_= _line(line['points_px'])
        observed_competitors |= _strip(gray.shape,*ends,
            float(line['width_px'])/2+float(line['uncertainty_px'])+math.sqrt(.5),work)
    blocked=heads|curves|observed_competitors
    for target in targets:
        models=target.get('carrier_models',[target]);profiles=[]
        body=np.ones(gray.shape,bool)
        for model in models:
            profile=_width_profile(ink,model['a'],model['b'],model['width'],heads|curves,work)
            if work['width_samples']>POLICY['maximum_width_samples']:
                raise _Limit('source_width_profile_budget')
            if not profile['accepted']:
                return None,'current_source_width_inconsistent'
            profiles.append(profile)
            body &= _strip(gray.shape,model['a'],model['b'],
                model['width']/2+model['uncertainty']+math.sqrt(.5),work)
        target['profile']=profiles[0]
        target['body']=body
        exclusive=body.copy();proofs=[]
        for model in models:
            section,proof=_exclusive_sections(ink,model,body,work)
            exclusive &= section;proofs.append(proof)
        target['section_proof']=proofs[0]
        if 'carrier_models' in target:
            target['carrier_consensus']={'method':POLICY['carrier_body_axis'],
                'tips_define_transverse_axis':False,'measured_coverage_continuous':True,
                'source_bands_intersected':True,'localization_uncertainty_expanded':False,
                'owners':[{'interval_id':m['id'],'observed_points_px':m['observed_points'].tolist(),
                    'bounded_axis_px':[m['a'].tolist(),m['b'].tolist()],
                    'width_px':m['width'],'uncertainty_px':m['uncertainty'],
                    'width_profile':profile,'section_proof':proof}
                    for m,profile,proof in zip(models,profiles,proofs)]}
        target['safe']=exclusive&~blocked
    # Heads, shared contacts and other target bodies remain shared, even when
    # their original working intervals happened to acquire different labels.
    for target in targets:
        for other in targets:
            if other is target:
                continue
            target['safe'] &= ~other['body']
    return (targets,ink>0,blocked),None



_bind = previous.bind


def bind(gray, working_gray, geometry, roles, chain_row, context_row, matrix, *, protection=None):
    """Retain V1 whole-cell guards and add the measured-carrier diagnostics."""
    consensus={}

    def targets(*args, **kwargs):
        result, failure=_source_target_masks(*args, **kwargs)
        if result is not None:
            for target in result[0]:
                if 'carrier_consensus' in target:
                    for identity in target['ids']:
                        consensus[identity]=target['carrier_consensus']
        return result, failure

    with (patch.object(previous,'POLICY',POLICY),
          patch.object(previous,'_source_target_masks',targets)):
        result=_bind(gray,working_gray,geometry,roles,chain_row,context_row,matrix,protection=protection)
    for row in result['report']['rows']:
        if row['interval_id'] in consensus:
            row['carrier_consensus']=consensus[row['interval_id']]
    return result
