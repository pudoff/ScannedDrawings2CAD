"""Bind native dimension bodies to original working intervals conservatively.

The caller supplies a freshly computed chain/context, never an external
certificate. These are geometric proposals; semantic release remains separate.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np

from .native_arc_extension_contact_v1 import observe as contact_observe
from .native_dimension_chain_v1 import _width_profile, _Limit as WidthLimit
from .native_dimension_context_v1 import _line
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix


POLICY = {'version':'native-dimension-transfer/1.0', 'maximum_roi_pixels':4_000_000,
    'maximum_working_pixels':4_000_000, 'maximum_intervals':4096,
    'maximum_source_points':2_000_000, 'maximum_cell_pixels':2_000_000,
    'maximum_source_mask_pixels':32_000_000, 'maximum_competitor_raster_pixels':40_000_000,
    'maximum_width_samples':2_000_000, 'minimum_source_scale':1.,
    'whole_source_ink_footprint_required':True, 'known_roles_immutable':True,
    'semantic_dimension_approved':False, 'physical_geometry_approved':False}


class _Limit(Exception):
    pass


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _supported_matrix(matrix):
    a=matrix[:2,:2]; nonzero=abs(a)>1e-9
    return bool(np.all(nonzero.sum(0)==1) and np.all(nonzero.sum(1)==1)
        and np.min(abs(a[nonzero]))>=POLICY['minimum_source_scale'])


def _footprint(q, matrix, origin, shape):
    """Bounds of all native cells overlapping a working pixel, or None if clipped.

    Matrix must be a validated axis permutation/scale. Cell-edge touches have
    zero area and do not count. Bounds are left, top, right, bottom (exclusive).
    """
    q=np.asarray(q,float)
    corners=q+np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    native=corners@matrix[:2,:2].T+matrix[:2,2]-origin
    low=np.ceil(native.min(0)-.5+1e-9).astype(int)
    high=np.floor(native.max(0)+.5-1e-9).astype(int)+1
    if (low<0).any() or (high>np.array(shape[::-1])).any() or (high<=low).any():
        return None
    return int(low[0]),int(low[1]),int(high[0]),int(high[1])


def _strip(shape, a, b, radius, work):
    work['source_mask_pixels']+=int(np.prod(shape))
    if work['source_mask_pixels']>POLICY['maximum_source_mask_pixels']:
        raise _Limit('source_mask_budget')
    a,b=np.asarray(a,float),np.asarray(b,float)
    delta=b-a; length=float(np.linalg.norm(delta))
    if length<=0 or not math.isfinite(radius) or radius<=0:
        raise ValueError('Finite nondegenerate measured body required')
    unit=delta/length; normal=np.array([-unit[1],unit[0]])
    y,x=np.indices(shape)
    along=(x-a[0])*unit[0]+(y-a[1])*unit[1]
    across=abs((x-a[0])*normal[0]+(y-a[1])*normal[1])
    return (along>=0)&(along<=length)&(across<=radius)


def _possible_working(points, uncertainty, shape, work):
    radius=math.ceil(max(0.,float(uncertainty))+math.sqrt(.5))
    if radius>max(shape):
        raise _Limit('competitor_support_budget')
    lo=np.maximum(0,np.floor(points.min(0)).astype(int)-radius)
    hi=np.minimum(np.array(shape[::-1]),np.ceil(points.max(0)).astype(int)+radius+1)
    cost=int(np.prod(hi-lo));work['competitor_raster_pixels']+=cost
    if work['competitor_raster_pixels']>POLICY['maximum_competitor_raster_pixels']:
        raise _Limit('competitor_support_budget')
    mask=np.zeros(tuple((hi-lo)[::-1]),np.uint8)
    cv2.polylines(mask,[np.rint(points-lo).astype(np.int32)],False,1,2*radius+1)
    if len(points)==1:
        cv2.circle(mask,tuple(np.rint(points[0]-lo).astype(int)),radius,1,-1)
    return tuple(map(int,(*lo,*hi))),mask>0


def _exclusive_sections(ink, target, body, work):
    """Keep only sections with one observed run, including unmodelled ink."""
    a,b=target['a'],target['b'];length=float(np.linalg.norm(b-a));unit=(b-a)/length
    normal=np.array([-unit[1],unit[0]])
    count=max(2,math.ceil(length)+1)
    reach=math.ceil(target['width']/2+target['uncertainty']+math.sqrt(.5))+1
    offsets=np.arange(-reach,reach+.25,.5)
    work['width_samples']+=count*len(offsets)
    if work['width_samples']>POLICY['maximum_width_samples']:
        raise _Limit('source_width_profile_budget')
    sites=np.linspace(a,b,count);valid=np.zeros(count,bool);center=int(np.argmin(abs(offsets)))
    for i,site in enumerate(sites):
        q=np.rint(site+offsets[:,None]*normal).astype(int)
        if ((q<0).any() or (q[:,0]>=ink.shape[1]).any() or (q[:,1]>=ink.shape[0]).any()):
            continue
        bits=ink[q[:,1],q[:,0]]>0
        changes=np.diff(np.r_[False,bits,False].astype(np.int8))
        starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
        if len(starts)!=1 or starts[0]==0 or ends[0]==len(offsets) or not bits[center]:
            continue
        observed=float((ends[0]-starts[0])*.5)
        valid[i]=abs(observed-target['width'])<=2*math.sqrt(.5)
    y,x=np.indices(ink.shape)
    along=(x-a[0])*unit[0]+(y-a[1])*unit[1]
    section=np.rint(np.clip(along/length,0,1)*(count-1)).astype(int)
    return body&valid[section],{'sample_count':count,'unique_supported_sections':int(valid.sum()),
        'secondary_runs_and_unexplained_sections_protected':True}


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
    carrier_width=max(float(measured[i]['width_px']) for i in owners)
    carrier_unc=max(float(measured[i]['uncertainty_px']) for i in owners)
    targets.append({'role':2,'ids':owners,'a':tips[0],'b':tips[1],
        'width':carrier_width,'uncertainty':carrier_unc})
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
        profile=_width_profile(ink,target['a'],target['b'],target['width'],heads|curves,work)
        if work['width_samples']>POLICY['maximum_width_samples']:
            raise _Limit('source_width_profile_budget')
        if not profile['accepted']:
            return None,'current_source_width_inconsistent'
        target['profile']=profile
        body=_strip(gray.shape,target['a'],target['b'],
            target['width']/2+target['uncertainty']+math.sqrt(.5),work)
        target['body']=body
        exclusive,proof=_exclusive_sections(ink,target,body,work)
        target['section_proof']=proof
        target['safe']=exclusive&~blocked
    # Heads, shared contacts and other target bodies remain shared, even when
    # their original working intervals happened to acquire different labels.
    for target in targets:
        for other in targets:
            if other is target:
                continue
            target['safe'] &= ~other['body']
    return (targets,ink>0,blocked),None


def bind(gray, working_gray, geometry, roles, chain_row, context_row, matrix, *, protection=None):
    """Return geometric proposals and UNKNOWN-only candidates, not released roles."""
    matrix=checked_matrix(matrix)
    for image in (gray,working_gray):
        if image.ndim!=2 or image.dtype!=np.uint8:
            raise ValueError('Two-dimensional uint8 rasters required')
    if roles.shape!=working_gray.shape or roles.dtype!=np.uint8:
        raise ValueError('Aligned uint8 working roles required')
    if not np.isin(roles,list(range(10))+[255]).all():
        raise ValueError('Invalid working role code')
    protected=np.zeros_like(roles,np.uint8) if protection is None else protection
    if protected.shape!=roles.shape or protected.dtype!=np.uint8:
        raise ValueError('Aligned uint8 protection required')
    origin=np.asarray(chain_row['source_box_px'][:2],float)
    box=np.asarray(chain_row['source_box_px'],float)
    if (not np.isfinite(box).all() or not np.equal(box,np.rint(box)).all()
            or tuple((box[2:]-box[:2])[::-1])!=gray.shape):
        raise ValueError('Source ROI and recorded box differ')
    if (chain_row['id']!=context_row['id'] or chain_row['source_box_px']!=context_row['source_box_px']
            or chain_row['source_window_sha256']!=_sha(gray)
            or context_row['source_window_sha256']!=_sha(gray)):
        raise ValueError('Current source window differs from fresh chain/context')
    inputs=object_sha([geometry,chain_row,context_row,matrix.tolist()])
    array_shas=[_sha(a) for a in (gray,working_gray,roles,protected)]
    report={'schema_version':POLICY['version'],'policy':deepcopy(POLICY),'chain_id':chain_row['id'],
        'source_box_px':deepcopy(chain_row['source_box_px']), 'source_window_sha256':_sha(gray),
        'working_pixels_sha256':_sha(working_gray),'roles_sha256':_sha(roles),
        'protection_sha256':_sha(protected),'protection_supplied':protection is not None,
        'geometry_sha256':object_sha(geometry),'chain_sha256':object_sha(chain_row),
        'context_sha256':object_sha(context_row),'working_to_source':matrix.tolist(),
        'semantic_dimension_approved':False,'physical_geometry_approved':False,
        'roles_changed':0,'source_pixels_erased':0,'original_intervals_modified':False,
        'source_geometry_modified':False,'projected_seeds_used_as_native_ink':False,
        'known_roles_preserved':True,'rows':[],'limits':[], 'holds':[]}
    work={'source_mask_pixels':0,'width_samples':0,'cell_pixels':0,'source_points':0,
          'competitor_raster_pixels':0}
    result=roles.copy(); owner=np.zeros(roles.shape,np.int32)
    proposed=np.full(roles.shape,255,np.uint8)
    intervals=geometry.get('ownership_intervals',geometry['candidates'])
    typed={x['id']:x for x in geometry['candidates']}; records=[]; points_seen={}
    if not _supported_matrix(matrix):
        report['holds'].append('unsupported_source_cell_transform')
    if (chain_row.get('status')!='source_linear_chain_observed'
            or not chain_row.get('geometry_chain_confirmed')
            or context_row.get('status')!='source_context_observed'
            or not context_row.get('stroke_observation',{}).get('complete')):
        report['holds'].append('current_geometric_chain_not_observed')
    try:
        if gray.size>POLICY['maximum_roi_pixels'] or working_gray.size>POLICY['maximum_working_pixels']:
            raise _Limit('raster_budget')
        if len(intervals)>POLICY['maximum_intervals']:
            raise _Limit('interval_budget')
        for number,interval in enumerate(intervals,1):
            identity=interval['id']
            if identity in points_seen:
                raise ValueError('Unique source interval IDs required')
            points=np.asarray(interval.get('support_points',interval.get('points_px',[])),float)
            if points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all():
                raise ValueError('Finite nonempty source interval points required')
            q=np.rint(points).astype(int)
            if ((points<0).any() or (points[:,0]>=roles.shape[1]).any() or (points[:,1]>=roles.shape[0]).any()
                    or (q<0).any() or (q[:,0]>=roles.shape[1]).any() or (q[:,1]>=roles.shape[0]).any()):
                raise ValueError('Rounded working support lies outside its role raster')
            points_seen[identity]=number;work['source_points']+=len(q)
            if work['source_points']>POLICY['maximum_source_points']:
                raise _Limit('source_point_budget')
            prior=typed.get(identity)
            uncertainty=float(prior.get('uncertainty_px',1.)) if prior else 1.
            if not math.isfinite(uncertainty) or uncertainty<0:
                raise ValueError('Finite nonnegative interval uncertainty required')
            records.append({'id':identity,'number':number,'q':np.unique(q,axis=0),
                'points':points,'uncertainty':uncertainty,'kind':prior['kind'] if prior else 'untyped'})
        if not typed.keys()<=points_seen.keys():
            raise ValueError('Complete ownership must contain every typed interval')
        if not report['holds']:
            observed,failure=_source_target_masks(gray,chain_row,context_row,work)
            if failure:
                report['holds'].append(failure)
            else:
                targets,ink,source_protection=observed
                working_ink=cv2.threshold(working_gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]>0
                shared=np.zeros(roles.shape,np.uint16)
                for record in records:
                    q=record['q'];shared[q[:,1],q[:,0]]+=1
                for target in targets:
                    possible=np.zeros(roles.shape,bool)
                    for other in records:
                        if other['id'] in target['ids']:
                            continue
                        (left,top,right,bottom),mask=_possible_working(other['points'],other['uncertainty'],roles.shape,work)
                        possible[top:bottom,left:right] |= mask
                    for record in records:
                        if record['id'] not in target['ids']:
                            continue
                        if record['kind']!='line':
                            report['holds'].append('target_is_not_original_line_interval')
                            continue
                        row={'interval_id':record['id'],'role':target['role'],
                             'width_profile':target['profile'],'admitted_working_points':[],
                             'source_section_proof':target['section_proof'],
                             'withheld_points':[],'previous_role_counts':{}}
                        for x,y in record['q']:
                            reason=None
                            if not working_ink[y,x]:reason='working_source_ink_missing'
                            elif protected[y,x]:reason='working_protection'
                            elif shared[y,x]>1:reason='shared_working_seed'
                            elif possible[y,x]:reason='competing_working_interval_footprint'
                            footprint=_footprint([x,y],matrix,origin,gray.shape)
                            if reason is None:
                                if footprint is None:reason='source_footprint_clipped'
                                else:
                                    left,top,right,bottom=footprint
                                    work['cell_pixels']+=(right-left)*(bottom-top)
                                    if work['cell_pixels']>POLICY['maximum_cell_pixels']:
                                        raise _Limit('source_cell_budget')
                                    actual=ink[top:bottom,left:right]
                                    safe=target['safe'][top:bottom,left:right]
                                    if not actual.any():reason='source_cell_ink_missing'
                                    elif np.any(actual&~safe):reason='source_cell_has_unowned_or_shared_ink'
                            point=[int(x),int(y)]
                            if reason:
                                row['withheld_points'].append({'point':point,'reason':reason})
                            else:
                                row['admitted_working_points'].append(point)
                                proposed[y,x]=target['role'];owner[y,x]=record['number']
                                if roles[y,x]==255:result[y,x]=target['role']
                        counts=Counter(int(roles[y,x]) for x,y in row['admitted_working_points'])
                        row['previous_role_counts']={str(k):v for k,v in counts.items()}
                        report['rows'].append(row)
    except (WidthLimit,_Limit) as exc:
        report['limits'].append(str(exc))
    if report['limits'] or report['holds']:
        result=roles.copy();owner[:]=0;proposed[:]=255
        for row in report['rows']:
            row['admitted_working_points']=[]
            row['previous_role_counts']={}
        report['rolled_back']=True
    if inputs!=object_sha([geometry,chain_row,context_row,matrix.tolist()]) or array_shas!=[_sha(a) for a in (gray,working_gray,roles,protected)]:
        raise ValueError('Native transfer inputs changed during binding')
    report.update(complete=not report['limits'],work=work,
        candidate_role_changes=int(np.count_nonzero(result!=roles)),
        proposed_role_pixels=int(np.count_nonzero(proposed!=255)),
        summary={str(int(k)):int(v) for k,v in zip(*np.unique(proposed[proposed!=255],return_counts=True))},
        candidate_roles_sha256=_sha(result),geometric_roles_sha256=_sha(proposed),binding_owner_sha256=_sha(owner))
    return {'report':report,'candidate_roles':result,'proposed_roles':proposed,'binding_owner':owner}
