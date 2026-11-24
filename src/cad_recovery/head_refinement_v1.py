"""Source-backed binding views of immutable raw head observations.

Localization and own-stem eligibility are independent ablations. Neither view
certifies semantic CAD ownership or a physical dimension.
"""
from copy import deepcopy
import math
import numpy as np
from .stroke_roles_v1 import checked_gray,foreground
from .head_eligibility_v2 import source_width
from .arrow_path_recovery_v1 import native_junction
from .local_width_heads_v1 import profiles

POLICY={'version':'head-refinement/1.0','maximum_heads':512,'maximum_tracks':1024,
        'maximum_track_checks':250000,'maximum_profile_samples':8000000,'maximum_raster_pixels':8000000,
        'maximum_head_extent_px':192.,'minimum_axis_ink_fraction':.95,'sample_step_px':.5,
        'minimum_stem_length_px':6.,'minimum_stem_length_in_pens':3.,
        'maximum_stem_to_base_width':.5,'raw_observations_immutable':True,
        'confidence_calibrated':False,'production_admitted':False,'physical_geometry_approved':False}


def ink_fraction(mask,a,b,audit):
    count=max(2,int(math.ceil(np.linalg.norm(b-a)/.5))+1)
    if audit['raster_pixels']+count>POLICY['maximum_raster_pixels']:
        audit['limits'].add('raster_pixels');return None
    audit['raster_pixels']+=count;p=np.rint(np.linspace(a,b,count)).astype(int)
    if (p<0).any() or (p[:,0]>=mask.shape[1]).any() or (p[:,1]>=mask.shape[0]).any():return None
    return float(np.mean(mask[p[:,1],p[:,0]]>0))


def localize(mask,head,tracks,audit):
    if head.get('evidence',{}).get('path_kind')=='arc':return deepcopy(head),{'status':'unchanged_arc'}
    polygon=np.asarray(head['polygon_px'],float);base=polygon[1:].mean(0);tip=np.asarray(head['tip_px'],float)
    length=float(np.linalg.norm(tip-base));width=source_width(head);candidates=[];rejected=[]
    if not 1<length<=192:return deepcopy(head),{'status':'head_extent_unavailable'}
    direction=np.asarray(head['direction'])
    for track in tracks:
        audit['track_checks']+=1
        if audit['track_checks']>POLICY['maximum_track_checks']:
            audit['limits'].add('track_checks');break
        points=np.asarray(track['points'],float);delta=points[1]-points[0];extent=float(np.linalg.norm(delta))
        if extent<max(20,2*width):continue
        axis=delta/extent
        if abs(axis@direction)>.5:continue
        t,s=np.linalg.solve(np.column_stack((direction,-axis)),points[0]-tip)
        if not (-max(3,head.get('uncertainty_px',1.))<=-t<=.5*length and -width<=s<=extent+width):continue
        point=tip+t*direction;support=ink_fraction(mask,base,point,audit)
        if support is None or support<POLICY['minimum_axis_ink_fraction']:
            rejected.append({'track_id':track['id'],'reason':'interrupted_base_to_junction_ink','axis_ink_fraction':support});continue
        corrected=polygon.copy();corrected[0]=point
        raster_cost=int(np.prod(np.ceil(corrected.max(0))-np.floor(corrected.min(0))+1))+32
        if audit['raster_pixels']+raster_cost>POLICY['maximum_raster_pixels']:
            audit['limits'].add('raster_pixels');break
        audit['raster_pixels']+=raster_cost
        # Reuse frozen native triangle/flank and transverse source checks.
        probe=deepcopy(head);probe['evidence']=deepcopy(probe.get('evidence',{}))
        probe['evidence'].pop('native_junction_lines',None)
        projected=native_junction(probe,mask,width,[{'id':track['id'],'kind':'line','points_px':track['points']}],())
        if projected.get('evidence',{}).get('native_junction_lines')==[track['id']]:
            candidates.append({'track_id':track['id'],'head':projected,'axis_ink_fraction':support})
    groups=[]
    for c in sorted(candidates,key=lambda c:tuple(c['head']['tip_px'])):
        point=np.asarray(c['head']['tip_px'])
        group=next((g for g in groups if all(np.linalg.norm(point-np.asarray(x['head']['tip_px']))<=width for x in g)),None)
        if group is None:groups.append([c])
        else:group.append(c)
    evidence={'status':'unique_source_connected_junction' if len(groups)==1 and not audit['limits'] else 'unchanged_ambiguous_or_missing_junction',
        'candidate_groups':[[{'track_id':c['track_id'],'tip_px':c['head']['tip_px'],'axis_ink_fraction':c['axis_ink_fraction']} for c in g] for g in groups],
        'rejected':rejected}
    if len(groups)!=1 or audit['limits']:return deepcopy(head),evidence
    # Retain one actually tested intersection rather than invent an average point.
    chosen=min(groups[0],key=lambda c:(-c['axis_ink_fraction'],tuple(c['head']['tip_px'])))
    result=chosen['head'];result['evidence']['refinement']={'method':POLICY['version'],'raw_tip_px':head['tip_px'],**evidence}
    return result,evidence


def own_stem(mask,head,audit):
    if head.get('evidence',{}).get('path_kind')=='arc':return True,{'reason':'arc_kept_for_separate_curved_validation'}
    p=np.asarray(head['polygon_px'],float);length=float(np.linalg.norm(p[0]-p[1:].mean(0)))
    breadth=float(np.linalg.norm(p[1]-p[2]));width=source_width(head)
    if not 1<length<=192 or not 0<breadth<=192:return True,{'reason':'stem_window_unavailable'}
    base=p[1:].mean(0);unit=np.asarray(head['direction']);normal=np.array([-unit[1],unit[0]])
    required=max(6.,3*width)
    corners=np.asarray([base-s*unit+t*normal for s in (0.,required) for t in (-breadth/2,breadth/2)])
    if (corners<0).any() or (corners[:,0]>=mask.shape[1]).any() or (corners[:,1]>=mask.shape[0]).any():
        return True,{'reason':'stem_window_clipped_unavailable'}
    estimated=math.ceil((min(40.,max(14.,length))+length+1)/.5)*math.ceil((2*min(96.,breadth+3)+1)/.5)
    if audit['profile_samples']+estimated>POLICY['maximum_profile_samples']:
        audit['limits'].add('profile_samples');return True,{'reason':'stem_budget_unchecked'}
    a,w,c,cost=profiles(mask,p);audit['profile_samples']+=cost
    narrow=(a<-.5)&(w>0)&(w<=.5*breadth)&(np.abs(c)<=max(1.,.75*width))
    changes=np.diff(np.r_[False,narrow,False].astype(int));runs=[]
    minimum=max(6.,3*width)
    for begin,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
        span=float(a[end-1]-a[begin]+.5)
        if a[end-1]>=-max(2.,width) and span>=minimum:runs.append({'length_px':span,'near_base_px':float(a[end-1])})
    return bool(runs),{'reason':'independent_straight_stem' if runs else 'review_without_independent_straight_stem',
        'observed_shaft_pen_px':width,'minimum_stem_length_px':minimum,'accepted_runs':runs}


def refine(gray,observations,tracks):
    checked_gray(gray)
    if len(observations)>512 or len(tracks)>1024:raise ValueError('Bounded heads/tracks required')
    if len({h['id'] for h in observations})!=len(observations):raise ValueError('Unique raw head IDs required')
    for track in tracks:
        points=np.asarray(track['points'],float)
        if points.shape!=(2,2) or not np.isfinite(points).all():raise ValueError('Finite source track required')
    mask=foreground(gray);audit={'track_checks':0,'raster_pixels':0,'profile_samples':0,'limits':set()}
    localized=[];stem=[];combined=[];decisions=[]
    for head in observations:
        p=np.asarray(head['polygon_px'],float);d=np.asarray(head['direction'],float)
        tip=np.asarray(head['tip_px'],float)
        if p.shape!=(3,2) or d.shape!=(2,) or tip.shape!=(2,) or not np.isfinite(tip).all() or not np.isfinite(p).all() or not np.isfinite(d).all() or abs(np.linalg.norm(d)-1)>.01:
            raise ValueError('Finite triangular observation with unit direction required')
        new,local=localize(mask,head,tracks,audit);eligible,reason=own_stem(mask,head,audit)
        localized.append(new)
        if eligible:stem.append(deepcopy(head));combined.append(deepcopy(new))
        decisions.append({'head_id':head['id'],'localization':local,'eligible_for_binding':eligible,'stem':reason,
            'raw_observation_retained':True,'semantic_text_class_confirmed':False})
    incomplete=bool(audit['limits'])
    if incomplete:localized=deepcopy(list(observations));stem=deepcopy(list(observations));combined=deepcopy(list(observations))
    audit['limits']=sorted(audit['limits'])
    return {'raw_observations':deepcopy(list(observations)),'localized':localized,'stem_only':stem,'combined':combined,
        'decisions':decisions,'audit':audit,'search_incomplete':incomplete,'production_admitted':False,
        'physical_geometry_approved':False,'source_pixels_erased':0,'new_head_observations':0}
