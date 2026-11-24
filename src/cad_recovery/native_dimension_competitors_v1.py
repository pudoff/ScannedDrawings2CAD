"""Resolve finite coarse vetoes using current source component ownership.

Only geometric exceptions are returned. Shared working/native ink, unrelated
owners of the same stroke, and incomplete source evidence remain withheld.
"""
from collections import Counter
from copy import deepcopy
import hashlib

import cv2
import numpy as np

from . import native_dimension_transfer_v1 as binding
from .part_admission_v1 import object_sha
from .source_arrow_observation_v1 import checked_matrix


POLICY={'version':'native-dimension-competitors/1.0','maximum_roi_pixels':4_000_000,
    'maximum_working_pixels':4_000_000,'maximum_intervals':4096,
    'maximum_query_points':4096,'maximum_component_count':16384,
    'maximum_source_cell_pixels':2_000_000,'maximum_component_pixels':8_000_000,
    'maximum_cached_ownership_pixels':32_000_000,
    'maximum_interval_point_checks':2_000_000,'maximum_source_points':2_000_000,
    'source_components_must_be_complete':True,'shared_working_seeds_revisable':False,
    'same_component_exception':'typed_adjacent_same_source_path_disjoint_edges_only',
    'roles_changed':False,'geometry_changed':False,'physical_geometry_approved':False}


class _Limit(Exception):
    pass


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _consume(work,name,count,maximum):
    work[name]=work.get(name,0)+int(count)
    if work[name]>POLICY[maximum]:raise _Limit(maximum)


def _adjacent(rival,target):
    """Source-index adjacency is required; collinearity alone is insufficient."""
    if not rival or rival.get('kind')!='line' or target.get('kind')!='line':return None
    members=rival.get('source_members',[]);parents=target.get('source_members',[])
    if len(members)!=1 or len(parents)!=1:return None
    child,parent=members[0],parents[0]
    keys=('candidate_id','source_observation_id','path_id')
    if any(not child.get(key) or child[key]!=parent.get(key) for key in keys):return None
    a,b=child.get('source_index_interval',[]),parent.get('source_index_interval',[])
    if (len(a)!=2 or len(b)!=2 or any(type(v) is not int or v<0 for v in [*a,*b])
            or not a[0]<a[1] or not b[0]<b[1]):return None
    if a[1]==b[0]:boundary=a[1]
    elif b[1]==a[0]:boundary=a[0]
    else:return None
    first=np.asarray(rival.get('support_points',[]),float)
    second=np.asarray(target.get('support_points',[]),float)
    if first.shape!=(a[1]-a[0]+1,2) or second.shape!=(b[1]-b[0]+1,2):return None
    shared=first[boundary-a[0]]
    if not np.array_equal(shared,second[boundary-b[0]]):return None
    if not np.array_equal(shared,np.rint(shared)):return None
    return {'source_identity':{key:child[key] for key in keys},
        'rival_source_index_interval':a,'target_source_index_interval':b,
        'shared_source_index':boundary,'shared_working_point':shared.astype(int).tolist(),
        'source_edge_overlap':False,'source_path_orientation_verified':True}


def _longitudinal_span(bounds,axis):
    left,top,right,bottom=bounds
    corners=np.array([[left-.5,top-.5],[right-.5,top-.5],
                      [right-.5,bottom-.5],[left-.5,bottom-.5]])
    origin=np.asarray(axis['a'],float);direction=np.asarray(axis['b'],float)-origin
    length=float(np.linalg.norm(direction))
    if length<=0 or not np.isfinite(length):raise ValueError('Finite observed carrier axis required')
    values=(corners-origin)@(direction/length)
    return [float(values.min()),float(values.max())]


def _rival_evidence(ink,labels,stats,target,rival,target_record,matrix,origin,work):
    """Localize all original rival cells; retain mixed and same-track owners.

    ``rival`` has id, points (original ordered working supports), and typed
    (the full candidate or None). Returned mask is diagnostic ownership only.
    This pure helper does not accept an approval flag or change any role.
    """
    proof={'interval_id':rival['id'],'status':'withheld','reasons':[],
        'source_cell_ink_counts':[],'source_components':[],
        'all_untyped_intervals_retained':True,'semantic_role_inferred':False}
    if rival.get('typed') and not np.array_equal(
            np.asarray(rival['typed'].get('support_points',[]),float),rival['points']):
        proof['reasons'].append('typed_rival_differs_from_ownership_support');return proof,None
    _consume(work,'cached_ownership_pixels',ink.size,'maximum_cached_ownership_pixels')
    owned=np.zeros(ink.shape,bool);component_ids=set()
    for point in np.unique(np.rint(rival['points']).astype(int),axis=0):
        bounds=binding._footprint(point,matrix,origin,ink.shape)
        if bounds is None:
            proof['reasons'].append('rival_source_cell_clipped');return proof,None
        left,top,right,bottom=bounds;region=np.s_[top:bottom,left:right]
        _consume(work,'source_cell_pixels',(right-left)*(bottom-top),'maximum_source_cell_pixels')
        actual=ink[region]
        if not actual.any():
            proof['reasons'].append('rival_source_seed_cell_has_no_ink');return proof,None
        owned[region] |= actual
        ids,counts=np.unique(labels[region][actual],return_counts=True)
        component_ids.update(map(int,ids))
        proof['source_cell_ink_counts'].append({'point':point.tolist(),
            'box_px':list(bounds),'component_ink':{str(int(i)):int(n) for i,n in zip(ids,counts)}})
    carrier_ids=set(map(int,np.unique(labels[ink&target['safe']])))
    carrier_ids.discard(0)
    same=component_ids&carrier_ids
    if same:
        adjacent=_adjacent(rival.get('typed'),target_record)
        if adjacent is None:
            proof['reasons'].append('same_source_component_without_adjacent_source_path');return proof,None
        carrier_ink=owned&np.isin(labels,list(same))
        if np.any(carrier_ink&~target['safe']):
            proof['reasons'].append('adjacent_source_ink_not_exclusively_on_observed_carrier');return proof,None
        proof['adjacent_source_interval']=adjacent
        proof['native_transverse_support_rechecked']=True
        proof['adjacent_longitudinal_domains']=[{
            'axis_id':axis.get('id','measured_carrier'),
            'source_cell_spans_px':[_longitudinal_span(cell['box_px'],axis)
                                  for cell in proof['source_cell_ink_counts']]}
            for axis in target.get('carrier_models',[target])]
    # Foreign ownership is the WHOLE connected component, not just whichever
    # pixels happen to lie inside a convenient projected seed cell.
    for number in sorted(component_ids):
        left,top,width,height,area=map(int,stats[number]);right=left+width;bottom=top+height
        carrier=number in carrier_ids
        entry={'component_id':number,'box_px':[left,top,right,bottom],'ink_pixels':area,
            'carrier_component':carrier,'touches_source_window_boundary':
            left==0 or top==0 or right==ink.shape[1] or bottom==ink.shape[0]}
        proof['source_components'].append(entry)
        if carrier:continue
        if entry['touches_source_window_boundary']:
            proof['reasons'].append('foreign_source_component_clipped');return proof,None
        _consume(work,'component_pixels',width*height,'maximum_component_pixels')
        owned[top:bottom,left:right] |= labels[top:bottom,left:right]==number
    proof.update(status='localized',mixed_component_interval=len(component_ids)>1,
        foreign_components_kept_whole=True,carrier_domain='original_registered_support_cells',
        source_pixels_changed=0,original_interval_extended=False)
    return proof,owned


def _query_evidence(proof,owned,ink,target,cell):
    """Check full native and along-track overlap without revising ownership."""
    diagnostic=deepcopy(proof)
    if owned is None:return diagnostic
    left,top,right,bottom=cell;region=np.s_[top:bottom,left:right]
    overlap=int(np.count_nonzero(ink[region]&owned[region]))
    diagnostic['target_cell_source_ink_overlap']=overlap
    longitudinal=[]
    for axis,domain in zip(target.get('carrier_models',[target]),proof.get('adjacent_longitudinal_domains',[])):
        span=_longitudinal_span(cell,axis)
        intersection=max((min(span[1],b)-max(span[0],a)
                          for a,b in domain['source_cell_spans_px']),default=0.)
        longitudinal.append({'axis_id':domain['axis_id'],'query_span_px':span,
            'maximum_positive_domain_overlap_px':max(0.,intersection)})
    if longitudinal:diagnostic['adjacent_query_domain_checks']=longitudinal
    if overlap:diagnostic.update(status='withheld',reasons=['shared_native_ownership_footprint'])
    elif any(check['maximum_positive_domain_overlap_px']>1e-9 for check in longitudinal):
        diagnostic.update(status='withheld',reasons=['adjacent_source_longitudinal_domain_overlap'])
    else:diagnostic['status']='source_separated'
    return diagnostic


def admit(gray,working_gray,geometry,roles,chain_row,context_row,matrix,*,
          protection,baseline,targets,ink,source_protection):
    """Second pass on fresh V2 competing-only holds, with all rivals accounted.

    Target masks and baseline are internal results from the current invocation,
    not public approval documents. Every output point still needs the caller's
    unchanged semantic authority and final CAD verification.
    """
    matrix=checked_matrix(matrix)
    for array in (gray,working_gray,roles,protection):
        if array.ndim!=2 or array.dtype!=np.uint8:raise ValueError('Aligned uint8 rasters required')
    if roles.shape!=working_gray.shape or protection.shape!=roles.shape:
        raise ValueError('Working raster, roles and protection must align')
    if ink.shape!=gray.shape or source_protection.shape!=gray.shape:
        raise ValueError('Current source masks must align with source ROI')
    current=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]>0
    if not np.array_equal(current,ink):raise ValueError('Captured source ink differs from current ROI')
    frozen=baseline['report']
    expected={'source_window_sha256':_sha(gray),'working_pixels_sha256':_sha(working_gray),
        'roles_sha256':_sha(roles),'protection_sha256':_sha(protection),
        'geometry_sha256':object_sha(geometry),'chain_sha256':object_sha(chain_row),
        'context_sha256':object_sha(context_row),'working_to_source':matrix.tolist()}
    if any(frozen.get(key)!=value for key,value in expected.items()):
        raise ValueError('Current baseline binding inputs differ')
    before=object_sha([geometry,chain_row,context_row]);array_hashes=[_sha(a) for a in (gray,working_gray,roles,protection,ink,source_protection)]
    report={'schema_version':POLICY['version'],'policy':deepcopy(POLICY),**expected,
        'baseline_binding_sha256':object_sha(frozen),'source_box_px':chain_row['source_box_px'],
        'rows':[],'limits':[],'shared_points_preserved':[],
        'roles_changed':0,'geometry_changed':False,'source_pixels_changed':0,
        'source_interval_ownership_extended':False,'reference_used_for_inference':False,
        'physical_geometry_approved':False,'production_changed':False}
    report['source_ink_sha256']=_sha(ink)
    report['source_protection_sha256']=_sha(source_protection)
    report['target_source_masks']=[{'interval_ids':sorted(target['ids']),
        'safe_sha256':_sha(target['safe']),'body_sha256':_sha(target['body']),
        'measured_carrier_consensus_sha256':object_sha(target.get('carrier_consensus'))}
        for target in targets if target['role']==2]
    result={};work={'source_cell_pixels':0,'component_pixels':0,'interval_point_checks':0,
                    'source_points':0,'competitor_raster_pixels':0,'cached_ownership_pixels':0}
    origin=np.asarray(chain_row['source_box_px'][:2]);typed={r['id']:r for r in geometry['candidates']}
    try:
        if gray.size>POLICY['maximum_roi_pixels'] or working_gray.size>POLICY['maximum_working_pixels']:
            raise _Limit('raster_budget')
        if not binding._supported_matrix(matrix):raise _Limit('unsupported_source_cell_transform')
        if not frozen['complete'] or frozen['holds'] or frozen['limits']:
            raise _Limit('baseline_binding_incomplete')
        queries=[]
        for row in frozen['rows']:
            if row['role']!=2:continue
            for point in row['withheld_points']:
                if point['reason']=='shared_working_seed':report['shared_points_preserved'].append(point['point'])
                if point['reason']=='competing_working_interval_footprint':queries.append((row['interval_id'],point['point']))
        if len(queries)>POLICY['maximum_query_points']:raise _Limit('query_point_budget')
        if not queries:
            report.update(complete=True,work=work,points_admitted=0,summary={},
                no_competing_queries=True,source_components_examined=False)
            if before!=object_sha([geometry,chain_row,context_row]) or array_hashes!=[_sha(a) for a in (gray,working_gray,roles,protection,ink,source_protection)]:
                raise ValueError('Competitor observation inputs changed before empty-query return')
            return {'points_by_interval':{},'report':report,'complete':True}
        intervals=geometry.get('ownership_intervals',geometry['candidates'])
        if len(intervals)>POLICY['maximum_intervals']:raise _Limit('interval_budget')
        records={};shared=Counter()
        for interval in intervals:
            identity=interval['id'];points=np.asarray(interval.get('support_points',interval.get('points_px',[])),float)
            if identity in records or points.ndim!=2 or points.shape[1]!=2 or not len(points) or not np.isfinite(points).all():
                raise ValueError('Unique finite nonempty source intervals required')
            q=np.rint(points).astype(int)
            if (points<0).any() or (points>=np.array(roles.shape[::-1])).any() or (q<0).any() or (q>=np.array(roles.shape[::-1])).any():
                raise ValueError('Source interval outside working raster')
            _consume(work,'source_points',len(q),'maximum_source_points')
            unique=np.unique(q,axis=0);shared.update(map(tuple,unique))
            records[identity]={'id':identity,'points':points,'q':unique,'typed':typed.get(identity)}
        if not typed.keys()<=records.keys():raise ValueError('Missing typed source ownership intervals')
        count,labels,stats,_=cv2.connectedComponentsWithStats(current.astype(np.uint8),8)
        if count-1>POLICY['maximum_component_count']:raise _Limit('source_component_budget')
        report['source_component_count']=count-1
        report['source_components_examined']=True
        by_id={identity:target for target in targets if target['role']==2 for identity in target['ids']}
        working_ink=cv2.threshold(working_gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]>0
        possible={}
        for identity,record in records.items():
            prior=record['typed'];uncertainty=float(prior.get('uncertainty_px',1.)) if prior else 1.
            possible[identity]=binding._possible_working(record['points'],uncertainty,roles.shape,work)
        localized={}
        for identity,point in queries:
            x,y=point;target=by_id[identity]
            if (identity not in typed or typed[identity].get('kind')!='line'
                    or not np.array_equal(np.asarray(typed[identity].get('support_points',[]),float),records[identity]['points'])
                    or not np.any(np.all(records[identity]['q']==point,axis=1))):
                raise ValueError('Target query differs from its original LINE ownership')
            entry={'interval_id':identity,'point':point,'status':'withheld','reasons':[],'rivals':[]}
            report['rows'].append(entry)
            if (not working_ink[y,x] or protection[y,x] or shared[(x,y)]!=1
                    or baseline['proposed_roles'][y,x]!=255):
                entry['reasons'].append('working_source_or_ownership_protected');continue
            cell=binding._footprint(point,matrix,origin,gray.shape)
            if cell is None:entry['reasons'].append('target_source_cell_clipped');continue
            left,top,right,bottom=cell;region=np.s_[top:bottom,left:right];actual=ink[region]
            _consume(work,'source_cell_pixels',(right-left)*(bottom-top),'maximum_source_cell_pixels')
            if not actual.any() or np.any(actual&(~target['safe'][region]|source_protection[region])):
                entry['reasons'].append('target_source_cell_not_exclusively_safe');continue
            rivals=[]
            for other,record in records.items():
                if other in target['ids']:continue
                _consume(work,'interval_point_checks',1,'maximum_interval_point_checks')
                (a,b,c,d),mask=possible[other]
                if a<=x<c and b<=y<d and mask[y-b,x-a]:rivals.append(record)
            if not rivals:entry['reasons'].append('no_current_competing_interval');continue
            for rival in rivals:
                key=(identity,rival['id'])
                if key not in localized:
                    localized[key]=_rival_evidence(ink,labels,stats,target,rival,typed[identity],matrix,origin,work)
                proof,owned=localized[key];diagnostic=_query_evidence(proof,owned,ink,target,cell)
                entry['rivals'].append(diagnostic)
            if all(r['status']=='source_separated' for r in entry['rivals']):
                entry.update(status='all_current_rivals_source_separated',all_rivals_accounted=True)
                result.setdefault(identity,[]).append(point)
            else:entry['reasons'].append('competing_source_ownership_unresolved')
    except (_Limit,binding._Limit) as exc:
        report['limits'].append(str(exc));result={}
        for row in report['rows']:
            if row['status']=='all_current_rivals_source_separated':
                row.update(status='withheld',reasons=['incomplete_competitor_domain_rollback'])
    if before!=object_sha([geometry,chain_row,context_row]) or array_hashes!=[_sha(a) for a in (gray,working_gray,roles,protection,ink,source_protection)]:
        raise ValueError('Competitor observation mutated its source inputs')
    report.update(complete=not report['limits'],work=work,
        points_admitted=sum(map(len,result.values())),summary=dict(Counter(row['status'] for row in report['rows'])))
    return {'points_by_interval':result,'report':report,'complete':report['complete']}
