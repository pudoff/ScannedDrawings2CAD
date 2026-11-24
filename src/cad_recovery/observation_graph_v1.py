"""Read-only observation graph, with explicit frames and unconfirmed roles.

DrawingGraph 0.2 entities can stage pixel observations. A complete physical
graph requires a separately reviewed Page/View contract; no pixel is a mm.
"""
from copy import deepcopy
import hashlib
import math
import numpy as np
from .dataset_layout_v1 import encoded
from .graph_v2 import Evidence, LineV2, CircleV2, ArcV2, TextV2

VERSION = 'observation-graph/1.0'
LIMITS = {'maximum_objects': 25_000, 'maximum_endpoints': 40_000, 'maximum_pair_checks': 100_000}
REFS = ('measured_candidate_ids','dimension_line_candidate_ids','extension_line_candidate_ids','arrow_candidate_ids')


class ObservationLimit(ValueError):pass


def matrix(value):
    m = np.asarray(value, dtype=float)
    if m.shape!=(3,3) or not np.isfinite(m).all() or not np.array_equal(m[2],[0,0,1]) or abs(np.linalg.det(m[:2,:2]))<1e-12:
        raise ValueError('A finite invertible affine frame is required')
    return m


def transform(points, affine):
    p = np.asarray(points, float)
    if p.size==0:return []
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all():raise ValueError('Finite pixel points required')
    return (np.c_[p,np.ones(len(p))]@affine.T)[:,:2].tolist()


def bounds(points):
    p=np.array(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or not len(p) or not np.isfinite(p).all():raise ValueError('Finite object extent required')
    return [float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]


def _geometry_extent(g):
    if g['kind']=='line':return bounds(g['points_px'])
    c=np.array(g['center_px'],float);r=float(g['radius_px'])
    if not np.isfinite(c).all() or not math.isfinite(r) or r<=0:raise ValueError('Invalid circular observation')
    return [float(c[0]-r),float(c[1]-r),float(c[0]+r),float(c[1]+r)]


def _checked_ids(rows):
    ids=[r['id'] for r in rows]
    if any(not isinstance(i,str) or not i or len(i)>1000 for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Nonempty unique source IDs required in each observation kind')


def _row(kind,source_id,box,original):
    return {'id':kind+':'+source_id,'source_id':source_id,'kind':kind,'status':'proposed',
        'semantic_status':'unknown','view_membership':'unknown','bbox_px':box,
        'accepted':False,'observation':deepcopy(original),'reasons':['automatic_observation_requires_review']}


def _link(choice,map_ref,affine):
    refs={key:[map_ref(key,x) for x in choice.get(key,[])] for key in REFS}
    unresolved=[x for values in refs.values() for x in values if x.startswith('unresolved:')]
    return {'anchors_px':transform(choice.get('anchors_px',[]),affine),'references':refs,
            'status':'proposed' if choice.get('status')=='proposed' else 'unknown',
            'producer_status':choice.get('status','not_provided'),'reasons':deepcopy(choice.get('reasons',[])),
            'unresolved_references':unresolved,'evidence':deepcopy(choice.get('evidence',{})),
            'evidence_coordinate_space':'producer frame retained; see local_to_observation',
            'local_to_observation':affine.tolist(),'accepted':False}


def build_flat(*, geometry, ocr, arrows, associations, support=None, page_index=0):
    """Accept c.12/c.13/joint source-bound reports without expected labels."""
    common={k:geometry[k] for k in ('source_sha256','processed_size_px')}
    for report in (ocr,arrows,associations):
        if any(report.get(k)!=v for k,v in common.items()):raise ValueError('Observation report frame mismatch')
    candidates=deepcopy(geometry['candidates'])
    if support is not None:
        if any(support.get(k)!=v for k,v in common.items()):raise ValueError('Annotation support frame mismatch')
        by_id={g['id']:g for g in candidates}
        for span in support.get('spans',[]):
            if span.get('role')!='annotation_support_only' or span.get('accepted_as_part_geometry') is not False:
                raise ValueError('Invalid annotation support role')
            if span['id'] in by_id:
                if by_id[span['id']]['points_px']!=span['points_px']:raise ValueError('Annotation support ID collision')
                continue
            candidates.append({**deepcopy(span),'kind':'line','classification':'annotation_support_only'})
    source={'geometry':candidates,'texts':ocr['lines'],'notations':ocr.get('notations',[]),'arrows':arrows['candidates'],
            'associations':associations['associations']}
    packet={**common,'source_size_px':common['processed_size_px'],**deepcopy(source),
        'coordinate_system':'processed_image_pixel_centers','page_index':page_index,
        'geometry_search_incomplete':geometry.get('analysis',{}).get('contour_limit_reached',False),
        'association_completeness':associations.get('complete'), 'aliases':{},'windows':[],
        'adapter_input':'flat_source_bound_reports','source_reports':{'geometry':deepcopy(geometry.get('analysis',{})),
            'association_issues':deepcopy(associations.get('issues',[]))}}
    return build(packet)


def _connections(objects,source_sha):
    endpoints=[]
    for obj in objects:
        if obj['kind']!='geometry' or obj['observation']['kind']!='line':continue
        g=obj['observation'];unc=g.get('uncertainty_px')
        if unc is not None and (not isinstance(unc,(int,float)) or not math.isfinite(unc) or unc<0):
            raise ValueError('Invalid endpoint localization uncertainty')
        for i,p in enumerate(g['points_px']):
            endpoints.append({'id':'end:'+obj['source_id']+':'+str(i),'kind':'endpoint','object_id':obj['id'],
                'point_px':p,'uncertainty_px':unc,'status':'unknown','candidate_ids':[],
                'reason':'no_endpoint_candidate_is_not_proof_of_a_broken_part'})
    complete=len(endpoints)<=LIMITS['maximum_endpoints'];endpoints=endpoints[:LIMITS['maximum_endpoints']]
    cell=max(1.,2*max((e['uncertainty_px'] or 0 for e in endpoints),default=0))
    grid={};pairs=[];checks=0
    for e in sorted(endpoints,key=lambda e:e['id']):
        x,y=e['point_px'];key=(math.floor(x/cell),math.floor(y/cell))
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for other in grid.get((key[0]+dx,key[1]+dy),[]):
                    if e['object_id']==other['object_id']:continue
                    checks+=1
                    if checks>LIMITS['maximum_pair_checks']:complete=False;break
                    distance=math.dist(e['point_px'],other['point_px'])
                    tolerance=(e['uncertainty_px'] or 0)+(other['uncertainty_px'] or 0)
                    if distance>max(1e-7,tolerance):continue
                    ids=sorted([e['id'],other['id']]);cid='contact:'+hashlib.sha256(encoded([source_sha,ids])).hexdigest()[:24]
                    pairs.append({'id':cid,'kind':'endpoint_contact','status':'proposed','accepted':False,
                        'endpoint_ids':ids,'object_ids':sorted([e['object_id'],other['object_id']]),
                        'distance_px':distance,'localization_tolerance_px':tolerance,
                        'bbox_px':bounds([e['point_px'],other['point_px']]),
                        'reason':'spatial_endpoint_contact_does_not_establish_a_semantic_junction'})
                    e['candidate_ids'].append(cid);other['candidate_ids'].append(cid)
                if not complete:break
            if not complete:break
        grid.setdefault(key,[]).append(e)
        if not complete:break
    if not complete:
        for pair in pairs:pair['status']='unknown'
    for e in endpoints:
        e['no_endpoint_neighbor']=bool(complete and not e['candidate_ids'])
        if e['candidate_ids'] and complete:e['status']='proposed'
    return {'complete':complete,'scope':'endpoint to endpoint only; no interior crossing becomes a node',
            'pair_checks':checks,'endpoints':endpoints,'contacts':pairs,
            'no_endpoint_neighbor_count':sum(e['no_endpoint_neighbor'] for e in endpoints)}


def _staging(objects,size,sha):
    flip=np.array([[1.,0,0],[0,-1.,size[1]],[0,0,1.]])
    evidence=lambda obj:Evidence(source_ref=obj['id'],method='ocr' if obj['kind']=='text' else 'geometry_fit',status='observed')
    primitives=[];texts=[];unrepresented=[]
    for obj in objects:
        g=obj['observation'];common={'id':obj['id'],'view_id':'pixel-frame','evidence':evidence(obj),'status':'proposed'}
        if obj['kind']=='geometry':
            if g.get('role')=='annotation_support_only' or g.get('classification')=='annotation_support_only':
                unrepresented.append(obj['id']);continue
            if g['kind']=='line':
                a,b=transform(g['points_px'],flip);entity=LineV2(**common,start=a,end=b)
            elif g['kind']=='circle':entity=CircleV2(**common,center=transform([g['center_px']],flip)[0],radius=g['radius_px'])
            elif g['kind']=='arc' and 0<abs(g['sweep_angle_deg'])<360:
                start=-g['start_angle_deg'];end=-(g['start_angle_deg']+g['sweep_angle_deg'])
                if g['sweep_angle_deg']>0:start,end=end,start
                entity=ArcV2(**common,center=transform([g['center_px']],flip)[0],radius=g['radius_px'],start_angle_deg=start,end_angle_deg=end)
            else:unrepresented.append(obj['id']);continue
            primitives.append(entity.model_dump(mode='json'))
        elif obj['kind']=='text':
            p=np.array(transform(g['polygon_px'],flip));edges=np.roll(p,-1,axis=0)-p
            axis=edges[np.argmax(np.linalg.norm(edges,axis=1))];normal=np.array([-axis[1],axis[0]])/np.linalg.norm(axis)
            height=max(.001,float(np.ptp(p@normal)))
            entity=TextV2(**common,raw=g['raw'],position=p.mean(axis=0),height=height,
                          rotation_deg=math.degrees(math.atan2(axis[1],axis[0])))
            texts.append(entity.model_dump(mode='json'))
    return {'entity_schema':'DrawingGraph/0.2','coordinate_space':'pixel','source_to_view':flip.tolist(),
            'view_id':'pixel-frame','semantic_view_membership':'unknown','primitives':primitives,'texts':texts,
            'unrepresented_ids':unrepresented,'physical_scale_confirmed':False,
            'drawing_graph_v2':None,'graph_state':'page_and_frame_review_required',
            'reason':'DrawingGraph0.2 requires physical Page dimensions; no pixel-to-mm substitution is made'}


def build(packet):
    """Native-original or flat pixel observations; never trust automatic accepted flags."""
    packet=deepcopy(packet);sha=packet['source_sha256'];size=packet.get('source_size_px') or packet.get('processed_size_px')
    if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Invalid observation source hash')
    if not isinstance(size,list) or len(size)!=2 or any(type(n)is not int or not 0<n<=1_000_000 for n in size):raise ValueError('Invalid observation dimensions')
    categories={'geometry':packet.get('geometry',[]),'text':packet.get('texts',[]),'arrow':packet.get('arrows',[])}
    for values in categories.values():_checked_ids(values)
    _checked_ids(packet.get('associations',[]))
    total=sum(map(len,categories.values()))+len(packet.get('associations',[]))
    if total>LIMITS['maximum_objects']:raise ObservationLimit('Observation graph object resource limit exceeded')
    objects=[];lookup={}
    for kind,values in categories.items():
        for obj in sorted(values,key=lambda o:o['id']):
            if kind=='geometry':box=_geometry_extent(obj)
            elif kind=='text':
                if not isinstance(obj['raw'],str) or len(obj['raw'])>4096:raise ValueError('Invalid OCR raw text')
                box=bounds(obj['polygon_px'])
            else:
                box=bounds([obj['tip_px']]);direction=np.asarray(obj['direction'],float)
                if direction.shape!=(2,) or not np.isfinite(direction).all():raise ValueError('Invalid arrow direction')
            row=_row(kind,obj['id'],box,obj);objects.append(row);lookup[row['id']]=row
    windows={w['id']:w for w in packet.get('windows',[])}
    aliases=packet.get('aliases',{})
    associations=[]
    for a in sorted(packet.get('associations',[]),key=lambda a:a['id']):
        native='local_evidence' in a
        evidence=a['local_evidence'] if native else a
        wid=a.get('source_window_id')
        window=windows.get(wid,{})
        affine=matrix(window.get('processed_to_original',window.get('processed_to_native',np.eye(3)))) if native else np.eye(3)
        if native and not window:raise ValueError('Association references a missing window frame')
        def ref(key,identifier):
            kind='arrow' if key=='arrow_candidate_ids' else 'geometry'
            mapped=identifier
            if native:
                group=aliases.get('arrows' if kind=='arrow' else 'geometry',{})
                found={value for name,value in group.items() if name.endswith('/'+str(wid)+'/'+identifier)}
                mapped=next(iter(found)) if len(found)==1 else 'unresolved:'+identifier
            out=kind+':'+mapped
            return out if out in lookup else 'unresolved:'+key+':'+identifier
        text_id='text:'+a.get('source_text_id',a.get('source_line_id',''))
        if text_id not in lookup:raise ValueError('Association source OCR ID is missing')
        active=_link(evidence,ref,affine)
        if native:
            active['anchors_px']=deepcopy(a.get('anchors_px',[]))
            active['references']={k:[('arrow:' if k=='arrow_candidate_ids' else 'geometry:')+v for v in a.get('references',{}).get(k,[])] for k in REFS}
            active['unresolved_references']=[v for values in active['references'].values() for v in values if v not in lookup]
        alternatives=[_link(c,ref,affine) for c in evidence.get('alternatives',[])]
        for tag,choice in [('active',active),*((f'alternatives/{i}',c) for i,c in enumerate(alternatives))]:
            # Technical IDs refer to the producer's ordered array, not inferred
            # semantic attachment points. Original producer IDs remain intact.
            choice['anchor_ids']=[f'anchor:{a["id"]}:{tag}:{i}' for i in range(len(choice['anchors_px']))]
            choice['anchor_id_origin']='adapter reference to association/choice/anchors_px index; no geometric inference'
            choice['producer_anchor_ids']=deepcopy((evidence if tag=='active' else evidence['alternatives'][int(tag.split('/')[1])]).get('anchor_ids',[]))
        b=lookup[text_id]['bbox_px'];all_points=active['anchors_px']+[p for c in alternatives for p in c['anchors_px']]
        if all_points:b=bounds([[b[0],b[1]],[b[2],b[3]],*all_points])
        dimension_kind={'angle':'angular'}.get(evidence.get('kind'),evidence.get('kind','unknown'))
        nominal=evidence.get('nominal_deg') if dimension_kind=='angular' else evidence.get('nominal_mm')
        row={'id':'association:'+a['id'],'source_id':a['id'],'kind':'association',
             'status':'proposed' if a['status']=='proposed' else 'unknown','producer_status':a['status'],
             'accepted':False,'source_text_id':text_id,'raw':evidence.get('raw',lookup[text_id]['observation']['raw']),
             'dimension_kind':dimension_kind,'nominal_reading':nominal,
             'nominal_unit_status':'degree_observation' if dimension_kind=='angular' else 'unconfirmed',
             'view_membership':'unknown','bbox_px':b,'active':active,'alternatives':alternatives,
             'reasons':deepcopy(evidence.get('reasons',[])),'source_window_id':wid,
             'local_evidence':deepcopy(evidence),'local_to_observation':affine.tolist()}
        associations.append(row)
    connections=_connections(objects,sha)
    return {'schema_version':VERSION,'source_sha256':sha,'source_size_px':size,'page_index':packet.get('page_index',0),
        'coordinate_system':packet.get('coordinate_system','source_pixel_centers'),
        'objects':objects,'associations':associations,'connectivity':connections,'staging_graph':_staging(objects,size,sha),
        'source_completeness':{k:deepcopy(packet.get(k)) for k in ('pixel_coverage_complete','ocr_windows_complete','geometry_search_incomplete','deferred_retries','association_completeness')},
        'source_aliases':deepcopy(aliases),'source_conflicts':deepcopy(packet.get('text_conflicts',[])),
        'source_coordinate_chain':deepcopy(packet.get('coordinate_chain',{})),
        'source_notations':deepcopy(packet.get('notations',[])),
        'source_reports':deepcopy(packet.get('source_reports',{})),
        'counts':{'objects':len(objects),'associations':len(associations),'accepted':0,
                  'proposed_relations':sum(a['status']=='proposed' for a in associations)},
        'limits':LIMITS,'physical_scale_confirmed':False,'solver_state':'requires_reviewed_graph_and_constraints',
        'accepted_geometry':False,'production_admitted':False}
