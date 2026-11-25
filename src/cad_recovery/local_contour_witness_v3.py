"""Independent, bounded local material-corner proposals on observed source ink.

The legacy semantic graph is immutable and may remain incomplete. Completeness
here describes only this source inventory / primitive / local-contact search.
This module does not admit PART past negative guards or change CAD geometry.
"""
from collections import Counter,defaultdict
from copy import deepcopy
import math
import cv2
import numpy as np
from .cad_object_assembly_v5 import assemble
from .grid_role_revision_v1 import check_authority,sha
from .local_contour_profiles_v2 import measure,junction_distance
from .contact_core_probes_v1 import probes,profile_junctions
from .part_admission_v1 import object_sha
from .stroke_roles_v1 import checked_gray,foreground

POLICY={'version':'local-contour-witness/3.0','maximum_intervals':4096,'maximum_objects':4096,
    'maximum_source_points':2000000,'maximum_lines':512,'maximum_pair_checks':250000,
    'maximum_profile_samples':2000000,'maximum_local_raster_pixels':64000000,
    'minimum_reference_length_in_pens':6.,'minimum_boundary_length_in_pens':6.,'minimum_seed_length_in_pens':12.,
    'minimum_pen_ratio':1.5,'maximum_pair_pen_ratio':1.67,'maximum_profile_ratio':2.,
    'minimum_corner_angle_deg':20.,'minimum_distance_from_right_angle_deg':20.,
    'minimum_thin_reference_sources':3,'source_walk_reach_in_pens':4.,'pixel_roles_modified':False,
    'global_semantic_graph_rebuilt':False,'physical_geometry_approved':False,'production_admitted':False}


def inventory(packet):
    """Validate every immutable observed interval against its source path."""
    geometry=packet['geometry'];graph=packet['graph'];items=graph['proposals']
    if len(items)>POLICY['maximum_intervals']:raise ValueError('Local source interval budget')
    lookup={c['id']:c for c in geometry}
    if len(lookup)!=len(geometry) or len({p['id'] for p in items})!=len(items):raise ValueError('Unique source identities required')
    parts=defaultdict(list);points=0
    for c in geometry:
        q=np.asarray(c['support_points'],float);a,b=c['source_index_interval']
        if (q.ndim!=2 or q.shape[1]!=2 or len(q)!=b-a+1 or not np.isfinite(q).all() or not np.array_equal(q,np.rint(q))
                or (q<0).any() or (q[:,0]>=packet['gray'].shape[1]).any() or (q[:,1]>=packet['gray'].shape[0]).any()):
            raise ValueError('Finite bounded observed source pixel centres required')
    for p in items:
        c=lookup.get(p['parent_primitive_id'])
        if c is None or p['source_observation_id']!=c['source_observation_id'] or p['path_id']!=c['path_id']:
            raise ValueError('Interval belongs to another source path')
        low,high=p['source_index_interval'];a,b=c['source_index_interval']
        if not a<=low<=high<=b or p['points_px']!=c['support_points'][low-a:high-a+1]:raise ValueError('Interval source support differs')
        parts[c['id']].append((low,high));points+=len(p['points_px'])
        if points>POLICY['maximum_source_points']:raise ValueError('Local source point budget')
    for c in geometry:
        spans=sorted(parts[c['id']])
        if not spans or spans[0][0]!=c['source_index_interval'][0] or spans[-1][1]!=c['source_index_interval'][1] or any(a[1]!=b[0] for a,b in zip(spans,spans[1:])):
            raise ValueError('Local inventory does not cover its observed source')
    return {'complete':True,'intervals':len(items),'source_points':points,'source_paths':len(geometry),
        'scope':'checked_observed_source_inventory_only','legacy_graph_complete':not graph.get('search_incomplete',True)}


def _contact(ink,blocked,a,b,pen,audit):
    pad=math.ceil(pen+math.sqrt(.5));lo=np.maximum(0,np.minimum(a,b)-pad);hi=np.minimum(ink.shape[::-1],np.maximum(a,b)+pad+1)
    audit['local_raster_pixels']+=int(np.prod(hi-lo))
    if audit['local_raster_pixels']>POLICY['maximum_local_raster_pixels']:return None
    local=((ink[lo[1]:hi[1],lo[0]:hi[0]]>0)&~blocked[lo[1]:hi[1],lo[0]:hi[0]]).astype(np.uint8)
    _,labels=cv2.connectedComponents(local,connectivity=8)
    aa=a-lo;bb=b-lo;x=int(labels[aa[1],aa[0]]);y=int(labels[bb[1],bb[0]])
    return {'points_px':[a.tolist(),b.tolist()],'source_connected':True,'bbox_px':[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])]} if x and x==y else None


def analyze(packet,*,authority,protected=None):
    gray,roles=packet['gray'],packet['roles'];checked_gray(gray);shape=gray.shape
    owner,ambiguous=packet['source_owner'],packet['ambiguous_owner_mask'];graph=packet['graph']
    if packet.get('input_sha256')!=sha(gray):raise ValueError('Local witness belongs to another image')
    if roles.shape!=shape or roles.dtype!=np.uint8 or owner.shape!=shape or owner.dtype.kind not in 'ui' or ambiguous.shape!=shape or ambiguous.dtype!=np.uint8:
        raise ValueError('Aligned source role/ownership arrays required')
    if owner.min(initial=0)<0 or owner.max(initial=0)>len(graph['proposals']) or np.any((ambiguous>0)&(owner!=0)):raise ValueError('Invalid local source ownership')
    if not np.isin(roles,[*range(10),255]).all():raise ValueError('Invalid source role value')
    protected=np.zeros(shape,bool) if protected is None else protected
    if protected.shape!=shape or protected.dtype!=bool:raise ValueError('Aligned boolean protection required')
    source_inventory=inventory(packet);allowed=check_authority(gray,roles,authority)
    immutable=object_sha([graph,packet['geometry'],packet['heads']]);snapshots=[sha(a) for a in (gray,roles,owner,ambiguous,protected)]
    blocked=protected.copy()|(roles==6)
    for h in packet['heads']:
        p=np.asarray(h.get('polygon_px'),float)
        if p.shape!=(3,2) or not np.isfinite(p).all():raise ValueError('Finite observed arrow polygon required')
        mask=np.zeros(shape,np.uint8);cv2.fillPoly(mask,[np.rint(p).astype(np.int32)],255);blocked|=mask>0
    result=assemble(gray,packet['geometry'],roles=roles,protected=blocked)
    audit={'limits':[],'rejections':{},'profile_samples':0,'pair_checks':0,'local_raster_pixels':0,'objects_visited':0,'source_walk_contacts':0}
    if packet.get('fresh_source_extraction_complete',True) is not True:audit['limits'].append('source_extraction_incomplete')
    if not result['search_complete']:audit['limits'].append('source_primitive_search')
    objects,probe_audit=probes(result,packet['geometry'],gray=gray,protected=blocked,maximum_segments=POLICY['maximum_objects'])
    audit['source_probe_search']=probe_audit
    if not probe_audit['complete'] or len(objects)>POLICY['maximum_objects']:audit['limits'].append('source_object_budget')
    ink=foreground(gray);dt=cv2.distanceTransform(ink,cv2.DIST_L2,5);work={'ink':ink,'distance':dt,'samples':0}
    junctions=junction_distance(packet['geometry'],shape)
    records=[]
    def reject(reason):audit['rejections'][reason]=audit['rejections'].get(reason,0)+1
    for o in objects if not audit['limits'] else []:
        audit['objects_visited']+=1
        if o['kind']!='line':reject('not_straight_branch');continue
        p=np.asarray(o['support_points'],int);ends=np.asarray(o['points_px'],float);length=float(np.linalg.norm(ends[1]-ends[0]))
        if length<POLICY['minimum_reference_length_in_pens']:reject('short_source');continue
        profile=measure(ends,p,gray,work,profile_junctions(o,shape,junctions),maximum_samples=POLICY['maximum_profile_samples'])
        if work['samples']>POLICY['maximum_profile_samples']:audit['limits'].append('profile_budget');break
        if profile is None:reject('profile_unresolved');continue
        low,high=map(float,np.quantile(profile['sections_px'],[.25,.95]))
        if high>POLICY['maximum_profile_ratio']*low or length<POLICY['minimum_reference_length_in_pens']*low:
            reject('unstable_or_short_profile');continue
        known=set(map(int,roles[p[:,1],p[:,0]]))-{255}
        actual=p[np.argmin(np.linalg.norm(p[:,None]-ends[None,:],axis=2),axis=0)]
        records.append({'id':o['id'],'object':o,'points':ends,'source_ends':actual,'u':(ends[1]-ends[0])/length,
            'length':length,'pen':low,'upper_pen':high,'profile':profile,'known':known,
            'source_ids':{m['source_observation_id'] for m in o['source_members']}})
    audit['profile_samples']=work['samples'];audit['measured_line_count']=len(records)
    if len(records)>POLICY['maximum_lines']:audit['limits'].append('direction_budget')
    boundaries=[r for r in records if not r['known'] and r['length']>=POLICY['minimum_boundary_length_in_pens']*r['pen']]
    source_lookup=defaultdict(list)
    for c in packet['geometry']:source_lookup[c['source_observation_id']].append(c)
    audit['boundary_candidates']=len(boundaries);witnesses=[];selected=set();seen=set()
    if boundaries and not audit['limits']:
        largest=max(r['pen'] for r in boundaries);cell=max(2,math.ceil(POLICY['source_walk_reach_in_pens']*largest+2*math.sqrt(.5)));index=defaultdict(list)
        for i,r in enumerate(boundaries):
            for end,p in enumerate(r['source_ends']):index[tuple(np.floor(p/cell).astype(int))].append((i,end))
        for i,a in enumerate(boundaries):
            for ae,p in enumerate(a['source_ends']):
                bucket=np.floor(p/cell).astype(int)
                for dx in (-1,0,1):
                    for dy in (-1,0,1):
                        for j,be in index.get(tuple(bucket+[dx,dy]),[]):
                            if j<=i or (i,j) in seen:continue
                            audit['pair_checks']+=1
                            if audit['pair_checks']>POLICY['maximum_pair_checks']:audit['limits'].append('pair_budget');break
                            b=boundaries[j];q=b['source_ends'][be];reach=a['pen']+b['pen']+2*math.sqrt(.5)
                            source_walk=None
                            if np.linalg.norm(p-q)>reach:
                                limit=POLICY['source_walk_reach_in_pens']*max(a['pen'],b['pen'])+2*math.sqrt(.5)
                                if np.linalg.norm(p-q)>limit:continue
                                # Only the same immutable observed path may explain
                                # a short intervening bend. Pixel proximity cannot.
                                carriers=[c for identity in a['source_ids']&b['source_ids'] for c in source_lookup[identity]]
                                for c in carriers:
                                    audit['pair_checks']+=1
                                    if audit['pair_checks']>POLICY['maximum_pair_checks']:audit['limits'].append('source_walk_budget');break
                                    path=np.asarray(c['support_points'],int)
                                    starts=np.flatnonzero(np.all(path==p,axis=1));stops=np.flatnonzero(np.all(path==q,axis=1))
                                    if not len(starts) or not len(stops):continue
                                    lo,hi=sorted((int(starts[0]),int(stops[0])));walk=path[lo:hi+1]
                                    length=float(np.linalg.norm(np.diff(walk,axis=0),axis=1).sum())
                                    if length<=limit and np.all(ink[walk[:,1],walk[:,0]]>0) and not blocked[walk[:,1],walk[:,0]].any():
                                        source_walk={'candidate_id':c['id'],'source_index_interval':[lo+c['source_index_interval'][0],hi+c['source_index_interval'][0]],'length_px':length};break
                                if source_walk is None:continue
                            if max(a['length']/a['pen'],b['length']/b['pen'])<POLICY['minimum_seed_length_in_pens']:continue
                            angle=math.degrees(math.acos(float(np.clip(abs(a['u']@b['u']),0,1))))
                            if not POLICY['minimum_corner_angle_deg']<=angle<=90-POLICY['minimum_distance_from_right_angle_deg']:continue
                            if max(a['pen'],b['pen'])/min(a['pen'],b['pen'])>POLICY['maximum_pair_pen_ratio']:continue
                            contact=_contact(ink,blocked,p,q,max(a['pen'],b['pen']),audit)
                            if audit['local_raster_pixels']>POLICY['maximum_local_raster_pixels']:audit['limits'].append('contact_raster_budget');break
                            if contact is None:reject('unsupported_source_contact');continue
                            endpoints=np.concatenate([a['points'],b['points']]);margin=max(a['length'],b['length'])/2
                            lo=endpoints.min(0)-margin;hi=endpoints.max(0)+margin;refs=[];source_ids=set(a['source_ids']|b['source_ids'])
                            for ref in sorted(records,key=lambda r:r['id']):
                                center=ref['points'].mean(0)
                                if (ref['source_ids']&source_ids or np.any(center<lo) or np.any(center>hi)
                                        or ref['upper_pen']*POLICY['minimum_pen_ratio']>min(a['pen'],b['pen'])):continue
                                refs.append(ref);source_ids|=ref['source_ids']
                            if len(refs)<POLICY['minimum_thin_reference_sources']:reject('insufficient_local_thin_evidence');continue
                            seen.add((i,j));selected|={a['id'],b['id']}
                            if source_walk is not None:audit['source_walk_contacts']+=1
                            witnesses.append({'branch_object_ids':[a['id'],b['id']],'angle_deg':angle,'contact':contact,
                                'boundary_profiles':[a['profile'],b['profile']],
                                'reference_object_ids':[r['id'] for r in refs],
                                'native_pen_ratio':min(a['pen'],b['pen'])/max(r['upper_pen'] for r in refs),
                                'context_bbox_px':[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],
                                'short_observed_source_walk':source_walk})
                        if audit['limits']:break
                    if audit['limits']:break
                if audit['limits']:break
            if audit['limits']:break
    proposal=np.zeros(shape,np.uint8);object_evidence=[]
    eligible=allowed&(roles==255)&~blocked&(owner>0)&(ambiguous==0)&(ink>0)
    for r in records if not audit['limits'] else []:
        if r['id'] not in selected:continue
        o=r['object'];source_numbers=set()
        for m in o['source_members']:
            a,b=m['source_index_interval']
            source_numbers|={n for n,p in enumerate(graph['proposals'],1) if p['parent_primitive_id']==m['candidate_id']
                and max(a,p['source_index_interval'][0])<min(b,p['source_index_interval'][1])}
        seed=np.zeros(shape,np.uint8);q=np.asarray(o['support_points'],int);seed[q[:,1],q[:,0]]=255
        distance=cv2.distanceTransform(255-seed,cv2.DIST_L2,5)
        accepted=eligible&np.isin(owner,list(source_numbers))&(distance<=r['pen']/2+math.sqrt(.5))
        if not o.get('witness_only',False):proposal[accepted]=255
        else:accepted[:]=False
        object_evidence.append({'kind':'line','witness_only':bool(o.get('witness_only',False)),'id':r['id'],'source_members':o['source_members'],'source_owner_numbers':sorted(source_numbers),
            'points_px':o['points_px'],'observed_pen_px':r['pen'],'proposed_pixels':int(accepted.sum())})
    if immutable!=object_sha([graph,packet['geometry'],packet['heads']]) or snapshots!=[sha(a) for a in (gray,roles,owner,ambiguous,protected)]:
        raise ValueError('Local positive search changed original observations')
    return {'mask':proposal,'complete':not audit['limits'],'scope':'independent_local_positive_stage_only',
        'source_inventory':source_inventory,'assembly_source_edge_partition_exact':result['source_edge_partition_exact'],
        'legacy_packet_complete':packet.get('complete'),'legacy_graph_search_incomplete':graph.get('search_incomplete'),
        'source_sha256':sha(gray),'source_graph_sha256':object_sha(graph),'source_owner_sha256':sha(owner),
        'source_heads_sha256':object_sha(packet['heads']),
        'witnesses':witnesses if not audit['limits'] else [],'objects':object_evidence,'audit':audit,'policy':deepcopy(POLICY),
        'raw_positive_pixels':int(np.count_nonzero(proposal)),'source_immutable':True,'full_roles_modified':False}
