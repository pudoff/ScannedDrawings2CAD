"""Source-bound eligibility; preserve every primitive and symbol observation."""
from collections import defaultdict
import math
import cv2
import numpy as np
from .geometry_bundle_v1 import inspect
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha
from .stroke_roles_v1 import checked_gray,foreground
from .arrow_path_recovery_v1 import native_head_check
from .head_eligibility_v2 import source_width

POLICY={'version':'primitive-ownership/1.0','maximum_primitives':4096,'maximum_heads':512,
    'maximum_profile_samples':2000000,'profile_stations':9,'maximum_normal_radius_px':64,
    'minimum_head_support_fraction':.9,'pixel_cell_radius_px':math.sqrt(.5),
    'minimum_taper_r_squared':.8,'minimum_taper_width_ratio':1.5,
    'minimum_taper_width_change_px':2*math.sqrt(.5),
    'protected_roles':[1,2,3,4,5,6,7,8,9],
    'source_erasure_allowed':False,'CAD_or_physical_approval':False}
ARMS=('unfiltered','head_bound','head_bound_and_taper_review')


def head_geometry_ids(head):
    evidence=head.get('evidence',{});ids=set(evidence.get('geometry_ids',[]))
    ids.update(evidence.get('legacy_profile_evidence',{}).get('geometry_ids',[]))
    if isinstance(evidence.get('path_id'),str):ids.add(evidence['path_id'])
    return ids


def normal_profile(ink,primitive,audit):
    a,b=np.asarray(primitive['points_px'],float);delta=b-a;length=float(np.linalg.norm(delta))
    if length<=0:return {'available':False,'reason':'zero_length'}
    unit=delta/length;normal=np.array([-unit[1],unit[0]])
    width=primitive.get('local_width',{}).get('width_px')
    if width is None or not math.isfinite(width) or width<=0:return {'available':False,'reason':'missing_width'}
    radius=min(64.,max(6.,2*width));offset=np.arange(-radius,radius+.25,.5)
    stations=np.linspace(.25,.75,9);centers=a+stations[:,None]*delta
    points=centers[:,None,:]+offset[None,:,None]*normal;pixel=np.rint(points).astype(int)
    cost=int(pixel.shape[0]*pixel.shape[1])
    if audit['profile_samples']+cost>POLICY['maximum_profile_samples']:
        audit['limits'].append('profile_sample_budget');return {'available':False,'reason':'profile_budget'}
    audit['profile_samples']+=cost
    valid=(pixel>=0).all(2)&(pixel[:,:,0]<ink.shape[1])&(pixel[:,:,1]<ink.shape[0])
    if not valid.all():return {'available':False,'reason':'clipped_source_roi'}
    values=ink[pixel[:,:,1],pixel[:,:,0]]>0;widths=[]
    for row in values:
        edges=np.diff(np.r_[False,row,False].astype(np.int8));runs=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
        if not runs:return {'available':False,'reason':'missing_source_run'}
        lo,hi=min(runs,key=lambda x:abs((offset[x[0]]+offset[x[1]-1])/2))
        if lo==0 or hi==len(offset):return {'available':False,'reason':'normal_span_clipped'}
        if not offset[lo]-.5<=0<=offset[hi-1]+.5:return {'available':False,'reason':'model_axis_outside_ink'}
        widths.append(float(offset[hi-1]-offset[lo]+.5))
    widths=np.asarray(widths);slope,intercept=np.polyfit(stations,widths,1)
    fit=slope*stations+intercept;variance=float(np.sum((widths-widths.mean())**2))
    r2=1-float(np.sum((fit-widths)**2))/variance if variance>1e-12 else 0.
    change=float(abs(fit[-1]-fit[0]));ratio=float(max(fit[0],fit[-1])/max(.5,min(fit[0],fit[-1])))
    return {'available':True,'widths_px':widths.tolist(),'r_squared':r2,'width_ratio':ratio,'width_change_px':change,
        'tapering_nonuniform_stroke':bool(r2>=.8 and ratio>=1.5 and change>=2*math.sqrt(.5)),
        'length_px':length,'coordinate_space':'input_pixel_centres','not_a_confirmed_arrow':True}


def analyze(gray,packet):
    checked_gray(gray)
    if not packet['complete'] or not inspect(gray,packet,packet['input_to_native'])['valid']:
        raise ValueError('Complete checked bundle required')
    geometry=packet['geometry'];heads=packet['heads']
    if len(geometry)>4096 or len(heads)>512:raise ValueError('Ownership input budget exceeded')
    before=object_sha([geometry,packet['source'],packet['graph'],heads,packet['chains']]);pixels=sha(gray);roles=sha(packet['roles'])
    ink=foreground(gray);index=defaultdict(list);head_checks={}
    for head in heads:
        if not head['id'].startswith(packet['prefix']+'/'):raise ValueError('Stale head identity')
        if any(i.startswith(('gb1-','gb2-')) and not i.startswith(packet['prefix']+'/') for i in head_geometry_ids(head)):
            raise ValueError('Foreign head geometry identity')
        valid,check=native_head_check(head,ink,source_width(head),())
        head_checks[head['id']]={'accepted':valid,'native_check':check}
        if valid:
            for identity in head_geometry_ids(head):index[identity].append(head)
    audit={'profile_samples':0,'limits':[]};decisions=[];typed={g['id']:g for g in geometry if g['kind'] in ('line','arc','circle')}
    for identity,g in typed.items():
        support=np.rint(g['support_points']).astype(int)
        protected=bool(np.any(np.isin(packet['roles'][support[:,1],support[:,0]],POLICY['protected_roles'])))
        owners=[];profile=None
        if g['kind']=='line' and not protected:
            for head in index.get(identity,[]):
                polygon=np.asarray(head['polygon_px'],np.float32)
                distances=[cv2.pointPolygonTest(polygon,tuple(map(float,p)),True) for p in g['support_points']]
                fraction=float(np.mean(np.asarray(distances)>=-POLICY['pixel_cell_radius_px']))
                if fraction>=.9:owners.append({'head_id':head['id'],'support_fraction':fraction,
                    'source_observation_id':g['source_observation_id'],'source_index_interval':g['source_index_interval']})
            if not owners:profile=normal_profile(ink,g,audit)
        status='head_owned_observation' if len(owners)==1 else ('ambiguous_head_overlap' if owners else
            ('ambiguous_taper_observation' if profile and profile.get('tapering_nonuniform_stroke') else 'independent_candidate'))
        decisions.append({'primitive_id':identity,'kind':g['kind'],'status':status,'owners':owners,'profile':profile,
            'protected_by_known_role':protected,'source_preserved':True,'physical_geometry_approved':False})
    complete=not audit['limits']
    owned={d['primitive_id'] for d in decisions if d['status'] in ('head_owned_observation','ambiguous_head_overlap')}
    ambiguous={d['primitive_id'] for d in decisions if d['status']=='ambiguous_taper_observation'}
    eligible={'unfiltered':list(typed),'head_bound':[i for i in typed if i not in owned],
        'head_bound_and_taper_review':[i for i in typed if i not in owned|ambiguous]}
    if not complete:eligible={arm:list(typed) for arm in ARMS}
    if before!=object_sha([geometry,packet['source'],packet['graph'],heads,packet['chains']]) or pixels!=sha(gray) or roles!=sha(packet['roles']):
        raise ValueError('Ownership changed its bound observations')
    revision=object_sha({'packet_revision':packet['revision'],'pixels':pixels,'geometry':object_sha(geometry),
        'graph':object_sha(packet['graph']),'heads':object_sha(heads),'roles':roles,'policy':POLICY})
    return {'schema':'primitive-ownership/1.0','revision':revision,'packet_revision':packet['revision'],
        'input_sha256':pixels,'input_to_native':packet['input_to_native'],'policy':POLICY,
        'eligible_primitive_ids':eligible,'decisions':decisions,'head_checks':head_checks,'audit':audit,'complete':complete,
        'source_geometry_graph_roles_heads_chains_immutable':True,'CAD_export_approved':False,
        'production_changed':False,'physical_geometry_approved':False}
