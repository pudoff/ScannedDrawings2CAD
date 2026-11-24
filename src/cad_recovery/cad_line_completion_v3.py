"""Source-supported LINE continuation; arrow shape never implies an axis."""
from collections import Counter,defaultdict
from copy import deepcopy
import math
import cv2
import numpy as np
from .cad_object_assembly_v2 import assemble as primitives
from .cad_primitive_fit_v2 import Fitter
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha

POLICY={'version':'cad-line-completion/3.0','junction_region_px':2,
    'maximum_pair_checks':131072,'maximum_passes':12,'maximum_degree':32,
    'direction_deg':8,'maximum_pen_ratio':1.67,
    'white_gap_completion':False,'axis_from_arrow':False}


def _pairs(objects):
    endings=defaultdict(list);connectors=[]
    for i,o in enumerate(objects):
        p=np.asarray(o['support_points'],int)
        if o['kind']=='line':
            for end in (0,-1):endings[tuple(p[end])].append((i,end))
        elif o['kind']=='polyline' and len(p)>1 and np.max(np.ptp(p,axis=0))<=POLICY['junction_region_px']:
            connectors.append((tuple(p[0]),tuple(p[-1])))
    pairs=set()
    def add(left,right):
        if len(left)>POLICY['maximum_degree'] or len(right)>POLICY['maximum_degree']:return
        for a in left:
            for b in right:
                if a[0]!=b[0]:pairs.add(tuple(sorted((a,b))))
    for node,incident in endings.items():
        for dx in (-1,0,1):
            for dy in (-1,0,1):add(incident,endings.get((node[0]+dx,node[1]+dy),[]))
        if len(pairs)>POLICY['maximum_pair_checks']:return None
    for a,b in connectors:
        add(endings.get(a,[]),endings.get(b,[]))
        if len(pairs)>POLICY['maximum_pair_checks']:return None
    return sorted(pairs)


def _connector(objects,i,j,p,q,ink,roles):
    if np.array_equal(p,q):return [p.tolist()],[]
    low,high=np.minimum(p,q),np.maximum(p,q)
    if np.max(high-low)>POLICY['junction_region_px']:return None,[]
    if not np.all(ink[low[1]:high[1]+1,low[0]:high[0]+1]):return None,[]
    internal=[];graph=defaultdict(set)
    for k,o in enumerate(objects):
        if k in (i,j) or o['kind']!='polyline':continue
        points=np.asarray(o['support_points'],int)
        if not ((points>=low)&(points<=high)).all():continue
        known=set(map(int,roles[points[:,1],points[:,0]]))&set(range(1,10))
        if len(known)>1:return None,[]
        internal.append(k)
        for a,b in zip(points,points[1:]):
            graph[tuple(a)].add(tuple(b));graph[tuple(b)].add(tuple(a))
    start,goal=tuple(p),tuple(q);routes=[(start,[start])];seen={start}
    for node,walk in routes:
        if node==goal:return [list(map(int,x)) for x in walk],internal
        for neighbor in sorted(graph[node]):
            if neighbor not in seen:seen.add(neighbor);routes.append((neighbor,walk+[neighbor]))
    if np.max(abs(p-q))<=1:return [p.tolist(),q.tolist()],internal
    return None,[]


def complete(objects,gray,roles):
    objects=deepcopy(objects);_,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    dt=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    fitter=Fitter(cv2.dilate(ink,np.ones((3,3),np.uint8)));checks=0;merges=0;complete=True
    for iteration in range(POLICY['maximum_passes']):
        pairs=_pairs(objects)
        if pairs is None:complete=False;break
        options=defaultdict(list);fits={}
        for (i,ei),(j,ej) in pairs:
            checks+=1
            if checks>POLICY['maximum_pair_checks']:complete=False;break
            a,b=objects[i],objects[j]
            p=np.asarray(a['support_points'],int);q=np.asarray(b['support_points'],int)
            p=p if ei==-1 else p[::-1];q=q if ej==0 else q[::-1]
            u=p[0].astype(float)-p[-1];v=q[-1].astype(float)-q[0]
            if not np.linalg.norm(u)*np.linalg.norm(v):continue
            if float(u@v)/(np.linalg.norm(u)*np.linalg.norm(v))>-math.cos(math.radians(POLICY['direction_deg'])):continue
            pen=lambda x:max(1.,float(np.median(dt[x[:,1],x[:,0]])*2-1))
            if max(pen(p),pen(q))/min(pen(p),pen(q))>POLICY['maximum_pen_ratio']:continue
            connector,internal=_connector(objects,i,j,p[-1],q[0],ink,roles)
            if connector is None:continue
            claimed=[a,b]+[objects[k] for k in internal]
            known=set()
            for o in claimed:
                points=np.asarray(o['support_points'],int)
                known|=set(map(int,roles[points[:,1],points[:,0]]))&set(range(1,10))
            if len(known)>1:continue
            points=np.r_[p,np.asarray(connector,int)[1:-1],q] if len(connector)>1 else np.r_[p,q[1:]]
            model=fitter.fit(points)
            if not model or model['kind']!='line':continue
            members=sorted([m for o in claimed for m in o['source_members']],key=lambda m:(m['candidate_id'],m['source_index_interval']))
            links=[link for o in claimed for link in o.get('source_junction_links',[])]+[{
                'points_px':connector,'evidence':'observed_collinear_junction_continuation',
                'source_path_ids':list(dict.fromkeys(path for o in claimed for path in o['path_ids']))}]
            identity='cad-line/'+object_sha({'source':sha(gray),
                'members':members,'geometry':model})[:24]
            obj={**a,**model,'id':identity,'source_members':members,'source_junction_links':links,
                'path_ids':list(dict.fromkeys(m['path_id'] for m in members)),
                'support_points':points.tolist(),'role':next(iter(known)) if known else 255,
                'line_continuation_evidence':'unique_collinear_source_support_same_pen_and_role'}
            options[i,ei].append((j,ej));options[j,ej].append((i,ei));fits[i,ei,j,ej]=(obj,internal)
        if not complete:break
        used=set();new=[]
        for (i,ei,j,ej),(obj,internal) in fits.items():
            if options[i,ei]==[(j,ej)] and options[j,ej]==[(i,ei)] and not ({i,j}|set(internal))&used:
                used|={i,j}|set(internal);new.append(obj);merges+=1
        if not used:break
        objects=[o for i,o in enumerate(objects) if i not in used]+new
    else:complete=False
    return objects,{'policy':deepcopy(POLICY),'merges':merges,'pair_checks':checks,
                    'search_complete':complete and not fitter.incomplete}


def assemble(gray,geometry,*,roles=None,protected=None):
    roles=np.full_like(gray,255) if roles is None else roles
    result=primitives(gray,geometry,roles=roles,protected=protected)
    before=Counter((m['candidate_id'],n) for o in result['objects'] for m in o['source_members'] for n in range(*m['source_index_interval']))
    result['objects'],evidence=complete(result['objects'],gray,roles)
    after=Counter((m['candidate_id'],n) for o in result['objects'] for m in o['source_members'] for n in range(*m['source_index_interval']))
    if before!=after or any(n!=1 for n in after.values()):raise ValueError('LINE completion changed source edge partition')
    result.update(object_count=len(result['objects']),kind_counts=dict(Counter(o['kind'] for o in result['objects'])),
        line_completion=evidence,search_complete=result['search_complete'] and evidence['search_complete'])
    return result
