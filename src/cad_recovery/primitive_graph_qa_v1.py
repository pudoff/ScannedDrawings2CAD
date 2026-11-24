"""Bounded geometric incidence QA; intersections are not semantic part junctions.

No pixel rounding, endpoint snapping or coordinate movement is performed.
Numerical equality is 1e-7 native pixels. Unresolved/budgeted work fails closed.
"""
from collections import Counter,defaultdict
import math

import cv2
import numpy as np

from .cad_validation_topology_v1 import loop_data

EPS=1e-7
POLICY={'version':'primitive-graph-qa/1.0','numerical_epsilon_native_px':EPS,
    'maximum_objects':4096,'maximum_segments':120000,'maximum_vertices':120000,
    'maximum_pair_checks':250000,'maximum_grid_entries':500000,'maximum_faces':2048,
    'maximum_nesting_checks':100000,'grid_cell_native_px':16,'face_chord_error_native_px':.001,
    'maximum_face_vertices':512,'semantics':'geometric_incidence_only_no_part_approval'}


class Incomplete(ValueError):pass


def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])
def key(point):return tuple(float(v) for v in point)


def point(segment,t):
    if segment['kind']=='line':return segment['a']+t*(segment['b']-segment['a'])
    angle=segment['start']+segment['sweep']*t
    return segment['center']+segment['radius']*np.array([math.cos(angle),math.sin(angle)])


def parameter(segment,p):
    if segment['kind']=='line':
        delta=segment['b']-segment['a'];denom=float(delta@delta)
        t=float((p-segment['a'])@delta/denom)
        if -EPS<=t<=1+EPS and np.linalg.norm(point(segment,t)-p)<=EPS:return min(1.,max(0.,t))
        return None
    if abs(float(np.linalg.norm(p-segment['center']))-segment['radius'])>EPS:return None
    angle=math.atan2(*(p-segment['center'])[::-1]);sweep=segment['sweep']
    delta=(angle-segment['start'])%(2*math.pi)
    if delta>=2*math.pi-EPS/max(1.,segment['radius']):delta=0.
    t=delta/sweep
    return min(1.,max(0.,t)) if t<=1+EPS/max(1.,segment['radius']) else None


def segments(candidates):
    if len(candidates)>POLICY['maximum_objects']:raise Incomplete('object_budget')
    result=[];isolated=[];definition_vertices=0
    if len({c['id'] for c in candidates})!=len(candidates):raise ValueError('Duplicate primitive IDs')
    for c in candidates:
        owner=c.get('path_id',c['id']);kind=c['kind']
        if kind in ('line','polyline'):
            p=np.asarray(c['points_px'],float)
            if p.ndim!=2 or p.shape[1]!=2 or not len(p) or not np.isfinite(p).all():raise ValueError('Invalid points')
            if kind=='line' and len(p)!=2:raise ValueError('LINE requires two endpoints')
            definition_vertices+=len(p)
            if c.get('closed',False) and np.linalg.norm(p[0]-p[-1])>EPS:p=np.r_[p,p[:1]]
            added=False
            for a,b in zip(p,p[1:]):
                if np.linalg.norm(a-b)<=EPS:continue
                result.append({'kind':'line','a':a,'b':b,'owner':owner,'primitive':c['id'],'cuts':[0.,1.]});added=True
            if not added:isolated.append((p[0],owner))
        elif kind in ('arc','circle'):
            center=np.asarray(c['center_px'],float);radius=float(c['radius_px'])
            start=math.radians(c.get('start_angle_deg',0))%(2*math.pi)
            sweep=math.radians(c.get('sweep_angle_deg',360))
            if center.shape!=(2,) or not np.isfinite(center).all() or not math.isfinite(radius+sweep+start) or radius<=EPS or not 0<sweep<=2*math.pi+EPS:raise ValueError('Invalid curve')
            if kind=='arc' and sweep>=2*math.pi-EPS:raise ValueError('A full circle requires CIRCLE')
            definition_vertices+=1 if kind=='circle' else 3
            result.append({'kind':'arc','center':center,'radius':radius,'start':start,'sweep':sweep,
                'owner':owner,'primitive':c['id'],'cuts':[0.,.25,.5,.75,1.] if kind=='circle' else [0.,1.]})
        else:raise ValueError('Unsupported geometric primitive')
        if len(result)>POLICY['maximum_segments']:raise Incomplete('segment_budget')
    for s in result:
        points=[point(s,t) for t in (0.,1.)]
        if s['kind']=='arc':
            for angle in (0,math.pi/2,math.pi,3*math.pi/2):
                p=s['center']+s['radius']*np.array([math.cos(angle),math.sin(angle)])
                if parameter(s,p) is not None:points.append(p)
        p=np.asarray(points)
        if abs(p).max()>1e7:raise Incomplete('coordinate_budget')
        s['box']=[*p.min(axis=0),*p.max(axis=0)]
    return result,isolated,definition_vertices


def intersections(a,b):
    """Finite analytic intersections, including coincident overlap endpoints."""
    if a['kind']=='line' and b['kind']=='line':
        u=a['b']-a['a'];v=b['b']-b['a'];det=cross(u,v)
        if abs(det)>EPS:
            t=cross(b['a']-a['a'],v)/det;p=point(a,t)
            return [p] if parameter(a,p) is not None and parameter(b,p) is not None else []
        if abs(cross(b['a']-a['a'],u))>EPS*max(1.,np.linalg.norm(u)):return []
        return [p for p in (a['a'],a['b'],b['a'],b['b']) if parameter(a,p) is not None and parameter(b,p) is not None]
    if a['kind']=='line' or b['kind']=='line':
        line,arc=(a,b) if a['kind']=='line' else (b,a)
        u=line['b']-line['a'];d=line['a']-arc['center'];aa=float(u@u);bb=2*float(d@u)
        cc=float(d@d)-arc['radius']**2;discriminant=bb*bb-4*aa*cc
        error=1e-12*max(1.,bb*bb,abs(4*aa*cc))
        if discriminant < -error:return []
        root=math.sqrt(max(0.,discriminant));points=[point(line,(-bb-root)/(2*aa)),point(line,(-bb+root)/(2*aa))]
    else:
        d=b['center']-a['center'];length=float(np.linalg.norm(d));r=a['radius'];q=b['radius']
        if length<=EPS:
            if abs(r-q)>EPS:return []
            # Split at all sector endpoints and circle seams; duplicate edges are coalesced later.
            points=[point(s,t) for s in (a,b) for t in s['cuts']]
        elif length>r+q+EPS or length<abs(r-q)-EPS:return []
        else:
            along=(r*r-q*q+length*length)/(2*length);h2=r*r-along*along
            if h2 < -EPS*max(1.,r):return []
            base=a['center']+along*d/length;normal=np.array([-d[1],d[0]])/length
            points=[base+math.sqrt(max(0.,h2))*normal,base-math.sqrt(max(0.,h2))*normal]
    return [p for p in points if parameter(a,p) is not None and parameter(b,p) is not None]


def split_intersections(parts):
    bins=defaultdict(list);checks=entries=0
    for i,s in enumerate(parts):
        lo=np.floor((np.asarray(s['box'][:2])-EPS)/16).astype(int);hi=np.floor((np.asarray(s['box'][2:])+EPS)/16).astype(int)
        cell_count=int(np.prod(hi-lo+1));entries+=cell_count
        if entries>POLICY['maximum_grid_entries']:raise Incomplete('spatial_index_budget')
        nearby=set()
        for x in range(lo[0],hi[0]+1):
            for y in range(lo[1],hi[1]+1):nearby.update(bins[(x,y)])
        for j in sorted(nearby):
            other=parts[j];a=s['box'];b=other['box']
            if a[0]>b[2]+EPS or b[0]>a[2]+EPS or a[1]>b[3]+EPS or b[1]>a[3]+EPS:continue
            checks+=1
            if checks>POLICY['maximum_pair_checks']:raise Incomplete('intersection_budget')
            for p in intersections(s,other):
                s['cuts'].append(parameter(s,p));other['cuts'].append(parameter(other,p))
        for x in range(lo[0],hi[0]+1):
            for y in range(lo[1],hi[1]+1):bins[(x,y)].append(i)
    return checks,entries


def assemble(parts,isolated):
    nodes=[];bins=defaultdict(list);edges=[];edge_index={};duplicates=0
    def vertex(p):
        cell=tuple(np.floor(p/EPS).astype(np.int64))
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for j in bins[(cell[0]+dx,cell[1]+dy)]:
                    if np.linalg.norm(nodes[j]-p)<=EPS:return j
        if len(nodes)>=POLICY['maximum_vertices']:raise Incomplete('vertex_budget')
        index=len(nodes);nodes.append(np.asarray(p));bins[cell].append(index);return index
    isolated_owners=defaultdict(set)
    for p,owner in isolated:isolated_owners[vertex(p)].add(owner)
    for s in parts:
        cuts=sorted(set(s['cuts']));selected=[cuts[0]]
        for t in cuts[1:]:
            if np.linalg.norm(point(s,t)-point(s,selected[-1]))>EPS:selected.append(t)
        for low,high in zip(selected,selected[1:]):
            a=vertex(point(s,low));b=vertex(point(s,high))
            if a==b:continue
            middle=point(s,(low+high)/2)
            # Curves with the same endpoints but different sectors remain distinct.
            fingerprint=(s['kind'],min(a,b),max(a,b),*(np.round(middle,7) if s['kind']=='arc' else []))
            if fingerprint in edge_index:
                e=edges[edge_index[fingerprint]];e['owners'].add(s['owner']);e['primitives'].add(s['primitive']);duplicates+=1;continue
            edge_index[fingerprint]=len(edges)
            edges.append({'a':a,'b':b,'segment':s,'low':low,'high':high,'owners':{s['owner']},'primitives':{s['primitive']}})
    return nodes,edges,isolated_owners,duplicates


def adjacency(nodes,edges):
    adj=[[] for _ in nodes]
    for i,e in enumerate(edges):adj[e['a']].append((i,1));adj[e['b']].append((i,-1))
    return adj


def faces(nodes,edges,adj):
    def tangent(half):
        i,sign=half;e=edges[i];s=e['segment'];t=e['low'] if sign==1 else e['high']
        if s['kind']=='line':d=sign*(s['b']-s['a'])
        else:
            a=s['start']+t*s['sweep'];d=sign*np.array([-math.sin(a),math.cos(a)])
        return math.atan2(d[1],d[0])
    ordered=[sorted(a,key=tangent) for a in adj];positions=[{half:i for i,half in enumerate(a)} for a in ordered]
    visited=set();result=[];unresolved=[]
    for i in range(len(edges)):
        for sign in (1,-1):
            first=(i,sign)
            if first in visited:continue
            walk=[];current=first
            while current not in visited:
                visited.add(current);walk.append(current);j,direction=current;e=edges[j]
                end=e['b'] if direction==1 else e['a'];order=ordered[end]
                current=order[(positions[end][(j,-direction)]-1)%len(order)]
            if current!=first:raise Incomplete('face_walk_inconsistent')
            # Remove dangling excursions without removing the graph edges themselves.
            reduced=[]
            for half in walk:
                if reduced and reduced[-1]==(half[0],-half[1]):reduced.pop()
                else:reduced.append(half)
            while len(reduced)>1 and reduced[0]==(reduced[-1][0],-reduced[-1][1]):reduced=reduced[1:-1]
            if not reduced:continue
            parts=[];vertices=[];owners=set()
            for j,direction in reduced:
                e=edges[j];s=e['segment'];low,high=(e['low'],e['high']) if direction==1 else (e['high'],e['low'])
                a,b=point(s,low),point(s,high);bulge=math.tan(s['sweep']*(high-low)/4) if s['kind']=='arc' else 0.
                parts.append((tuple(a),tuple(b),bulge));vertices.append(e['a'] if direction==1 else e['b']);owners.update(e['owners'])
            # Existing independent CAD QA supplies exact circular-sector area and a reported chord envelope.
            try:loop=loop_data(parts,EPS)
            except ValueError as exc:
                unresolved.append(str(exc));continue
            if loop['signed_area']<=EPS:continue
            if len(result)>=POLICY['maximum_faces']:raise Incomplete('face_budget')
            result.append({'id':f'face-{len(result):05d}','area_px2':loop['signed_area'],'owners':sorted(owners),
                'vertices':vertices,'flat':loop['flat'],'chord_error_bound_px':loop['chord_error_bound'],
                'simple':len(set(vertices))==len(vertices),'parent':None,'depth':None})
    return result,unresolved


def nesting(loops):
    checks=0;uncertain=[]
    for child in loops:
        if not child['simple']:uncertain.append(child['id']);continue
        point0=child['flat'][0];parents=[]
        for outer in loops:
            if outer is child or outer['area_px2']<=child['area_px2']:continue
            checks+=1
            if checks>POLICY['maximum_nesting_checks']:raise Incomplete('nesting_budget')
            polygon=np.asarray(outer['flat'],np.float32)
            clearance=float(cv2.pointPolygonTest(polygon,tuple(point0),True))
            envelope=child['chord_error_bound_px']+outer['chord_error_bound_px']+1e-5
            if abs(clearance)<=envelope:uncertain.extend([outer['id'],child['id']]);continue
            if clearance>envelope:
                # All intersections were split earlier. Shared graph vertices mean touching, not strict nesting.
                if set(child['vertices'])&set(outer['vertices']):uncertain.extend([outer['id'],child['id']]);continue
                parents.append(outer)
        if parents:child['parent']=min(parents,key=lambda f:f['area_px2'])['id']
    lookup={f['id']:f for f in loops}
    for loop in loops:
        depth=0;parent=loop['parent'];seen=set()
        while parent:
            if parent in seen:raise Incomplete('cyclic_nesting')
            seen.add(parent);depth+=1;parent=lookup[parent]['parent']
        loop['depth']=depth
    return checks,sorted(set(uncertain))


def build(candidates):
    """Return explicit incomplete status instead of treating truncated work as a valid graph."""
    try:
        parts,isolated,definition_vertices=segments(candidates)
        checks,grid_entries=split_intersections(parts)
        nodes,edges,isolated_owners,duplicates=assemble(parts,isolated);adj=adjacency(nodes,edges)
        component=[-1]*len(nodes);components=0
        for start in range(len(nodes)):
            if component[start]>=0:continue
            stack=[start];component[start]=components
            while stack:
                a=stack.pop()
                for j,_ in adj[a]:
                    e=edges[j];b=e['b'] if e['a']==a else e['a']
                    if component[b]<0:component[b]=components;stack.append(b)
            components+=1
        loops,unresolved=faces(nodes,edges,adj);nest_checks,uncertain=nesting(loops)
        structural=[]
        for i,p in enumerate(nodes):
            degree=len(adj[i])
            if degree==2:continue
            owners=set(isolated_owners[i])
            for j,_ in adj[i]:owners.update(edges[j]['owners'])
            structural.append({'id':f'node-{i:06d}','point_px':p.tolist(),'degree':degree,'owners':sorted(owners),'component':component[i]})
        public_faces=[{k:v for k,v in f.items() if k not in ('flat','vertices')} for f in loops]
        component_owners=[set() for _ in range(components)]
        for e in edges:component_owners[component[e['a']]].update(e['owners'])
        for i,owners in isolated_owners.items():component_owners[component[i]].update(owners)
        return {'complete':True,'NoP':len(candidates),'NoV_definition':definition_vertices,
            'graph_vertices':len(nodes),'graph_edges':len(edges),'components':components,
            'cycle_rank':len(edges)-len(nodes)+components,'degree_histogram':dict(sorted(Counter(map(len,adj)).items())),
            'structural_nodes':structural,'faces':public_faces,'bounded_faces':len(loops),
            'component_owners':sorted(sorted(owners) for owners in component_owners),
            'nesting_complete':not unresolved and not uncertain,'unresolved_face_reasons':dict(Counter(unresolved)),
            'uncertain_nesting_ids':uncertain,'duplicate_geometric_pieces':duplicates,
            'pair_checks':checks,'grid_entries':grid_entries,'nesting_checks':nest_checks,
            'geometric_crossings_are_semantic_junctions':False,'endpoint_snapping_performed':False,
            'physical_geometry_approved':False}
    except Incomplete as exc:
        return {'complete':False,'reason':str(exc),'physical_geometry_approved':False}
