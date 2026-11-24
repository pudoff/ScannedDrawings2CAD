"""Bounded geometric boundary hypotheses, never semantic or physical approval."""
import math
import numpy as np

from .primitive_graph_qa_v1 import segments,split_intersections,assemble,adjacency,Incomplete
from .cad_validation_topology_v1 import loop_data

POLICY={'maximum_boundaries':512,'maximum_boundary_edges':256,'area_minimum_in_strokes2':64,
    'hypothesis_only':True,'physical_geometry_approved':False}


def boundaries(geometry,shape,stroke):
    try:
        parts,isolated,_=segments(geometry);checks,_=split_intersections(parts)
        nodes,edges,_,_=assemble(parts,isolated);adj=adjacency(nodes,edges)
        def tangent(half):
            index,sign=half;e=edges[index];s=e['segment'];t=e['low'] if sign==1 else e['high']
            if s['kind']=='line':d=sign*(s['b']-s['a'])
            else:
                a=s['start']+s['sweep']*t;d=sign*np.array([-math.sin(a),math.cos(a)])
            return math.atan2(d[1],d[0])
        ordered=[sorted(a,key=tangent) for a in adj];positions=[{h:i for i,h in enumerate(a)} for a in ordered]
        seen=set();loops=[];rejected=[];fingerprints=set()
        from .primitive_graph_qa_v1 import point
        for index in range(len(edges)):
            for sign in (1,-1):
                first=(index,sign)
                if first in seen:continue
                current=first;walk=[]
                while current not in seen:
                    seen.add(current);walk.append(current);j,d=current;e=edges[j]
                    end=e['b'] if d==1 else e['a'];order=ordered[end]
                    current=order[(positions[end][(j,-d)]-1)%len(order)]
                if current!=first:raise Incomplete('inconsistent_boundary_walk')
                reduced=[]
                for h in walk:
                    if reduced and reduced[-1]==(h[0],-h[1]):reduced.pop()
                    else:reduced.append(h)
                while len(reduced)>1 and reduced[0]==(reduced[-1][0],-reduced[-1][1]):reduced=reduced[1:-1]
                if not reduced:continue
                if len(reduced)>256:rejected.append('boundary_edge_budget');continue
                pieces=[];owners=set();vertices=[]
                for j,d in reduced:
                    e=edges[j];s=e['segment'];lo,hi=(e['low'],e['high']) if d==1 else (e['high'],e['low'])
                    a,b=point(s,lo),point(s,hi)
                    bulge=math.tan(s['sweep']*(hi-lo)/4) if s['kind']=='arc' else 0.
                    pieces.append((tuple(a),tuple(b),bulge));owners.update(e['owners']);vertices.append(e['a'] if d==1 else e['b'])
                if len(set(vertices))!=len(vertices):rejected.append('non_simple_boundary_walk');continue
                try:data=loop_data(pieces,1e-7)
                except ValueError as error:rejected.append(str(error));continue
                area=abs(data['signed_area'])
                if area<64*stroke*stroke:continue
                flat=np.asarray(data['flat'],float);lo=flat.min(axis=0);hi=flat.max(axis=0)
                if min(hi-lo)<6*stroke:continue
                signature=(tuple(sorted(owners)),round(area,4))
                if signature in fingerprints:continue
                fingerprints.add(signature)
                near_edge=min(lo[0],lo[1],shape[1]-1-hi[0],shape[0]-1-hi[1])
                frame=bool((hi[0]-lo[0])>=.7*shape[1] and (hi[1]-lo[1])>=.7*shape[0] and near_edge<=3*stroke)
                loops.append({'id':f'boundary-{len(loops):04d}','owners':sorted(owners),'polygon':flat.tolist(),
                    'area_px2':area,'box':[*lo.tolist(),*hi.tolist()],'frame_hypothesis':frame,
                    'chord_error_px':data['chord_error_bound'],'orientation':'bounded_face' if data['signed_area']>0 else 'component_outline',
                    'semantic_ownership_confirmed':False})
                if len(loops)>=512:raise Incomplete('boundary_count_budget')
        return {'boundaries':loops,'rejected':rejected,'pair_checks':checks,'search_incomplete':False}
    except Incomplete as error:
        return {'boundaries':[],'rejected':[str(error)],'search_incomplete':True}
