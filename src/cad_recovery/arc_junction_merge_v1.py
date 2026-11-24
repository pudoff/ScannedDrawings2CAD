"""Continue the same ARC through a source-connected small raster junction."""
from collections import defaultdict
import math
import numpy as np
from .cad_object_assembly_v5 import POLICY
from .cad_primitive_fit_v2 import checked, sector

def context(items,i,j,p,q,ink):
    low=np.minimum(p,q);high=np.maximum(p,q);fragments=[];directions=[];annotation=True
    for k,item in enumerate(items):
        if k in (i,j):continue
        points=item['points'];inside=((points>=low)&(points<=high)).all(1)
        near=(np.max(abs(points-p),axis=1)<=1)|(np.max(abs(points-q),axis=1)<=1)
        if not near.any():continue
        if inside.all():fragments.append(k);continue
        annotation &= bool(item['roles']&{2,3,4,5})
        contact=np.flatnonzero(near)
        for samples in (points[:contact[0]+1][::-1],points[contact[-1]:]):
            delta=samples-(p+q)/2;distance=np.linalg.norm(delta,axis=1)
            if not len(distance) or distance.max()<2:continue
            model=item.get('model')
            if model and model['kind']=='line':
                ends=np.asarray(model['points_px'],float);axis=ends[1]-ends[0]
                vector=axis*np.sign(float(delta[int(np.argmax(distance))]@axis))
            else:
                index=np.flatnonzero(distance>=8)
                vector=delta[int(index[0])] if len(index) else delta[int(np.argmax(distance))]
            if np.linalg.norm(vector)>0:directions.append(vector/np.linalg.norm(vector))
    # White corners of the bounding rectangle are not a broken stroke.
    # Continuation still needs an owned or actual foreground walk below.
    if not directions or annotation:return True,fragments
    if len(directions)==2 and float(directions[0]@directions[1])<=-math.cos(math.radians(8)):
        return True,fragments
    return False,[]

def raw_route(ink,blocked,p,q,audit,forbidden=()):
    if audit['routes_checked']>=8192:
        audit['search_complete']=False;return None
    audit['routes_checked']+=1
    low=np.minimum(p,q);high=np.maximum(p,q)
    if np.max(high-low)>POLICY['fit_max_px']:return None
    start,goal=tuple(p),tuple(q);queue=[(start,[start])];seen={start}
    for node,walk in queue:
        if node==goal:
            audit['foreground_routes_found']+=1;return walk
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
            v=(node[0]+dx,node[1]+dy)
            if (v not in seen and v not in forbidden and low[0]<=v[0]<=high[0] and low[1]<=v[1]<=high[1]
                    and ink[v[1],v[0]] and not blocked[v[1],v[0]]):
                seen.add(v);queue.append((v,walk+[v]))
    return None

def merge(items,fitter,ink,blocked,audit):
    for _ in range(12):
        endings=defaultdict(list)
        for i,item in enumerate(items):
            if len(item['points'])<2:continue
            endings[tuple(item['points'][0])].append((i,0));endings[tuple(item['points'][-1])].append((i,-1))
        options=defaultdict(list);fits={}
        pairs=set()
        for node,incident in endings.items():
            if len(incident)>POLICY['maximum_junction_degree']:continue
            neighbors=[(node,incident)]+[(other,endings[other]) for dx in (-1,0,1) for dy in (-1,0,1)
                if (dx or dy) and (other:=(node[0]+dx,node[1]+dy)) in endings and other>node]
            for other,following in neighbors:
                for pos,(i,ei) in enumerate(incident):
                    for j,ej in (incident[pos+1:] if other==node else following):
                        pairs.add((i,ei,j,ej))
            # A source-supported junction connector can span several pixels;
            # it is not a tolerance-based bridge over an unobserved gap.
            for fragment in items:
                endpoints=fragment['points'][[0,-1]]
                if len(fragment['points'])<2 or np.max(abs(endpoints[1]-endpoints[0]))>POLICY['fit_max_px']:continue
                for a,b in (endpoints,endpoints[::-1]):
                    if tuple(a)!=node or tuple(b) not in endings:continue
                    for i,ei in incident:
                        for j,ej in endings[tuple(b)]:pairs.add((i,ei,j,ej))
        # Connector lookup visits both ends. Count each half-edge pair once;
        # otherwise a unique continuation appears spuriously ambiguous.
        pairs={(*a,*b) if a<b else (*b,*a) for i,ei,j,ej in pairs for a,b in [((i,ei),(j,ej))]}
        for i,ei,j,ej in sorted(pairs):
            a,b=items[i],items[j];ma,mb=a['model'],b['model']
            if i==j or not ma or not mb or ma['kind']!=mb['kind'] or ma['kind'] not in ('line','arc'):continue
            if len(a['roles']|b['roles'])>1:continue
            p=a['points'] if ei==-1 else a['points'][::-1];q=b['points'] if ej==0 else b['points'][::-1]
            same=np.array_equal(p[-1],q[0]);internal=[]
            if ma['kind']=='line' and not same:continue
            connector=np.asarray([p[-1],q[0]])
            if ma['kind']=='arc':
                safe,internal=context(items,i,j,p[-1],q[0],ink)
                if not safe:continue
                if not same:
                    graph=defaultdict(set)
                    for k in internal:
                        for u,v in zip(items[k]['points'],items[k]['points'][1:]):
                            graph[tuple(u)].add(tuple(v));graph[tuple(v)].add(tuple(u))
                    start,goal=tuple(p[-1]),tuple(q[0]);routes=[(start,[start])];seen={start};route=None
                    for node,walk in routes:
                        if node==goal:route=walk;break
                        for adjacent in sorted(graph[node]):
                            if adjacent not in seen:seen.add(adjacent);routes.append((adjacent,walk+[adjacent]))
                    if route is not None:connector=np.asarray(route)
                    else:
                        forbidden={tuple(point) for point in np.r_[p[:-1],q[1:]]}
                        route=raw_route(ink,blocked,p[-1],q[0],audit,forbidden)
                        if route is None:continue
                        connector=np.asarray(route)
                if ma.get('locally_stable',True) and mb.get('locally_stable',True):
                    if np.linalg.norm(np.asarray(ma['center_px'])-mb['center_px'])>2 or abs(ma['radius_px']-mb['radius_px'])>2:continue
            joined=np.r_[p,q[1:]] if same else np.r_[p,connector[1:-1],q]
            closed=np.array_equal(joined[0],joined[-1])
            if (not closed and ma['kind']=='arc' and ma.get('circle_family_support')
                    and ma.get('circle_family_support')==mb.get('circle_family_support')):
                # Independent sectors already determine the circle. A local
                # endpoint-constrained fit may fail at a raster corner; it must
                # not hide a source-validated model from the stronger evidence.
                fitted=checked(sector(joined,np.asarray(ma['center_px']),ma['radius_px']),joined,fitter.support)
                if fitted:fitted.update(locally_stable=True,circle_family_support=ma['circle_family_support'],
                    circle_family_source_sectors=ma.get('circle_family_source_sectors',[]),
                    parameter_sensitivity_native_px=ma.get('parameter_sensitivity_native_px'))
            else:fitted=fitter.fit(joined,closed)
            if not fitted or (ma['kind']=='line' and fitted['kind']!='line') or (ma['kind']=='arc' and fitted['kind'] not in ('arc','circle')):continue
            options[i,ei].append((j,ej));options[j,ej].append((i,ei))
            links=a.get('source_junction_links',[])+b.get('source_junction_links',[])
            if not same:links=links+[{'points_px':np.asarray([u,v]).astype(int).tolist(),
                'evidence':'owned_or_foreground_pixel_walk_same_circle',
                'source_path_ids':list(dict.fromkeys(a['path_ids']+b['path_ids']))}
                for u,v in zip(connector,connector[1:])]
            fragments=a.get('junction_fragments',[])+b.get('junction_fragments',[])+[
                {'points':items[k]['points'],'path_ids':items[k]['path_ids']} for k in internal]
            fits[i,ei,j,ej]=({'points':joined,'model':fitted,'roles':a['roles']|b['roles'],
                'path_ids':list(dict.fromkeys(a['path_ids']+b['path_ids']+[path for k in internal for path in items[k]['path_ids']])),
                'source_junction_links':links,'junction_fragments':fragments},internal)
        used=set();output=[]
        for (i,ei,j,ej),(item,internal) in fits.items():
            if options[i,ei]==[(j,ej)] and options[j,ej]==[(i,ei)] and not ({i,j}|set(internal))&used:
                output.append(item);used|={i,j}|set(internal)
        if not used:break
        items=[o for i,o in enumerate(items) if i not in used]+output
    return items
