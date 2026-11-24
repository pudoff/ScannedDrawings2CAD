"""Spatially indexed variant of the frozen V5 primitive continuation loop.

Each pass builds indices over original integer source coordinates. Matching and
source-ledger rules are kept in the original order for experimental replay.
"""
from collections import defaultdict

import numpy as np

from .cad_object_assembly_v5 import POLICY, _junction_context
from .cad_primitive_fit_v2 import checked, sector


def _indexed_junction_context(items, owners, i, j, p, q, ink):
    nearby = {i, j}
    for x, y in (p, q):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nearby.update(owners.get((x + dx, y + dy), ()))
    indices = sorted(nearby)
    safe, fragments = _junction_context(
        [items[k] for k in indices], indices.index(i), indices.index(j), p, q, ink)
    return safe, [indices[k] for k in fragments]


def merge_curves(items,fitter,ink):
    for _ in range(12):
        endings=defaultdict(list)
        connectors=defaultdict(set)
        point_owners=defaultdict(set)
        for i,item in enumerate(items):
            points = item['points']
            for point in points:
                point_owners[tuple(point)].add(i)
            if len(points) < 2:
                continue
            first, last = (tuple(point) for point in points[[0, -1]])
            endings[first].append((i, 0))
            endings[last].append((i, -1))
            if np.max(abs(points[-1] - points[0])) <= POLICY['fit_max_px']:
                connectors[first].add(last)
                connectors[last].add(first)
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
            for adjacent in connectors[node]:
                for i, ei in incident:
                    for j, ej in endings[adjacent]:
                        pairs.add((i, ei, j, ej))
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
                safe,internal=_indexed_junction_context(items,point_owners,i,j,p[-1],q[0],ink)
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
                    elif np.max(abs(p[-1]-q[0]))>1:continue
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
            if not same:links=links+[{'points_px':connector.astype(int).tolist(),
                'evidence':'owned_junction_walk_or_adjacent_foreground_same_circle',
                'source_path_ids':list(dict.fromkeys(a['path_ids']+b['path_ids']))}]
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
