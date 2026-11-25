"""Geometric primitives with exact source-edge partitions and separate curve sectors."""
from collections import Counter,defaultdict
from copy import deepcopy
import math
import cv2
import numpy as np
from . import cad_object_assembly_v1 as old
from .cad_primitive_fit_v4 import Fitter
from .cad_primitive_corners_v3 import refine as refine_corners
from .cad_primitive_fit_v2 import checked,sector
from .curve_graph_metrics_v1 import distance as geometry_distance
from .stroke_curve_control_v1 import distance_to_polyline
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha

POLICY={**old.POLICY,'version':'cad-object-assembly/4.0','maximum_objects':16384,
        'unit':'homogeneous_LINE_ARC_CIRCLE_or_explicit_unresolved_source',
        'junctions':'model_incidence_within_previous_2px_source_region'}


def _member_ledger(geometry):
    edges=defaultdict(list);points=defaultdict(list);expected=set()
    for c in geometry:
        p=old._points(c.get('support_points',c.get('points_px')));lo,hi=c['source_index_interval']
        for n,(a,b) in enumerate(zip(p,p[1:]),lo):
            key=tuple(sorted((tuple(a),tuple(b))))
            edges[key].append((c,n));expected.add((c['id'],n))
        if len(p)==1:points[tuple(p[0])].append(c)
    return edges,points,expected,{c['id']:c for c in geometry}


def _members(item,ledger):
    edges,points,_,candidate_lookup=ledger;p=item['points'];allowed=set(item['path_ids']);spans=defaultdict(list);lookup={}
    if len(p)==1:
        candidates=[c for c in points[tuple(p[0])] if c['path_id'] in allowed]
        for c in candidates:spans[c['id']].append(c['source_index_interval'][0]);lookup[c['id']]=c
    for a,b in zip(p,p[1:]):
        options=[(c,n) for c,n in edges[tuple(sorted((tuple(a),tuple(b))))] if c['path_id'] in allowed]
        if not options and tuple(sorted((tuple(a),tuple(b)))) in {
                tuple(sorted(map(tuple,link['points_px']))) for link in item.get('source_junction_links',[])}:
            continue
        if len(options)!=1:raise ValueError('Ambiguous or absent source edge ownership')
        c,n=options[0];spans[c['id']].append(n);lookup[c['id']]=c
    # Internal edges of a filled raster junction remain traceable even when
    # their triangle/square skeleton is not a material loop.
    for fragment in item.get('junction_fragments',[]):
        for member in _members(fragment,ledger):
            c=candidate_lookup[member['candidate_id']]
            lookup[c['id']]=c
            for n in range(*member['source_index_interval']):
                if n not in spans[c['id']]:spans[c['id']].append(n)
    result=[]
    for identity,indices in sorted(spans.items()):
        c=lookup[identity];indices=sorted(indices)
        if len(indices)!=len(set(indices)):raise ValueError('Source edge reused within primitive')
        groups=[]
        for n in indices:
            if groups and n==groups[-1][-1]+1:groups[-1].append(n)
            else:groups.append([n])
        for group in groups:
            low,high=group[0],group[-1]+(len(p)>1)
            interval_ranges=c.get('graph_interval_ranges',[])
            intersect=[r for r in interval_ranges if max(low,r['source_index_interval'][0])<min(high,r['source_index_interval'][1])]
            result.append({'candidate_id':identity,'source_observation_id':c.get('source_observation_id',c['path_id']),
                'path_id':c['path_id'],'source_index_interval':[low,high],
                'interval_ids':[r['id'] for r in intersect] if interval_ranges else c.get('interval_ids',[]),
                'graph_interval_scope':'intersected_source_edges' if interval_ranges else 'parent_candidate_references',
                'graph_interval_ranges':[{'id':r['id'],'source_index_interval':[
                    max(low,r['source_index_interval'][0]),min(high,r['source_index_interval'][1])]} for r in intersect]})
    return result


def _junction_context(items,i,j,p,q,ink):
    low=np.minimum(p,q);high=np.maximum(p,q);fragments=[];directions=[];annotation=True
    for k,item in enumerate(items):
        if k in (i,j):continue
        points=item['points'];inside=((points>=low)&(points<=high)).all(1)
        near=(np.max(abs(points-p),axis=1)<=1)|(np.max(abs(points-q),axis=1)<=1)
        if not near.any():continue
        if inside.all():
            fragments.append(k);continue
        annotation &= bool(item['roles']&{2,3,4,5})
        contact=np.flatnonzero(near)
        for samples in (points[:contact[0]+1][::-1],points[contact[-1]:]):
            delta=samples-(p+q)/2;distance=np.linalg.norm(delta,axis=1)
            if not len(distance) or distance.max()<2:continue
            model=item.get('model')
            if model and model['kind']=='line':
                # A whole source-validated LINE is independent evidence of
                # direction. An 8px raster prefix may wobble by a whole pixel.
                ends=np.asarray(model['points_px'],float);axis=ends[1]-ends[0]
                vector=axis*np.sign(float(delta[int(np.argmax(distance))]@axis))
            else:
                index=np.flatnonzero(distance>=8)
                vector=delta[int(index[0])] if len(index) else delta[int(np.argmax(distance))]
            if np.linalg.norm(vector)>0:directions.append(vector/np.linalg.norm(vector))
    if fragments and not np.all(ink[low[1]:high[1]+1,low[0]:high[0]+1]):return False,[]
    if not directions or annotation:return True,fragments
    # An unknown transverse stroke must have corroborated continuation on both
    # sides. A single unknown T leg can instead be a material contour corner.
    if len(directions)==2 and float(directions[0]@directions[1])<=-math.cos(math.radians(8)):
        return True,fragments
    return False,[]


def _merge_curves(items,fitter,ink):
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
                safe,internal=_junction_context(items,i,j,p[-1],q[0],ink)
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


def _circle_line(circle,line,node):
    center=np.asarray(circle['center_px']);radius=circle['radius_px'];a,b=np.asarray(line['points_px'])
    u=b-a;aa=float(u@u);v=a-center
    if aa==0:return []
    bb=2*float(v@u);cc=float(v@v)-radius**2;disc=bb*bb-4*aa*cc
    if disc<0:return []
    return [a+t*u for t in ((-bb-math.sqrt(disc))/(2*aa),(-bb+math.sqrt(disc))/(2*aa))]


def _align_endpoints(items,support):
    endings=defaultdict(list);changes=[]
    for i,item in enumerate(items):
        if len(item['points'])>1 and not np.array_equal(item['points'][0],item['points'][-1]):
            endings[tuple(item['points'][0])].append((i,0));endings[tuple(item['points'][-1])].append((i,-1))
    for node,incident in endings.items():
        if len(incident)!=2:continue
        (i,ei),(j,ej)=incident;a,b=items[i],items[j];ma,mb=a['model'],b['model']
        if not ma or not mb or {ma['kind'],mb['kind']}!={'line','arc'}:continue
        if len(a['roles']|b['roles'])>1 or not set(a['path_ids'])&set(b['path_ids']):continue
        li,le,ci,ce=(i,ei,j,ej) if ma['kind']=='line' else (j,ej,i,ei)
        line,arc=items[li]['model'],items[ci]['model'];point=np.asarray(node,float)
        choices=_circle_line(arc,line,point)
        center=np.asarray(arc['center_px']);radial=point-center
        if np.linalg.norm(radial)>0:choices.append(center+radial/np.linalg.norm(radial)*arc['radius_px'])
        choices.sort(key=lambda q:float(np.linalg.norm(q-point)))
        accepted=False
        for q in choices:
            if np.linalg.norm(q-point)>2:continue
            new_line=deepcopy(line);ends=np.asarray(line['points_px']);ends[0 if le==0 else 1]=q
            new_line['points_px']=ends.tolist();error=distance_to_polyline(items[li]['points'],ends)
            if error.max()>2 or np.percentile(error,95)>1.25:continue
            new_line.update(fit_max_px=float(error.max()),fit_p95_px=float(np.percentile(error,95)))
            adjusted=items[ci]['points'].astype(float).copy();adjusted[ce]=q
            new_arc=sector(adjusted,center,arc['radius_px'])
            if new_arc is None:continue
            new_arc={**arc,**new_arc}
            if not checked(new_line,items[li]['points'],support) or not checked(new_arc,items[ci]['points'],support):continue
            items[li]['model']=new_line;items[ci]['model']=new_arc;accepted=True
            changes.append({'source_node':list(map(int,node)),'model_node':q.tolist(),'distance_px':float(np.linalg.norm(q-point)),
                            'items':[li,ci]});break
        if not accepted:
            # Exact residual preserves the common source node when no safe common
            # CAD endpoint can be established; never leave an unreported gap.
            items[li]['model']=None;items[ci]['model']=None
    return changes


def assemble(gray,geometry,*,roles=None,protected=None):
    if gray.dtype!=np.uint8 or gray.ndim!=2 or not 0<gray.size<=POLICY['maximum_pixels']:raise ValueError('Bounded gray required')
    if len(geometry)>POLICY['maximum_candidates']:raise ValueError('Candidate budget exceeded')
    roles=np.full_like(gray,255) if roles is None else roles;protected=np.zeros_like(gray,bool) if protected is None else protected
    if roles.shape!=gray.shape or roles.dtype!=np.uint8 or protected.shape!=gray.shape or protected.dtype!=bool:
        raise ValueError('Aligned roles and protection required')
    if not np.isin(roles,list(range(10))+[255]).all():raise ValueError('Invalid role')
    before=object_sha(geometry);snapshots=[sha(a) for a in (gray,roles,protected.astype(np.uint8))]
    excluded=protected|(roles==6);paths,ink=old._paths(gray,geometry,roles,excluded)
    # Keep original source paths separate until their primitive types are known.
    # The old degree-two join could absorb a construction arc at a contour corner.
    joined=paths;ledger=_member_ledger(geometry)
    support=cv2.dilate(ink,np.ones((3,3),np.uint8));fitter=Fitter(support);items=[];withheld=[]
    for path in joined:
        p=path['points']
        if len(p)==1:
            item={'points':p,'path_ids':path['path_ids'],'roles':set(map(int,roles[p[:,1],p[:,0]]))&set(range(1,10)),
                  'model':{'kind':'point','points_px':p.tolist(),'fit_max_px':0.,'fit_p95_px':0.}}
            if excluded[p[0,1],p[0,0]]:withheld.append(item)
            else:items.append(item)
            continue
        blocked=excluded[p[:,1],p[:,0]];bad=blocked[:-1]|blocked[1:]
        cuts=np.r_[0,np.flatnonzero(bad[1:]!=bad[:-1])+1,len(bad)]
        for lo,hi in zip(cuts,cuts[1:]):
            q=p[lo:hi+1]
            base={'path_ids':path['path_ids'],'roles':set(map(int,roles[q[:,1],q[:,0]]))&set(range(1,10))}
            if bad[lo]:withheld.append({**base,'points':q});continue
            parts=fitter.refine_curve_change(fitter.refine_tangent(fitter.merge(fitter.split(q,np.array_equal(q[0],q[-1])))))
            items.extend({**base,**part} for part in parts)
    family=fitter.families(items)
    for group in family['groups']:
        group['source_sectors']=[_members(items[i],ledger) for i in group['members']]
    for item in items:
        model=item['model']
        if model and model.get('circle_family_support'):
            group=next(g for g in family['groups'] if g['members']==model['circle_family_support'])
            model['circle_family_source_sectors']=group['source_sectors']
    items=refine_corners(items,fitter)
    items=_merge_curves(items,fitter,ink)
    adjustments=_align_endpoints(items,support)
    objects=[]
    for item in items:
        p=item['points'];typed=item['model']
        if typed and typed['kind']=='arc' and not typed.get('locally_stable',True):typed=None
        if typed is None:typed={'kind':'polyline','points_px':old.compact(p),'closed':np.array_equal(p[0],p[-1]),
            'fit_max_px':0.,'fit_p95_px':0.,'reason':'unsupported_or_unstable_primitive'}
        members=_members(item,ledger)
        identity='cad-primitive/'+object_sha({'source':sha(gray),'members':members,'geometry':typed})[:24]
        objects.append({**typed,'id':identity,'source_members':members,'path_ids':list(dict.fromkeys(m['path_id'] for m in members)),
            'source_junction_links':item.get('source_junction_links',[]),
            'support_points':p.tolist(),'role':next(iter(item['roles'])) if len(item['roles'])==1 else 255,
            'status':'draft_requires_review','semantic_ownership_confirmed':False,'physical_geometry_approved':False})
    if len(objects)>POLICY['maximum_objects']:raise ValueError('Output object budget exceeded')
    withheld=[{'source_members':_members(o,ledger),'reason':'protected_source_subpath'} for o in withheld]
    observed=Counter()
    for obj in objects+withheld:
        for m in obj['source_members']:
            lo,hi=m['source_index_interval']
            for n in range(lo,hi):observed[m['candidate_id'],n]+=1
    if set(observed)!=ledger[2] or any(n!=1 for n in observed.values()):raise ValueError('Source edge partition failed')
    if before!=object_sha(geometry) or snapshots!=[sha(a) for a in (gray,roles,protected.astype(np.uint8))]:raise ValueError('Input changed')
    return {'objects':objects,'withheld':withheld,'source_sha256':sha(gray),'policy':deepcopy(POLICY),
        'input_candidate_count':len(geometry),'input_source_path_count':len(paths),'joined_path_count':len(joined),
        'object_count':len(objects),'kind_counts':dict(Counter(o['kind'] for o in objects)),
        'multi_candidate_objects':sum(len(o['source_members'])>1 for o in objects),
        'input_immutable':True,'physical_geometry_approved':False,'units':'working_pixels_unitless',
        'source_edge_partition_exact':True,'source_edges':len(ledger[2]),'fit_calls':fitter.calls,
        'search_complete':not fitter.incomplete,'circle_families':family,'junction_adjustments':adjustments}
