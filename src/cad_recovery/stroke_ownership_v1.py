"""Contextual roles with explicit conflict/unknown ink ownership; no CAD promotion."""
from collections import Counter,defaultdict
import math

import cv2
import numpy as np

from .stroke_roles_v1 import CLASSES,checked_gray,foreground,text_components,stroke_floor,line_distance,intersection
from .ownership_contours_v1 import boundaries
from .curve_graph_v1 import sample

POLICY={'version':'stroke-ownership/1.0','maximum_objects':4096,'maximum_lines':1024,
    'maximum_pair_checks_per_stage':250000,'minimum_hatch_lines':4,'hatch_angle_deg':5,
    'hatch_gap_in_strokes':[1.5,14.],'hatch_gap_CV_max':.35,'head_angle_deg':20,
    'head_contact_in_strokes':1.5,'extension_angle_cos_max':.2,
    'minimum_extension_reach_in_strokes':3.,'minimum_text_components':3,
    'minimum_part_extent_in_strokes':10.,'minimum_part_page_fraction':.15,
    'source_pixels_erased':0,'production_admitted':False,'semantic_ownership_confirmed':False}


def polygon_distance(point,boundary):
    return float(cv2.pointPolygonTest(np.asarray(boundary['polygon'],np.float32),tuple(map(float,point)),True))


def strict_text(gray,geometry,stroke):
    original=text_components(gray,stroke);ink=foreground(gray)
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=8)
    accepted=[];rejected=[];mask=np.zeros_like(gray);ambiguous=np.zeros_like(gray)
    for group in original['groups']:
        axes=[];straight=[];closed_round=[]
        for cid in group['component_ids']:
            y,x=np.nonzero(labels==cid);p=np.c_[x,y].astype(float)
            values,vectors=np.linalg.eigh(np.cov(p.T));axis=vectors[:,-1]
            straight.append(bool(values[-1]>4*max(1e-9,values[0])));axes.append(axis)
            x0,y0,w0,h0,_=stats[cid];patch=(labels[y0:y0+h0,x0:x0+w0]==cid).astype(np.uint8)
            holes=cv2.connectedComponents(np.pad(1-patch,1,constant_values=1),connectivity=4)[0]-2
            closed_round.append(bool(holes>0 and .6<=w0/max(1,h0)<=1.4))
        parallel=bool(all(straight) and all(abs(axes[0]@u)>.98 for u in axes[1:]))
        x,y,r,b=group['box'];h=max(1,min(r-x,b-y));structural=False
        for c in geometry:
            if c['kind']!='line':continue
            p=np.asarray(c['points_px']);length=float(np.linalg.norm(p[1]-p[0]))
            if length<3*h:continue
            q=np.rint(sample(c,1.)).astype(int)
            valid=(q[:,0]>=x)&(q[:,0]<r)&(q[:,1]>=y)&(q[:,1]<b)
            if valid.any() and np.any(np.isin(labels[q[valid,1],q[valid,0]],group['component_ids'])):structural=True;break
        round_group=all(closed_round)
        reason='parallel_stroke_group' if parallel else 'touches_long_geometry' if structural else 'ambiguous_closed_glyph_or_hole_group' if round_group else None
        if reason:
            rejected.append({**group,'reason':reason})
            if round_group:ambiguous[np.isin(labels,group['component_ids'])]=255
            continue
        accepted.append(group);mask[np.isin(labels,group['component_ids'])]=255
    return {'component_mask':mask,'ambiguity_mask':ambiguous,'groups':accepted,'rejected':rejected,'search_incomplete':original['search_incomplete']}


def line_objects(geometry,widths):
    lookup={w['id']:w for w in widths};lines=[]
    for c in geometry:
        if c['kind']!='line':continue
        p=np.asarray(c['points_px'],float);delta=p[1]-p[0];length=float(np.linalg.norm(delta))
        if length<=0:continue
        w=lookup.get(c['id'],{});lines.append({'id':c['id'],'owner':c.get('path_id',c['id']),
            'points':p,'u':delta/length,'length':length,'width':w.get('width_px')})
    return lines


def hatch_bundles(lines,stroke):
    eligible=[l for l in lines if 6*stroke<=l['length']<=45*stroke];adj=defaultdict(set);checks=0
    for i,a in enumerate(eligible):
        for j in range(i+1,len(eligible)):
            checks+=1
            if checks>250000:return [],True
            b=eligible[j]
            if abs(a['u']@b['u'])<math.cos(math.radians(5)):continue
            normal=np.array([-a['u'][1],a['u'][0]]);offset=abs(float((a['points'].mean(axis=0)-b['points'].mean(axis=0))@normal))
            projections=[p@a['u'] for p in (a['points'],b['points'])]
            overlap=min(p.max() for p in projections)-max(p.min() for p in projections)
            if 1.5*stroke<=offset<=14*stroke and overlap>=.3*min(a['length'],b['length']):adj[i].add(j);adj[j].add(i)
    visited=set();groups=[]
    for i in range(len(eligible)):
        if i in visited:continue
        stack=[i];component=set()
        while stack:
            j=stack.pop()
            if j in component:continue
            component.add(j);stack.extend(adj[j]-component)
        visited.update(component)
        if len(component)<4:continue
        chosen=[eligible[j] for j in sorted(component)];u=chosen[0]['u'];normal=np.array([-u[1],u[0]])
        offsets=np.array(sorted(float(l['points'].mean(axis=0)@normal) for l in chosen));gaps=np.diff(offsets)
        if np.min(gaps)<1.5*stroke or np.std(gaps)/np.mean(gaps)>.35:continue
        vertices=np.concatenate([l['points'] for l in chosen]);hull=cv2.convexHull(vertices.astype(np.float32))[:,0,:]
        groups.append({'id':f'hatch-bundle-{len(groups):03d}','line_ids':[l['id'] for l in chosen],
            'owners':sorted(set(l['owner'] for l in chosen)),'polygon':hull.tolist(),'spacing_px':float(np.median(gaps)),
            'semantic_ownership_confirmed':False})
    return groups,False


def sample_fraction(mask,candidate):
    p=np.rint(sample(candidate,1.)).astype(int)
    valid=(p>=0).all(axis=1)&(p[:,0]<mask.shape[1])&(p[:,1]<mask.shape[0])
    return float(np.mean(mask[p[valid,1],p[valid,0]]>0)) if valid.any() else 0.


def unresolved_result(gray,geometry,heads,reason,incomplete):
    ink=foreground(gray);total=int((ink>0).sum())
    return {'proposals':[{'id':c['id'],'path_id':c.get('path_id',c['id']),'role':'unknown','alternatives':[],
        'status':'proposed_only','semantic_ownership_confirmed':False,'physical_geometry_approved':False,'reason':reason} for c in geometry],
        'pixel_roles':np.full(gray.shape,255,np.uint8),'text_mask':np.zeros_like(gray),'role_counts':{'unknown':len(geometry)},
        'boundaries':{'boundaries':[],'search_incomplete':incomplete},'text':{'groups':[],'rejected':[]},
        'hatch_bundles':[],'hatch_support':{},'head_events':[],'head_ids_used':[],
        'head_observations_retained':len(heads),'dimension_chains':[],'stroke_px':None,'conflicting_ink_px':0,
        'assigned_ink_px':0,'unknown_ink_px':total,'input_ink_px':total,'exact_ink_partition':True,
        'reason':reason,'search_incomplete':incomplete,'source_pixels_erased':0,
        'production_admitted':False,'physical_geometry_approved':False}


def head_context_gate(heads,text_mask,bundles,stroke,*,protected=True):
    usable=[];events=[]
    for h in heads:
        tip=np.asarray(h['tip_px'],float);base=np.asarray(h['polygon_px'][1:],float).mean(axis=0)
        midpoint=(tip+base)/2;reason=None
        if protected and any(cv2.pointPolygonTest(np.asarray(g['polygon'],np.float32),tuple(midpoint),True)>=-stroke for g in bundles):reason='head_in_parallel_stroke_bundle'
        q=np.rint(midpoint).astype(int)
        if 0<=q[0]<text_mask.shape[1] and 0<=q[1]<text_mask.shape[0] and text_mask[q[1],q[0]]:reason='head_in_text_component'
        if reason:events.append({'id':h['id'],'reason':reason,'observation_retained':True})
        else:usable.append(h)
    return usable,events


def title_grid_owners(loops,text_groups,heads,stroke,shape):
    rects=[b for b in loops if not b['frame_hypothesis'] and
        b['area_px2']>=.92*(b['box'][2]-b['box'][0])*(b['box'][3]-b['box'][1])]
    adj=defaultdict(set)
    for i,a in enumerate(rects):
        x,y,r,b=a['box']
        for j in range(i+1,len(rects)):
            q,t,s,u=rects[j]['box']
            side=(min(abs(r-q),abs(s-x))<=2*stroke and min(b,u)-max(y,t)>.5*min(b-y,u-t))
            row=(min(abs(b-t),abs(u-y))<=2*stroke and min(r,s)-max(x,q)>.5*min(r-x,s-q))
            if side or row:adj[i].add(j);adj[j].add(i)
    visited=set();owners=set();groups=[]
    for i in range(len(rects)):
        if i in visited:continue
        pending=[i];component=set()
        while pending:
            j=pending.pop()
            if j in component:continue
            component.add(j);pending.extend(adj[j]-component)
        visited.update(component)
        if len(component)<4:continue
        boxes=np.asarray([rects[j]['box'] for j in component]);x,y=boxes[:,:2].min(axis=0);r,b=boxes[:,2:].max(axis=0)
        if (r-x)*(b-y)>.4*shape[0]*shape[1]:continue
        if not any(x<(g['box'][0]+g['box'][2])/2<r and y<(g['box'][1]+g['box'][3])/2<b for g in text_groups):continue
        if any(x<=h['tip_px'][0]<=r and y<=h['tip_px'][1]<=b for h in heads):continue
        group_owners=sorted({o for j in component for o in rects[j]['owners']});owners.update(group_owners)
        groups.append({'boundary_ids':[rects[j]['id'] for j in sorted(component)],'owners':group_owners,'box':[float(x),float(y),float(r),float(b)]})
    return owners,groups


def analyze(gray,geometry,widths,heads,*,protected=True):
    checked_gray(gray)
    if len(geometry)>4096 or len(heads)>2048:raise ValueError('Object/head budget exceeded')
    for h in heads:
        if (np.asarray(h['tip_px']).shape!=(2,) or np.asarray(h['polygon_px']).shape!=(3,2)
            or np.asarray(h['direction']).shape!=(2,) or not np.isfinite([*h['tip_px'],*np.ravel(h['polygon_px']),*h['direction']]).all()
            or np.linalg.norm(h['direction'])<=0):raise ValueError('Finite observed filled head required')
    if not np.any(foreground(gray)):return unresolved_result(gray,geometry,heads,'empty_ink',False)
    stroke=stroke_floor(widths)
    if stroke is None:return unresolved_result(gray,geometry,heads,'missing_stroke_scale',True)
    lines=line_objects(geometry,widths)
    if len(lines)>1024:return unresolved_result(gray,geometry,heads,'line_budget',True)
    text=strict_text(gray,geometry,stroke) if protected else text_components(gray,stroke)
    text_mask=text['component_mask'];topology=boundaries(geometry,gray.shape,stroke)
    loops=topology['boundaries'];bundles,hatch_incomplete=hatch_bundles(lines,stroke)
    by_id={c['id']:c for c in geometry};by_owner=defaultdict(list)
    for c in geometry:by_owner[c.get('path_id',c['id'])].append(c)
    eligible=[];frame_owners=set()
    for loop in loops:
        if loop['frame_hypothesis']:frame_owners.update(loop['owners']);continue
        x,y,r,b=loop['box'];extent=max(r-x,b-y)
        if extent<10*stroke or extent<.15*min(gray.shape):continue
        members=[c for owner in loop['owners'] for c in by_owner[owner]]
        if members and np.mean([sample_fraction(text_mask,c) for c in members])>.15:continue
        eligible.append(loop)
    # Preserve every head observation, but never attach it to a role chain across a text/hatch conflict.
    usable,head_events=head_context_gate(heads,text_mask,bundles,stroke,protected=protected)
    grid_owners,grid_groups=title_grid_owners(loops,text['groups'],usable,stroke,gray.shape)
    contacts=defaultdict(list)
    for l in lines:
        for h in usable:
            tip=np.asarray(h['tip_px']);base=np.asarray(h['polygon_px'][1:]).mean(axis=0);direction=np.asarray(h['direction'],float).copy()
            direction/=np.linalg.norm(direction)
            if abs(l['u']@direction)<math.cos(math.radians(20)):continue
            if line_distance(base,l['points'])[0]>max(1.,1.5*stroke):continue
            if float(np.min((l['points']-tip)@direction))>=-np.linalg.norm(base-tip):continue
            contacts[l['id']].append(h)
    dimension=set();extension=set();leaders=set();confirmed_loops=set();chain_evidence=[];pair_checks=0
    for shaft in lines:
        attached=contacts[shaft['id']]
        if not attached:continue
        p=shaft['points'];options=[]
        # The observed head body connects its base to the tip. Extend only the semantic
        # test span to that tip; never add a CAD line or bridge an unobserved gap.
        projections=[float(q@shaft['u']) for q in p]
        projections.extend(float(np.asarray(h['tip_px'])@shaft['u']) for h in attached
            if line_distance(np.asarray(h['tip_px']),p)[0]<=np.linalg.norm(np.asarray(h['polygon_px'])[1:].mean(axis=0)-h['tip_px'])+2*stroke)
        origin=p[0]-float(p[0]@shaft['u'])*shaft['u']
        semantic=np.array([origin+min(projections)*shaft['u'],origin+max(projections)*shaft['u']])
        for ext in lines:
            if ext is shaft or abs(ext['u']@shaft['u'])>.2:continue
            pair_checks+=1
            if pair_checks>250000:break
            hit=intersection(semantic,ext['points'])
            if hit is None or min(np.linalg.norm(hit-e) for e in semantic)>3*stroke:continue
            for boundary in eligible:
                if polygon_distance(p.mean(axis=0),boundary)>=-.5*stroke:continue
                feet=[q for q in ext['points'] if abs(polygon_distance(q,boundary))<=max(1.5*stroke,boundary['chord_error_px']*2)
                      and np.linalg.norm(q-hit)>=3*stroke]
                if feet:options.append((ext,boundary,hit,feet[0]))
        for i,a in enumerate(options):
            for b in options[i+1:]:
                if a[0]['id']==b[0]['id'] or a[1]['id']!=b[1]['id']:continue
                if np.linalg.norm(a[2]-b[2])<.6*shaft['length']:continue
                # Two explicit extension contacts distinguish a one-head dimension from a leader.
                dimension.add(shaft['id']);extension.update([a[0]['id'],b[0]['id']]);confirmed_loops.add(a[1]['id'])
                chain_evidence.append({'shaft_id':shaft['id'],'extension_ids':[a[0]['id'],b[0]['id']],
                    'boundary_id':a[1]['id'],'head_ids':[h['id'] for h in attached],'visible_heads':len(attached),
                    'feet':[a[3].tolist(),b[3].tolist()],'status':'geometric_hypothesis_only'})
        for i,head in enumerate(attached):
            for other in attached[i+1:]:
                first=np.asarray(head['direction'],float);second=np.asarray(other['direction'],float)
                if (first@second)/(np.linalg.norm(first)*np.linalg.norm(second))>-.9:continue
                for boundary in eligible:
                    if abs(polygon_distance(head['tip_px'],boundary))<=2*stroke and abs(polygon_distance(other['tip_px'],boundary))<=2*stroke:
                        dimension.add(shaft['id']);confirmed_loops.add(boundary['id'])
                        chain_evidence.append({'shaft_id':shaft['id'],'extension_ids':[],
                            'boundary_id':boundary['id'],'head_ids':[head['id'],other['id']],
                            'visible_heads':2,'kind':'direct_two_head_span','status':'geometric_hypothesis_only'})
        if shaft['id'] not in dimension:
            for head in attached:
                tip=np.asarray(head['tip_px']);back=p[np.argmax(np.linalg.norm(p-tip,axis=1))]
                for boundary in eligible:
                    if abs(polygon_distance(tip,boundary))<=2*stroke and polygon_distance(back,boundary)<-2*stroke:
                        leaders.add(shaft['id']);confirmed_loops.add(boundary['id'])
    hatch_ids=set();hatch_support={}
    for group in bundles:
        chosen=[next(l for l in lines if l['id']==i) for i in group['line_ids']]
        for boundary in eligible:
            if all(polygon_distance(l['points'].mean(axis=0),boundary)>stroke for l in chosen):
                hatch_ids.update(group['line_ids']);hatch_support[group['id']]=boundary['id'];confirmed_loops.add(boundary['id'])
    part_owners=set()
    annotation_owners={c.get('path_id',c['id']) for c in geometry if c['id'] in dimension|extension|leaders|hatch_ids}
    for loop in eligible:
        if set(loop['owners'])&(annotation_owners|grid_owners):continue
        # Large closed geometry is a contour proposal. Width alone never makes an open line a part.
        if loop['id'] in confirmed_loops or not protected or max(loop['box'][2]-loop['box'][0],loop['box'][3]-loop['box'][1])>=.3*min(gray.shape):
            part_owners.update(loop['owners'])
    # Curves nested in a supported outline may be small boundaries; text/hatch conflicts still veto ownership.
    for loop in loops:
        if loop['frame_hypothesis']:continue
        if set(loop['owners'])&(annotation_owners|grid_owners):continue
        if any(parent['id'] in confirmed_loops and polygon_distance(loop['polygon'][0],parent)>2*stroke for parent in eligible):part_owners.update(loop['owners'])
    axis=set()
    for l in lines:
        if pair_checks>250000:break
        if l['id'] in dimension|extension|leaders|hatch_ids or contacts[l['id']]:continue
        collinear=[]
        for other in lines:
            pair_checks+=1
            if pair_checks>250000:break
            if other is l or abs(l['u']@other['u'])<.995:continue
            normal=np.array([-l['u'][1],l['u'][0]])
            if abs(float((other['points'].mean(axis=0)-l['points'][0])@normal))>stroke:continue
            gap=min(np.linalg.norm(a-b) for a in l['points'] for b in other['points'])
            if stroke<gap<=12*stroke:collinear.append(other)
        if len(collinear)>=2:
            lengths=[l['length'],*[o['length'] for o in collinear]]
            if max(lengths)>=1.8*min(lengths) and any(polygon_distance(l['points'].mean(axis=0),b)>0 for b in eligible):axis.add(l['id'])
    ink=foreground(gray);pixel=np.full(gray.shape,255,np.uint8);conflicts=np.zeros_like(gray,bool);proposals=[]
    for c in geometry:
        hints=[];owner=c.get('path_id',c['id']);fraction=sample_fraction(text_mask,c)
        if owner in frame_owners:hints.append('sheet_frame')
        if owner in grid_owners:hints.append('title_grid')
        if owner in part_owners:hints.append('part_contour')
        for ids,role in [(dimension,'dimension_line'),(extension,'extension_line'),(leaders,'leader_line'),(hatch_ids,'hatching'),(axis,'axis_construction')]:
            if c['id'] in ids:hints.append(role)
        if fraction>=.5:hints.append('text')
        ambiguous_text=protected and sample_fraction(text.get('ambiguity_mask',np.zeros_like(gray)),c)>=.5
        # An annotation owner on a closed component is expected; closedness cannot override chain evidence.
        specific=[h for h in hints if h in ('dimension_line','extension_line','leader_line','hatching','axis_construction')]
        if specific:hints=[h for h in hints if h!='part_contour']
        hints=sorted(set(hints));role=hints[0] if len(hints)==1 else 'unknown'
        if ambiguous_text:role='unknown'
        if topology['search_incomplete'] or hatch_incomplete or pair_checks>250000:role='unknown'
        proposals.append({'id':c['id'],'path_id':owner,'role':role,'alternatives':hints,
            'text_overlap_fraction':fraction,'status':'proposed_only','semantic_ownership_confirmed':False,
            'ambiguous_closed_glyph_or_hole':bool(ambiguous_text),
            'physical_geometry_approved':False,'head_ids':[h['id'] for h in contacts[c['id']]]})
        if role=='unknown':continue
        drawing=np.zeros_like(gray);points=np.rint(sample(c,.5)).astype(np.int32)
        width=next((w.get('width_px') for w in widths if w['id']==c['id']),None)
        width=width if width is not None else c.get('local_width',{}).get('width_px')
        if width is None:continue
        cv2.polylines(drawing,[points],c['kind']=='circle' or c.get('closed',False),255,max(1,min(32,int(round(width)))))
        selected=(drawing>0)&(ink>0);cls=CLASSES.index(role)
        conflicts|=selected&(pixel!=255)&(pixel!=cls);pixel[selected&~conflicts]=cls
    # Only actual component ink is marked; no rectangular erasure or classification of unassigned whitespace.
    selected=(text_mask>0)&(ink>0);conflicts|=selected&(pixel!=255)&(pixel!=6);pixel[selected&~conflicts]=6
    pixel[conflicts]=255
    assigned=(pixel!=255)&(ink>0);unknown=(pixel==255)&(ink>0)
    return {'proposals':proposals,'pixel_roles':pixel,'role_counts':dict(Counter(p['role'] for p in proposals)),
        'boundaries':topology,'text':{k:v for k,v in text.items() if k not in ('component_mask','ambiguity_mask')},'text_mask':text_mask,
        'hatch_bundles':bundles,'hatch_support':hatch_support,'title_grids':grid_groups,'head_events':head_events,
        'head_ids_used':[h['id'] for h in usable],'head_observations_retained':len(heads),
        'dimension_chains':chain_evidence,'stroke_px':stroke,'conflicting_ink_px':int((conflicts&(ink>0)).sum()),
        'assigned_ink_px':int(assigned.sum()),'unknown_ink_px':int(unknown.sum()),'input_ink_px':int((ink>0).sum()),
        'exact_ink_partition':bool(np.array_equal(assigned|unknown,ink>0)),
        'search_incomplete':topology['search_incomplete'] or hatch_incomplete or text['search_incomplete'] or pair_checks>250000,
        'boundary_analysis_complete':not topology['rejected'],
        'source_pixels_erased':0,'production_admitted':False,'physical_geometry_approved':False}
