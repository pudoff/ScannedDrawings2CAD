"""Observed centerline paths and bounded line/arc/circle proposals, never CAD approval."""
import copy
import math

import cv2
import numpy as np

from .centerline_comparison_v1 import foreground,topology
from .geometry_detection import _fit_circle,_observed_angles
from .local_stroke_evidence_v1 import line_width

POLICY={'version':'curve-graph/1.0','maximum_pixels':4194304,'maximum_skeleton_pixels':120000,
    'maximum_paths':4096,'maximum_fit_calls':8192,'maximum_fit_points':1024,'maximum_depth':8,
    'fit_p95_px':1.25,'fit_max_px':2.,'minimum_arc_sweep_deg':20.,'maximum_radius_in_sides':4,
    'curve_line_error_ratio':.65,'minimum_arc_sagitta_px':1.,'maximum_merge_passes':3,
    'minimum_typed_path_length_px':8.,'source_support_required':1.,
    'production_admitted':False,'physical_geometry_approved':False}


def graph_paths(skeleton):
    yy,xx=np.nonzero(skeleton);pixels=np.c_[xx,yy]
    if len(pixels)>120000:return [],{'search_incomplete':True,'reason':'skeleton_pixel_budget','pixels':len(pixels)}
    index={(int(x),int(y)):i for i,(x,y) in enumerate(pixels)};neighbors=[[] for _ in pixels]
    for i,(x,y) in enumerate(pixels):
        for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)):
            j=index.get((int(x+dx),int(y+dy)))
            if j is None:continue
            # Remove diagonal shortcuts when an observed orthogonal path exists.
            if dx and dy and ((int(x+dx),int(y)) in index or (int(x),int(y+dy)) in index):continue
            neighbors[i].append(j)
    critical={i for i,n in enumerate(neighbors) if len(n)!=2};used=set();paths=[];complete=True
    def trace(first,second):
        nonlocal complete
        chain=[first,second];used.add(tuple(sorted((first,second))))
        while chain[-1] not in critical:
            options=[j for j in neighbors[chain[-1]] if j!=chain[-2]]
            if not options:break
            nxt=options[0];edge=tuple(sorted((chain[-1],nxt)))
            if edge in used:break
            used.add(edge);chain.append(nxt)
            if nxt==first:break
        paths.append({'points':pixels[chain].astype(float),'closed':chain[0]==chain[-1],
            'start_degree':len(neighbors[chain[0]]),'end_degree':len(neighbors[chain[-1]])})
    for i in sorted(critical):
        for j in neighbors[i]:
            if tuple(sorted((i,j))) not in used:
                if len(paths)>=4096:complete=False;break
                trace(i,j)
        if not complete:break
    if complete:
        for i,adjacent in enumerate(neighbors):
            for j in adjacent:
                if tuple(sorted((i,j))) not in used:
                    if len(paths)>=4096:complete=False;break
                    trace(i,j)
            if not complete:break
    return paths,{'pixels':len(pixels),'pixel_edges':sum(map(len,neighbors))//2,'traced_edges':len(used),
        'paths':len(paths),'junction_pixels':sum(len(n)>=3 for n in neighbors),
        'isolated_pixels':sum(len(n)==0 for n in neighbors),'search_incomplete':not complete,
        'white_gap_edges_added':0}


def sample(candidate,spacing=.5):
    if candidate['kind']=='line':
        p=np.asarray(candidate['points_px'],float);length=float(np.linalg.norm(p[1]-p[0]))
        return p[0]+np.linspace(0,1,max(2,math.ceil(length/spacing)+1))[:,None]*(p[1]-p[0])
    if candidate['kind']=='polyline':return np.asarray(candidate['points_px'],float)
    start=math.radians(candidate.get('start_angle_deg',0));sweep=math.radians(candidate.get('sweep_angle_deg',360))
    count=max(8,math.ceil(candidate['radius_px']*sweep/spacing)+1)
    angles=np.linspace(start,start+sweep,count)
    return np.asarray(candidate['center_px'])+candidate['radius_px']*np.c_[np.cos(angles),np.sin(angles)]


def native_support(candidate,ink):
    points=np.rint(sample(candidate)).astype(int)
    valid=(points>=0).all(axis=1)&(points[:,0]<ink.shape[1])&(points[:,1]<ink.shape[0])
    return bool(valid.all() and np.all(ink[points[:,1],points[:,0]]>0)) if valid.all() else False


def fit(points,closed,ink):
    points=np.asarray(points,float)
    if len(points)<3:return None
    chosen=points[np.linspace(0,len(points)-1,min(len(points),1024)).astype(int)]
    mean=chosen.mean(axis=0);_,_,vectors=np.linalg.svd(chosen-mean,full_matrices=False);axis=vectors[0]
    normal=np.array([-axis[1],axis[0]]);offset=np.abs((points-mean)@normal)
    projection=(points-mean)@axis
    line={'kind':'line','points_px':np.array([mean+projection.min()*axis,mean+projection.max()*axis]).tolist()}
    line_error=float(np.sqrt(np.mean(offset**2)))
    line_ok=not closed and np.percentile(offset,95)<=1.25 and offset.max()<=2 and native_support(line,ink)
    circle=_fit_circle(chosen);curve=None
    if circle is not None:
        center,radius,_=circle
        residual=np.abs(np.linalg.norm(points-center,axis=1)-radius)
        if 2<=radius<=max(ink.shape)*4 and np.percentile(residual,95)<=1.25 and residual.max()<=2:
            start,sweep=_observed_angles(points,center)
            if closed:
                candidate={'kind':'circle','center_px':center.tolist(),'radius_px':radius}
            else:
                candidate={'kind':'arc','center_px':center.tolist(),'radius_px':radius,
                    'start_angle_deg':math.degrees(start),'sweep_angle_deg':math.degrees(sweep)}
            error=float(np.sqrt(np.mean(residual**2)))
            sagitta=radius*(1-math.cos(min(sweep,math.pi)/2))
            supported=native_support(candidate,ink)
            if supported and (closed or sweep>=math.radians(20) and sagitta>=1 and (not line_ok or error<=line_error*.65)):
                curve={**candidate,'fit_p95_px':float(np.percentile(residual,95)),'fit_max_px':float(residual.max()),
                    'fit_rmse_px':error,'source_support':1.,'observed_closed_path':bool(closed),
                    'observed_start_angle_deg':math.degrees(start),'observed_sweep_angle_deg':math.degrees(sweep)}
    if curve is not None:return curve
    if line_ok:return {**line,'fit_p95_px':float(np.percentile(offset,95)),'fit_max_px':float(offset.max()),
        'fit_rmse_px':line_error,'source_support':1.,'observed_closed_path':False}
    return None


def split_fit(points,closed,ink,account,depth=0):
    if account['calls']>=8192:return [{'kind':'polyline','points_px':points.tolist(),'closed':closed,'reason':'fit_budget'}]
    account['calls']+=1
    length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    candidate=fit(points,closed,ink) if length>=8 else None
    if candidate is not None:return [{**candidate,'support_points':points.tolist()}]
    if depth>=8 or len(points)<12 or length<16 or closed:
        return [{'kind':'polyline','points_px':points.tolist(),'closed':closed,'reason':'unresolved_source_path'}]
    middle=len(points)//2
    return split_fit(points[:middle+1],False,ink,account,depth+1)+split_fit(points[middle:],False,ink,account,depth+1)


def merge_adjacent(parts,ink,account):
    parts=list(parts)
    for _ in range(3):
        merged=[];i=0;changed=False
        while i<len(parts):
            if i+1<len(parts) and account['calls']<8192:
                a,b=parts[i:i+2]
                p=np.asarray(a.get('support_points',a.get('points_px')),float)
                q=np.asarray(b.get('support_points',b.get('points_px')),float)
                if len(p) and len(q) and np.array_equal(p[-1],q[0]):
                    joined=np.r_[p,q[1:]];account['calls']+=1;candidate=fit(joined,False,ink)
                    if candidate is not None:
                        merged.append({**candidate,'support_points':joined.tolist()});i+=2;changed=True;continue
            merged.append(parts[i]);i+=1
        parts=merged
        if not changed:break
    return parts


def recover(gray):
    if gray.dtype!=np.uint8 or gray.ndim!=2 or not 0<gray.size<=4194304:raise ValueError('Bounded uint8 gray required')
    from skimage.morphology import skeletonize
    ink=foreground(gray);skel=skeletonize(ink>0,method='zhang').astype(np.uint8)*255
    paths,stats=graph_paths(skel);trace=[];merged=[];account={'calls':0}
    for i,path in enumerate(paths):
        parts=split_fit(path['points'],path['closed'],ink,account)
        for p in parts:p.update(path_id=f'observed-path-{i:05d}',classification='unclassified',physical_geometry_approved=False)
        trace.extend(copy.deepcopy(parts))
        second=merge_adjacent(copy.deepcopy(parts),ink,account)
        for p in second:p.update(path_id=f'observed-path-{i:05d}',classification='unclassified',physical_geometry_approved=False)
        merged.extend(second)
    distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    for mode,values in [('trace',trace),('merged',merged)]:
        for i,c in enumerate(values,1):
            c['id']=f'{mode}-geometry-{i:05d}'
            c['review_status']='unresolved' if c['kind']=='polyline' else 'proposal_requires_review'
            c['local_width']=width_evidence(c,ink,distance)
    return {'trace':trace,'merged':merged,'analysis':{**stats,'fit_calls':account['calls'],
        'search_incomplete':stats['search_incomplete'] or account['calls']>=8192,
        'input_topology':topology(ink),'skeleton_topology':topology(skel)},'ink':ink,'skeleton':skel}


def width_evidence(candidate,ink,distance):
    if candidate['kind']=='line':return line_width(candidate['points_px'],ink,distance)
    points=sample(candidate)
    indices=np.unique(np.linspace(0,len(points)-1,min(9,len(points))).astype(int))
    observations=[line_width([points[a],points[b]],ink,distance) for a,b in zip(indices,indices[1:])
                  if np.linalg.norm(points[b]-points[a])>0]
    values=[r['width_px'] for r in observations if r['width_px'] is not None]
    return {'method':'local_chord_normal_spans/1.0','width_px':float(np.median(values)) if values else None,
        'local_observations':observations,'line_role':'unknown','physical_width_mm':None,
        'uncertainty_note':'chord probes are observational; curvature and junctions may bias spans'}


def curve_distance(points,candidate):
    points=np.asarray(points,float);delta=points-np.asarray(candidate['center_px'])
    radial=np.abs(np.linalg.norm(delta,axis=1)-candidate['radius_px'])
    if candidate['kind']=='circle':return radial
    angles=np.degrees(np.arctan2(delta[:,1],delta[:,0]))%360
    inside=(angles-candidate['start_angle_deg'])%360<=candidate['sweep_angle_deg']+1e-6
    endpoints=sample(candidate)[[0,-1]]
    end_distance=np.min(np.linalg.norm(points[:,None,:]-endpoints[None,:,:],axis=2),axis=1)
    return np.where(inside,radial,end_distance)


def hybrid(baseline,recovered):
    curves=[c for c in recovered if c['kind'] in ('arc','circle')];added=[]
    for c in curves:
        if any(old['kind']==c['kind'] and max(curve_distance(sample(c),old).max(),curve_distance(sample(old),c).max())<=1
               for old in baseline if old['kind'] in ('arc','circle')):continue
        added.append(c)
    all_curves=[c for c in baseline if c['kind'] in ('arc','circle')]+added
    suppressed=[];active=[]
    for c in baseline:
        if c['kind']=='line':
            points=sample(c)
            matches=[curve for curve in all_curves if float(curve_distance(points,curve).max())<=1.25
                     and np.linalg.norm(points[-1]-points[0])>=4]
            if len(matches)==1:
                suppressed.append({'id':c['id'],'reason':'entire_LINE_within_observed_curve_band','curve_id':matches[0]['id']});continue
        active.append(c)
    return {'candidates':active+added,'suppressed':suppressed,'observations':baseline+added,
        'new_LINE_connectors':0,'source_pixels_erased':False}


def residual_ledger(candidates,ink):
    explained=np.zeros_like(ink,bool)
    for c in candidates:
        points=np.rint(sample(c)).astype(int)
        valid=(points>=0).all(axis=1)&(points[:,0]<ink.shape[1])&(points[:,1]<ink.shape[0])
        p=points[valid];explained[p[:,1],p[:,0]]=True
    # Evidence corridor is diagnostic; it never erases or assigns a semantic role.
    corridor=cv2.dilate(explained.astype(np.uint8),np.ones((5,5),np.uint8))>0
    assigned=(ink>0)&corridor;residual=(ink>0)&~corridor
    n,_,stats,_=cv2.connectedComponentsWithStats(residual.astype(np.uint8),connectivity=8)
    return {'assigned_mask':assigned.astype(np.uint8)*255,'residual_mask':residual.astype(np.uint8)*255,
        'assigned_ink_px':int(assigned.sum()),'residual_ink_px':int(residual.sum()),'input_ink_px':int((ink>0).sum()),
        'residual_components':[{'id':f'residual-{i:05d}','box':stats[i,:4].tolist(),'area_px':int(stats[i,4]),'status':'unknown_requires_review'} for i in range(1,n)],
        'exact_partition':bool(np.array_equal(assigned|residual,ink>0) and not np.any(assigned&residual)),
        'source_pixels_erased':False}
