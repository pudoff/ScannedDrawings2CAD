"""Conservative head eligibility from source paths and positive glyph evidence.

Missing a long straight tail is not a negative label. A compact complete
component with a cavity or extra branches and aligned glyph peers supplies counterevidence;
an independently observed short/curved stem protects a head from exclusion.
All input observations and source pixels remain available unchanged.
"""
from copy import deepcopy
from collections import deque
import math
import cv2
import numpy as np
from skimage.morphology import skeletonize
from .stroke_roles_v1 import checked_gray,foreground,text_components
from .head_eligibility_v2 import source_width
from .junction_topology_v1 import adjacency_bits,OFFSETS,POPCOUNT

POLICY={'version':'head-source-context/1.0','maximum_heads':512,'maximum_components':4096,
    'maximum_component_side':192,'maximum_path_roi_side':256,'maximum_roi_pixels':10000000,
    'maximum_path_visits':500000,'minimum_text_peers':3,'source_erasure_allowed':False,
    'production_admitted':False,'physical_geometry_approved':False}


def component_for_head(head,labels,stats):
    polygon=np.asarray(head['polygon_px'],float);q=np.rint(polygon.mean(0)).astype(int)
    if (q<0).any() or q[0]>=labels.shape[1] or q[1]>=labels.shape[0]:return None
    identity=int(labels[q[1],q[0]])
    if not identity:return None
    x,y,w,h,area=map(int,stats[identity])
    return {'id':identity,'box':[x,y,x+w,y+h],'width':w,'height':h,'area':area,
        'complete':x>0 and y>0 and x+w<labels.shape[1] and y+h<labels.shape[0]}


def cavity_evidence(component,labels,pen):
    if not component or not component['complete'] or max(component['width'],component['height'])>192:
        return {'available':False,'cavities':[]}
    x,y,right,bottom=component['box'];local=labels[y:bottom,x:right]==component['id']
    background=np.pad((~local).astype(np.uint8),1,constant_values=1)
    count,ids,stats,_=cv2.connectedComponentsWithStats(background,connectivity=4)
    outside=set(np.r_[ids[0],ids[-1],ids[:,0],ids[:,-1]].tolist());cavities=[]
    minimum=max(4.,(.75*pen)**2)
    for i in range(1,count):
        if i in outside or stats[i,cv2.CC_STAT_AREA]<minimum:continue
        bx,by,w,h,area=map(int,stats[i]);cavities.append({'box':[x+bx-1,y+by-1,x+bx+w-1,y+by+h-1],'area':area})
    return {'available':True,'cavities':cavities,'minimum_area_from_pen':minimum}


def outside_branches(head,component,labels,audit):
    if not component or not component['complete'] or max(component['width'],component['height'])>192:
        return {'available':False,'outside_endpoints_px':[]}
    x,y,right,bottom=component['box'];local=labels[y:bottom,x:right]==component['id']
    cost=int(local.size)
    if audit['roi_pixels']+cost>POLICY['maximum_roi_pixels']:
        audit['limits'].append('component_topology_pixels');return {'available':False,'outside_endpoints_px':[]}
    audit['roi_pixels']+=cost
    skeleton=skeletonize(local,method='zhang').astype(np.uint8)*255;degree=POPCOUNT[adjacency_bits(skeleton)]
    endpoints=np.argwhere((skeleton>0)&(degree==1));polygon=np.asarray(head['polygon_px'],np.float32)
    outside=[];pen=source_width(head)
    for yy,xx in endpoints:
        point=(float(x+xx),float(y+yy))
        if cv2.pointPolygonTest(polygon,point,True)<-pen:outside.append(list(point))
    return {'available':True,'outside_endpoints_px':outside,'head_margin_from_pen_px':pen,
        'single_stem_topology_supported':len(outside)<=1}


def tail_path(head,ink,labels,dt,component,has_cavity,audit):
    polygon=np.asarray(head['polygon_px'],float);base=polygon[1:].mean(0);unit=np.asarray(head['direction'],float)
    length=float(np.linalg.norm(polygon[0]-base));breadth=float(np.linalg.norm(polygon[1]-polygon[2]));pen=source_width(head)
    if not component or not 1<length<=192 or not 0<breadth<=192:return {'supported':False,'status':'unavailable_head_component'}
    pad=min(128,max(24,math.ceil(2.5*length+2*breadth+4*pen)))
    lo=np.maximum(0,np.floor(base-pad)).astype(int);hi=np.minimum(ink.shape[::-1],np.ceil(base+pad)).astype(int)
    cost=int(np.prod(hi-lo))
    if audit['roi_pixels']+cost>POLICY['maximum_roi_pixels']:
        audit['limits'].append('roi_pixels');return {'supported':False,'status':'path_budget_unchecked'}
    audit['roi_pixels']+=cost
    local=(labels[lo[1]:hi[1],lo[0]:hi[0]]==component['id'])&(ink[lo[1]:hi[1],lo[0]:hi[0]]>0)
    skeleton=skeletonize(local,method='zhang').astype(np.uint8)*255;bits=adjacency_bits(skeleton);degree=POPCOUNT[bits]
    yy,xx=np.indices(local.shape);points=np.stack([xx+lo[0],yy+lo[1]],axis=-1);along=(points-base)@unit
    triangle=np.zeros(local.shape,np.uint8);cv2.fillConvexPoly(triangle,np.rint(polygon-lo).astype(np.int32),255)
    measured_pen=2*dt[lo[1]:hi[1],lo[0]:hi[0]]-1
    allowed=(skeleton>0)&(measured_pen<=.65*breadth+math.sqrt(.5))&((triangle==0)|(along<=pen))
    distance=np.linalg.norm(points-base,axis=2);radius=max(3.,1.5*pen,.5*breadth)
    starts=np.argwhere(allowed&(distance<=radius)&(along<=-min(1.,.5*pen)))
    if not len(starts):return {'supported':False,'status':'no_observed_backward_seed'}
    if has_cavity:
        _,parts=cv2.connectedComponents(allowed.astype(np.uint8),connectivity=8)
        for identity in set(int(parts[y,x]) for y,x in starts)-{0}:
            body=parts==identity
            # The closed route must be observed outside the head, connected to
            # its backward seed; it is retained as an ambiguous curved shaft.
            holes=cv2.connectedComponents(np.pad((~body).astype(np.uint8),1,constant_values=1),connectivity=4)[0]-2
            if holes>0:
                return {'supported':True,'status':'observed_closed_curved_stem','source_only':True,
                    'semantic_arrow_confirmed':False,'closed_source_routes':int(holes),'head_length_px':length}
    order=sorted((tuple(map(int,p)) for p in starts),key=lambda p:(distance[p],p));previous={p:None for p in order};queue=deque(order)
    while queue:
        y,x=queue.popleft();audit['path_visits']+=1
        if audit['path_visits']>POLICY['maximum_path_visits']:
            audit['limits'].append('path_visits');return {'supported':False,'status':'path_budget_unchecked'}
        escaped=distance[y,x]>=2*length
        # A width cutoff must not manufacture an endpoint. Short support requires
        # a real endpoint in the original local skeleton, not the allowed subset.
        short=not has_cavity and degree[y,x]==1 and along[y,x]<=-max(2.,pen) and distance[y,x]>=max(2.,pen)
        if escaped or short:
            path=[];p=(y,x)
            while p is not None:path.append([p[1]+int(lo[0]),p[0]+int(lo[1])]);p=previous[p]
            path.reverse()
            return {'supported':True,'status':'observed_curved_or_long_stem' if escaped else 'observed_short_terminal_stem',
                'path_px':path,'escape_distance_px':float(distance[y,x]),'head_length_px':length,
                'source_only':True,'semantic_arrow_confirmed':False}
        for bit,(dy,dx) in enumerate(OFFSETS):
            if not int(bits[y,x])&(1<<bit):continue
            other=(y+dy,x+dx)
            if allowed[other] and other not in previous:previous[other]=(y,x);queue.append(other)
    return {'supported':False,'status':'no_independent_stem_extent_observed','head_length_px':length}


def analyze(gray,observations):
    checked_gray(gray)
    if len(observations)>512 or len({h['id'] for h in observations})!=len(observations):raise ValueError('Bounded unique head observations required')
    for head in observations:
        p=np.asarray(head['polygon_px'],float);d=np.asarray(head['direction'],float);tip=np.asarray(head['tip_px'],float)
        if p.shape!=(3,2) or d.shape!=(2,) or tip.shape!=(2,) or not np.isfinite(p).all() or not np.isfinite(tip).all() or not np.isfinite(d).all() or abs(np.linalg.norm(d)-1)>.01:
            raise ValueError('Finite triangular observation with unit direction required')
    audit={'roi_pixels':0,'path_visits':0,'limits':[],'threshold_pixels':0}
    if not observations:return {'observations':[],'context_only':[],'combined':[],'decisions':[],'audit':audit,'complete':True,
        'source_pixels_erased':0,'production_admitted':False,'physical_geometry_approved':False}
    ink=foreground(gray);audit['threshold_pixels']=int(gray.size)
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=8)
    stroke=float(np.clip(np.median([source_width(h) for h in observations]),1,24));text=text_components(gray,stroke)
    dt=cv2.distanceTransform(ink,cv2.DIST_L2,5);decisions=[];context=[];combined=[]
    if count-1>POLICY['maximum_components'] or text['search_incomplete']:audit['limits'].append('component_or_text_budget')
    for head in observations:
        if audit['limits']:break
        component=component_for_head(head,labels,stats);cavity=cavity_evidence(component,labels,source_width(head))
        peers=[g for g in text['groups'] if component and component['id'] in g['component_ids'] and len(g['component_ids'])>=4]
        branches=outside_branches(head,component,labels,audit)
        multibranch=branches['available'] and len(branches['outside_endpoints_px'])>=2
        counter=bool(peers and (cavity['available'] and cavity['cavities'] or multibranch))
        path=tail_path(head,ink,labels,dt,component,bool(cavity['cavities']),audit)
        if not counter:context.append(deepcopy(head))
        independent=path['supported'] and (path['status'] in ('observed_curved_or_long_stem','observed_closed_curved_stem') or not multibranch)
        # Multiple observations in the same compact component can represent a
        # small complete dimension; preserve that unresolved case for binding.
        shared_component=False
        if component and counter:
            for other in observations:
                if other['id']==head['id']:continue
                c=component_for_head(other,labels,stats)
                if c and c['id']==component['id']:shared_component=True;break
        eligible=not counter or independent or shared_component
        if eligible:combined.append(deepcopy(head))
        decisions.append({'head_id':head['id'],'component':component,'component_cavities':cavity,'aligned_peer_groups':peers,
            'outside_branches':branches,'multiple_heads_share_component':shared_component,
            'independent_stem_supported':independent,
            'positive_text_component_counterevidence':counter,'source_path':path,'eligible_for_binding':eligible,
            'status':'path_supported' if independent else 'shared_component_review' if shared_component else 'source_counterevidence' if counter else 'review_retained',
            'observation_retained':True,'semantic_text_recognized':False,'eligible_for_CAD':False})
    complete=not audit['limits']
    if not complete:context=combined=deepcopy(list(observations))
    return {'observations':deepcopy(list(observations)),'context_only':context,'combined':combined,
        'decisions':decisions,'audit':audit,'complete':complete,'source_pixels_erased':0,
        'production_admitted':False,'physical_geometry_approved':False}
