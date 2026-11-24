"""Read-only raster/primitive baselines; source rays remain nonsemantic."""
import cv2
import numpy as np
from .geometry_detection import _skeleton
from .stroke_roles_v1 import checked_gray,foreground
from .raster_junctions import _circular_runs,localize_junction
from . import primitive_graph_qa_v1 as kernel


def pixel_nodes(gray):
    checked_gray(gray)
    if gray.size>4194304:raise ValueError('Pixel observation budget')
    ink=foreground(gray);skeleton,limited=_skeleton(ink);binary=(skeleton>0).astype(np.uint8)
    degree=cv2.filter2D(binary,cv2.CV_16S,np.ones((3,3),np.int16),borderType=cv2.BORDER_CONSTANT)-binary
    nodes=[];clipped=0
    for kind,mask in [('endpoint',(binary>0)&(degree==1)),('junction',(binary>0)&(degree>=3))]:
        count,labels,_,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
        for i in range(1,count):
            p=centers[i];ys,xs=np.nonzero(labels==i)
            if min(xs)==0 or min(ys)==0 or max(xs)==gray.shape[1]-1 or max(ys)==gray.shape[0]-1:clipped+=1;continue
            directions=np.linspace(0,2*np.pi,720,endpoint=False);q=np.rint(p+12*np.c_[np.cos(directions),np.sin(directions)]).astype(int)
            inside=(q>=0).all(1)&(q[:,0]<gray.shape[1])&(q[:,1]<gray.shape[0]);branches=None
            if inside.all():branches=len(_circular_runs(ink[q[:,1],q[:,0]]>0))
            nodes.append({'id':'pixel-'+str(len(nodes)),'point_px':p.tolist(),'kind':kind,
                'degree':1 if kind=='endpoint' else branches})
    complete=not limited and len(nodes)<=512
    return nodes if complete else [],skeleton,{'complete':complete,'raw_node_count':len(nodes),'border_clipped_clusters':clipped,
        'skeleton_iteration_limit':limited,'ring_radius_input_px':12,'physical_geometry_approved':False}


def tiled_pixel_nodes(gray):
    split=gray.shape[1]//2;output=[];audits=[]
    for lo,hi in ((0,min(gray.shape[1],split+32)),(max(0,split-32),gray.shape[1])):
        nodes,_,audit=pixel_nodes(gray[:,lo:hi]);audits.append(audit)
        for n in nodes:
            p=np.asarray(n['point_px'])+[lo,0]
            if (lo==0 and p[0]<split) or (lo>0 and p[0]>=split):
                output.append({**n,'id':'tile-'+str(len(output)),'point_px':p.tolist()})
    return output,{'complete':all(a['complete'] for a in audits),'tiles':audits,'overlap_input_px':64}


def primitive_nodes(candidates):
    graph=kernel.build(candidates)
    nodes=[{**n,'kind':'endpoint' if n['degree']==1 else 'isolated' if n['degree']==0 else 'junction'}
        for n in graph.get('structural_nodes',[])]
    if len(nodes)>512:return [],{**graph,'complete':False,'reason':'node_budget'}
    return nodes,graph


def guard_nodes(gray,candidates,nodes):
    parts,_,_=kernel.segments(candidates);rgb=np.repeat(gray[:,:,None],3,axis=2);kept=[];decisions=[]
    if len(nodes)>512:raise ValueError('Guard node budget')
    for n in nodes:
        if n['kind']!='junction':kept.append(n);continue
        p=np.asarray(n['point_px']);directions=[]
        for part in parts:
            t=kernel.parameter(part,p)
            if t is None:continue
            if part['kind']=='line':d=part['b']-part['a'];d/=np.linalg.norm(d)
            else:
                theta=part['start']+t*part['sweep'];d=np.array([-np.sin(theta),np.cos(theta)])
            for ray in ([-d] if t>=1-kernel.EPS else [d] if t<=kernel.EPS else [-d,d]):
                if not any(ray@old>np.cos(np.deg2rad(8)) for old in directions):directions.append(ray)
        if not 2<=len(directions)<=6:
            decisions.append({'node_id':n['id'],'status':'ambiguous','reason':'distinct_ray_count'});continue
        report=localize_junction(rgb,p,[d.tolist() for d in directions],search_radius=8,inner_radius=12,outer_radius=52)
        decisions.append({'node_id':n['id'],**report})
        if report['status']=='localized':kept.append({**n,'point_px':report['observed_xy']})
    return kept,decisions


def pixel_connection(gray,anchors):
    ink=foreground(gray);_,labels=cv2.connectedComponents(ink,connectivity=8);components=[]
    for p in np.rint(anchors).astype(int):
        lo=np.maximum(0,p-1);hi=np.minimum(gray.shape[::-1],p+2)
        ids=set(labels[lo[1]:hi[1],lo[0]:hi[0]].ravel())-{0}
        if len(ids)!=1:return None
        components.append(next(iter(ids)))
    return bool(components[0]==components[1])


def primitive_connection(candidates,graph,anchors,tolerance):
    if not graph.get('complete'):return None
    parts,_,_=kernel.segments(candidates);groups=[]
    for p in np.asarray(anchors):
        owners=set()
        for part in parts:
            if part['kind']=='line':
                d=part['b']-part['a'];t=np.clip((p-part['a'])@d/(d@d),0,1);distance=np.linalg.norm(p-kernel.point(part,t))
            else:
                # Curved-anchor relation is left unresolved unless the exact
                # finite arc parameter is observed; no full-circle shortcut.
                t=kernel.parameter(part,p);distance=np.inf if t is None else 0.
            if distance<=tolerance:owners.add(part['owner'])
        ids={i for i,keys in enumerate(graph['component_owners']) if owners&set(keys)}
        if len(ids)!=1:return None
        groups.append(next(iter(ids)))
    return bool(groups[0]==groups[1])
