"""Bounded topology observations, with an explicit shared whole-frame prepass.

Raw source pixels and production geometry remain untouched. Digital adjacency
retains 8-connectivity while omitting a diagonal shortcut when an observed
orthogonal two-edge path already exists. Shared tiles partition cached graph
observations; they are not independent tiled thinning.
"""
from copy import deepcopy
import cv2
import numpy as np
from skimage.morphology import skeletonize
from .stroke_roles_v1 import checked_gray,foreground
from .raster_junctions import _circular_runs

POLICY={'version':'junction-topology/1.0','maximum_pixels':4194304,
    'maximum_active_skeleton_pixels':200000,'maximum_critical_pixels':4096,'maximum_nodes':512,
    'graph_connectivity':'8-connected without redundant diagonals','source_erasure_allowed':False,
    'production_admitted':False,'physical_geometry_approved':False}
OFFSETS=((-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1))
POPCOUNT=np.array([i.bit_count() for i in range(256)],np.uint8)


def topology(mask):
    binary=(mask>0).astype(np.uint8)
    return {'components':cv2.connectedComponents(binary,connectivity=8)[0]-1,
        'holes':cv2.connectedComponents(np.pad(1-binary,1,constant_values=1),connectivity=4)[0]-2}


def legacy_nodes_on_skeleton(ink,skeleton):
    """The frozen observation definition, so thinning can be ablated alone."""
    binary=(skeleton>0).astype(np.uint8)
    degree=cv2.filter2D(binary,cv2.CV_16S,np.ones((3,3),np.int16),borderType=cv2.BORDER_CONSTANT)-binary
    nodes=[];raw_clusters=0
    for kind,mask in [('endpoint',(binary>0)&(degree==1)),('junction',(binary>0)&(degree>=3))]:
        count,labels,stats,centers=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
        raw_clusters+=count-1
        if raw_clusters>POLICY['maximum_critical_pixels']:
            return [],{'complete':False,'reason':'critical_cluster_budget'}
        for i in range(1,count):
            x,y,w,h,_=stats[i];p=centers[i]
            if x==0 or y==0 or x+w==ink.shape[1] or y+h==ink.shape[0]:continue
            angles=np.linspace(0,2*np.pi,720,endpoint=False)
            q=np.rint(p+12*np.c_[np.cos(angles),np.sin(angles)]).astype(int)
            inside=(q>=0).all(1)&(q[:,0]<ink.shape[1])&(q[:,1]<ink.shape[0]);branches=None
            if inside.all():branches=len(_circular_runs(ink[q[:,1],q[:,0]]>0))
            nodes.append({'id':'pixel-'+str(len(nodes)),'point_px':p.tolist(),'kind':kind,'degree':1 if kind=='endpoint' else branches})
    if len(nodes)>POLICY['maximum_nodes']:return [],{'complete':False,'reason':'node_budget'}
    return nodes,{'complete':True,'raw_cluster_count':raw_clusters,'ring_radius_input_px':12}


def adjacency_bits(skeleton):
    mask=skeleton>0;height,width=mask.shape;padded=np.pad(mask,1);bits=np.zeros(mask.shape,np.uint8)
    for bit,(dy,dx) in enumerate(OFFSETS):
        neighbor=padded[1+dy:1+dy+height,1+dx:1+dx+width]
        allowed=mask&neighbor
        if dx and dy:
            vertical=padded[1+dy:1+dy+height,1:1+width]
            horizontal=padded[1:1+height,1+dx:1+dx+width]
            allowed &= ~(vertical|horizontal)
        bits[allowed] |= np.uint8(1<<bit)
    return bits


def digital_nodes(skeleton):
    active=skeleton>0;count=int(active.sum())
    if count>POLICY['maximum_active_skeleton_pixels']:
        return [],{'complete':False,'reason':'active_skeleton_budget','active_pixels':count}
    bits=adjacency_bits(skeleton);degree=POPCOUNT[bits];critical=active&(degree!=2)
    critical_count=int(critical.sum())
    if critical_count>POLICY['maximum_critical_pixels']:
        return [],{'complete':False,'reason':'critical_pixel_budget','critical_pixels':critical_count}
    visited=np.zeros(active.shape,bool);nodes=[];transit_cores=0;clipped=0;visits=0
    for y,x in np.argwhere(critical):
        if visited[y,x]:continue
        junction=degree[y,x]>=3;queue=[(int(y),int(x))];visited[y,x]=True;cluster=[]
        while queue:
            cy,cx=queue.pop();cluster.append((cy,cx))
            if not junction:continue
            for bit,(dy,dx) in enumerate(OFFSETS):
                if not int(bits[cy,cx])&(1<<bit):continue
                ny,nx=cy+dy,cx+dx;visits+=1
                if degree[ny,nx]>=3 and not visited[ny,nx]:visited[ny,nx]=True;queue.append((ny,nx))
        members=set(cluster);boundary=[];internal=0
        for cy,cx in cluster:
            for bit,(dy,dx) in enumerate(OFFSETS):
                if not int(bits[cy,cx])&(1<<bit):continue
                other=(cy+dy,cx+dx)
                if other in members:internal+=1
                else:boundary.append(((cy,cx),other))
        arity=len(boundary)
        if arity==2:transit_cores+=1;continue
        if any(cy in (0,active.shape[0]-1) or cx in (0,active.shape[1]-1) for cy,cx in cluster):
            clipped+=1;continue
        points=np.array([[cx,cy] for cy,cx in cluster],float);point=points.mean(0)
        nodes.append({'id':'digital-'+str(len(nodes)),'point_px':point.tolist(),
            'kind':'endpoint' if arity==1 else 'isolated' if arity==0 else 'junction','degree':arity,
            'source_cluster_px':points.tolist(),'boundary_edge_count':arity,
            'internal_cycle_rank':max(0,internal//2-len(cluster)+1)})
    if len(nodes)>POLICY['maximum_nodes']:return [],{'complete':False,'reason':'node_budget','raw_nodes':len(nodes)}
    return nodes,{'complete':True,'active_pixels':count,'critical_pixels':critical_count,
        'critical_edge_visits':visits,'transit_cores':transit_cores,'border_clipped_clusters':clipped,
        'only_observed_neighbor_edges':True,'diagonal_removal_has_observed_orthogonal_witness':True}


def prepare_from_mask(ink):
    if not isinstance(ink,np.ndarray) or ink.ndim!=2 or ink.dtype!=np.uint8 or not 0<ink.size<=POLICY['maximum_pixels']:
        raise ValueError('Bounded uint8 source mask required')
    if not np.isin(ink,[0,255]).all():raise ValueError('Binary source mask required')
    skeleton=skeletonize(ink>0,method='zhang').astype(np.uint8)*255
    legacy,old_audit=legacy_nodes_on_skeleton(ink,skeleton)
    nodes,audit=digital_nodes(skeleton)
    before=topology(ink);after=topology(skeleton);subset=not bool(np.any((skeleton>0)&(ink==0)))
    return {'nodes':nodes,'legacy_nodes':legacy,'skeleton':skeleton,'ink':ink,
        'audit':audit,'legacy_audit':old_audit,'source_topology':before,'skeleton_topology':after,
        'source_subset':subset,'topology_preserved':before==after,'complete':audit['complete'] and old_audit['complete'] and subset,
        'global_prepass_calls':1,'production_admitted':False,'physical_geometry_approved':False}


def prepare(gray):
    checked_gray(gray)
    return prepare_from_mask(foreground(gray))


def shared_tiles(packet,rows=1,columns=2,reverse=False):
    if rows not in (1,2) or columns not in (1,2):raise ValueError('Bounded 1x1,1x2,2x1,2x2 views required')
    if not packet['complete']:return [],{'complete':False,'reason':'global_prepass_incomplete'}
    height,width=packet['ink'].shape;tiles=[]
    for row in range(rows):
        for column in range(columns):
            tiles.append([column*width//columns,row*height//rows,(column+1)*width//columns,(row+1)*height//rows])
    if reverse:tiles.reverse()
    output=[];owners=[]
    for index,(x,y,right,bottom) in enumerate(tiles):
        owned=[n for n in packet['nodes'] if x<=n['point_px'][0]<right and y<=n['point_px'][1]<bottom]
        output+=deepcopy(owned);owners.append({'tile':index,'box':[x,y,right,bottom],'node_ids':[n['id'] for n in owned]})
    output.sort(key=lambda n:int(n['id'].split('-')[-1]))
    return output,{'complete':len(output)==len(packet['nodes']),'ownership':owners,
        'shared_global_skeleton_and_graph':True,'local_thinning_calls':0,'global_prepass_calls_reused':1,
        'out_of_core_processing':False}


def independent_tiles(packet):
    """Diagnostic local thinning on crops of the common mask, with the old halo."""
    height,width=packet['ink'].shape;split=width//2;output=[];audits=[]
    for lo,hi in ((0,min(width,split+32)),(max(0,split-32),width)):
        local=prepare_from_mask(packet['ink'][:,lo:hi].copy());audits.append(local['audit'])
        for n in local['nodes']:
            point=np.asarray(n['point_px'])+[lo,0]
            if (lo==0 and point[0]<split) or (lo>0 and point[0]>=split):
                output.append({**n,'id':'independent-'+str(len(output)),'point_px':point.tolist(),
                    'source_cluster_px':(np.asarray(n['source_cluster_px'])+[lo,0]).tolist(),'tile_input_offset_px':[lo,0]})
    return output,{'complete':all(a['complete'] for a in audits),'tiles':audits,
        'local_thinning_calls':2,'shared_mask_only':True,'halo_input_px':32}
