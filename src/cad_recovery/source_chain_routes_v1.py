"""Local connectivity and observed source-edge routes, never gap filling."""
from collections import defaultdict
import heapq
import math
import numpy as np
import cv2

from .interval_roles_v1 import finite_hit

POLICY={'maximum_local_pixels':40000000,'maximum_route_edge_visits':250000,
        'maximum_route_length_in_strokes':4.,'maximum_route_turn_deg':30.,
        'maximum_intervals':4096,'maximum_source_points':2000000,
        'new_chains_require_two_opposed_heads':True,'source_erasure_allowed':False}


class Routes:
    def __init__(self,intervals,ink):
        self.ink=ink;self.lookup={i['id']:i for i in intervals};self.graph=defaultdict(list)
        self.audit={'local_pixels':0,'route_edge_visits':0,'source_points':0,'recovered_junctions':0,'recovered_feet':0,'limits':[]}
        self.incomplete=False;self.valid={}
        if len(intervals)>4096 or len(self.lookup)!=len(intervals):raise ValueError('Bounded unique source intervals required')
        for item in intervals:
            p=np.asarray(item['points_px'],float);self.audit['source_points']+=len(p)
            if self.audit['source_points']>2000000:self.limit('source_point_budget');return
            if p.ndim!=2 or p.shape[1]!=2 or not len(p) or not np.isfinite(p).all():raise ValueError('Finite source interval required')
            q=np.rint(p).astype(int)
            if (q<0).any() or (q[:,0]>=ink.shape[1]).any() or (q[:,1]>=ink.shape[0]).any():raise ValueError('Source point outside image')
            continuous=bool(np.all(ink[q[:,1],q[:,0]]>0) and (len(q)<2 or np.max(np.abs(np.diff(q,axis=0)))<=1))
            self.valid[item['id']]=continuous
            if len(p)<2 or not continuous:continue
            length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
            a,b=tuple(p[0]),tuple(p[-1])
            self.graph[a].append((b,item['id'],length));self.graph[b].append((a,item['id'],length))
        if self.audit['source_points']>2000000:self.limit('source_point_budget')

    def limit(self,reason):
        if reason not in self.audit['limits']:self.audit['limits'].append(reason)
        self.incomplete=True

    def intersection(self,span,ext,tolerance,shaft,heads):
        old=finite_hit(span,ext['points'],tolerance,self.ink)
        if old is not None:return old,{'method':'legacy_observed_hit'}
        p,q=np.asarray(span,float),np.asarray(ext['points'],float);u,v=p[1]-p[0],q[1]-q[0]
        design=np.c_[u,-v]
        if abs(np.linalg.det(design))<1e-9:return None,None
        t,s=np.linalg.solve(design,q[0]-p[0]);hit=p[0]+t*u
        nearest_head=min(heads,key=lambda h:np.linalg.norm(np.asarray(h['tip_px'])-hit))
        polygon=np.asarray(nearest_head['polygon_px'],float)
        observed_head_extent=float(np.linalg.norm(polygon[0]-polygon[1:].mean(0)))
        # A linear extension can stop at the side of an observed filled head;
        # the missing analytic reach must fit that very head, then pass actual
        # source connectivity. This does not extend a stroke through white pixels.
        if min(t,1-t)*np.linalg.norm(u)<-tolerance or min(s,1-s)*np.linalg.norm(v)<-(observed_head_extent+tolerance):return None,None
        radius=observed_head_extent+max(shaft['width'],ext['width'])+max(h.get('uncertainty_px',0) for h in heads)+2**.5/2
        lo=np.maximum(0,np.floor(hit-radius-1)).astype(int);hi=np.minimum(self.ink.shape[::-1],np.ceil(hit+radius+2)).astype(int)
        if (hi<=lo).any():return None,None
        cost=int(np.prod(hi-lo))*4
        if self.audit['local_pixels']+cost>40000000:self.limit('junction_pixel_budget');return None,None
        self.audit['local_pixels']+=cost
        patch=(self.ink[lo[1]:hi[1],lo[0]:hi[0]]>0).astype(np.uint8)
        _,labels=cv2.connectedComponents(patch,connectivity=8)
        def seeds(track):
            mask=np.zeros_like(patch)
            for key in track['interval_ids']:
                points=np.rint(np.asarray(self.lookup[key]['points_px'])-lo).astype(np.int32)
                cv2.polylines(mask,[points],False,1,1)
            return mask
        a=seeds(shaft);b=seeds(ext)
        for head in heads:cv2.fillConvexPoly(a,np.rint(np.asarray(head['polygon_px'])-lo).astype(np.int32),1)
        left=set(np.unique(labels[(a>0)&(patch>0)]))-{0};right=set(np.unique(labels[(b>0)&(patch>0)]))-{0}
        common=left&right
        if len(common)!=1:return None,None
        yy,xx=np.nonzero(labels==next(iter(common)));points=np.c_[xx,yy]+lo
        nearest=points[np.argmin(np.linalg.norm(points-hit,axis=1))]
        self.audit['recovered_junctions']+=1
        return nearest.astype(float),{'method':'connected_source_ink_in_local_junction','analytic_hit_px':hit.tolist(),
            'observed_hit_px':nearest.tolist(),'observed_head_extent_px':observed_head_extent,
            'roi_xyxy':[*lo.tolist(),*hi.tolist()],'source_pixels':int(patch.sum())}

    def feet(self,ext,boundary,hit,forbidden):
        targets={tuple(np.asarray(self.lookup[key]['points_px'],float)[end])
                 for key in boundary['owners'] if key in self.lookup for end in (0,-1)}
        starts={tuple(np.asarray(self.lookup[key]['points_px'],float)[end]) for key in ext['interval_ids'] for end in (0,-1)}
        width=float(ext['width']);max_length=4*width;u=np.asarray(ext['u'],float);origin=np.asarray(ext['points'][0],float)
        found=[]
        for start in sorted(starts):
            queue=[(0.,start,())];best={start:0.}
            while queue:
                length,node,route=heapq.heappop(queue)
                if length!=best[node]:continue
                if node in targets:
                    if route and np.linalg.norm(np.asarray(node)-hit)>=3*width:
                        found.append((np.asarray(node),list(route),length));self.audit['recovered_feet']+=1
                    continue
                for other,key,extent in self.graph[node]:
                    self.audit['route_edge_visits']+=1
                    if self.audit['route_edge_visits']>250000:self.limit('source_route_edge_budget');return []
                    if key in forbidden or key in ext['interval_ids'] or key in boundary['owners'] or length+extent>max_length:continue
                    item=self.lookup[key];p=np.asarray(item['points_px'],float);delta=np.asarray(other)-np.asarray(node)
                    norm=np.linalg.norm(delta)
                    if norm<=0 or abs(delta@u)/norm<math.cos(math.radians(30)):continue
                    # A short observed connector must stay in the extension's
                    # stroke corridor. No coordinate snapping or virtual edge.
                    normal=np.array([-u[1],u[0]])
                    if np.max(np.abs((p-origin)@normal))>width+2**.5/2:continue
                    distance=length+extent
                    if distance<best.get(other,float('inf')):
                        best[other]=distance;heapq.heappush(queue,(distance,other,route+(key,)))
        unique={tuple(p):(p,route,length) for p,route,length in found}
        return list(unique.values())
