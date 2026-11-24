"""Thin title rules proposed by Hough and checked against original page pixels."""
import math
import cv2
import numpy as np
from . import page_frame_context_v1 as first,page_frame_context_v2 as previous
from .stroke_roles_v1 import checked_gray,foreground
from .grid_role_revision_v1 import sha

POLICY={**previous.POLICY,'version':'page-frame-context/3.0','maximum_hough_segments':1024,
    'maximum_support_samples':2000000,'maximum_axis_angle_deg':8.,'minimum_source_support':.9}


def observed_pen(local,a,b,horizontal,footprint,audit):
    widths=[]
    for t in (.2,.5,.8):
        point=a+t*(b-a);along=int(round(point[0 if horizontal else 1]))
        profile=local[:,along]>0 if horizontal else local[along]>0
        audit['support_samples']+=len(profile)
        if audit['support_samples']>POLICY['maximum_support_samples']:
            audit['limits'].append('source_support_budget');return None
        changes=np.diff(np.r_[False,profile,False].astype(int))
        runs=list(zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)))
        if not runs:continue
        centre=point[1 if horizontal else 0]
        lo,hi=min(runs,key=lambda r:abs((r[0]+r[1]-1)/2-centre))
        if abs((lo+hi-1)/2-centre)<=(hi-lo)/2+2*footprint:widths.append(hi-lo)
    return float(np.median(widths)) if widths else 1.


def corner_grid(ink,quad,audit):
    options=[]
    distance=cv2.distanceTransform((ink==0).astype(np.uint8),cv2.DIST_L2,5)
    for turn in range(4):
        p=np.roll(quad,-turn,axis=0);w=max(16,round(np.linalg.norm(p[1]-p[0])));h=max(16,round(np.linalg.norm(p[3]-p[0])))
        scale=min(1.,POLICY['maximum_side']/max(w,h));w,h=round(w*scale),round(h*scale)
        audit['warp_pixels']+=w*h
        if audit['warp_pixels']>POLICY['maximum_warp_pixels']:
            audit['limits'].append('warp_pixel_budget');return None
        matrix=cv2.getPerspectiveTransform(p.astype(np.float32),np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32))
        local=cv2.warpPerspective(ink,matrix,(w,h),flags=cv2.INTER_NEAREST)
        inverse=np.linalg.inv(matrix)
        footprint=max(np.linalg.norm(p[1]-p[0])/max(1,w-1),np.linalg.norm(p[3]-p[0])/max(1,h-1))*math.sqrt(.5)
        segments=cv2.HoughLinesP(local,1,np.pi/720,threshold=max(8,round(.05*min(w,h))),
            minLineLength=max(8,round(.05*min(w,h))),maxLineGap=max(2,math.ceil(2*footprint)))
        segments=[] if segments is None else np.asarray(segments).reshape(-1,4).tolist()
        audit['hough_segments']+=len(segments)
        if len(segments)>POLICY['maximum_hough_segments']:
            audit['limits'].append('hough_segment_budget');return None
        groups={'rows':[],'columns':[]}
        for values in sorted(segments):
            a,b=np.array(values,float).reshape(2,2);delta=b-a;length=np.linalg.norm(delta)
            if length<1:continue
            horizontal=abs(delta[1])<=math.sin(math.radians(8))*length
            vertical=abs(delta[0])<=math.sin(math.radians(8))*length
            if horizontal:
                lo,hi=sorted([a[0],b[0]])
                if not .15*w<=hi-lo<=.75*w or max(a[0],b[0])<.95*w or (a[1]+b[1])/2<.55*h:continue
                key='rows';position=float((a[1]+b[1])/2)
            elif vertical:
                if min(a[0],b[0])<.25*w or max(a[1],b[1])<.95*h or abs(delta[1])<.05*h:continue
                key='columns';position=float((a[0]+b[0])/2)
            else:continue
            samples=a+np.linspace(0,1,max(2,math.ceil(length)+1))[:,None]*delta
            audit['support_samples']+=len(samples)
            if audit['support_samples']>POLICY['maximum_support_samples']:
                audit['limits'].append('source_support_budget');return None
            q=np.c_[samples,np.ones(len(samples))]@inverse.T;q=q[:,:2]/q[:,2:]
            indices=np.rint(q).astype(int)
            inside=(indices>=0).all(1)&(indices[:,0]<ink.shape[1])&(indices[:,1]<ink.shape[0])
            if not inside.all():continue
            support=float(np.mean(distance[indices[:,1],indices[:,0]]<=footprint+math.sqrt(.5)))
            if support<POLICY['minimum_source_support']:continue
            pen=observed_pen(local,a,b,horizontal,footprint,audit)
            if pen is None:return None
            x,y=np.floor(np.minimum(a,b)).astype(int);right,bottom=np.ceil(np.maximum(a,b)).astype(int)
            groups[key].append({'position':position,'bbox':[int(x),int(y),int(right-x+1),int(bottom-y+1),int(length)],
                                'source_support':support,'length':float(length),'observed_pen_local_px':pen})
        unique={}
        for key,items in groups.items():
            kept=[]
            for item in sorted(items,key=lambda r:(-r['length'],r['position'])):
                # One thick outer rule is not many independent table columns.
                if all(abs(item['position']-old['position'])>(item['observed_pen_local_px']+old['observed_pen_local_px'])/2+2*footprint for old in kept):kept.append(item)
            unique[key]=kept
        if len(unique['rows'])>=POLICY['corner_grid_rows'] and len(unique['columns'])>=POLICY['corner_grid_columns']:
            options.append({'quarter_turn':turn,'rows':[r['bbox'] for r in unique['rows']],
                'columns':[r['bbox'] for r in unique['columns']],'source_support':unique,
                'supported':True,'width_evidence':'source_corridor_verified_hough_proposals'})
    return max(options,key=lambda o:(len(o['rows'])+len(o['columns']),-o['quarter_turn'])) if options else None


def detect(gray):
    checked_gray(gray);legacy=previous.detect(gray)
    if legacy['frames'] or not legacy['complete']:
        return {**legacy,'v2_frames_exact':True,'thin_fallback_used':False}
    ink=foreground(gray);yy,xx=np.nonzero(ink)
    if not len(xx):return {**legacy,'v2_frames_exact':True,'thin_fallback_used':False}
    audit={'warp_pixels':legacy['audit']['warp_pixels'],'quad_checks':0,'hough_segments':0,'support_samples':0,'limits':[]}
    content=max(1,int(np.ptp(xx)+1)*int(np.ptp(yy)+1))
    contours,_=cv2.findContours(ink,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=8)
    if len(contours)>POLICY['maximum_contours'] or count-1>POLICY['maximum_components']:
        audit['limits'].append('component_or_contour_budget')
    quads=[]
    for contour in sorted(contours,key=lambda c:-abs(cv2.contourArea(c))):
        if abs(cv2.contourArea(contour))<POLICY['minimum_content_area_fraction']*content:continue
        poly=cv2.approxPolyDP(contour,.015*cv2.arcLength(contour,True),True)
        if len(poly)!=4 or not cv2.isContourConvex(poly):continue
        q=first.ordered_quad(poly);e=np.roll(q,-1,axis=0)-q;u=e/np.linalg.norm(e,axis=1)[:,None]
        if np.max(np.abs(np.sum(u*np.roll(u,-1,axis=0),axis=1)))>math.sin(math.radians(8)):continue
        if not any(np.max(np.linalg.norm(q-old,axis=1))<2 for old in quads):quads.append(q)
    for outer in list(quads):
        for inner in first.inset_quads(ink,outer,audit):
            if not any(np.max(np.linalg.norm(inner-old,axis=1))<2 for old in quads):quads.append(inner)
    if len(quads)>POLICY['maximum_quads']:audit['limits'].append('quad_budget')
    frames=[];rejected=[]
    for q in quads[:POLICY['maximum_quads']]:
        if audit['limits']:break
        audit['quad_checks']+=1;w=np.linalg.norm(q[1]-q[0]);h=np.linalg.norm(q[3]-q[0])
        corners=np.clip(np.rint(q).astype(int),[0,0],np.array(ink.shape[::-1])-1)
        attached=set(labels[corners[:,1],corners[:,0]].tolist());views=[]
        for i in range(1,count):
            if i in attached:continue
            x,y,ww,hh,pixels=map(int,stats[i])
            if min(ww,hh)<POLICY['minimum_view_extent_fraction']*min(w,h) or ww*hh>.5*w*h:continue
            if cv2.pointPolygonTest(q.astype(np.float32),(x+ww/2,y+hh/2),False)<0:continue
            views.append({'component':i,'bbox_overview':[x,y,x+ww,y+hh],'ink_pixels':pixels})
        if not views:rejected.append('no_detached_view');continue
        grid=corner_grid(ink,q,audit)
        if not grid:rejected.append('source_corner_grid');continue
        sides=first.observed_sides(ink,q) or previous.recentered_sides(ink,q)
        if not sides:rejected.append('side_sampling');continue
        frames.append({'quad_overview':q.tolist(),'sides':sides,'view_components':views,
            'corner_grid':grid,'frame_semantics_confirmed':False,'evidence':'source_supported_thin_corner_rules'})
    complete=not audit['limits']
    return {'source_sha256':sha(gray),'size_px':list(gray.shape[::-1]),'frames':frames if complete else[],
        'audit':audit,'complete':complete,'v2_frames_exact':True,'thin_fallback_used':True,
        'thin_rejections':rejected,'production_admitted':False,'physical_geometry_approved':False}
