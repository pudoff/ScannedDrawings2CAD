"""Preserve v3 frames and check native source fragments for unsupported pages."""
import math
import cv2
import numpy as np
from . import page_frame_context_v1 as first,page_frame_context_v2 as second,page_frame_context_v3 as previous
from .source_corner_rules_v1 import segments,corner_grid
from .source_frame_sides_v1 import thin_sides
from .stroke_roles_v1 import checked_gray,foreground
from .grid_role_revision_v1 import sha

POLICY={**previous.POLICY,'version':'page-frame-context/4.0','maximum_source_pixels':4194304}


def detect(gray):
    checked_gray(gray);legacy=previous.detect(gray)
    if legacy['frames'] or not legacy['complete']:return legacy
    ink=foreground(gray);yy,xx=np.nonzero(ink)
    if not len(xx):return legacy
    audit={'warp_pixels':legacy['audit']['warp_pixels'],'quad_checks':0,'source_segments':0,'source_support_samples':0,'limits':[]}
    if ink.size>POLICY['maximum_source_pixels']:
        return {**legacy,'complete':False,'audit':{**audit,'limits':['source_pixel_budget']}}
    source_segments=segments(ink,audit);distance=cv2.distanceTransform((ink==0).astype(np.uint8),cv2.DIST_L2,5)
    content=max(1,int(np.ptp(xx)+1)*int(np.ptp(yy)+1));contours,_=cv2.findContours(ink,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=8)
    if len(contours)>POLICY['maximum_contours'] or count-1>POLICY['maximum_components']:audit['limits'].append('component_or_contour_budget')
    quads=[]
    for contour in sorted(contours,key=lambda c:-abs(cv2.contourArea(c))):
        if abs(cv2.contourArea(contour))<POLICY['minimum_content_area_fraction']*content:continue
        poly=cv2.approxPolyDP(contour,.015*cv2.arcLength(contour,True),True)
        if len(poly)!=4 or not cv2.isContourConvex(poly):continue
        q=first.ordered_quad(poly);e=np.roll(q,-1,axis=0)-q;u=e/np.linalg.norm(e,axis=1)[:,None]
        if np.max(np.abs(np.sum(u*np.roll(u,-1,axis=0),axis=1)))>math.sin(math.radians(8)):continue
        if not any(np.max(np.linalg.norm(q-old,axis=1))<2 for old in quads):quads.append(q)
    for outer in list(quads):
        if audit['limits']:break
        for inner in first.inset_quads(ink,outer,audit):
            if not any(np.max(np.linalg.norm(inner-old,axis=1))<2 for old in quads):quads.append(inner)
    if len(quads)>POLICY['maximum_quads']:audit['limits'].append('quad_budget')
    frames=[];rejected=[];hairlines=[]
    for q in quads[:POLICY['maximum_quads']]:
        if audit['limits']:break
        audit['quad_checks']+=1;w=np.linalg.norm(q[1]-q[0]);h=np.linalg.norm(q[3]-q[0])
        corners=np.clip(np.rint(q).astype(int),[0,0],np.array(ink.shape[::-1])-1);attached=set(labels[corners[:,1],corners[:,0]].tolist());views=[]
        for i in range(1,count):
            if i in attached:continue
            x,y,ww,hh,pixels=map(int,stats[i])
            if min(ww,hh)<POLICY['minimum_view_extent_fraction']*min(w,h) or ww*hh>.5*w*h:continue
            if cv2.pointPolygonTest(q.astype(np.float32),(x+ww/2,y+hh/2),False)<0:continue
            views.append({'component':i,'bbox_overview':[x,y,x+ww,y+hh],'ink_pixels':pixels})
        if not views:rejected.append({'quad':q.tolist(),'reason':'no_detached_view'});continue
        grid=corner_grid(ink,q,source_segments,distance,audit)
        if not grid:rejected.append({'quad':q.tolist(),'reason':'native_source_corner_grid'});continue
        sides=first.observed_sides(ink,q) or second.recentered_sides(ink,q)
        if not sides:
            sides,proof=thin_sides(ink,q);hairlines.append({'quad':q.tolist(),'proof':proof})
            if not proof['complete']:audit['limits'].append('thin_side_template_budget');break
        if not sides:rejected.append({'quad':q.tolist(),'reason':'source_side_sampling'});continue
        frames.append({'quad_overview':q.tolist(),'sides':sides,'view_components':views,'corner_grid':grid,
            'frame_semantics_confirmed':False,'evidence':'native_source_rules_and_observed_sides'})
    complete=not audit['limits']
    return {'source_sha256':sha(gray),'size_px':list(gray.shape[::-1]),'frames':frames if complete else[],
        'audit':audit,'complete':complete,'source_fallback_used':True,'source_rejections':rejected,
        'hairline_side_proofs':hairlines,'production_admitted':False,'physical_geometry_approved':False}
