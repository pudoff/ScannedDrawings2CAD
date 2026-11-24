"""Offline component proposals; annotation masks are never classifier inputs."""
from __future__ import annotations
import cv2
import numpy as np

PARAMETERS = {'max_area_strokes2':16.,'max_extent_strokes':12.,
              'max_contrast':.12,'min_background_spread':12./255,
              'min_isolation_strokes':3.,'context_margin_strokes':2.,
              'line_elongation_guard':4.,'max_pixels':1_000_000,'max_components':2048}
VARIANTS=('baseline','context','context_line_guard')


def context_mask(shape, ocr, geometry, arrows):
    result=np.zeros(shape,dtype=np.uint8)
    for item in ocr.get('lines',[]):
        polygon=np.asarray(item['polygon_px'],dtype=float)
        if polygon.shape!=(4,2) or not np.isfinite(polygon).all():raise ValueError('Invalid OCR polygon')
        cv2.fillPoly(result,[np.rint(polygon).astype('int32')],1)
    for item in arrows.get('candidates',[]):
        polygon=np.asarray(item['polygon_px'],dtype=float)
        if polygon.ndim!=2 or polygon.shape[1]!=2 or not np.isfinite(polygon).all():raise ValueError('Invalid arrow polygon')
        cv2.fillPoly(result,[np.rint(polygon).astype('int32')],1)
    for item in geometry.get('candidates',[]):
        if item.get('kind')=='line':
            points=np.asarray(item['points_px'],dtype=float)
            if points.shape!=(2,2) or not np.isfinite(points).all():raise ValueError('Invalid line')
            cv2.line(result,tuple(np.rint(points[0]).astype(int)),tuple(np.rint(points[1]).astype(int)),1,1)
        elif item.get('kind') in ('circle','arc'):
            # Conservative: protect whole circle even if observation is an arc.
            center=np.asarray(item['center_px'],dtype=float);radius=float(item['radius_px'])
            if not np.isfinite(center).all() or not 0<radius<=max(shape)*2:raise ValueError('Invalid circle')
            cv2.circle(result,tuple(np.rint(center).astype(int)),max(1,round(radius)),1,1)
    return result>0


def clean_components(gray, ink, context, variant):
    if variant not in VARIANTS:raise ValueError('Unknown variant')
    if gray.dtype!=np.uint8 or gray.ndim!=2 or ink.dtype!=bool or context.dtype!=bool or gray.shape!=ink.shape or context.shape!=ink.shape:
        raise ValueError('Expected matching grayscale/boolean images')
    if gray.size>PARAMETERS['max_pixels']:raise ValueError('Pixel budget exceeded')
    n,labels,stats,_=cv2.connectedComponentsWithStats(ink.astype('uint8'),connectivity=8)
    if n-1>PARAMETERS['max_components']:raise ValueError('Component budget exceeded')
    distance=cv2.distanceTransform(ink.astype('uint8'),cv2.DIST_L2,5)
    widths=[]
    for x,y,w,h,area in stats[1:]:
        if area>=24 and max(w,h)>=4*max(1,min(w,h)):
            values=distance[y:y+h,x:x+w]
            widths.append(max(1.,2*float(values.max())-1))
    stroke=float(np.median(widths)) if widths else 1.
    radius=max(3,min(24,int(np.ceil(6*stroke))))
    margin=max(1,min(16,int(np.ceil(PARAMETERS['context_margin_strokes']*stroke))))
    nearby=cv2.dilate(context.astype('uint8'),np.ones((2*margin+1,2*margin+1),dtype=np.uint8))>0
    removed=np.zeros(ink.shape,dtype=bool);ledger=[]
    for label,(x,y,w,h,area) in enumerate(stats[1:],1):
        x,y,w,h,area=map(int,(x,y,w,h,area));l=max(0,x-radius);t=max(0,y-radius)
        r=min(ink.shape[1],x+w+radius);b=min(ink.shape[0],y+h+radius)
        patch=labels[t:b,l:r];component=patch==label
        # All components are kept intact: never clip a connected mixed object.
        reasons=[];area_norm=area/stroke**2;extent=max(w,h)/stroke
        if area_norm>PARAMETERS['max_area_strokes2'] or extent>PARAMETERS['max_extent_strokes']:reasons.append('large_or_long')
        if nearby[t:b,l:r][component].any():reasons.append('observation_context')
        _,hierarchy=cv2.findContours(component.astype('uint8'),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None and (hierarchy[0,:,3]>=0).any():reasons.append('possible_loop_or_symbol')
        coordinates=np.column_stack(np.nonzero(component)).astype(float)
        if len(coordinates)>=3:
            eigen=np.linalg.eigvalsh(np.cov(coordinates.T,bias=True))
            elongation=float((eigen[-1]+.25)/(eigen[0]+.25))
        else:elongation=1.
        if variant=='context_line_guard' and elongation>=PARAMETERS['line_elongation_guard']:reasons.append('possible_dash_or_edge')
        background=gray[t:b,l:r][patch==0]
        contrast=None;spread=None;isolation=None
        if len(background)<16:reasons.append('insufficient_paper')
        else:
            dark=gray[t:b,l:r][component]
            contrast=float(max(0,np.median(background)-np.percentile(dark,10))/255)
            spread=float((np.percentile(background,90)-np.percentile(background,10))/255)
            if contrast>PARAMETERS['max_contrast']:reasons.append('visible_ink_contrast')
            if spread<PARAMETERS['min_background_spread']:reasons.append('no_shadow_texture')
        others=(patch!=0)&~component
        if others.any():
            other_distance=cv2.distanceTransform((~others).astype('uint8'),cv2.DIST_L2,5)
            isolation=float(other_distance[component].min()/stroke)
            if isolation<PARAMETERS['min_isolation_strokes']:reasons.append('neighbouring_ink')
        else:isolation=float(radius/stroke)
        # A tiny isolated dot can be a degree/decimal mark. No semantic claim.
        if area_norm<3:reasons.append('possible_small_symbol')
        if variant=='baseline':reasons.append('baseline')
        decision='keep_or_review' if reasons else 'remove_proposal'
        if not reasons:removed[t:b,l:r]|=component
        ledger.append({'component':label,'box':[x,y,x+w,y+h],'area_px':area,
            'area_strokes2':area_norm,'extent_strokes':extent,'contrast':contrast,
            'background_spread':spread,'isolation_strokes':isolation,'elongation':elongation,
            'decision':decision,'reasons':reasons,'estimated_stroke_px':stroke})
    return ink&~removed,removed,ledger
