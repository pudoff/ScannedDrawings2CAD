"""Bounded CV context proposals, independent of annotations and OCR.

Morphology groups context only. Its masks never replace source pixels or CAD
geometry. A view and a magnified detail remain an unresolved semantic family.
"""
import math

import cv2
import numpy as np

from .family_catalog_v2 import require
from .preprocessing import _regions, _binary

PARAMETERS = {
    'max_pixels':4194304, 'max_cases':16, 'max_total_pixels':24000000,
    'max_candidates':24, 'max_components':20000, 'max_match_targets':12,
    'grid_horizontal_fraction':.25, 'grid_vertical_fraction':.25,
    'grid_min_horizontal':5, 'grid_min_vertical':4, 'grid_min_crossings':14,
    'table_min_crossings':50, 'table_min_horizontal':12,
    'context_close_width_fraction':.035, 'context_close_height_fraction':.015,
    'context_margin_fraction':.008, 'minimum_component_ink':30,
    'minimum_component_area_fraction':.0012, 'maximum_component_area_fraction':.65,
    'content_window':[.06,.065,.98,.975],
    'minimum_component_width_fraction':.02,'minimum_component_height_fraction':.025,
}


def validated_rgb(rgb):
    require(isinstance(rgb,np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3
            and rgb.shape[2] == 3 and 0 < rgb.shape[0]*rgb.shape[1] <= PARAMETERS['max_pixels'],
            'Bounded native-preparation RGB array required')
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)


def grid_features(gray):
    h,w = gray.shape
    ink = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    horizontal = cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(3,math.ceil(w*PARAMETERS['grid_horizontal_fraction']))),np.uint8))
    vertical = cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(3,math.ceil(h*PARAMETERS['grid_vertical_fraction'])),1),np.uint8))
    def components(mask):
        return cv2.connectedComponentsWithStats(mask,connectivity=8)[0]-1
    return {'horizontal':components(horizontal),'vertical':components(vertical),
            'crossings':components(cv2.bitwise_and(horizontal,vertical)),
            'ink_fraction':round(float(np.count_nonzero(ink)/ink.size),8)}


def classify(rgb):
    gray = validated_rgb(rgb); evidence = grid_features(gray)
    if evidence['ink_fraction'] == 0 or evidence['ink_fraction'] > .6:
        role = 'unknown'
    elif (evidence['crossings'] >= PARAMETERS['table_min_crossings']
          and evidence['horizontal'] >= PARAMETERS['table_min_horizontal']):
        role = 'table'
    elif (evidence['horizontal'] >= PARAMETERS['grid_min_horizontal']
          and evidence['vertical'] >= PARAMETERS['grid_min_vertical']
          and evidence['crossings'] >= PARAMETERS['grid_min_crossings']):
        role = 'title_block'
    else:
        role = 'view_or_detail'
    return {'role':role,'evidence':evidence,'score_meaning':'fixed heuristic; not calibrated probability',
            'detail_subtype_resolved':False}


def _proposal(box,role,method,evidence):
    return {'box':list(map(int,box)),'role':role,'method':method,'evidence':evidence,
            'status':'proposal','physical_geometry_approved':False,'production_admitted':False}


def legacy(rgb):
    gray = validated_rgb(rgb)
    rows = _regions(gray,_binary(gray))
    return [_proposal(r['bbox_px'],'title_block','legacy_bottom_right_grid',r['evidence'])
            for r in rows if r['kind'] == 'title_block_grid_candidate']


def candidate(rgb):
    gray = validated_rgb(rgb); h,w = gray.shape
    result = legacy(rgb)
    # A dense global grid is a table proposal, never a full-page fallback.
    global_grid = grid_features(gray)
    if (global_grid['crossings'] >= PARAMETERS['table_min_crossings']
            and global_grid['horizontal'] >= PARAMETERS['table_min_horizontal']):
        ink = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
        ys,xs = np.nonzero(ink)
        if len(xs): result.append(_proposal([xs.min(),ys.min(),xs.max()+1,ys.max()+1],
                                             'table','dense_global_grid',global_grid))
    ink = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    a,b,c,d = PARAMETERS['content_window']
    window = np.zeros_like(ink); window[math.floor(h*b):math.ceil(h*d),math.floor(w*a):math.ceil(w*c)] = 255
    ink &= window
    for r in result:
        if r['role'] == 'title_block':
            x,y,right,bottom = r['box']; ink[y:bottom,x:right] = 0
    # Long frame observations may remain inside the window. Remove only their
    # morphology for grouping, never from the preserved source evidence.
    long_h = cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(3,math.ceil(w*.8))),np.uint8))
    long_v = cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(3,math.ceil(h*.8)),1),np.uint8))
    ink[cv2.dilate(long_h|long_v,np.ones((3,3),np.uint8))>0] = 0
    kw = max(3,math.ceil(w*PARAMETERS['context_close_width_fraction'])|1)
    kh = max(3,math.ceil(h*PARAMETERS['context_close_height_fraction'])|1)
    grouped = cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((kh,kw),np.uint8))
    n,labels,stats,_ = cv2.connectedComponentsWithStats(grouped,connectivity=8)
    require(n-1 <= PARAMETERS['max_components'],'Context component budget exceeded')
    choices = []; mx = math.ceil(w*PARAMETERS['context_margin_fraction']); my = math.ceil(h*PARAMETERS['context_margin_fraction'])
    for i in range(1,n):
        x,y,bw,bh,_ = map(int,stats[i]); area = bw*bh
        if not (PARAMETERS['minimum_component_area_fraction']*w*h <= area <= PARAMETERS['maximum_component_area_fraction']*w*h
                and bw >= w*PARAMETERS['minimum_component_width_fraction'] and bh >= h*PARAMETERS['minimum_component_height_fraction']):continue
        support = int(np.count_nonzero(ink[y:y+bh,x:x+bw][labels[y:y+bh,x:x+bw]==i]))
        if support < PARAMETERS['minimum_component_ink']:continue
        box = [max(0,x-mx),max(0,y-my),min(w,x+bw+mx),min(h,y+bh+my)]
        patch = rgb[box[1]:box[3],box[0]:box[2]]; semantic = classify(patch)
        choices.append((support,box,semantic))
    for support,box,semantic in sorted(choices,key=lambda r:(-r[0],r[1])):
        if len(result) >= PARAMETERS['max_candidates']:break
        result.append(_proposal(box,semantic['role'],'context_morphology',
                                {'observed_ink':support,**semantic['evidence']}))
    return result


def map_box(box,matrix,size):
    """Exact pixel-edge mapping; matrices are separate from centre sampling."""
    x,y,r,b = box; m = np.asarray(matrix,dtype=float)
    require(m.shape == (3,3) and np.isfinite(m).all() and np.array_equal(m[2],[0,0,1]),'Invalid affine edge transform')
    p = np.array([[x,y,1],[r,y,1],[r,b,1],[x,b,1]])@m.T
    return [max(0.,float(p[:,0].min())),max(0.,float(p[:,1].min())),
            min(float(size[0]),float(p[:,0].max())),min(float(size[1]),float(p[:,1].max()))]
