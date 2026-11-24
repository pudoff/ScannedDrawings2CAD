"""Verify novel geometry-chain proposals against pixels, leaving v3 untouched."""
from copy import deepcopy
import math
import cv2
import numpy as np
from . import dimension_support_v2 as previous
from .dimension_support_v1 import PixelSupport,_demote,v3

ARMS=previous.ARMS
POLICY={'version':'dimension-support-verified/1.0','base_policy':previous.POLICY,
        'face_ink_fraction':.85,'carrier_ink_fraction':.85,'arrow_ink_fraction':.65,
        'maximum_arrow_mask_pixels':50000,'maximum_arrow_samples':1000,
        'physical_geometry_approved':False,'production_admitted':False}


def _arrow_ink(pixels,arrow):
    poly=np.array(arrow['polygon_px']);lo=np.floor(poly.min(axis=0)).astype(int);hi=np.ceil(poly.max(axis=0)).astype(int)
    w,h=hi-lo+1
    if w*h>POLICY['maximum_arrow_mask_pixels']:raise v3._SearchLimit()
    mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(mask,np.rint(poly-lo).astype(np.int32),1)
    y,x=np.where(mask);step=max(1,math.ceil(len(x)/POLICY['maximum_arrow_samples']))
    points=np.c_[x[::step]+lo[0],y[::step]+lo[1]]
    if pixels.samples+len(points)>previous.base.POLICY['maximum_pixel_samples']:raise v3._SearchLimit()
    pixels.samples+=len(points);height,width=pixels.distance.shape
    inside=(points[:,0]>=0)&(points[:,0]<width)&(points[:,1]>=0)&(points[:,1]<height)
    ink=np.zeros(len(points),bool)
    ink[inside]=pixels.distance[points[inside,1],points[inside,0]]<=.5
    return float(ink.mean()) if len(ink) else 0.


def propose(source_image,*,geometry,ocr,arrows,baseline,arm='all'):
    result=previous.propose(source_image,geometry=geometry,ocr=ocr,arrows=arrows,baseline=baseline,arm=arm)
    result['algorithm_version']='dimension-support-verified-v1';result['policy']=deepcopy(POLICY)
    pixels=PixelSupport(source_image);geo={g['id']:g for g in geometry['candidates']}
    arrow_by={a['id']:a for a in arrows['candidates']}
    faces={f['id']:f for f in (result.get('contour_topology') or {}).get('faces',[])}
    events=[];complete=True;cache={}
    def segment(points):
        key=tuple(tuple(p) for p in points)
        if key not in cache:cache[key]=pixels.segment(*np.array(points,dtype=float))
        return cache[key]
    for row in result['associations']:
        if row['status']!='proposed' or not row.get('support_version'):continue
        fractions={'face':[],'lines':{},'arrows':{}};valid=True
        try:
            for face_id in row['evidence'].get('contour_face_ids',[]):
                for edge in faces[face_id]['boundaries']:
                    fraction=segment(edge['points_px']);fractions['face'].append(fraction)
                    valid &= fraction>=POLICY['face_ink_fraction']
            for gid in set(row['dimension_line_candidate_ids']+row['extension_line_candidate_ids']):
                if geo[gid]['kind']!='line':raise ValueError('A chain carrier must be a line')
                fraction=segment(geo[gid]['points_px']);fractions['lines'][gid]=fraction
                valid &= fraction>=POLICY['carrier_ink_fraction']
            for aid in row['arrow_candidate_ids']:
                fraction=_arrow_ink(pixels,arrow_by[aid]);fractions['arrows'][aid]=fraction
                valid &= fraction>=POLICY['arrow_ink_fraction']
        except v3._SearchLimit:
            complete=False;valid=False;fractions['budget_exhausted']=True
        row['evidence']['observed_chain_pixels']=fractions
        if not valid:_demote(row,'observed_chain_pixel_support_missing_or_incomplete')
        events.append({'association_id':row['id'],'supported':bool(valid),'fractions':fractions})
    if not complete:
        result['complete']=False
        for row in result['associations']:
            if row.get('support_version'):_demote(row,'page_pixel_verification_incomplete')
    result['pixel_verification']={'complete':complete,'samples':pixels.samples,'events':events}
    return result
