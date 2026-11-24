"""Bounded joint integer-corner stroke hypotheses checked on original pixels."""
import hashlib
import itertools
import math
import cv2
import numpy as np
from . import native_frame_v1 as old

POLICY={'version':'frame-raster-template/1.0','maximum_raster_pixels':1000000000,
    'maximum_joint_models':512,'maximum_unique_masks':64,'minimum_sections':24,
    'model':'integer endpoints, closed four-side LINE_8 strokes; observed-source consistency only',
    'production_admitted':False,'physical_geometry_approved':False}


def profiles(ink,points,width,error,audit):
    delta=points[1]-points[0];major=int(np.argmax(np.abs(delta)));minor=1-major
    if abs(delta[major])<1:return None
    data=ink.T if major==0 else ink
    normal=abs(delta[major])/np.linalg.norm(delta)
    reach=max(4,math.ceil((1.5*width+error+2)/normal))
    if 2*reach+1>old.POLICY['maximum_cross_section_pixels']:return None
    slope=delta[minor]/delta[major];intercept=points[0,minor]-slope*points[0,major]
    start=max(0,math.floor(min(points[:,major])-width));stop=min(len(data),math.ceil(max(points[:,major])+width)+1)
    columns=np.arange(start,stop);sections={}
    for x in columns:
        centre=slope*x+intercept;low=max(0,math.floor(centre-reach));high=min(data.shape[1],math.ceil(centre+reach)+1)
        if high<=low:continue
        audit['source_samples']+=high-low
        if audit['source_samples']>old.POLICY['maximum_source_samples']:
            audit['limits'].append('source_sample_budget');return None
        a=data[x,low:high];change=np.diff(np.r_[False,a,False].astype(int))
        runs=[(lo+low,hi+low-1) for lo,hi in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)) if lo>0 and hi<high-low]
        if not runs:continue
        lo,hi=min(runs,key=lambda r:abs((r[0]+r[1])/2-centre))
        if abs((lo+hi)/2-centre)>(width/2+error+1)/normal:continue
        sections[int(x)]=(lo,hi)
    if len(sections)<POLICY['minimum_sections']:return None
    median=float(np.median([b-a+1 for a,b in sections.values()]))
    valid=np.array([int(x in sections and median-1<=sections[x][1]-sections[x][0]+1<=median+1) for x in columns],np.uint8)
    contaminated=(valid==0).astype(np.uint8)[None,:]
    valid=cv2.dilate(contaminated,cv2.getStructuringElement(cv2.MORPH_RECT,(2*math.ceil(median)+1,1)))[0]==0
    xx=columns[valid]
    if len(xx)<POLICY['minimum_sections']:return None
    return {'major':major,'x':xx,'low':np.array([sections[x][0] for x in xx]),
            'high':np.array([sections[x][1] for x in xx]),'width_across':median}


def corners_from_sides(sides):
    output=[]
    for i in range(4):
        a,b=np.asarray(sides[i-1]['points_native'],float);c,d=np.asarray(sides[i]['points_native'],float)
        u=b-a;v=d-c
        if abs(np.linalg.det(np.c_[u,-v]))<1e-6:return None
        t,_=np.linalg.solve(np.c_[u,-v],c-a);output.append(a+t*u)
    return np.array(output)


def match_side(ink,a,b,pen,profile,audit):
    margin=math.ceil(pen/2)+2
    lo=np.maximum(0,np.minimum(a,b)-margin);hi=np.minimum(ink.shape[::-1],np.maximum(a,b)+margin+1)
    if np.any(hi<=lo):return False
    area=int(np.prod(hi-lo));audit['raster_pixels']+=area
    if audit['raster_pixels']>POLICY['maximum_raster_pixels']:
        audit['limits'].append('template_raster_budget');return False
    image=np.zeros(tuple((hi-lo)[::-1]),np.uint8)
    cv2.line(image,tuple(a-lo),tuple(b-lo),255,pen,cv2.LINE_8)
    if np.any((image>0)&~ink[lo[1]:hi[1],lo[0]:hi[0]]):return False
    major=profile['major'];minor=1-major;data=image.T if major==0 else image
    xx=profile['x']-lo[major]
    if (xx<0).any() or (xx>=len(data)).any():return False
    rows=data[xx]>0
    if not rows.any(1).all():return False
    low=np.argmax(rows,axis=1)+lo[minor]
    high=rows.shape[1]-1-np.argmax(rows[:,::-1],axis=1)+lo[minor]
    return bool(np.array_equal(low,profile['low']) and np.array_equal(high,profile['high']))


def fit(ink,sides):
    audit={'source_samples':0,'raster_pixels':0,'side_candidates':0,'joint_models':0,'limits':[]}
    empty={'complete':True,'fitted':False,'audit':audit,'reason':'source_model_unavailable'}
    if len(sides)!=4:return empty
    corners=corners_from_sides(sides)
    if corners is None or not np.isfinite(corners).all():return empty
    choices=[]
    for point in corners:
        values=sorted(set(itertools.product(*[(math.floor(v),math.ceil(v)) for v in point])))
        values=[np.array(v,int) for v in values if np.all(np.array(v)>=0) and np.all(np.array(v)<ink.shape[::-1])]
        if not values:return {**empty,'reason':'whole_frame_outside_bound_source'}
        choices.append(values)
    feasible=[]
    for i,side in enumerate(sides):
        p=np.asarray(side['points_native'],float);width=side['width_native_px'];error=side['error_native_px']
        profile=profiles(ink,p,width,error,audit)
        if profile is None:return {**empty,'reason':'uncontaminated_sections_unavailable','complete':not audit['limits']}
        pens=range(max(1,math.floor(width-2*math.sqrt(2))),max(2,math.ceil(width+math.sqrt(2)))+1)
        models=[]
        for ia,a in enumerate(choices[i]):
            for ib,b in enumerate(choices[(i+1)%4]):
                for pen in pens:
                    audit['side_candidates']+=1
                    if match_side(ink,a,b,pen,profile,audit):models.append({'a':ia,'b':ib,'pen':pen})
                    if audit['limits']:return {**empty,'reason':'template_search_budget','complete':False}
        feasible.append(models)
        if not models:return {**empty,'reason':'no_exact_source_side_template'}
    solutions=[]
    for a in feasible[0]:
        for b in feasible[1]:
            if a['b']!=b['a'] or a['pen']!=b['pen']:continue
            for c in feasible[2]:
                if b['b']!=c['a'] or a['pen']!=c['pen']:continue
                for d in feasible[3]:
                    if c['b']!=d['a'] or d['b']!=a['a'] or d['pen']!=a['pen']:continue
                    audit['joint_models']+=1
                    if audit['joint_models']>POLICY['maximum_joint_models']:
                        audit['limits'].append('joint_model_budget');return {**empty,'complete':False}
                    q=np.array([choices[0][a['a']],choices[1][a['b']],choices[2][b['b']],choices[3][c['b']]])
                    cost=float(np.sum((q-corners)**2))
                    solutions.append({'quad':q.tolist(),'pen':a['pen'],'source_corner_residual':cost})
    if not solutions:return {**empty,'reason':'no_joint_closed_frame'}
    unique={};best=None;safe=np.ones_like(ink,bool);possible=np.zeros_like(ink,bool)
    for model in sorted(solutions,key=lambda m:(m['source_corner_residual'],m['pen'],m['quad'])):
        image=np.zeros(ink.shape,np.uint8);cv2.polylines(image,[np.array(model['quad'],np.int32)],True,255,model['pen'],cv2.LINE_8)
        mask=image>0
        if np.any(mask&~ink):continue
        key=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        if key in unique:continue
        if len(unique)>=POLICY['maximum_unique_masks']:
            audit['limits'].append('unique_mask_budget');return {**empty,'complete':False}
        unique[key]=model;safe &= mask;possible |= mask
        if best is None:best=mask.copy()
    if best is None:return {**empty,'reason':'joint_template_not_contained_in_source'}
    return {'complete':True,'fitted':True,'mid':best,'safe':safe,'possible':possible,
        'models':list(unique.values()),'unique_source_masks':len(unique),'audit':audit,
        'corner_points_source':corners.tolist(),'GT_used':False,'native_prediction_transfer':False}
