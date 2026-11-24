"""Recover an observed straight source interval without painting missing pixels."""
import math
import numpy as np

POLICY={'minimum_sections':24,'maximum_samples':4000000,'maximum_cross_section':256,
        'refits':2,'pixel_radius':math.sqrt(.5),'GT_used':False,'source_pixels_erased':0}


def sections(ink,major,xs,centres,reach,audit):
    data=ink.T if major==0 else ink;found=[]
    for x,centre in zip(xs,centres):
        lo=max(0,math.floor(centre-reach));hi=min(data.shape[1],math.ceil(centre+reach)+1)
        if hi<=lo:continue
        audit['source_support_samples']+=hi-lo
        if audit['source_support_samples']>POLICY['maximum_samples']:
            audit['limits'].append('extension_sample_budget');return None
        a=data[x,lo:hi]>0;change=np.diff(np.r_[False,a,False].astype(int))
        runs=[(a+lo,b+lo-1) for a,b in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)) if a>0 and b<hi-lo]
        if not runs:continue
        runs.sort(key=lambda r:abs((r[0]+r[1])/2-centre))
        if len(runs)>1 and abs((runs[1][0]+runs[1][1])/2-centre)<=reach/2:continue
        a,b=runs[0];found.append((int(x),(a+b)/2,b-a+1))
    return np.asarray(found,float).reshape(-1,3)


def fit_sections(values,normal):
    if len(values)<POLICY['minimum_sections']:return None
    width=float(np.median(values[:,2]));clean=values[np.abs(values[:,2]-width)<=1]
    if len(clean)<POLICY['minimum_sections']:return None
    origin=float(clean[:,0].mean());slope,offset=np.polyfit(clean[:,0]-origin,clean[:,1],1)
    residual=np.abs(clean[:,1]-(slope*(clean[:,0]-origin)+offset))
    if np.quantile(residual,.95)>math.sqrt(2)/normal:return None
    return {'slope':float(slope),'intercept':float(offset-slope*origin),'width':width,
            'sections':len(clean),'residual':float(np.quantile(residual,.95))}


def observed_interval(distance,major,xs,centres,seed,audit):
    audit['source_support_samples']+=len(xs)
    if audit['source_support_samples']>POLICY['maximum_samples']:
        audit['limits'].append('extension_sample_budget');return None
    minor=1-major;q=np.zeros((len(xs),2),int);q[:,major]=xs;q[:,minor]=np.rint(centres).astype(int)
    inside=(q>=0).all(1)&(q[:,0]<distance.shape[1])&(q[:,1]<distance.shape[0]);good=np.zeros(len(xs),bool)
    good[inside]=distance[q[inside,1],q[inside,0]]<=2*POLICY['pixel_radius']
    # Quantisation-sized gaps may belong to one source footprint; longer gaps split it.
    holes=np.diff(np.r_[False,~good,False].astype(int));starts=np.flatnonzero(holes==1);ends=np.flatnonzero(holes==-1)
    for a,b in zip(starts,ends):
        if a>0 and b<len(good) and b-a<=math.ceil(2*math.sqrt(2)):good[a:b]=True
    change=np.diff(np.r_[False,good,False].astype(int))
    for a,b in zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)):
        if xs[a]<=seed<=xs[b-1]:return slice(int(a),int(b))
    return None


def recover(ink,distance,points,width,audit):
    points=np.asarray(points,float);delta=points[1]-points[0];major=int(np.argmax(np.abs(delta)));minor=1-major
    if abs(delta[major])<POLICY['minimum_sections']:return None
    normal=abs(delta[major])/np.linalg.norm(delta);reach=math.ceil((1.5*max(1.,width)+math.sqrt(2))/normal)
    if 2*reach+1>POLICY['maximum_cross_section']:return None
    slope=delta[minor]/delta[major];intercept=points[0,minor]-slope*points[0,major]
    seed_x=np.arange(max(0,math.ceil(min(points[:,major]))),min(ink.shape[1-major],math.floor(max(points[:,major]))+1))
    values=sections(ink,major,seed_x,slope*seed_x+intercept,reach,audit)
    if values is None:return None
    model=fit_sections(values,normal)
    if model is None:return None
    seed=float(np.median(values[:,0]));all_x=np.arange(ink.shape[1-major]);interval=None
    for _ in range(POLICY['refits']):
        centres=model['slope']*all_x+model['intercept']
        interval=observed_interval(distance,major,all_x,centres,seed,audit)
        if interval is None:return None
        xs=all_x[interval]
        if len(xs)<POLICY['minimum_sections']:return None
        values=sections(ink,major,xs,model['slope']*xs+model['intercept'],reach,audit)
        if values is None:return None
        refined=fit_sections(values,normal)
        if refined is None:return None
        model=refined
    # Re-check after the last fit; the new endpoints must still be observed.
    interval=observed_interval(distance,major,all_x,model['slope']*all_x+model['intercept'],seed,audit)
    if interval is None:return None
    bounds=all_x[interval][[0,-1]];output=np.zeros((2,2),float);output[:,major]=bounds
    output[:,minor]=model['slope']*bounds+model['intercept']
    return {'points':output,'width':model['width']*normal,'source_sections':model['sections'],
            'residual_minor_px':model['residual'],'seed_major_px':seed,'source_pixels_changed':0}
