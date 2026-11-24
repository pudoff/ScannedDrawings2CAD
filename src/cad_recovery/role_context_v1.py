"""Bounded source-only neighbourhood descriptors, with no role/label input."""
import math
import numpy as np
from .role_interval_dataset_v1 import FEATURES as BASE_FEATURES

RADII=(24,64,128)
NEIGHBOUR_STATS=('log_count','mean_cos_2angle','mean_sin_2angle','straight_fraction',
                 'branch_fraction','mean_parent_length','mean_relative_width','clipped')
FEATURES=BASE_FEATURES+tuple(f'neighbours_{r}_{s}' for r in RADII for s in NEIGHBOUR_STATS)+(
    'nearest_parallel_in_widths','nearest_perpendicular_in_widths','parallel_fraction_64',
    'perpendicular_fraction_64','orientation_coherence_64','collinear_fraction_64')
POLICY=dict(maximum_intervals_per_patch=4096,maximum_patch_side=1024,
    radii_in_stroke_widths=list(RADII),maximum_radius_px=256,
    straight_chord_minimum=.95,orientation_tolerance_degrees=10,
    collinear_offset_in_widths=2,maximum_nearest_distance_in_widths=128,
    input='all observed source intervals in the patch, including unreviewed intervals',
    unavailable_context='zero neighbour statistics with explicit clipped flags',
    color_or_role_features=False,target_input=False)


def augment(base,points,*,size_px):
    """Returns one feature row per interval in its input order; never reads masks."""
    x=np.asarray(base,np.float64)
    if (x.ndim!=2 or x.shape[1]!=len(BASE_FEATURES) or not np.isfinite(x).all()
            or len(points)!=len(x) or len(x)>POLICY['maximum_intervals_per_patch']):
        raise ValueError('Bounded finite source intervals required')
    if (len(size_px)!=2 or any(type(n)is not int or not 0<n<=1024 for n in size_px)):
        raise ValueError('Bounded native patch dimensions required')
    centers=[];widths=[]
    for row,path in zip(x,points,strict=True):
        p=np.asarray(path,float)
        if (p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all()
                or (p<0).any() or (p>np.array(size_px)-1).any()):
            raise ValueError('Source interval points lie outside patch')
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        if length<=0 or not 0<row[0]<30:raise ValueError('Invalid stroke-normalized source length')
        centers.append(p.mean(0));widths.append(max(1.,length/np.expm1(row[0])))
    if not len(x):return np.empty((0,len(FEATURES)),np.float32)
    centers=np.asarray(centers);widths=np.asarray(widths)
    orientation=x[:,3:5];norm=np.linalg.norm(orientation,axis=1)
    if np.any(abs(norm-1)>1e-5):raise ValueError('Unit doubled source orientations required')
    straight=x[:,1]>=.95;branch=np.maximum(x[:,12],x[:,13])>=3
    threshold=math.cos(math.radians(20));out=[]
    for i,center in enumerate(centers):
        delta=centers-center;distance=np.linalg.norm(delta,axis=1)
        distinct=np.arange(len(x))!=i;dot=orientation@orientation[i]
        parallel=dot>=threshold;perpendicular=dot<=-threshold
        extras=[];near64=None
        for factor in RADII:
            radius=min(256.,factor*widths[i]);near=distinct&(distance<=radius)
            clipped=bool((center-radius<0).any() or (center+radius>np.array(size_px)-1).any())
            if near.any():
                extras.extend([math.log1p(int(near.sum())),*orientation[near].mean(0),
                    float(straight[near].mean()),float(branch[near].mean()),
                    float(x[near,9].mean()),float(x[near,5].mean()),float(clipped)])
            else:extras.extend([0.]*7+[float(clipped)])
            if factor==64:near64=near
        for direction in (parallel,perpendicular):
            selected=distinct&direction
            nearest=float(distance[selected].min()/widths[i]) if selected.any() else 128.
            extras.append(math.log1p(min(128.,nearest)))
        if near64.any():
            theta=.5*math.atan2(orientation[i,1],orientation[i,0])
            normal=np.array([-math.sin(theta),math.cos(theta)])
            collinear=parallel&(np.abs(delta@normal)<=2*widths[i])
            extras.extend([float(parallel[near64].mean()),float(perpendicular[near64].mean()),
                float(np.linalg.norm(orientation[near64].mean(0))),float(collinear[near64].mean())])
        else:extras.extend([0.]*4)
        out.append(np.r_[x[i],extras])
    result=np.asarray(out,np.float32)
    if result.shape!=(len(x),len(FEATURES)) or not np.isfinite(result).all():
        raise ValueError('Invalid context descriptors')
    return result
