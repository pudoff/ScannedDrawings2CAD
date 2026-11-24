"""Source cross-sections may propose an axis before the heuristic taper gate."""
import math
import cv2
import numpy as np

from .local_head_axis_profiles_v1 import Profiles, signature
from .local_head_axis_profiles_v1 import prepare as first_prepare, finish as first_finish
from .sparse_local_heads_v1 import runs


def prepare(base,unit,length,width,profiles):
    prepared,reason=first_prepare(base,unit,length,width,profiles)
    if reason not in ('no_coarse_source_taper','no_near_base_source_run'):return prepared,reason
    normal=np.array([-unit[1],unit[0]])
    along=np.array([-.5,-2.,-4.,-6.]);radius=min(16,math.ceil(max(4.,.75*width+2)))
    across=np.arange(-radius,radius+1,dtype=float)
    if not profiles.charge(len(along)*len(across),'cheap'):return None,'profile_sample_budget'
    points=base+along[:,None,None]*unit+across[None,:,None]*normal
    q=np.rint(points).astype(int)
    inside=(q>=0).all(2)&(q[:,:,0]<profiles.mask.shape[1])&(q[:,:,1]<profiles.mask.shape[0])
    ink=np.zeros(inside.shape,bool);ink[inside]=profiles.mask[q[:,:,1][inside],q[:,:,0][inside]]>0
    coverage=cv2.remap(profiles.field,points[:,:,0].astype(np.float32),points[:,:,1].astype(np.float32),
                       cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    stations=[]
    for i,row in enumerate(ink):
        spans=list(runs(row))
        if not spans:continue
        a,b=min(spans,key=lambda ab:abs((across[ab[0]]+across[ab[1]-1])/2))
        lo,hi=max(0,a-1),min(len(across),b+1);mass=float(coverage[i,lo:hi].sum())
        if mass<=0:continue
        center=float(coverage[i,lo:hi]@across[lo:hi]/mass);observed_width=b-a
        if (not inside[i,a:b].all() or a==0 or b==len(across) or observed_width>width*.8
            or abs(center)>radius-1):continue
        stations.append((float(along[i]),center,observed_width))
    if len(stations)<3 or np.ptp([s[0] for s in stations])<3.5:return None,reason
    values=np.array(stations);slope,intercept=np.polyfit(values[:,0],values[:,1],1)
    if abs(slope)>math.tan(math.radians(20)) or abs(intercept)>max(2.,float(np.median(values[:,2]))):
        return None,'pre_taper_source_axis_unstable'
    adjusted=unit+slope*normal;adjusted/=np.linalg.norm(adjusted)
    profiles.axis_proposals+=1
    prepared,reason=first_prepare(base+intercept*normal,adjusted,length,width,profiles)
    if prepared is not None:
        profiles.recovered_axes[signature(prepared)]={
            'method':'sparse_source_sections_before_coarse_gate_then_full_shaft_validation',
            'preliminary_run_extent_px':float(np.ptp(values[:,0])),
            'source_section_stations_px':values[:,0].tolist(),'seed_direction':unit.tolist(),
            'proposed_direction':adjusted.tolist(),'final_shaft_extent_px':float(prepared[4][3]),
            'final_shaft_width_px':float(prepared[4][2]),'full_shaft_checks_unchanged':True}
    return prepared,reason


def finish(prepared,profiles,text_polygons):
    head,reason=first_finish(prepared,profiles,text_polygons)
    if head is not None:head['evidence']['method']='local-head-context/2.0'
    return head,reason
