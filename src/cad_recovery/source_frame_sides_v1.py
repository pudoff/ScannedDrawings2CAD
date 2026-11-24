"""An unsupported hairline outline needs an exact closed source-raster witness."""
import math
import numpy as np
from .frame_raster_template_v1 import fit,profiles


def thin_sides(ink,quad):
    yy,xx=np.nonzero(ink);pixels=np.c_[xx,yy];lines=[]
    for a,b in zip(quad,np.roll(quad,-1,axis=0)):
        u=(b-a)/np.linalg.norm(b-a);normal=np.array([-u[1],u[0]]);along=(pixels-a)@u
        points=pixels[(along>=.08*np.linalg.norm(b-a))&(along<=.92*np.linalg.norm(b-a))&
                      (np.abs((pixels-a)@normal)<=2*math.sqrt(2))]
        if len(points)<24:return [],{'complete':True,'fitted':False,'reason':'few_source_side_pixels'}
        centre=points.mean(0);_,_,v=np.linalg.svd(points-centre,full_matrices=False);axis=v[0]
        n=np.array([-axis[1],axis[0]]);error=float(np.quantile(np.abs((points-centre)@n),.95))
        if error>math.sqrt(2):return [],{'complete':True,'fitted':False,'reason':'not_an_observed_hairline'}
        ends=centre+((np.array([a,b])-centre)@axis)[:,None]*axis
        lines.append({'points_native':ends.tolist(),'width_native_px':1.,'error_native_px':error+math.sqrt(.5)})
    result=fit(ink>0,lines)
    audit={k:v for k,v in result.items() if k not in ('mid','safe','possible')}
    if not result['complete'] or not result['fitted']:return [],audit
    sides=[]
    for line in lines:
        counts={'source_samples':0,'limits':[]}
        profile=profiles(ink>0,np.array(line['points_native']),1.,line['error_native_px'],counts)
        if profile is None:return [],{'complete':not counts['limits'],'fitted':False,'reason':'source_profile_not_reproduced'}
        sides.append({'points_overview':line['points_native'],'stroke_width_overview_px':1.,
            'position_uncertainty_overview_px':line['error_native_px'],
            'source_sample_count':len(profile['x']),'observed_fraction':len(profile['x'])/max(1,np.ptp(profile['x'])+1),
            'evidence':'joint_exact_source_hairline_template'})
    return sides,audit
