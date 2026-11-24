"""Source-validated curve change points, retaining the frozen v2 fit gates."""
import math
import numpy as np
from .cad_primitive_fit_v2 import Fitter as Previous, POLICY as PRIOR
from .stroke_curve_control_v1 import distance_to_polyline

POLICY={**PRIOR,'version':'cad-primitive-fit/3.0','maximum_change_point_positions':128,
        'curve_change_requires_lower_source_SSE':True}


def _squared_error(points,model):
    if model is None:return float('inf')
    if model['kind']=='line':d=distance_to_polyline(points,model['points_px'])
    elif model['kind']=='arc':d=np.linalg.norm(points-model['center_px'],axis=1)-model['radius_px']
    else:return float('inf')
    return float(d@d)


class Fitter(Previous):
    def refine_curve_change(self,parts):
        parts=super().refine_curve_change(parts);out=[];index=0
        while index<len(parts):
            best=None
            if index+2<len(parts):
                a,b,c=parts[index:index+3];ma,mb,mc=[o['model'] for o in (a,b,c)]
                eligible=(ma and mc and ma['kind']==mc['kind']=='arc'
                          and (mb is None or mb['kind']=='line' or mb['kind']=='arc' and not mb.get('locally_stable',True)))
                if eligible:
                    points=np.r_[a['points'],b['points'][1:],c['points'][1:]]
                    lo=len(a['points'])-1;hi=lo+len(b['points'])-1
                    baseline=sum(_squared_error(o['points'],o['model']) for o in (a,b,c))
                    cuts=np.unique(np.linspace(lo,hi,min(hi-lo+1,POLICY['maximum_change_point_positions'])).astype(int))
                    for cut in cuts:
                        if self.incomplete:break
                        left=self.fit(points[:cut+1]);right=self.fit(points[cut:])
                        if not all(m and m['kind']=='arc' and m.get('locally_stable',True) for m in (left,right)):continue
                        # This refinement removes a raster-ambiguous flat spot,
                        # not a true corner between two different curve pieces.
                        u=points[cut]-left['center_px'];v=points[cut]-right['center_px']
                        if abs(float(u@v))/(np.linalg.norm(u)*np.linalg.norm(v))<math.cos(math.radians(PRIOR['tangent_tolerance_deg'])):continue
                        if np.linalg.norm(np.asarray(left['center_px'])-right['center_px'])<=PRIOR['family_parameter_ambiguity_px']:continue
                        score=_squared_error(points[:cut+1],left)+_squared_error(points[cut:],right)
                        if score>=baseline-1e-9:continue
                        if best is None or score<best[0]:best=(score,cut,left,right,points)
            if best is None:out.append(parts[index]);index+=1;continue
            score,cut,left,right,p=best
            for m in (left,right):m['parameter_evidence']='joint_source_curve_change_lower_SSE'
            out.extend([{'points':p[:cut+1],'model':left},{'points':p[cut:],'model':right}]);index+=3
        return out
