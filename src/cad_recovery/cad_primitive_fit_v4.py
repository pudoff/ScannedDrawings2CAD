"""Joint LINE corner localization within the existing source fit uncertainty."""
import math
import numpy as np
from .cad_primitive_fit_v3 import Fitter as Previous,POLICY as PRIOR,_squared_error

POLICY={**PRIOR,'version':'cad-primitive-fit/4.0','line_corner_search_radius':'previous_max_fit_error'}


class Fitter(Previous):
    def refine_curve_change(self,parts):
        parts=super().refine_curve_change(parts);output=[];index=0
        while index<len(parts):
            best=None
            if index+1<len(parts):
                a,b=parts[index:index+2];ma,mb=a['model'],b['model']
                eligible=(any(m and m['kind']=='line' for m in (ma,mb))
                          and all(m is None or m['kind']=='line' for m in (ma,mb)))
                if eligible:
                    points=np.r_[a['points'],b['points'][1:]];node=len(a['points'])-1
                    cuts=np.flatnonzero(np.linalg.norm(points-points[node],axis=1)<=PRIOR['max_px'])
                    cuts=cuts[(cuts>=2)&(cuts<len(points)-2)]
                    cuts=cuts[np.linspace(0,len(cuts)-1,min(len(cuts),PRIOR['maximum_change_point_positions'])).astype(int)] if len(cuts) else []
                    baseline=_squared_error(a['points'],ma)+_squared_error(b['points'],mb)
                    for cut in cuts:
                        if self.incomplete:break
                        left=self.fit(points[:cut+1]);right=self.fit(points[cut:])
                        if not all(m and m['kind']=='line' for m in (left,right)):continue
                        u=points[0]-points[cut];v=points[-1]-points[cut]
                        if abs(float(u@v))/(np.linalg.norm(u)*np.linalg.norm(v))>=math.cos(math.radians(PRIOR['tangent_tolerance_deg'])):continue
                        score=_squared_error(points[:cut+1],left)+_squared_error(points[cut:],right)
                        if score>=baseline-1e-9:continue
                        if best is None or score<best[0]:best=(score,cut,left,right,points)
            if best is None:output.append(parts[index]);index+=1;continue
            _,cut,left,right,p=best
            output.extend([{'points':p[:cut+1],'model':left},{'points':p[cut:],'model':right}]);index+=2
        return output
