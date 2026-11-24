"""Recover independent straight context hidden in an unstable flat arc fit."""
import math
import numpy as np
from .cad_primitive_fit_v4 import Fitter as Previous,POLICY as PRIOR
from .cad_primitive_fit_v2 import checked

POLICY={**PRIOR,'version':'cad-primitive-fit/5.0',
    'dominant_straight_tail_fraction':.5,'straight_tail_error':'half_diagonal_of_source_pixel'}


class Fitter(Previous):
    def merge(self,parts):
        # An independently validated straight tail must not be swallowed again
        # by the very unstable circle hypothesis it was extracted from.
        for _ in range(12):
            output=[];index=0;changed=False
            while index<len(parts):
                if index+1<len(parts):
                    a,b=parts[index:index+2];ma,mb=a['model'],b['model']
                    locked=any(m and m.get('independent_straight_tail') for m in (ma,mb))
                    weak=lambda m:m and m['kind']=='line' and m['fit_max_px']>math.sqrt(.5)
                    compatible=(ma is None or mb is None or ma['kind']==mb['kind'] or weak(ma) or weak(mb))
                    if locked:compatible=all(m and m['kind']=='line' for m in (ma,mb))
                    if compatible and np.array_equal(a['points'][-1],b['points'][0]):
                        p=np.r_[a['points'],b['points'][1:]];model=self.fit(p,np.array_equal(p[0],p[-1]))
                        if model and (not locked or model['kind']=='line'):
                            if locked:model['independent_straight_tail']=True
                            output.append({'points':p,'model':model});index+=2;changed=True;continue
                output.append(parts[index]);index+=1
            parts=output
            if not changed or self.incomplete:break
        return parts

    def refine_tangent(self,parts):
        expanded=[]
        for index,item in enumerate(parts):
            model=item['model'];best=None
            opposite=(parts[-1] if index==0 else parts[0])['model']
            two_sided_context=(len(parts)>=3 and index in (0,len(parts)-1)
                               and opposite and opposite['kind']=='line')
            if two_sided_context and model and model['kind']=='arc' and not model.get('locally_stable',True):
                for side,neighbor in ((1,index-1),(-1,index+1)):
                    if not 0<=neighbor<len(parts):continue
                    other=parts[neighbor]['model']
                    if not other or other['kind']!='arc':continue
                    # Do not split sectors that agree within their existing
                    # parameter uncertainty. Here the flat arc and its curved
                    # neighbor must be genuinely inconsistent hypotheses.
                    uncertainty=model.get('parameter_sensitivity_native_px',2)+other.get('parameter_sensitivity_native_px',2)
                    if model['radius_px']-other['radius_px']<=uncertainty:continue
                    p=item['points'] if side==1 else item['points'][::-1]
                    cuts=np.unique(np.linspace(2,len(p)//2,min(max(0,len(p)//2-1),PRIOR['maximum_change_point_positions'])).astype(int))
                    for cut in cuts:
                        if self.calls>=PRIOR['maximum_fit_calls']:self.incomplete=True;break
                        self.calls+=1;q=p[cut:];center=q.mean(0)
                        _,_,vectors=np.linalg.svd(q-center,full_matrices=False);u=vectors[0]
                        projection=(q-center)@u;errors=np.linalg.norm(q-center-projection[:,None]*u,axis=1)
                        if errors.max()>math.sqrt(.5) or np.ptp(projection)<PRIOR['minimum_length_px']:continue
                        ends=center+projection[[0,-1],None]*u
                        linear=checked({'kind':'line','points_px':ends.tolist(),'fit_max_px':float(errors.max()),
                            'fit_p95_px':float(np.percentile(errors,95)),
                            'independent_straight_tail':True,
                            'parameter_evidence':'dominant_observed_straight_tail'},q,self.support)
                        if linear is None:continue
                        if best is None or len(q)>best[0]:best=(len(q),side,p,cut,None,linear)
            if best is None:expanded.append(item);continue
            _,side,p,cut,curved,linear=best
            pieces=[{'points':p[:cut+1],'model':curved},{'points':p[cut:],'model':linear}]
            if side==-1:pieces=[{**o,'points':o['points'][::-1]} for o in pieces[::-1]]
            expanded.extend(pieces)
        return super().refine_tangent(self.merge(expanded))
