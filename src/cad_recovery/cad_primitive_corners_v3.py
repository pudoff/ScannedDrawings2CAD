"""Locate a LINE/ARC corner hidden by a tiny unresolved raster transition."""
import numpy as np
from .cad_primitive_fit_v2 import checked,sector,POLICY
from .cad_object_assembly_v2 import _circle_line
from .stroke_curve_control_v1 import distance_to_polyline


def refine(items,fitter):
    output=[];index=0
    while index<len(items):
        best=None
        if index+2<len(items):
            triple=items[index:index+3];a,b,c=triple;ma,mb,mc=[o['model'] for o in triple]
            if (ma and mc and mb is None and {ma['kind'],mc['kind']}=={'line','arc'}
                    and a['path_ids']==b['path_ids']==c['path_ids'] and len(a['roles']|b['roles']|c['roles'])<=1):
                reverse=ma['kind']=='arc';ordered=triple[::-1] if reverse else triple
                line,_,arc=ordered;circle=arc['model']
                if circle.get('locally_stable',True):
                    groups=[o['points'][::-1] if reverse else o['points'] for o in ordered]
                    points=np.r_[groups[0],groups[1][1:],groups[2][1:]]
                    lo=len(groups[0])-1;hi=lo+len(groups[1])-1
                    cuts=np.unique(np.linspace(lo,hi,min(hi-lo+1,128)).astype(int))
                    for cut in cuts:
                        if fitter.calls>=POLICY['maximum_fit_calls']:fitter.incomplete=True;break
                        fitter.calls+=1
                        left,right=points[:cut+1],points[cut:]
                        if np.linalg.norm(left[-1]-left[0])<POLICY['minimum_length_px']:continue
                        initial={'kind':'line','points_px':left[[0,-1]].astype(float).tolist()}
                        for node in _circle_line(circle,initial,points[cut]):
                            if np.linalg.norm(node-points[cut])>POLICY['max_px']:continue
                            ends=np.array([left[0],node]);error=distance_to_polyline(left,ends)
                            if error.max()>POLICY['max_px'] or np.percentile(error,95)>POLICY['p95_px']:continue
                            center=np.asarray(circle['center_px']);radius=circle['radius_px']
                            actual=sector(right,center,radius)
                            adjusted=right.astype(float).copy();adjusted[0]=node
                            curve=sector(adjusted,center,radius)
                            if actual is None or curve is None:continue
                            linear=checked({**initial,'points_px':ends.tolist(),
                                'fit_max_px':float(error.max()),'fit_p95_px':float(np.percentile(error,95))},left,fitter.support)
                            curve=checked({**circle,**curve,'fit_max_px':actual['fit_max_px'],
                                'fit_p95_px':actual['fit_p95_px'],'parameter_evidence':'source_supported_LINE_circle_corner'},right,fitter.support)
                            if linear is None or curve is None:continue
                            score=float(error@error)+float(np.sum((np.linalg.norm(right-center,axis=1)-radius)**2))
                            if best is None or score<best[0]:best=(score,cut,linear,curve,points,reverse,triple)
        if best is None:output.append(items[index]);index+=1;continue
        _,cut,linear,curve,p,reverse,triple=best
        a,b,c=triple
        first={**a,'points':p[:cut+1],'model':linear};second={**c,'points':p[cut:],'model':curve}
        if reverse:
            first={**c,'points':first['points'][::-1],'model':linear}
            second={**a,'points':second['points'][::-1],'model':curve}
            output.extend([second,first])
        else:output.extend([first,second])
        index+=3
    return output
