"""Explicit source-bound axis review, including measured dash/gap validation."""
from collections import Counter
from copy import deepcopy
import numpy as np
from .grid_role_revision_v1 import sha
from .part_admission_v1 import object_sha


def apply_review(result,gray,roles,geometry,groups):
    if not isinstance(groups,list) or len(groups)>32:raise ValueError('Axis review budget exceeded')
    result=deepcopy(result);lookup={c['id']:c for c in geometry};used=set();accepted=[]
    for group in groups:
        ids=group.get('source_candidate_ids',[])
        if (group.get('source_pixel_sha256')!=sha(gray) or group.get('kind')!='axis'
                or not isinstance(group.get('reviewer'),str) or not 0<len(group['reviewer'])<=128):
            raise ValueError('Explicit source-bound axis review required')
        if not isinstance(ids,list) or not 0<len(ids)<=8192 or len(set(ids))!=len(ids) or set(ids)-set(lookup) or used&set(ids):
            raise ValueError('Unique complete axis source candidates required')
        selected=[]
        for index,o in enumerate(result['objects']):
            owners={m['candidate_id'] for m in o['source_members']}
            if not owners&set(ids):continue
            if owners-set(ids) or o['kind'] not in ('line','polyline','point') or o.get('closed'):
                raise ValueError('Axis review requires whole open collinear observations')
            selected.append(index)
        members=[m for i in selected for m in result['objects'][i]['source_members']]
        # Selecting a partially protected candidate is forbidden even if the
        # visible pieces happen to carry every candidate ID.
        covered=Counter((m['candidate_id'],n) for m in members for n in range(*m['source_index_interval']))
        expected=Counter((c,n) for c in ids for n in range(*lookup[c]['source_index_interval']))
        if covered!=expected or {m['candidate_id'] for m in members}!=set(ids):
            raise ValueError('Axis includes withheld or foreign source edges')
        points=[]
        for m in members:
            c=lookup[m['candidate_id']];lo,hi=m['source_index_interval'];start=c['source_index_interval'][0]
            points.extend(c['support_points'][lo-start:hi-start+1])
        p=np.unique(np.asarray(points,int),axis=0)
        if len(p)<8 or len(p)>120000:raise ValueError('Axis point budget')
        known=set(map(int,roles[p[:,1],p[:,0]]))&set(range(1,10))
        if known-{5}:raise ValueError('Axis conflicts with a non-axis source role')
        u=np.asarray(group.get('direction_px'),float)
        if u.shape!=(2,) or not np.isfinite(u).all() or np.linalg.norm(u)==0:raise ValueError('Finite axis direction required')
        u/=np.linalg.norm(u);origin=p.mean(0);projection=(p-origin)@u;low,high=float(projection.min()),float(projection.max())
        error=np.linalg.norm(p-origin-projection[:,None]*u,axis=1)
        if error.max()>2 or np.percentile(error,95)>1.25:raise ValueError('Axis source is not collinear')
        pattern=np.asarray(group.get('pattern_px'),float)
        if (pattern.shape!=(4,) or not np.isfinite(pattern).all() or np.any(abs(pattern)<1)
                or np.any(np.sign(pattern)!=[1,-1,1,-1]) or sum(abs(pattern))>120000):
            raise ValueError('Measured dash-gap-dot-gap pattern required')
        boundaries=np.r_[0,np.cumsum(abs(pattern))];period=float(boundaries[-1])
        def expected_on(t):
            phase=np.mod(t,period);part=np.minimum(np.searchsorted(boundaries,phase,side='right')-1,3)
            margin=np.min(abs(phase[:,None]-boundaries),axis=1)
            return pattern[part]>0,margin
        observed=np.sort(projection-low);on,margin=expected_on(observed)
        if np.any(~on&(margin>.8)):raise ValueError('Axis review would erase intentional source strokes')
        samples=np.arange(0,high-low+.01,.5)
        if len(samples)>120000:raise ValueError('Axis sampling budget')
        on,margin=expected_on(samples);pos=np.searchsorted(observed,samples)
        distance=np.minimum(abs(samples-observed[np.maximum(0,pos-1)]),abs(samples-observed[np.minimum(len(observed)-1,pos)]))
        if np.any(on&(margin>.8)&(distance>.8)):raise ValueError('Axis pattern would invent unsupported dashes')
        model={'kind':'line','points_px':[(origin+low*u).tolist(),(origin+high*u).tolist()],
            'fit_max_px':float(error.max()),'fit_p95_px':float(np.percentile(error,95)),
            'linetype_pattern_px':pattern.tolist(),'axis_review':deepcopy(group)}
        identity='cad-axis/'+object_sha({'members':members,'model':model})[:24]
        obj={**result['objects'][selected[0]],**model,'id':identity,'role':5,
            'source_members':members,'support_points':p[np.argsort(projection)].tolist(),
            'path_ids':list(dict.fromkeys(m['path_id'] for m in members)),
            'source_junction_links':[link for i in selected for link in result['objects'][i].get('source_junction_links',[])],
            'semantic_ownership_confirmed':False,'status':'agent_reviewed_axis_draft'}
        result['objects']=[o for i,o in enumerate(result['objects']) if i not in selected]+[obj]
        used|=set(ids);accepted.append(identity)
    result.update(object_count=len(result['objects']),kind_counts=dict(Counter(o['kind'] for o in result['objects'])),
        axis_review={'accepted':accepted,'automatic_axis_inference':False})
    return result
