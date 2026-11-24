"""Conservative dash-dot LINE drafts from source ink, never from arrow shape."""
from collections import Counter
from copy import deepcopy
import math
import cv2
import numpy as np
from .cad_primitive_fit_v2 import checked
from .stroke_curve_control_v1 import distance_to_polyline
from .part_admission_v1 import object_sha
from .grid_role_revision_v1 import sha

POLICY={'version':'cad-axis-pattern/4.0','maximum_candidates':512,'maximum_support_candidates':4096,'maximum_axes':32,
    'maximum_samples':120000,'maximum_pair_checks':262144,'angle_deg':8.,
    'max_error_px':2.,'p95_error_px':1.25,'minimum_strokes':5,
    'pixel_roles_modified':False,'physical_geometry_approved':False}


def _edges(objects):
    return Counter((m['candidate_id'],n) for o in objects for m in o['source_members'] for n in range(*m['source_index_interval']))


def _runs(ink,origin,u,lo,hi,pen=1.):
    t=np.arange(lo,hi+.01,.5)
    if len(t)>POLICY['maximum_samples']:return None
    q=np.rint(origin+t[:,None]*u).astype(int)
    if (q<0).any() or (q[:,0]>=ink.shape[1]).any() or (q[:,1]>=ink.shape[0]).any():return None
    # Measure a continuous profile against nearby observed ink centres. Nearest
    # pixel sampling alone creates artificial gaps on one-pixel staircases.
    # This changes analysis only; source ownership and source pixels stay exact.
    positions=origin+t[:,None]*u
    values=np.zeros(len(t),bool);radius=POLICY['max_error_px']+math.sqrt(.5)
    for dx in (-3,-2,-1,0,1,2,3):
        for dy in (-3,-2,-1,0,1,2,3):
            neighbor=q+[dx,dy]
            valid=(neighbor[:,0]>=0)&(neighbor[:,1]>=0)&(neighbor[:,0]<ink.shape[1])&(neighbor[:,1]<ink.shape[0])
            ids=np.flatnonzero(valid)
            values[ids]|=(ink[neighbor[ids,1],neighbor[ids,0]]>0)&(np.linalg.norm(neighbor[ids]-positions[ids],axis=1)<=radius)
    cuts=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1,len(values)]
    runs=[{'on':bool(values[a]),'lo':float(t[a]-.25),'hi':float(t[b-1]+.25)} for a,b in zip(cuts,cuts[1:])]
    # Undo the longitudinal expansion of the analysis support band. Otherwise
    # a robust one-pixel profile would incorrectly produce thicker/longer dashes.
    expansion=radius-.5
    strokes=[]
    for i,run in enumerate(runs):
        if not run['on']:continue
        start=run['lo']+(expansion if i else 0.)
        end=run['hi']-(expansion if i<len(runs)-1 else 0.)
        if end<=start:return None
        if strokes:strokes.append({'on':False,'lo':strokes[-1]['hi'],'hi':start})
        strokes.append({'on':True,'lo':start,'hi':end})
    return strokes


def _measure(ink,origin,u,lo,hi,pen):
    runs=_runs(ink,origin,u,lo,hi,pen)
    if not runs or not runs[0]['on'] or not runs[-1]['on']:return None
    strokes=runs[::2];n=len(strokes)
    if n<POLICY['minimum_strokes'] or n%2==0 or len(runs)>65:return None
    # Pen width bounds transverse uncertainty, not an axial endpoint. Use the
    # existing geometric boundary error, then check the actual native dashes
    # against original pixel cells with both max and p95 limits below.
    tolerance=POLICY['max_error_px']
    widths=np.array([r['hi']-r['lo'] for r in strokes]);long=float(np.median(widths[2:-1:2]));short=float(np.median(widths[1::2]))
    if short<=0 or long-short<=2*tolerance:return None
    # A run length is the difference of two independently quantized endpoints.
    # Its uncertainty is twice a single boundary's uncertainty. Final boundary
    # and actual-source checks below still use their own unchanged limits.
    if np.max(abs(widths[1::2]-short))>2*tolerance or np.max(abs(widths[2:-1:2]-long))>2*tolerance:return None
    centers=np.array([(r['lo']+r['hi'])/2 for r in strokes[1::2]])
    period,first=np.linalg.lstsq(np.c_[np.arange(len(centers)),np.ones(len(centers))],centers,rcond=None)[0]
    if np.max(abs(centers-(first+period*np.arange(len(centers)))))>tolerance:return None
    # The first gap adjoins an A-aligned terminal dash and may include an
    # extension. Estimate the repeating phase from internal gaps instead.
    gap1=float(np.median([runs[i]['hi']-runs[i]['lo'] for i in range(5,len(runs),4)]))
    gap2=float(period-long-short-gap1)
    if min(gap1,gap2)<=2*math.sqrt(.5):return None
    for i,r in enumerate(runs):
        if not r['on'] and abs(r['hi']-r['lo']-(gap1 if i%4==1 else gap2))>2*tolerance:return None
    # Autodesk A-alignment lengthens terminal dashes. If one terminal includes
    # an additional solid extension, keep that extension as its original LINE.
    k=len(centers);predicted=lo+(hi-lo-k*period+long)/2+gap1+short/2
    shift=2*(first-predicted)
    if shift>tolerance:lo+=shift
    elif shift<-tolerance:hi+=shift
    if hi<=lo or not k*period<=hi-lo<(k+1)*period:return None
    terminal=(hi-lo-k*period+long)/2
    if terminal<long/2:return None
    native=[(lo,lo+terminal)];cursor=lo+terminal
    for index in range(k):
        cursor+=gap1;native.append((cursor,cursor+short));cursor+=short+gap2
        end=hi if index==k-1 else cursor+long
        native.append((cursor,end));cursor=end
    # Compare modeled A-aligned endpoints to every observed source run.
    cropped=[(max(lo,r['lo']),min(hi,r['hi'])) for r in strokes if r['hi']>lo and r['lo']<hi]
    if len(native)!=len(cropped) or max(abs(a-b) for x,y in zip(native,cropped) for a,b in zip(x,y))>tolerance:return None
    samples=np.concatenate([origin+np.arange(a,b+.01,.5)[:,None]*u for a,b in native])
    errors=ink_cell_errors(ink,samples)
    if errors.max()>POLICY['max_error_px'] or np.percentile(errors,95)>POLICY['p95_error_px']:return None
    return {'lo':float(lo),'hi':float(hi),'pattern_px':[long,-gap1,short,-gap2],
        'period_px':float(period),'cycles':k,'terminal_dash_px':float(terminal),
        'phase_tolerance_px':float(tolerance),'source_ink_runs':cropped,
        'A_aligned_runs':native,'shared_ink_preserved_as_source':True,
        'profile_method':'quantization_aware_support_band_with_longitudinal_expansion_removed',
        'native_dash_to_source_cell_max_px':float(errors.max()),
        'native_dash_to_source_cell_p95_px':float(np.percentile(errors,95))}


def ink_cell_errors(ink,points):
    """Actual visible native dashes to observed foreground pixel squares."""
    if len(points)>POLICY['maximum_samples']:raise ValueError('Axis source-support sampling budget')
    q=np.rint(points).astype(int);errors=np.full(len(points),POLICY['max_error_px']+1.)
    for dx in (-3,-2,-1,0,1,2,3):
        for dy in (-3,-2,-1,0,1,2,3):
            neighbor=q+[dx,dy]
            valid=(neighbor[:,0]>=0)&(neighbor[:,1]>=0)&(neighbor[:,0]<ink.shape[1])&(neighbor[:,1]<ink.shape[0])
            ids=np.flatnonzero(valid);ids=ids[ink[neighbor[ids,1],neighbor[ids,0]]>0]
            delta=np.maximum(abs(points[ids]-neighbor[ids])-.5,0.)
            errors[ids]=np.minimum(errors[ids],np.linalg.norm(delta,axis=1))
    return errors


def _slice(member,low,high):
    value=deepcopy(member);value['source_index_interval']=[low,high]
    if member.get('graph_interval_ranges'):
        value['graph_interval_ranges']=[{'id':r['id'],'source_index_interval':[max(low,r['source_index_interval'][0]),min(high,r['source_index_interval'][1])]}
            for r in member['graph_interval_ranges'] if max(low,r['source_index_interval'][0])<min(high,r['source_index_interval'][1])]
        value['interval_ids']=[r['id'] for r in value['graph_interval_ranges']]
    return value


def _members_points(members,lookup):
    values=[]
    for m in members:
        c=lookup[m['candidate_id']];lo,hi=m['source_index_interval'];start=c['source_index_interval'][0]
        values+=c['support_points'][lo-start:hi-start+1]
    return np.unique(np.asarray(values,int),axis=0)


def _partition(objects,indices,origin,u,evidence,lookup,support):
    members=[];remainders=[];lo,hi=evidence['lo'],evidence['hi']
    for i in indices:
        obj=objects[i];remaining=[]
        for m in obj['source_members']:
            c=lookup[m['candidate_id']];a,b=m['source_index_interval'];start=c['source_index_interval'][0]
            p=np.asarray(c['support_points'][a-start:b-start+1],float);t=(p-origin)@u
            selected=(t>=lo-1e-7)&(t<=hi+1e-7);inside=selected[:-1]&selected[1:]
            cuts=np.r_[0,np.flatnonzero(inside[1:]!=inside[:-1])+1,len(inside)]
            for left,right in zip(cuts,cuts[1:]):
                (members if inside[left] else remaining).append(_slice(m,a+int(left),a+int(right)))
        if remaining:
            if remaining==obj['source_members']:
                remainders.append(deepcopy(obj));continue
            p=_members_points(remaining,lookup);t=(p-origin)@u
            if t.min()<lo and t.max()>hi:return None
            # A separate terminal observation must not be stretched to the
            # axis boundary through other observations or genuine white gaps.
            endpoints=np.array([origin+t.min()*u,origin+t.max()*u])
            errors=distance_to_polyline(p,endpoints)
            if errors.max()>2 or np.percentile(errors,95)>1.25:return None
            model=checked({'kind':'line','points_px':endpoints.tolist(),'fit_max_px':float(errors.max()),
                'fit_p95_px':float(np.percentile(errors,95))},p,support)
            if model is None:return None
            remainders.append({**obj,**model,'source_members':remaining,'support_points':p.tolist(),
                'id':'axis-terminal/'+object_sha(remaining)[:24]})
    if not members:return None
    p=_members_points(members,lookup);ends=np.array([origin+lo*u,origin+hi*u]);errors=distance_to_polyline(p,ends)
    if errors.max()>2 or np.percentile(errors,95)>1.25:return None
    return members,p,ends,errors,remainders


def apply(result,gray,roles,geometry,heads=(),*,protected=None,text_protection=None):
    if result.get('source_sha256')!=sha(gray) or roles.shape!=gray.shape or roles.dtype!=np.uint8:
        raise ValueError('Axis assembly belongs to another source or role frame')
    for mask in (protected,text_protection):
        if mask is not None and (mask.shape!=gray.shape or mask.dtype!=bool):raise ValueError('Aligned boolean axis protection required')
    result=deepcopy(result);before=_edges(result['objects']);objects=result['objects'];lookup={c['id']:c for c in geometry}
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    dt=cv2.distanceTransform(ink,cv2.DIST_L2,5);support=cv2.dilate(ink,np.ones((3,3),np.uint8));candidates=[]
    for i,obj in enumerate(objects):
        p=np.asarray(obj['support_points'],int)
        if obj['kind'] not in ('line','polyline') or len(p)<3 or obj.get('closed'):continue
        known=set(map(int,roles[p[:,1],p[:,0]]))&set(range(1,10))
        if known-{5}:continue
        center=p.mean(0);_,_,vectors=np.linalg.svd(p-center,full_matrices=False);u=vectors[0];t=(p-center)@u
        pen=max(1.,float(np.median(dt[p[:,1],p[:,0]])*2-1))
        errors=np.linalg.norm(p-center-t[:,None]*u,axis=1)
        if np.ptp(t)<max(1.,pen/2) or errors.max()>POLICY['max_error_px'] or np.percentile(errors,95)>POLICY['p95_error_px']:continue
        candidates.append({'index':i,'p':p,'u':u,'length':float(np.ptp(t)),'pen':pen})
    supports=candidates
    # A tiny dot does not have an independently reliable direction. It may
    # support a measured long-stroke seed, but must not consume the seed budget.
    candidates=[c for c in supports if c['length']>=2*POLICY['max_error_px']]
    audit={'policy':deepcopy(POLICY),'candidate_count':len(candidates),'support_candidate_count':len(supports),
        'accepted':[],'pair_checks':0,'complete':True}
    if len(candidates)>POLICY['maximum_candidates'] or len(supports)>POLICY['maximum_support_candidates']:
        audit.update(complete=False,reason='candidate_budget');result['axis_pattern']=audit;result['search_complete']=False;return result
    used=set();new=[]
    for seed in sorted(candidates,key=lambda c:(-c['length'],c['index'])):
        if seed['index'] in used:continue
        origin=seed['p'].mean(0);u=seed['u'].copy()
        if u[np.argmax(abs(u))]<0:u=-u
        group=[]
        for candidate in supports:
            audit['pair_checks']+=1
            if audit['pair_checks']>POLICY['maximum_pair_checks']:audit.update(complete=False,reason='pair_budget');break
            if candidate['index'] in used:continue
            if candidate['length']>=2*POLICY['max_error_px'] and abs(float(candidate['u']@u))<math.cos(math.radians(POLICY['angle_deg'])):continue
            q=candidate['p'];t=(q-origin)@u
            if np.linalg.norm(q-origin-t[:,None]*u,axis=1).max()<=2:group.append(candidate)
        if not audit['complete']:break
        if len(group)<POLICY['minimum_strokes']:continue
        # Collinear geometry elsewhere on the sheet is not an axis extension.
        # Split only at source-gap outliers, using the measured gap distribution.
        spans=sorted(((float(((c['p']-origin)@u).min()),float(((c['p']-origin)@u).max()),c) for c in group),
            key=lambda span:(span[0],span[1],span[2]['index']))
        gaps=[];end=spans[0][1]
        for left,right,_ in spans[1:]:
            if left>end:gaps.append(left-end)
            end=max(end,right)
        if len(gaps)>=4:
            q1,q3=np.quantile(gaps,[.25,.75]);limit=float(q3+1.5*(q3-q1)+2*POLICY['max_error_px'])
            clusters=[[]];end=spans[0][1]
            for left,right,c in spans:
                if clusters[-1] and left-end>limit:clusters.append([])
                clusters[-1].append(c);end=max(end,right) if len(clusters[-1])>1 else right
            group=next(g for g in clusters if any(c['index']==seed['index'] for c in g))
        if len(group)<POLICY['minimum_strokes']:continue
        p=np.concatenate([c['p'] for c in group]);origin=p.mean(0);_,_,vectors=np.linalg.svd(p-origin,full_matrices=False);u=vectors[0]
        if u[np.argmax(abs(u))]<0:u=-u
        t=(p-origin)@u
        if np.linalg.norm(p-origin-t[:,None]*u,axis=1).max()>2:continue
        lo,hi=float(t.min()),float(t.max());pen=float(np.median([c['pen'] for c in group]));blocked=False
        for head in heads:
            polygon=np.asarray(head.get('polygon_px',[]),float);direction=np.asarray(head.get('direction',[0,0]),float)
            if polygon.shape!=(3,2) or direction.shape!=(2,) or not np.isfinite(polygon).all() or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:continue
            direction=direction/np.linalg.norm(direction)
            if abs(float(direction@u))<math.cos(math.radians(20)):continue
            along=(polygon-origin)@u;normal=polygon-origin-along[:,None]*u
            if along.max()>=lo-2 and along.min()<=hi+2 and np.linalg.norm(normal,axis=1).min()<=pen/2+2:blocked=True;break
        if blocked:continue
        evidence=_measure(ink,origin,u,lo,hi,pen)
        if evidence is None:continue
        blocked_mask=np.zeros_like(gray,bool) if protected is None else protected.copy()
        transverse=[]
        for head in heads:
            polygon=np.asarray(head.get('polygon_px',[]),float);direction=np.asarray(head.get('direction',[0,0]),float)
            if polygon.shape!=(3,2) or direction.shape!=(2,) or not np.isfinite(polygon).all() or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:continue
            direction=direction/np.linalg.norm(direction)
            if abs(float(direction@u))>=math.cos(math.radians(20)):continue
            mask=np.zeros_like(gray);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255);area=mask>0
            if area.any() and np.mean(ink[area]>0)>=.7:
                blocked_mask[area]=False;transverse.append(head.get('id'))
        blocked_mask|=roles==6
        if text_protection is not None:blocked_mask|=text_protection
        samples=np.arange(evidence['lo'],evidence['hi']+.01,.5);q=np.rint(origin+samples[:,None]*u).astype(int)
        if blocked_mask[q[:,1],q[:,0]].any():continue
        evidence['transverse_heads_retained_as_source']=transverse
        indices=[c['index'] for c in group];split=_partition(objects,indices,origin,u,evidence,lookup,support)
        if split is None:continue
        members,points,ends,errors,remainders=split
        identity='axis-pattern/'+object_sha({'source':sha(gray),'members':members,'pattern':evidence})[:24]
        new.append({**objects[indices[0]],'kind':'line','id':identity,'role':5,
            'points_px':ends.tolist(),'support_points':points.tolist(),'source_members':members,
            'path_ids':list(dict.fromkeys(m['path_id'] for m in members)),'linetype_pattern_px':evidence['pattern_px'],
            'axis_pattern_evidence':evidence,'fit_max_px':float(errors.max()),'fit_p95_px':float(np.percentile(errors,95)),
            'status':'automatic_axis_pattern_draft','semantic_ownership_confirmed':False})
        new+=remainders;used.update(indices);audit['accepted'].append(identity)
        if len(audit['accepted'])>=POLICY['maximum_axes']:audit.update(complete=False,reason='axis_budget');break
    if not audit['complete']:
        result['axis_pattern']=audit;result['search_complete']=False;return result
    result['objects']=[o for i,o in enumerate(objects) if i not in used]+new
    if _edges(result['objects'])!=before:raise ValueError('Axis pattern changed source edge ownership')
    result.update(axis_pattern=audit,object_count=len(result['objects']),kind_counts=dict(Counter(o['kind'] for o in result['objects'])))
    return result
