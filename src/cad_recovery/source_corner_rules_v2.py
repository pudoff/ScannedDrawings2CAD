"""Source-space line fragments, without resampling a hairline into dotted ink."""
import math
import cv2
import numpy as np
from .source_line_extension_v1 import recover

POLICY={'maximum_segments':2048,'maximum_support_samples':4000000,'minimum_source_support':.95,
        'maximum_groups':256,'axis_angle_deg':8.,'GT_used':False}


def project(points,matrix):
    q=np.c_[points,np.ones(len(points))]@matrix.T
    return q[:,:2]/q[:,2:]


def segments(ink,audit):
    span=min(ink.shape)
    found=cv2.HoughLinesP(ink,1,np.pi/720,threshold=max(8,round(.025*span)),
        minLineLength=max(8,round(.025*span)),maxLineGap=2)
    found=np.empty((0,4)) if found is None else np.asarray(found).reshape(-1,4)
    # Progressive Hough can omit a complete hairline. Gradient proposals supply
    # independent seeds; they still need the same original-ink evidence.
    extra=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(ink)[0]
    extra=np.empty((0,4)) if extra is None else np.asarray(extra).reshape(-1,4)
    extra=extra[np.linalg.norm(extra[:,:2]-extra[:,2:],axis=1)>=max(8,.025*span)]
    audit['gradient_segments']=len(extra)
    found=np.concatenate((found,extra))
    audit['source_segments']=len(found)
    if len(found)>POLICY['maximum_segments']:
        audit['limits'].append('source_segment_budget');return []
    return found


def support(ink,distance,a,b,audit):
    length=np.linalg.norm(b-a);n=max(2,math.ceil(length)+1)
    audit['source_support_samples']+=n
    if audit['source_support_samples']>POLICY['maximum_support_samples']:
        audit['limits'].append('source_support_budget');return None
    q=np.rint(a+np.linspace(0,1,n)[:,None]*(b-a)).astype(int)
    if (q<0).any() or (q[:,0]>=ink.shape[1]).any() or (q[:,1]>=ink.shape[0]).any():return None
    good=distance[q[:,1],q[:,0]]<=math.sqrt(2)
    change=np.diff(np.r_[False,~good,False].astype(int));gaps=np.flatnonzero(change==-1)-np.flatnonzero(change==1)
    if good.mean()<POLICY['minimum_source_support'] or (len(gaps) and gaps.max()>math.ceil(2*math.sqrt(2))):return None
    return float(good.mean())


def pen(ink,a,b,audit):
    u=(b-a)/np.linalg.norm(b-a);normal=np.array([-u[1],u[0]])
    # A bounded cross-section in original pixel centres, independent of GT.
    reach=max(4.,.025*np.linalg.norm(b-a));offsets=np.arange(-reach,reach+.25,.5);widths=[]
    for t in (.23,.51,.77):
        q=np.rint(a+t*(b-a)+offsets[:,None]*normal).astype(int)
        valid=(q>=0).all(1)&(q[:,0]<ink.shape[1])&(q[:,1]<ink.shape[0])
        dark=np.zeros(len(q),bool);dark[valid]=ink[q[valid,1],q[valid,0]]>0
        audit['source_support_samples']+=len(q)
        if audit['source_support_samples']>POLICY['maximum_support_samples']:
            audit['limits'].append('source_support_budget');return None
        change=np.diff(np.r_[False,dark,False].astype(int));runs=list(zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)))
        if not runs:continue
        lo,hi=min(runs,key=lambda r:abs((offsets[r[0]]+offsets[r[1]-1])/2))
        if lo>0 and hi<len(offsets):widths.append((hi-lo)*.5)
    return float(np.median(widths)) if widths else 1.


def corner_grid(ink,quad,source_segments,distance,audit,*,cache=None):
    options=[];extensions={} if cache is None else cache
    for turn in range(4):
        p=np.roll(quad,-turn,axis=0);w=np.linalg.norm(p[1]-p[0]);h=np.linalg.norm(p[3]-p[0])
        matrix=cv2.getPerspectiveTransform(p.astype(np.float32),np.array([[0,0],[w,0],[w,h],[0,h]],np.float32));inverse=np.linalg.inv(matrix)
        groups={'rows':[],'columns':[]}
        for values in source_segments:
            native=np.asarray(values,float).reshape(2,2);local=project(native,matrix);a,b=local;delta=b-a;length=np.linalg.norm(delta)
            if length<1:continue
            if abs(delta[1])<=math.sin(math.radians(POLICY['axis_angle_deg']))*length:
                key='rows';major=0
                if (a[1]+b[1])/2<.55*h:continue
            elif abs(delta[0])<=math.sin(math.radians(POLICY['axis_angle_deg']))*length:
                key='columns';major=1
                if min(a[0],b[0])<.25*w:continue
            else:continue
            token=tuple(values)
            if token not in extensions:
                if support(ink,distance,*native,audit) is None:
                    extensions[token]=None;continue
                width=pen(ink,*native,audit)
                if width is None:return None
                verified=extensions.setdefault('verified_intervals',[]);extended=None
                for candidate in verified:
                    ca,cb=candidate['points'];u=(cb-ca)/np.linalg.norm(cb-ca);n=np.array([-u[1],u[0]])
                    along=(native-ca)@u
                    if (abs(width-candidate['width'])<=2*math.sqrt(2) and
                        max(abs((native-ca)@n))<=candidate['width']/2+math.sqrt(.5) and
                        along.min()>=-math.sqrt(2) and along.max()<=np.linalg.norm(cb-ca)+math.sqrt(2)):
                        extended=candidate;break
                if extended is None:
                    extended=recover(ink,distance,native,width,audit)
                    if extended is not None:verified.append(extended)
                extensions[token]={'width':width,'extended':extended}
            entry=extensions[token]
            if entry is None:continue
            width=entry['width'];extended=entry['extended']
            if audit['limits']:return None
            if extended is not None:
                native=extended['points'];local=project(native,matrix);a,b=local;width=extended['width']
            if a[major]>b[major]:a,b=b,a
            groups[key].append({'a':a,'b':b,'width':width,'major':major})
        unique={}
        for key,items in groups.items():
            clusters=[]
            for item in sorted(items,key=lambda t:-np.linalg.norm(t['b']-t['a'])):
                a,b=item['a'],item['b'];u=(b-a)/np.linalg.norm(b-a);normal=np.array([-u[1],u[0]])
                cluster=None
                for old in clusters:
                    oa,ob=old['anchor'];ou=(ob-oa)/np.linalg.norm(ob-oa);on=np.array([-ou[1],ou[0]])
                    tolerance=(item['width']+old['width'])/2+math.sqrt(2)
                    if max(abs((a-oa)@on),abs((b-oa)@on))<=tolerance and max(abs((oa-a)@normal),abs((ob-a)@normal))<=tolerance:
                        cluster=old;break
                if cluster is None:
                    cluster={'anchor':(a,b),'points':[],'width':item['width']};clusters.append(cluster)
                cluster['points'].extend([a,b])
                if len(clusters)>POLICY['maximum_groups']:
                    audit['limits'].append('source_group_budget');return None
            accepted=[]
            for c in clusters:
                a,b=c['anchor'];axis=(b-a)/np.linalg.norm(b-a);points=np.array(c['points']);ts=(points-a)@axis
                a,b=a+ts.min()*axis,a+ts.max()*axis;dx,dy=abs(b-a)
                if key=='rows':
                    if not .15*w<=dx<=.75*w or max(a[0],b[0])<.95*w:continue
                elif dy<.05*h or max(a[1],b[1])<.95*h:continue
                native=project(np.array([a,b]),inverse);observed=support(ink,distance,*native,audit)
                if observed is None:continue
                low=np.floor(np.minimum(a,b)).astype(int);high=np.ceil(np.maximum(a,b)).astype(int)
                accepted.append({'bbox':[int(low[0]),int(low[1]),int(high[0]-low[0]+1),int(high[1]-low[1]+1),int(np.linalg.norm(b-a))],
                    'source_endpoints':native.tolist(),'source_support':observed,'source_width_px':c['width'],
                    'fragment_endpoints':points.tolist()})
            unique[key]=accepted
        if len(unique['rows'])>=3 and len(unique['columns'])>=3:
            options.append({'quarter_turn':turn,'rows':[r['bbox'] for r in unique['rows']],
                'columns':[r['bbox'] for r in unique['columns']],'source_support':unique,'supported':True,
                'width_evidence':'unresampled_source_collinear_fragments'})
    return max(options,key=lambda o:(len(o['rows'])+len(o['columns']),-o['quarter_turn'])) if options else None
