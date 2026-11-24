"""Conservative role proposals and component text masks; no nominal/CAD inference."""
from collections import Counter
import math

import cv2
import numpy as np

from .centerline_comparison_v1 import foreground
from .arrow_detection import _line_arrows
from . import arrow_head_completion_v1 as heads
from . import arrow_path_recovery_v1 as paths

CLASSES=['background','part_contour','dimension_line','extension_line','leader_line',
         'axis_construction','text','sheet_frame','title_grid','hatching']
POLICY={'version':'stroke-role-proposals/1.0','maximum_pixels':4194304,'maximum_components':4096,
    'maximum_text_candidates':256,'maximum_lines':512,'maximum_DT_seeds':64,'maximum_DT_profile_pixels':2_000_000,
    'text_minimum_group':3,'text_height_in_strokes':[1.5,24],'text_component_aspect':[.12,2.0],
    'text_maximum_height_ratio':1.6,'text_gap_in_heights':2.,'text_alignment_in_heights':.6,
    'DT_core_radius_in_strokes':1.25,'DT_body_radius_in_strokes':.70,'DT_axis_ratio_min':2.5,
    'part_to_thin_width_ratio':1.5,'head_contact_in_strokes':1.5,'direction_tolerance_deg':20,
    'axis_minimum_segments':3,'axis_maximum_gap_in_strokes':12,'maximum_role_conflicts':0,
    'production_admitted':False,'physical_geometry_approved':False,'confidence_calibrated':False}


def checked_gray(gray):
    if not isinstance(gray,np.ndarray) or gray.dtype!=np.uint8 or gray.ndim!=2 or not 0<gray.size<=POLICY['maximum_pixels']:
        raise ValueError('Bounded uint8 grayscale required')


def stroke_floor(widths):
    values=[w['width_px'] for w in widths if w['width_px'] is not None and w['coverage']>=.7 and not w['heterogeneous']]
    return max(1.,min(24.,float(np.quantile(values,.25)))) if values else None


def text_components(gray,stroke):
    checked_gray(gray)
    if not np.isfinite(stroke) or not 1<=stroke<=24:raise ValueError('Invalid search stroke')
    ink=foreground(gray); n,labels,stats,centers=cv2.connectedComponentsWithStats(ink,connectivity=8)
    eligible=[]
    for i in range(1,min(n,POLICY['maximum_components']+1)):
        x,y,w,h,area=map(int,stats[i]);extent=max(w,h);minor=min(w,h)
        if not max(5,1.5*stroke)<=extent<=min(120,24*stroke) or minor<2:continue
        aspect=w/max(h,1)
        if not .12<=aspect<=2 or not .06<=area/(w*h)<=.92:continue
        eligible.append({'id':i,'box':[x,y,x+w,y+h],'width':w,'height':h,'center':centers[i].tolist()})
    truncated=len(eligible)>256
    eligible=sorted(eligible,key=lambda c:(-(c['width']*c['height']),c['id']))[:256]
    groups=[];selected=set()
    for vertical in (False,True):
        items=sorted(eligible,key=lambda c:(c['center'][0] if vertical else c['center'][1],c['id']))
        for seed in items:
            h=seed['width'] if vertical else seed['height'];across=seed['center'][0 if vertical else 1]
            aligned=[c for c in items if abs(c['center'][0 if vertical else 1]-across)<=.6*h
                and 1/1.6<=(c['width'] if vertical else c['height'])/h<=1.6]
            aligned.sort(key=lambda c:c['box'][1 if vertical else 0])
            runs=[];run=[];last=None
            for c in aligned:
                lo=c['box'][1 if vertical else 0];hi=c['box'][3 if vertical else 2]
                if last is not None and lo-last>2*h:
                    if len(run)>=3:runs.append(run)
                    run=[]
                run.append(c);last=hi
            if len(run)>=3:runs.append(run)
            for run in runs:
                ids=tuple(sorted(c['id'] for c in run))
                if any(tuple(g['component_ids'])==ids for g in groups):continue
                boxes=np.array([c['box'] for c in run]);box=[int(boxes[:,0].min()),int(boxes[:,1].min()),int(boxes[:,2].max()),int(boxes[:,3].max())]
                groups.append({'component_ids':list(ids),'box':box,'vertical':vertical,'authority':'automatic_shape_alignment_only'})
                selected.update(ids)
    component=np.isin(labels,list(selected)).astype(np.uint8)*255
    bbox=np.zeros_like(gray)
    for group in groups:
        x,y,r,b=group['box'];bbox[y:b,x:r]=255
    return {'component_mask':component,'bbox_mask':bbox,'groups':groups,
        'component_count':len(eligible),'search_incomplete':truncated or n-1>POLICY['maximum_components']}


def dt_heads(gray,stroke):
    checked_gray(gray)
    if not np.isfinite(stroke) or not 1<=stroke<=24:raise ValueError('Invalid search stroke')
    ink=foreground(gray);dt=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    cores=(dt>=max(1.25,1.25*stroke)).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(cores,connectivity=8)
    order=sorted(range(1,n),key=lambda i:(-int(stats[i,cv2.CC_STAT_AREA]),i))
    proposals=[];events=[];cost=0
    for seed in order[:64]:
        x,y,w,h,area=map(int,stats[seed]);pad=min(96,max(12,math.ceil(12*stroke)))
        left,top=max(0,x-pad),max(0,y-pad);right,bottom=min(gray.shape[1],x+w+pad),min(gray.shape[0],y+h+pad)
        body=(dt[top:bottom,left:right]>=max(1.,stroke*.70)).astype(np.uint8)
        _,components=cv2.connectedComponents(body,connectivity=8)
        core_ids=set(np.unique(components[labels[top:bottom,left:right]==seed]))-{0}
        yy,xx=np.nonzero(np.isin(components,list(core_ids)))
        if len(xx)<8:events.append({'seed':seed,'reason':'small_body'});continue
        points=np.c_[xx+left,yy+top].astype(float);center=points.mean(axis=0)
        values,vectors=np.linalg.eigh(np.cov((points-center).T))
        if values[0]<=0 or values[1]/values[0]<2.5:events.append({'seed':seed,'reason':'not_elongated'});continue
        axis=vectors[:,1];extent=min(160.,max(24.,12*stroke,float(np.ptp((points-center)@axis))*1.5))
        endpoints=np.clip([center-axis*extent/2,center+axis*extent/2],[0,0],[gray.shape[1]-1,gray.shape[0]-1])
        length=float(np.linalg.norm(endpoints[1]-endpoints[0]));sampling=81*(math.ceil(length)+97)
        if cost+sampling>2_000_000:events.append({'seed':seed,'reason':'profile_budget'});break
        cost+=sampling
        for c in _line_arrows(ink,{'id':f'DT-seed-{seed}','points_px':endpoints.tolist()}):
            c['classification']='filled_arrow_candidate'
            c['evidence'].update(head_length_px=c['evidence']['arrow_length_px'],geometry_ids=[],DT_seed=seed,
                proposal_method='DT_core_PCA_seed_existing_transverse_profile',independent_of_geometry_LINE=True)
            if not heads.tip_has_native_ink(c,ink,stroke):events.append({'seed':seed,'reason':'no_native_apex'});continue
            allowed,check=paths.native_head_check(c,ink,stroke,())
            if not allowed:events.append({'seed':seed,'reason':check['reason']});continue
            allowed,shaft=heads.shaft_evidence(c,ink,stroke)
            if not allowed:events.append({'seed':seed,'reason':shaft['reason']});continue
            c['evidence'].update(native_head_check=check,native_shaft=shaft)
            proposals.append(c)
    return {'candidates':paths.combine(proposals),'core_count':n-1,'seeds_examined':min(n-1,64),
        'profile_pixels':cost,'events':events,'search_incomplete':n-1>64 or any(e['reason']=='profile_budget' for e in events)}


def append_heads(baseline,additional):
    combined=heads.append_preserving_baseline(baseline,additional)
    for i,c in enumerate(combined):c['id']=baseline[i]['id'] if i<len(baseline) else f'DT-added-{i-len(baseline)+1:04d}'
    return combined


def guard_heads(candidates,mask):
    accepted=[];excluded=[]
    for c in candidates:
        points=[np.asarray(c['tip_px']),np.mean(c['polygon_px'],axis=0)]
        hit=False
        for p in points:
            x,y=np.rint(p).astype(int)
            if 0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x]>0:hit=True
        (excluded if hit else accepted).append(c)
    return accepted,excluded


def line_distance(point,p):
    delta=p[1]-p[0];t=float(np.clip((point-p[0])@delta/(delta@delta),0,1))
    return float(np.linalg.norm(point-p[0]-t*delta)),t


def intersection(a,b):
    design=np.c_[a[1]-a[0],-(b[1]-b[0])]
    if abs(np.linalg.det(design))<1e-8:return None
    t,s=np.linalg.solve(design,b[0]-a[0])
    return a[0]+t*(a[1]-a[0]) if 0<=t<=1 and 0<=s<=1 else None


def role_proposals(gray,geometry,widths,arrows,text_mask,*,context=True):
    checked_gray(gray);floor=stroke_floor(widths)
    observed_floor=floor
    floor=3. if floor is None else floor
    lookup={w['id']:w for w in widths};lines=[]
    for c in geometry:
        if c['kind']!='line':continue
        p=np.asarray(c['points_px'],float);delta=p[1]-p[0];length=float(np.linalg.norm(delta))
        if p.shape!=(2,2) or not np.isfinite(p).all():raise ValueError('Finite pixel line required')
        if length<=0:continue
        lines.append({'id':c['id'],'points':p,'u':delta/length,'length':length,'width':lookup.get(c['id'])})
    shaft_contacts={}
    for line in lines[:512]:
        values=[]
        for a in arrows:
            base=np.asarray(a['polygon_px'])[1:].mean(axis=0)
            if (line_distance(base,line['points'])[0]<=max(2.,1.5*floor)
                    and abs(line['u']@np.asarray(a['direction']))>=math.cos(math.radians(20))):values.append(a)
        shaft_contacts[line['id']]=values
    paired_shafts=[line for line in lines[:512] if any(np.asarray(a['direction'])@np.asarray(b['direction'])<-.9
        for i,a in enumerate(shaft_contacts[line['id']]) for b in shaft_contacts[line['id']][i+1:])]
    counts=Counter();proposals=[];pixel=np.full(gray.shape,255,np.uint8);conflicts=np.zeros_like(gray,bool)
    # Known background is not inferred from missing annotations; role scoring ignores background.
    for line in lines[:512]:
        p,u,length,w=line['points'],line['u'],line['length'],line['width'];hints=[];evidence={}
        width=None if not w else w['width_px']
        samples=p[0]+np.linspace(0,1,min(128,max(3,int(length))))[:,None]*(p[1]-p[0])
        q=np.rint(samples).astype(int);valid=(q>=0).all(axis=1)&(q[:,0]<gray.shape[1])&(q[:,1]<gray.shape[0])
        text_fraction=float(np.mean(text_mask[q[valid,1],q[valid,0]]>0)) if valid.any() else 0.
        if width is not None and w['coverage']>=.9 and not w['heterogeneous'] and width>=1.5*floor:hints.append('part_contour')
        if context:
            contacts=shaft_contacts[line['id']]
            if contacts:
                directions=[np.asarray(a['direction']) for a in contacts]
                paired=any(a@b<-.9 for i,a in enumerate(directions) for b in directions[i+1:])
                hints.append('dimension_line' if paired else 'leader_line')
            collinear=[];parallel=[];crosses=0
            for other in lines[:512]:
                if other is line:continue
                dot=abs(u@other['u']);offset=other['points'].mean(axis=0)-p[0]
                distance=abs(u[0]*offset[1]-u[1]*offset[0])
                if dot>math.cos(math.radians(3)):
                    parallel.append(other)
                    center=other['points'].mean(axis=0);_,t=line_distance(center,p)
                    gap=min(np.linalg.norm(a-b) for a in p for b in other['points'])
                    if distance<=max(2.,.6*floor) and gap<=12*floor and gap>floor:collinear.append(other)
                elif dot<.05:
                    if intersection(p,other['points']) is not None:crosses+=1
            if width is not None and width<=1.3*floor:
                for shaft in paired_shafts:
                    if abs(u@shaft['u'])>.15:continue
                    point=intersection(p,shaft['points'])
                    if point is not None and min(np.linalg.norm(point-e) for e in shaft['points'])<=3*floor:
                        hints.append('extension_line');break
            if len(collinear)>=2:
                lengths=[length,*[o['length'] for o in collinear]]
                if max(lengths)/max(1,min(lengths))>=1.8:hints.append('axis_construction')
            orientation=abs(u[0])<.05 or abs(u[1])<.05
            if orientation and len(parallel)>=2 and crosses>=2 and np.count_nonzero(text_mask)>0:
                hints.append('title_grid')
            if not orientation and len(parallel)>=3 and length<.35*max(gray.shape) and width is not None and width<=1.3*floor:
                hints.append('hatching')
            if orientation and length>=.7*max(gray.shape) and crosses>=1:
                near_edge=min(p[:,0].min(),gray.shape[1]-1-p[:,0].max(),p[:,1].min(),gray.shape[0]-1-p[:,1].max())
                if near_edge<=3*floor:hints.append('sheet_frame')
            evidence.update(head_ids=[a['id'] for a in contacts],collinear_segments=len(collinear),parallel_segments=len(parallel),crossings=crosses)
            if text_fraction>=.5:hints=['text']
        hints=sorted(set(hints));role=hints[0] if len(hints)==1 else 'unknown';counts[role]+=1
        proposals.append({'id':line['id'],'role':role,'alternatives':hints,'width_px':width,
            'text_overlap_fraction':text_fraction,'evidence':evidence,'status':'proposed_only','physical_geometry_approved':False})
        if role=='unknown' or width is None:continue
        raster=np.zeros_like(gray);cv2.line(raster,tuple(np.rint(p[0]).astype(int)),tuple(np.rint(p[1]).astype(int)),255,max(1,min(32,int(round(width)))))
        selected=(raster>0)&(gray<240);cls=CLASSES.index(role)
        conflicts|=selected&(pixel!=255)&(pixel!=cls)
        pixel[selected & ~conflicts]=cls
    if context:
        selected=(text_mask>0)&(gray<240);conflicts|=selected&(pixel!=255)&(pixel!=6)
        pixel[selected & ~conflicts]=6
    pixel[conflicts]=255
    return {'lines':proposals,'role_counts':dict(counts),'pixel_roles':pixel,'conflicting_pixels':int(conflicts.sum()),
        'global_thin_width_px':observed_floor,'configured_search_scale_px':floor,
        'training':False,'production_admitted':False}
