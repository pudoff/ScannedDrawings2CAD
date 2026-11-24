"""Source-only structural page observer. No geometry branch suppression."""
import hashlib
import math
import cv2
import numpy as np

POLICY=dict(version='sheet-router/1.0',mode='observe',max_pixels=4_000_000,
    line_fraction=.025,minimum_kernel=12,border_fraction=.02,long_border_fraction=.85,
    minimum_grid_area_fraction=.025,minimum_rows=3,minimum_columns=3,
    minimum_junctions=6,dominant_grid_area_fraction=.45,
    large_residual_extent_fraction=.045,max_components=30000,max_grids=64,
    max_native_windows=8,absence_certification_supported=False)
ROUTES={'auto','general','drawing','table','mixed'}

def _runs(profile):
    edges=np.diff(np.r_[False,profile,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1).tolist(),np.flatnonzero(edges==-1).tolist()))

def _segments(mask,horizontal,extent):
    count,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if count>POLICY['max_components']:raise ValueError('line_component_budget')
    l,t,r,b=extent;w,h=r-l,b-t
    out=mask.copy()
    for x,y,ww,hh,area in stats[1:]:
        if horizontal:
            boundary=min(abs(y-t),abs(y+hh-b))<POLICY['border_fraction']*h
            long=ww>POLICY['long_border_fraction']*w
        else:
            boundary=min(abs(x-l),abs(x+ww-r))<POLICY['border_fraction']*w
            long=hh>POLICY['long_border_fraction']*h
        if boundary and long:out[y:y+hh,x:x+ww]=0
    return out

def observe(gray):
    if not isinstance(gray,np.ndarray) or gray.dtype!=np.uint8 or gray.ndim!=2 or min(gray.shape)<16 or gray.size>POLICY['max_pixels']:
        raise ValueError('Bounded uint8 grayscale page required')
    h,w=gray.shape;pixel_sha=hashlib.sha256(gray.tobytes()).hexdigest()
    ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    ys,xs=np.nonzero(ink);grids=[];unexplained=[];limits=[]
    if len(xs):
        extent=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(12,round(w*.025))),np.uint8))
        vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(12,round(h*.025)),1),np.uint8))
        try:
            horizontal=_segments(horizontal,True,extent);vertical=_segments(vertical,False,extent)
            network=cv2.dilate(horizontal|vertical,np.ones((3,3),np.uint8))
            n,_,stats,_=cv2.connectedComponentsWithStats(network,8)
            if n>POLICY['max_components']:raise ValueError('grid_component_budget')
            for x,y,ww,hh,area in stats[1:]:
                if ww*hh<gray.size*POLICY['minimum_grid_area_fraction']:continue
                hm=horizontal[y:y+hh,x:x+ww]>0;vm=vertical[y:y+hh,x:x+ww]>0
                rows=_runs(hm.sum(1)>=max(8,.25*ww));cols=_runs(vm.sum(0)>=max(8,.25*hh))
                joints=cv2.connectedComponents((hm&vm).astype(np.uint8),8)[0]-1
                if len(rows)>=3 and len(cols)>=3 and joints>=6:
                    grids.append(dict(box=[int(x),int(y),int(x+ww),int(y+hh)],
                        rows=len(rows),columns=len(cols),junctions=joints,
                        area_fraction=float(ww*hh/gray.size),semantic_status='candidate_not_confirmed'))
            if len(grids)>POLICY['max_grids']:raise ValueError('grid_candidate_budget')
            residual=ink.copy()
            for g in grids:
                l,t,r,b=g['box'];residual[t:b,l:r]=0
            # Source-visible large components are unexplained, not approved detail objects.
            residual=cv2.morphologyEx(residual,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
            count,_,stats,_=cv2.connectedComponentsWithStats(residual,8)
            if count>POLICY['max_components']:raise ValueError('residual_component_budget')
            for x,y,ww,hh,area in stats[1:]:
                if min(ww,hh)>=min(w,h)*POLICY['large_residual_extent_fraction']:
                    unexplained.append(dict(box=[int(x),int(y),int(x+ww),int(y+hh)],ink_pixels=int(area)))
        except ValueError as error:limits.append(str(error))
    grids=sorted(grids,key=lambda g:(-g['area_fraction'],g['box']))
    unexplained=sorted(unexplained,key=lambda r:(-r['ink_pixels'],r['box']))
    dominant=bool(grids and grids[0]['area_fraction']>=POLICY['dominant_grid_area_fraction'])
    if limits or not len(xs):kind='unknown';reason='incomplete_or_empty'
    elif dominant and not unexplained:kind='table_document';reason='dominant_grid_no_large_outside_component'
    elif grids and unexplained:kind='mixed';reason='grid_and_unexplained_outside_components'
    elif grids:kind='unknown';reason='grid_present_geometry_unresolved'
    elif unexplained:kind='drawing';reason='large_components_without_verified_table'
    else:kind='unknown';reason='insufficient_structure'
    return dict(schema_version='sheet-observation/1.0',policy=POLICY,pixel_sha256=pixel_sha,size_px=[w,h],
        proposed_page_kind=kind,score=None,score_calibrated=False,reason=reason,
        grids=grids,unexplained=unexplained,complete=not limits,limits=limits,
        geometry_presence='unknown',may_disable_geometry=False,source_only=True,
        native_review_required=bool(grids),absence_certification_supported=False)

def route(observation,requested='auto'):
    if requested not in ROUTES:raise ValueError('Unsupported route override')
    automatic={'table_document':'table','mixed':'mixed','drawing':'drawing'}.get(observation['proposed_page_kind'],'general')
    return dict(schema_version='sheet-route/1.0',mode='observe',automatic_route=automatic,
        requested_route=requested,proposed_route=automatic if requested=='auto' else requested,
        effective_route='general_geometry_preserving',geometry_enabled=True,
        table_branch_enabled=False,reason='Unqualified observer; current CAD processing retained',
        original_observation=observation,override_recorded=requested!='auto',production_admitted=False)

def inspect_native(source,observation,source_to_overview,expected_sha):
    """Inspect bounded source windows of unexplained components; never certify absence."""
    from .raster_windows_v2 import native_windows
    m=np.asarray(source_to_overview,float)
    if m.shape!=(3,3) or not np.isfinite(m).all() or not np.array_equal(m[2],[0,0,1]) or abs(np.linalg.det(m))<1e-15:
        raise ValueError('Finite invertible affine source transform required')
    from .raster_input_v2 import inspect
    meta=inspect(source)
    if meta['sha256']!=expected_sha:raise ValueError('Native source identity differs')
    if meta['format']!='PNG' or meta['orientation']!=1:
        return dict(status='deferred',reason='Native oriented JPEG/PDF dispatch belongs to shared native integration',absence_confirmed=False)
    inverse=np.linalg.inv(m);w,h=meta['size_px'];boxes=[]
    for region in observation['unexplained'][:POLICY['max_native_windows']]:
        l,t,r,b=region['box'];points=np.array([[l,t,1],[r-1,b-1,1]])@inverse.T
        cx,cy=points[:,:2].mean(0);half=512
        box=[max(0,int(cx)-half),max(0,int(cy)-half),min(w,int(cx)+half),min(h,int(cy)+half)]
        if box[0]<box[2] and box[1]<box[3] and box not in boxes:boxes.append(box)
    # Also inspect inside the grid: an inset drawing can be enclosed by a table.
    for grid in observation['grids']:
        l,t,r,b=grid['box']
        for fy,fx in [( .5,.5),(.2,.2),(.2,.8),(.8,.2),(.8,.8)]:
            point=inverse@np.array([l+fx*(r-l),t+fy*(b-t),1.])
            x,y=map(int,point[:2]);box=[max(0,x-512),max(0,y-512),min(w,x+512),min(h,y+512)]
            if len(boxes)<POLICY['max_native_windows'] and box[0]<box[2] and box[1]<box[3] and box not in boxes:boxes.append(box)
    if not boxes:return dict(status='deferred',reason='No unexplained proposal; absence not established',absence_confirmed=False,source_coverage_complete=False)
    windows,proof=native_windows(source,boxes,expected_sha)
    return dict(status='sampled_unexplained',proof=proof,source_coverage_complete=False,absence_confirmed=False,
        windows=[dict(box=item['box'],source_to_tile=item['source_to_tile'],
            pixel_sha256=hashlib.sha256(item['array'].tobytes()).hexdigest(),
            size_px=list(item['array'].shape[::-1]),contrast_range=int(np.ptp(item['array'])),
            local_observation=observe(item['array'])) for item in windows])
