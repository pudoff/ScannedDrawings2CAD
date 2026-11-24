"""Bounded source-backed alternatives to untrusted hatching assignments.

A grid is not automatically a title block. All revisions are 9 -> unknown;
original maps, reviewed protections and source geometry stay untouched.
"""
from copy import deepcopy
import hashlib
import math
import cv2
import numpy as np
from skimage.morphology import skeletonize
from .stroke_roles_v1 import checked_gray,foreground
from .supported_role_overlay_v1 import source_ownership
from .junction_topology_v1 import adjacency_bits,POPCOUNT

POLICY={'version':'grid-role-revision/1.0','maximum_tracks':1024,'maximum_intervals':4096,
    'maximum_pairs':250000,'maximum_cells':250000,'maximum_components':4096,
    'maximum_glyphs':256,'maximum_glyph_roi_pixels':2000000,'minimum_line_in_widths':8.,
    'minimum_line_in_image_fraction':.15,'angle_deg':8.,'minimum_offsets_per_axis':3,
    'minimum_cells':2,'minimum_glyphs':6,'minimum_text_rows':2,'minimum_row_glyphs':3,
    'minimum_rule_to_glyph_width':1.4,'revision':'untrusted_hatching_to_unknown_only',
    'production_admitted':False,'physical_geometry_approved':False}


def sha(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def cv_authority(gray,roles,producer_pin,*,protected=None):
    """Caller verifies the snapshot pin; GT masks are never authority inputs."""
    if set(producer_pin)!={'path','sha256'} or len(producer_pin['sha256'])!=64:raise ValueError('Explicit producer snapshot pin required')
    mask=np.ones(gray.shape,np.uint8)
    if protected is not None:
        if protected.shape!=gray.shape or protected.dtype!=np.uint8:raise ValueError('Aligned review-protection mask required')
        mask[protected>0]=2
    return {'mask':mask,'producer_kind':'frozen_cv_inference','producer':deepcopy(producer_pin),
        'source_sha256':sha(gray),'roles_sha256':sha(roles),'mask_sha256':sha(mask),'size_px':list(gray.shape[::-1])}


def check_authority(gray,roles,authority):
    if authority is None:return np.zeros(gray.shape,bool)
    mask=authority['mask']
    if mask.shape!=gray.shape or mask.dtype!=np.uint8 or np.any(mask>2):raise ValueError('Invalid role-authority mask')
    if authority['size_px']!=list(gray.shape[::-1]) or authority['source_sha256']!=sha(gray) or authority['roles_sha256']!=sha(roles) or authority['mask_sha256']!=sha(mask):
        raise ValueError('Authority does not match this exact source/role map')
    return (mask==1) if authority.get('producer_kind')=='frozen_cv_inference' else np.zeros(gray.shape,bool)


def crossing(a,b,ink):
    if abs(a['u']@b['u'])>math.sin(math.radians(8)):return None
    t,s=np.linalg.solve(np.column_stack((a['u'],-b['u'])),b['p'][0]-a['p'][0])
    margin=min(a['width'],b['width'])/2+math.sqrt(.5)
    if not -margin<=t<=a['length']+margin or not -margin<=s<=b['length']+margin:return None
    point=a['p'][0]+t*a['u'];q=np.rint(point).astype(int)
    left,top=np.maximum(0,np.floor(point-margin)).astype(int);right,bottom=np.minimum(ink.shape[::-1],np.ceil(point+margin)+1).astype(int)
    if right<=left or bottom<=top:return None
    y,x=np.indices((bottom-top,right-left));near=(x+left-point[0])**2+(y+top-point[1])**2<=margin**2
    return point if np.any((ink[top:bottom,left:right]>0)&near) else None


def offset_groups(lines,axis,normal):
    output=[]
    for line in sorted(lines,key=lambda t:(float(t['p'].mean(0)@normal),t['id'])):
        offset=float(line['p'].mean(0)@normal)
        if output and offset-output[-1]['minimum']<=max(line['width'],output[-1]['width']):
            output[-1]['members'].append(line);output[-1]['width']=max(output[-1]['width'],line['width'])
        else:output.append({'minimum':offset,'width':line['width'],'members':[line]})
    return output


def grids(ink,graph,audit):
    raw=graph['tracks']
    if len(raw)>1024 or len(graph['proposals'])>4096:raise ValueError('Bounded source tracks/intervals required')
    interval_ids={p['id'] for p in graph['proposals']};lines=[]
    for t in raw:
        p=np.asarray(t['points'],float);width=float(t['width'])
        if p.shape!=(2,2) or not np.isfinite(p).all() or not math.isfinite(width) or width<=0:raise ValueError('Finite source track required')
        length=float(np.linalg.norm(p[1]-p[0]))
        if length<max(8*width,.15*min(ink.shape)):continue
        if not set(t['interval_ids'])<=interval_ids:raise ValueError('Track refers to missing source intervals')
        lines.append({'id':t['id'],'p':p,'u':(p[1]-p[0])/length,'length':length,'width':width,'interval_ids':list(t['interval_ids'])})
    parent=list(range(len(lines)));edges={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,a in enumerate(lines):
        for j in range(i+1,len(lines)):
            audit['pair_checks']+=1
            if audit['pair_checks']>POLICY['maximum_pairs']:audit['limits'].append('track_pair_budget');return []
            point=crossing(a,lines[j],ink)
            if point is not None:edges[i,j]=point;parent[root(j)]=root(i)
    clusters={}
    for i in range(len(lines)):clusters.setdefault(root(i),[]).append(i)
    output=[];cos=math.cos(math.radians(8));sin=math.sin(math.radians(8))
    for members in clusters.values():
        if len(members)<6:continue
        u=lines[members[0]]['u'].copy()
        if u[0]<0:u=-u
        n=np.array([-u[1],u[0]]);aa=[];bb=[];valid=True
        for i in members:
            dot=abs(lines[i]['u']@u)
            if dot>=cos:aa.append(lines[i])
            elif dot<=sin:bb.append(lines[i])
            else:valid=False;break
        if not valid:continue
        a=offset_groups(aa,u,n);b=offset_groups(bb,n,u)
        if len(a)<3 or len(b)<3:continue
        membership={t['id']:('a',k) for k,g in enumerate(a) for t in g['members']}
        membership.update({t['id']:('b',k) for k,g in enumerate(b) for t in g['members']});links={};cross_points={}
        for (i,j),point in edges.items():
            if i not in members or j not in members:continue
            x,y=membership[lines[i]['id']],membership[lines[j]['id']]
            if x[0]==y[0]:continue
            if x[0]=='b':x,y=y,x
            links[x[1],y[1]]=True;cross_points.setdefault((x[1],y[1]),[]).append(point)
        cells=0
        for i in range(len(a)-1):
            for j in range(len(b)-1):
                audit['cell_checks']+=1
                if audit['cell_checks']>POLICY['maximum_cells']:audit['limits'].append('cell_budget');return []
                cells+=all((x,y) in links for x in (i,i+1) for y in (j,j+1))
        if cells<2:continue
        perimeter=all((i,j) in links for i in (0,len(a)-1) for j in range(len(b))) and all((i,j) in links for i in range(len(a)) for j in (0,len(b)-1))
        for family,groups,axis,last in [('a',a,u,len(b)-1),('b',b,n,len(a)-1)]:
            for k in (0,len(groups)-1):
                group=groups[k];points=np.concatenate([t['p'] for t in group['members']]);lo,hi=float((points@axis).min()),float((points@axis).max())
                first=(k,0) if family=='a' else (0,k);end=(k,last) if family=='a' else (last,k)
                if first not in cross_points or end not in cross_points:perimeter=False;continue
                hits=np.array(cross_points[first]+cross_points[end])@axis
                if abs(lo-float(hits.min()))>2*group['width']+math.sqrt(.5) or abs(hi-float(hits.max()))>2*group['width']+math.sqrt(.5):perimeter=False
        pts=np.concatenate([lines[i]['p'] for i in members]);ids=sorted({s for i in members for s in lines[i]['interval_ids']})
        output.append({'track_ids':[lines[i]['id'] for i in members],'interval_ids':ids,'u':u.tolist(),'n':n.tolist(),
            'local_box':[float((pts@u).min()),float((pts@n).min()),float((pts@u).max()),float((pts@n).max())],
            'offsets':[len(a),len(b)],'cells':int(cells),'perimeter_supported':bool(perimeter)})
    return output


def glyph_context(grid,ink,labels,stats,distance,items,audit):
    grid_labels=set();widths=[]
    for identity in grid['interval_ids']:
        p=np.rint(items[identity]['points_px']).astype(int)
        inside=(p>=0).all(1)&(p[:,0]<ink.shape[1])&(p[:,1]<ink.shape[0]);p=p[inside]
        if len(p):
            grid_labels.update(labels[p[:,1],p[:,0]].tolist());r=distance[p[:,1],p[:,0]];widths.extend((2*r[r>0]-1).tolist())
    pen=float(np.median(widths)) if widths else 1.;u=np.asarray(grid['u']);n=np.asarray(grid['n']);box=grid['local_box'];facts=[]
    for identity in range(1,len(stats)):
        if identity in grid_labels:continue
        x,y,w,h,area=map(int,stats[identity])
        if x<=0 or y<=0 or x+w>=ink.shape[1] or y+h>=ink.shape[0] or max(w,h)>192 or area<6:continue
        center=np.array([x+w/2,y+h/2]);s,t=center@u,center@n
        if not (box[0]<s<box[2] and box[1]<t<box[3]):continue
        audit['glyph_roi_pixels']+=w*h
        if audit['glyph_roi_pixels']>POLICY['maximum_glyph_roi_pixels']:audit['limits'].append('glyph_roi_budget');break
        local=labels[y:y+h,x:x+w]==identity;skeleton=skeletonize(local,method='zhang').astype(np.uint8)*255
        degree=POPCOUNT[adjacency_bits(skeleton)];ends=int(np.count_nonzero((skeleton>0)&(degree==1)))
        branches=int(np.count_nonzero((skeleton>0)&(degree>=3)))
        holes=cv2.connectedComponents(np.pad((~local).astype(np.uint8),1,constant_values=1),connectivity=4)[0]-2
        if ends<1 or not (branches or holes):continue
        yy,xx=np.nonzero(local);points=np.column_stack((xx+x,yy+y));a=points@u;b=points@n
        sy,sx=np.nonzero(skeleton);measured=2*distance[y+sy,x+sx]-1
        facts.append({'id':identity,'box':[float(a.min()),float(b.min()),float(a.max()+1),float(b.max()+1)],
            'stroke':float(np.quantile(measured,.25)),'stroke_median':float(np.median(measured)),
            'endpoints':ends,'branches':branches,'holes':int(holes)})
        if len(facts)>POLICY['maximum_glyphs']:audit['limits'].append('glyph_budget');break
    options=[]
    for swap in (False,True):
        glyphs=[]
        for f in facts:
            b=f['box'] if not swap else [f['box'][1],f['box'][0],f['box'][3],f['box'][2]];width,height=b[2]-b[0],b[3]-b[1]
            if max(5.,1.5*pen)<=height<=min(120.,20*pen) and .15<=width/height<=2.2:glyphs.append({**f,'box':b,'height':height})
        candidates={}
        for seed in glyphs:
            peers=[g for g in glyphs if .6<=g['height']/seed['height']<=1.6 and abs(g['box'][3]-seed['box'][3])<=.5*seed['height']]
            peers.sort(key=lambda g:g['box'][0]);groups=[]
            for g in peers:
                if groups and g['box'][0]-groups[-1][-1]['box'][2]<=2*max(g['height'],groups[-1][-1]['height']):groups[-1].append(g)
                else:groups.append([g])
            for group in groups:
                if len(group)>=3:candidates[tuple(sorted(g['id'] for g in group))]=group
        rows=[];seen=set()
        for group in sorted(candidates.values(),key=lambda p:(np.median([g['box'][3] for g in p]),tuple(g['id'] for g in p))):
            baseline=float(np.median([g['box'][3] for g in group]));height=float(np.median([g['height'] for g in group]))
            if seen&{g['id'] for g in group} or any(abs(baseline-r['baseline'])<.8*height for r in rows):continue
            rows.append({'component_ids':[g['id'] for g in group],'baseline':baseline,'height':height});seen.update(g['id'] for g in group)
        selected=[g for g in glyphs if g['id'] in seen];stroke=float(np.median([g['stroke'] for g in selected])) if selected else None
        options.append({'rows':rows,'glyph_count':len(seen),'glyph_component_ids':sorted(seen),
            'glyph_stroke_px':stroke,'glyph_width_estimator':'median_of_component_DT_q25_2r_minus_1','swapped_axes':swap})
    best=max(options,key=lambda p:(len(p['rows']),p['glyph_count']));best['grid_stroke_px']=pen
    best['supported']=len(best['rows'])>=2 and best['glyph_count']>=6 and best['glyph_stroke_px'] is not None
    return best


def analyze(gray,graph,roles,*,authority=None):
    checked_gray(gray)
    if roles.shape!=gray.shape or roles.dtype!=np.uint8:raise ValueError('Aligned uint8 role map required')
    permitted=check_authority(gray,roles,authority);audit={'pair_checks':0,'cell_checks':0,'glyph_roi_pixels':0,'limits':[]}
    baseline=roles.copy();structure=roles.copy();guarded=roles.copy();decisions=[]
    if not np.any(permitted&(roles==9)):
        return {'baseline':baseline,'structure_only':structure,'guarded':guarded,'decisions':[],
            'revision_owner':np.zeros(gray.shape,np.int32),'audit':audit,'complete':True,'source_pixels_erased':0,
            'production_admitted':False,'physical_geometry_approved':False}
    ink=foreground(gray);found=grids(ink,graph,audit);owner_map=np.zeros(gray.shape,np.int32)
    if found and not audit['limits']:
        count,labels,stats,_=cv2.connectedComponentsWithStats(ink,connectivity=8)
        if count-1>POLICY['maximum_components']:audit['limits'].append('component_budget')
        ownership,work=source_ownership(ink,graph['proposals']);audit['ownership']=work
        if work.get('incomplete'):audit['limits'].append('source_ownership_budget')
        if ownership is not None and not audit['limits']:
            distance=cv2.distanceTransform(ink,cv2.DIST_L2,5);items={p['id']:p for p in graph['proposals']};numbers={p['id']:i+1 for i,p in enumerate(graph['proposals'])}
            radii=np.zeros(len(items)+1,float)
            for identity in {s for g in found for s in g['interval_ids']}:
                p=np.rint(items[identity]['points_px']).astype(int);r=distance[p[:,1],p[:,0]];r=r[r>0]
                if len(r):radii[numbers[identity]]=float(np.quantile(r,.25))+math.sqrt(.5)
            nearest=ownership['owner'];safe=np.clip(nearest,0,len(items));near=(nearest>0)&(ownership['distance']<=radii[safe])&(ink>0)
            eligible=permitted&(roles==9)&near
            for grid in found:
                ids=[numbers[s] for s in grid['interval_ids']];selected=eligible&np.isin(nearest,ids)
                structure[selected]=255;context=glyph_context(grid,ink,labels,stats,distance,items,audit)
                # Merged cells and cropped title blocks need not have a full
                # rectangular perimeter. Geometry alone grants no revision.
                strong=context['supported'];thick=[]
                if strong:
                    for i in ids:
                        # Per-member contrast protects thin hatch lines even if
                        # a nearby thick table shares this geometric component.
                        observed=max(0.,2*(radii[i]-math.sqrt(.5))-1)
                        if observed>=POLICY['minimum_rule_to_glyph_width']*context['glyph_stroke_px']:thick.append(i)
                revised=selected&np.isin(nearest,thick)
                if thick:
                    glyphs=np.isin(labels,context['glyph_component_ids'])
                    glyph_source=(nearest>0)&(ownership['distance']<=context['glyph_stroke_px']/2+math.sqrt(.5))&(ink>0)
                    revised|=permitted&(roles==9)&glyphs&glyph_source
                guarded[revised]=255;owner_map[revised]=nearest[revised]
                decisions.append({**grid,'glyph_context':context,'guard_context_supported':bool(strong),
                    'structural_withdrawals':int(selected.sum()),'guarded_withdrawals':int(revised.sum()),
                    'source_owner_numbers':thick,'revision_source_owner_numbers':sorted(np.unique(nearest[revised]).tolist()),
                    'assigned_title_grid':False,'revision':'untrusted_hatching_to_unknown'})
                if audit['limits']:break
    complete=not audit['limits']
    if not complete:structure[:]=guarded[:]=roles;owner_map[:]=0
    return {'baseline':baseline,'structure_only':structure,'guarded':guarded,'decisions':decisions,
        'revision_owner':owner_map,'audit':audit,'complete':complete,'source_pixels_erased':0,
        'production_admitted':False,'physical_geometry_approved':False}
