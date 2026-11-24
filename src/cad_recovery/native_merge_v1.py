"""Deterministic native observations; overlap dedup never approves CAD geometry."""
from copy import deepcopy
import hashlib
import math
import numpy as np
from .dataset_layout_v1 import encoded
from .family_catalog_v2 import require

MAX_COMPARISONS=1_000_000

def stable_id(kind,value):return kind+'-'+hashlib.sha256(encoded(value)).hexdigest()[:20]

def points(values,matrix):
    a=np.asarray(values,float);m=np.asarray(matrix,float)
    require(m.shape==(3,3) and np.isfinite(m).all() and np.array_equal(m[2],[0,0,1]),'Affine matrix required')
    require(a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all(),'Finite point pairs required')
    return (np.column_stack((a,np.ones(len(a))))@m.T)[:,:2].tolist()

def vectors(values,matrix):
    m=np.asarray(matrix,float);require(m.shape==(3,3) and np.array_equal(m[2],[0,0,1]),'Affine matrix required')
    a=np.asarray(values,float)@m[:2,:2].T
    require(a.ndim==2 and a.shape[1]==2 and np.isfinite(a).all() and np.all(np.linalg.norm(a,axis=1)>0),'Invalid vectors')
    return (a/np.linalg.norm(a,axis=1)[:,None]).tolist()

def bounds(polygon):
    p=np.asarray(polygon,float);return [float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]

def overlap(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(1e-12,aa+bb-inter),inter/max(1e-12,min(aa,bb))

def geometry_box(g):
    if g['kind']=='line':return bounds(g['points_px'])
    x,y=g['center_px'];r=g['radius_px'];return [x-r,y-r,x+r,y+r]

def _cells(box):
    # Tile proposals are bounded; spatial buckets keep full-page merge finite.
    x0,y0,x1,y1=box
    return [(x,y) for x in range(math.floor((x0-2)/256),math.floor((x1+2)/256)+1)
                  for y in range(math.floor((y0-2)/256),math.floor((y1+2)/256)+1)]

def _line_union(a,b):
    p=np.asarray(a['points_px'],float);q=np.asarray(b['points_px'],float);u=p[1]-p[0];length=np.linalg.norm(u)
    v=q[1]-q[0];other=np.linalg.norm(v)
    if min(length,other)<1:return None
    u/=length;v/=other
    if abs(float(np.dot(u,v)))<math.cos(math.radians(2)):return None
    if max(abs(float(u[0]*(x-p[0])[1]-u[1]*(x-p[0])[0])) for x in q)>1.5:return None
    positions=(q-p[0])@u;lo,hi=sorted(positions)
    if min(length,hi)-max(0,lo)<2:return None # Never bridge a blank gap.
    endpoints=[p[0]+min(0,lo)*u,p[0]+max(length,hi)*u]
    return {**a,'points_px':[x.tolist() for x in endpoints]}

def _geometry_union(a,b):
    if a['kind']!=b['kind']:return None
    if a['kind']=='line':return _line_union(a,b)
    if math.dist(a['center_px'],b['center_px'])>1.5 or abs(a['radius_px']-b['radius_px'])>1.5:return None
    if a['kind']=='circle':return a
    # Only identical arc observations are deduplicated. Different sweeps remain
    # explicit fragments, rather than inventing unsupported missing arc pixels.
    delta=abs((a['start_angle_deg']-b['start_angle_deg']+180)%360-180)
    return a if delta<=1 and abs(a['sweep_angle_deg']-b['sweep_angle_deg'])<=1 else None

def merge(rows,kind):
    """Stable priority, spatially bounded comparisons, retained source aliases."""
    require(kind in ('text','geometry','arrow'),'Unknown observation kind')
    def priority(r):
        if kind=='text':return (-r['confidence'],-r.get('edge_margin_px',0),r['uid'])
        if kind=='geometry' and r['kind']=='line':return (-math.dist(*r['points_px']),0,r['uid'])
        return (0,0,r['uid'])
    groups=[];grid={};checks=0
    for row in sorted(rows,key=priority):
        box=geometry_box(row) if kind=='geometry' else bounds(row['polygon_px']) if kind=='text' else [*row['tip_px'],*row['tip_px']]
        cells=_cells(box);possible=sorted({i for c in cells for i in grid.get(c,[])})
        found=None;combined=None
        for index in possible:
            checks+=1;require(checks<=MAX_COMPARISONS,'Native merge comparison budget exceeded')
            item=groups[index]['value']
            if kind=='geometry':combined=_geometry_union(item,row)
            elif kind=='text':
                iou,covered=overlap(bounds(item['polygon_px']),box)
                combined=item if item['raw']==row['raw'] and (iou>=.6 or covered>=.85) else None
            else:
                combined=item if math.dist(item['tip_px'],row['tip_px'])<=2 and np.dot(item['direction'],row['direction'])>=.98 else None
            if combined is not None:found=index;break
        if found is None:
            found=len(groups);groups.append({'value':deepcopy(row),'members':[row['uid']]})
        else:
            groups[found]['value']=combined;groups[found]['members'].append(row['uid'])
        updated=groups[found]['value']
        updatebox=geometry_box(updated) if kind=='geometry' else bounds(updated['polygon_px']) if kind=='text' else box
        for c in _cells(updatebox):grid.setdefault(c,set()).add(found)
    result=[];aliases={}
    for group in groups:
        value=group['value'];members=sorted(set(group['members']))
        identifier=stable_id(kind,members);aliases.update({m:identifier for m in members})
        result.append({**value,'id':identifier,'observation_uids':members,'accepted':False})
    return sorted(result,key=lambda x:x['id']),aliases,checks

def conflicts(texts):
    """Conflicting OCR readings stay visible, without a target-text oracle."""
    grid={};pairs=set();checks=0
    for t in texts:
        cells=_cells(bounds(t['polygon_px']))
        for other in {i for c in cells for i in grid.get(c,[])}:
            checks+=1;require(checks<=MAX_COMPARISONS,'Text conflict budget exceeded')
            if t['raw']==other[1]:continue
            iou,covered=overlap(bounds(t['polygon_px']),other[2])
            if iou>=.6 or covered>=.85:pairs.add(tuple(sorted([t['id'],other[0]])))
        entry=(t['id'],t['raw'],tuple(bounds(t['polygon_px'])))
        for c in cells:grid.setdefault(c,set()).add(entry)
    return [{'ids':list(pair),'status':'needs_review','resolved_automatically':False} for pair in sorted(pairs)]
