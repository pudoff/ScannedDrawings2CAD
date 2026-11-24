"""Partial raster annotation QA. Unreviewed pixels are never negatives.

Geometric support is necessary, not proof of the semantic reading. A failed
source annotation stays in the inventory but contributes no role pixels.
"""
import math

import cv2
import numpy as np

IGNORE = 255
RADIUS = 3
MINIMUM_FRACTION = .95


def samples(g):
    kind = g['kind']
    if kind in ('LINE', 'DASHED_LINE'):
        a,b = np.asarray(g['points'], dtype=float)
        if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite([a,b]).all() or np.linalg.norm(b-a)==0:
            raise ValueError('Finite nonzero straight primitive required')
        return np.linspace(a,b,math.ceil(float(np.linalg.norm(b-a)))+1)
    if kind in ('CIRCLE','ARC'):
        c=np.asarray(g['centre'],dtype=float);r=float(g['radius'])
        start,end=(0.,360.) if kind=='CIRCLE' else (float(g['start_deg']),float(g['end_deg']))
        if c.shape!=(2,) or not np.isfinite([*c,r,start,end]).all() or r<=0 or not 0<end-start<=360:
            raise ValueError('Finite circle or one open arc required')
        theta=np.radians(np.linspace(start,end,math.ceil(r*math.radians(end-start))+1))
        return c+np.column_stack((np.cos(theta),np.sin(theta)))*r
    if kind=='INK_BOX':return None
    raise ValueError('Unsupported primitive; composite paths cannot be a LINE')


def local_band(g,box):
    """Euclidean source-pixel band, not a resized mask or a rectangular fill."""
    x0,y0,x1,y1=box
    yy,xx=np.mgrid[y0:y1,x0:x1]
    if g['kind']=='INK_BOX':return np.ones(xx.shape,dtype=bool)
    if 'points' in g:
        a,b=np.asarray(g['points'],float);v=b-a;den=float(v@v)
        if not math.isfinite(den) or den<=0:raise ValueError('Nonzero line required')
        t=np.clip(((xx-a[0])*v[0]+(yy-a[1])*v[1])/den,0,1)
        distance=np.hypot(xx-a[0]-t*v[0],yy-a[1]-t*v[1])
    else:
        cx,cy=g['centre'];distance=np.abs(np.hypot(xx-cx,yy-cy)-g['radius'])
        if g['kind']=='ARC':
            # Wrapped angles follow image coordinates (positive y downwards).
            start,end=g['start_deg'],g['end_deg'];angles=(np.degrees(np.arctan2(yy-cy,xx-cx))-start)%360
            distance=np.where(angles<=end-start,distance,np.inf)
            for p in samples(g)[[0,-1]]:distance=np.minimum(distance,np.hypot(xx-p[0],yy-p[1]))
    return distance<=RADIUS


def object_qa(ink,obj):
    x0,y0,x1,y1=obj['box'];g=obj['geometry'];local=ink[y0:y1,x0:x1]
    if obj['label_status']!='known':return dict(admitted=False,status='retained_ambiguity',sample_count=0,supported_samples=0,fraction=None)
    points=samples(g)
    if points is None:
        pixels=int(local.sum())
        return dict(admitted=pixels>0,status='nonempty visually reviewed text-ink box; not OCR accuracy',sample_count=pixels,supported_samples=pixels,fraction=1. if pixels else 0.)
    positions=np.rint(points).astype(int)
    if ((positions[:,0]<x0)|(positions[:,0]>=x1)|(positions[:,1]<y0)|(positions[:,1]>=y1)).any():
        raise ValueError('Primitive samples outside declared source box')
    distance=cv2.distanceTransform((~local).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    supported=distance[positions[:,1]-y0,positions[:,0]-x0]<=RADIUS
    if g['kind']=='DASHED_LINE':
        # Human-confirmed axis: measure only the original observed ink runs.
        # Gaps are explicitly not supposed to have foreground support.
        padded=np.r_[False,supported,False].astype(int);starts=np.flatnonzero(np.diff(padded)==1);ends=np.flatnonzero(np.diff(padded)==-1)
        runs=[dict(first_sample=int(a),last_sample=int(b-1),start=points[a].tolist(),end=points[b-1].tolist()) for a,b in zip(starts,ends)]
        return dict(admitted=len(runs)>=2,status='observed dash runs; gap pixels excluded',sample_count=len(points),supported_samples=int(supported.sum()),fraction=float(supported.mean()),visible_runs=runs,axis_unit_count=1)
    fraction=float(supported.mean())
    return dict(admitted=fraction>=MINIMUM_FRACTION,status='passed' if fraction>=MINIMUM_FRACTION else 'source_support_rejected_no_refit',sample_count=len(points),supported_samples=int(supported.sum()),fraction=fraction)


def masks(ink,objects,qa,contacts):
    """One bit plane per role. Shared and explicit exclusions remain IGNORE."""
    bits=np.zeros(ink.shape,dtype=np.uint16)
    for obj in objects:
        # Explicit ambiguity still excludes overlapping admitted labels; it is
        # never a single-role target and cannot become a background negative.
        if not qa[obj['id']]['admitted'] and obj['label_status']!='multi_role':continue
        x0,y0,x1,y1=obj['box'];part=ink[y0:y1,x0:x1]&local_band(obj['geometry'],obj['box'])
        role_bits=sum(1<<(r-1) for r in obj['roles'])
        bits[y0:y1,x0:x1][part] |= role_bits
    shared=(bits!=0)&((bits&(bits-1))!=0)
    labels=np.full(ink.shape,IGNORE,dtype=np.uint8)
    coverage=np.zeros(ink.shape,dtype=np.uint8)
    for role in range(1,10):
        only=bits==(1<<(role-1));labels[only]=role;coverage[only]=1
    coverage[shared]=2
    # Exclude contact neighbourhoods regardless of whether the two objects
    # passed QA. Do not force ownership at actual intersections.
    for contact in contacts:
        x,y=contact['source_point'];x0=max(0,math.floor(x-6));x1=min(ink.shape[1],math.ceil(x+7));y0=max(0,math.floor(y-6));y1=min(ink.shape[0],math.ceil(y+7))
        yy,xx=np.mgrid[y0:y1,x0:x1];exclude=(xx-x)**2+(yy-y)**2<=36
        labels[y0:y1,x0:x1][exclude]=IGNORE
        covered=coverage[y0:y1,x0:x1];covered[exclude&(covered!=0)]=3
    for obj in objects:
        if obj['label_status'] not in ('unknown','ambiguous'):continue
        x0,y0,x1,y1=obj['box'];exclude=local_band(obj['geometry'],obj['box'])
        labels[y0:y1,x0:x1][exclude]=IGNORE
        covered=coverage[y0:y1,x0:x1];covered[exclude&(covered!=0)]=3
    return labels,bits,coverage


def point_geometry_distance(point,g):
    p=np.asarray(point,float)
    if 'points' in g:
        a,b=np.asarray(g['points'],float);v=b-a
        return float(np.linalg.norm(p-a-np.clip((p-a)@v/(v@v),0,1)*v))
    return float(np.linalg.norm(samples(g)-p,axis=1).min())


def contact_qa(ink,contact,objects,qa):
    selected=[objects[n] for n in contact['objects']]
    distances=[point_geometry_distance(contact['source_point'],o['geometry']) for o in selected]
    x,y=np.rint(contact['source_point']).astype(int);h,w=ink.shape
    local=ink[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)]
    yy,xx=np.mgrid[max(0,y-3):min(h,y+4),max(0,x-3):min(w,x+4)]
    source_supported=bool((local&((xx-contact['source_point'][0])**2+(yy-contact['source_point'][1])**2<=RADIUS**2)).any())
    return dict(admitted=source_supported and max(distances)<=RADIUS and all(qa[o['id']]['admitted'] for o in selected),
                geometry_distances_px=distances,source_ink_supported=source_supported,
                both_source_units_admitted=all(qa[o['id']]['admitted'] for o in selected))


def role_unit_support(sources,reports,folds):
    """Unique source units/parents, never the number of pixels or windows."""
    by_id={r['source_id']:r for r in sources};counts=[]
    for role in range(1,10):
        units=set();parents=set();groups=set();source_ids=set()
        for sid,row in by_id.items():
            for obj in row['objects']:
                if obj['label_status']=='known' and obj['roles']==[role] and reports[sid]['objects'][obj['id']]['admitted']:
                    units.add(obj['source_unit_id']);parents.add(obj['correlation_parent_id']);groups.add(row['family']);source_ids.add(sid)
        fold_rows=[]
        for f in folds:
            fit=len(source_ids&set(f['fit_sources']));held=len(source_ids&set(f['held_sources']))
            fold_rows.append(dict(held_family=f['held_family'],fit_sources_with_role=fit,held_sources_with_role=held,both_sides_present=bool(fit and held)))
        counts.append(dict(role=role,source_units=len(units),correlation_parents=len(parents),groups=len(groups),source_count=len(source_ids),folds=fold_rows))
    return counts
