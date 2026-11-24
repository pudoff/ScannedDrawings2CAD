"""Source-only straight-anchor dimension bodies with an independent PART witness.

Experimental component. It never revises existing working roles or ARC chains.
"""
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np
from PIL import Image

from .native_dimension_context_v1 import _intersection, _native_geometry, _span
from .source_arrow_observation_v1 import checked_matrix, digest, transform
from .part_admission_v1 import object_sha

POLICY = dict(maximum_contexts=32, maximum_roi_pixels=4_000_000,
              minimum_pen_ratio=1.5, minimum_prior_points=3,
              minimum_visible_fraction=.95, maximum_witness_span_factor=3.,
              physical_geometry_approved=False, production_changed=False)


def refined_head_domain(context, heads):
    """Use agreeing freshly shape-checked contacts for the intervening veto."""
    output=deepcopy(heads)
    for identity,head in output.items():
        choices=[]
        for c in context['contexts']:
            for binding in c.get('head_extension_bindings',[]):
                if binding['head_id']!=identity:continue
                for o in binding['extension_options']:
                    if o['source_shape_rechecked']:
                        choices.append((np.asarray(o['intersection_px'])+c['source_box_px'][:2],c['id']))
        if not choices:continue
        if max(np.linalg.norm(a[0]-b[0]) for a in choices for b in choices)>math.sqrt(.5):continue
        head['tip_px']=choices[0][0].tolist()
        head['fresh_contact_context_ids']=sorted({c[1] for c in choices})
    return output


def _ink(gray):
    return cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]>0


def _part_witness(source, source_hash, size, anchors, width, geometry, roles, matrix):
    """Thick connected source ink must reach independent existing PART samples.

    Opening only measures connectivity at a pen scale; no opened pixel becomes
    inferred material or changes the original image. Rectangular/grid evidence
    without a corroborated existing PART prior cannot qualify an anchor.
    """
    a=np.asarray(anchors,float);span=np.linalg.norm(a[1]-a[0])
    low=np.floor(a.min(0)-[POLICY['maximum_witness_span_factor']*span,12*width]).astype(int)
    high=np.ceil(a.max(0)+[POLICY['maximum_witness_span_factor']*span,12*width]+1).astype(int)
    low=np.maximum(0,low);high=np.minimum(size,high)
    box=[*low.tolist(),*high.tolist()]
    if np.prod(high-low)>POLICY['maximum_roi_pixels']:
        return dict(accepted=False,reason='part_witness_window_budget',box=box)
    with Image.open(source) as im:gray=np.array(im.crop(box).convert('L'))
    ink=_ink(gray).astype('uint8')
    radius=math.ceil(width/2)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    opened=cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)
    _,labels=cv2.connectedComponents(opened,8)
    q=np.rint(a-low).astype(int);ids=labels[q[:,1],q[:,0]]
    report=dict(accepted=False,reason='anchors_not_on_one_thick_component',box=box,
        source_sha256=source_hash,pixels_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
        opening_radius_px=radius,opening_is_analysis_only=True,
        inferred_from_dimension_heads=False,prior_samples=[])
    if not ids[0] or ids[0]!=ids[1]:return report
    label=int(ids[0]);seen=set();conflicts=[]
    for line in geometry['candidates']:
        for point in line.get('support_points',line.get('points_px',[])):
            p=np.rint(point).astype(int)
            if not (0<=p[0]<roles.shape[1] and 0<=p[1]<roles.shape[0]):continue
            native=transform([point],matrix)[0];local=np.rint(native-low).astype(int)
            if not ((local>=0).all() and (local<high-low).all()):continue
            if labels[local[1],local[0]]!=label:continue
            value=int(roles[p[1],p[0]])
            if value not in (1,255):conflicts.append([*p.tolist(),value])
            if value!=1 or tuple(p) in seen:continue
            # Prior must be away from both proposed anchors; no circular self-support.
            if np.min(np.linalg.norm(a-native,axis=1))<6*width:continue
            seen.add(tuple(p));report['prior_samples'].append(dict(
                working_px=p.tolist(),source_px=native.tolist(),source_owner=line['id']))
    if conflicts:
        report.update(reason='known_competing_role_on_thick_component',conflicts=conflicts);return report
    if len(seen)<POLICY['minimum_prior_points']:
        report['reason']='independent_existing_PART_support_missing';return report
    report.update(accepted=True,reason='source_thick_connectivity_to_existing_PART',
        component_pixels=int(np.count_nonzero(labels==label)))
    return report


def observe_context(gray,context,heads,geometry,roles,matrix,*,source,source_hash,size):
    """Require every seed examined; unresolved ink or competing anchor withholds."""
    row=dict(id=context['id'],head_ids=context['head_ids'],status='withheld',reasons=[],
             source_box_px=context['source_box_px'],coordinate_space='original_source_pixels',
             source_sha256=source_hash,context_sha256=object_sha(context),
             geometry_sha256=object_sha(geometry),input_roles_sha256=hashlib.sha256(roles.tobytes()).hexdigest(),
             working_to_source=np.asarray(matrix).tolist())
    empty=np.zeros_like(gray,dtype='uint8')
    def held(reason):row['reasons'].append(reason);return dict(report=row,roles=empty)
    if hashlib.sha256(gray.tobytes()).hexdigest()!=context['source_window_sha256']:
        raise ValueError('Context pixels differ from current source window')
    if digest(source)!=source_hash:raise ValueError('Original source changed')
    checked_matrix(matrix)
    origin=np.asarray(context['source_box_px'][:2],float)
    ink=_ink(gray);native={c['id']:c for c in _native_geometry(geometry,matrix)}
    measured={c['id']:c for c in context['stroke_observation']['lines']}
    options=[];alternatives=[]
    for hid,seeds in zip(context['head_ids'],context['extension_seed_ids']):
        bindings=[b for b in context['head_extension_bindings'] if b['head_id']==hid]
        good=[o for b in bindings for o in b['extension_options'] if o['source_shape_rechecked']]
        if len(good)!=1:return held('unique_source_head_extension_missing')
        choice=good[0];head=heads[hid];axis=np.asarray(head['direction'],float)
        old=np.asarray(head['tip_px'])-origin
        for eid in seeds:
            if eid==choice['extension_id']:continue
            if eid not in native:return held('extension_seed_absent')
            ends=np.asarray(native[eid]['points_px'])-origin
            point=_intersection([old,old+axis],ends)
            if point is None:return held('unresolved_extension_direction')
            # Entire conservative intersection box must be white, not just its centre.
            v=ends[1]-ends[0];v/=np.linalg.norm(v)
            normals=np.array([[-axis[1],axis[0]],[-v[1],v[0]]])
            if abs(np.linalg.det(normals))<.98:return held('unstable_alternative_intersection')
            bounds=np.array([head['uncertainty_px'],native[eid]['uncertainty_px']])+math.sqrt(.5)
            offsets=np.array([np.linalg.solve(normals,bounds*np.array([s,t]))
                              for s,t in [(-1,-1),(-1,1),(1,1),(1,-1)]])
            polygon=point+offsets
            lo=np.floor(polygon.min(0)).astype(int);hi=np.ceil(polygon.max(0)+1).astype(int)
            if (lo<0).any() or (hi>gray.shape[::-1]).any():return held('alternative_intersection_clipped')
            possible=np.zeros_like(empty);cv2.fillConvexPoly(possible,np.rint(polygon).astype(int),1)
            occupied=int(np.count_nonzero(ink&(possible>0)))
            alternatives.append(dict(head_id=hid,extension_id=eid,box_px=[*lo.tolist(),*hi.tolist()],
                polygon_px=polygon.tolist(),uncertainty_strip_bounds_px=bounds.tolist(),ink_pixels=occupied))
            row['alternative_extension_checks']=alternatives
            if occupied:return held('alternative_extension_not_excluded_by_source_background')
        options.append(choice)
    tips=np.array([o['intersection_px'] for o in options]);delta=tips[1]-tips[0]
    span=np.linalg.norm(delta)
    if span<=0:return held('zero_span')
    axis=delta/span
    # Current bounded implementation supports near-horizontal linear dimensions.
    if abs(axis[0])<.999:return held('non_horizontal_short_dimension')
    directions=[np.asarray(o['candidate']['direction']) for o in options]
    if directions[0]@directions[1]>-.98 or any(abs(d@axis)<.98 for d in directions):return held('heads_not_opposed')
    thin=max(float(measured[o['extension_id']]['width_px']) for o in options)
    if span<8*thin:return held('source_pair_not_separated')
    # Different heads at the same contact are retained; a true inner contact is a veto.
    inner=[]
    for h in heads.values():
        if h['id'] in context['head_ids']:continue
        p=np.asarray(h['tip_px'])-origin
        along=(p-tips[0])@axis;across=abs(np.linalg.det(np.column_stack((axis,p-tips[0]))))
        bound=float(h['uncertainty_px'])+2*thin
        if bound<along<span-bound and across<bound and abs(np.asarray(h['direction'])@axis)>.98:inner.append(h['id'])
    if inner:return held('intervening_distinct_head')
    candidates=[]
    for line in measured.values():
        ends=np.asarray(line['points_px']);v=ends[1]-ends[0];v/=np.linalg.norm(v)
        if abs(v@axis)<.999 or line['width_px']<POLICY['minimum_pen_ratio']*thin:continue
        points=[_intersection(ends,measured[o['extension_id']]['points_px']) for o in options]
        if any(p is None for p in points):continue
        # Both anchors must lie on the same side, sufficiently outside the head ink.
        offsets=[p[1]-t[1] for p,t in zip(points,tips)]
        if offsets[0]*offsets[1]<=0 or min(abs(x) for x in offsets)<4*thin:continue
        if any(np.min(ends[:,0])-thin>p[0] or np.max(ends[:,0])+thin<p[0] for p in points):continue
        key=tuple(np.rint(np.array(points).ravel()*2).astype(int))
        if not any(x[0]==key for x in candidates):candidates.append((key,line,points))
    if len(candidates)!=1:return held('unique_thick_straight_anchor_missing')
    _,contour,anchors=candidates[0]
    spans=[_span(ink.astype('uint8')*255,*tips,thin/2+math.sqrt(.5))]
    spans.extend(_span(ink.astype('uint8')*255,t,a,thin/2+math.sqrt(.5)) for t,a in zip(tips,anchors))
    row['source_span_support']=spans
    if not all(s['passed'] for s in spans):return held('source_body_gap')
    witness=_part_witness(source,source_hash,size,np.array(anchors)+origin,thin,geometry,roles,matrix)
    row.update(part_witness=witness,alternative_extension_checks=alternatives)
    if not witness['accepted']:return held(witness['reason'])
    blocked=np.zeros_like(gray,dtype='uint8')
    for option,hid in zip(options,context['head_ids']):
        h=option['candidate'];mask=np.zeros_like(blocked)
        cv2.fillConvexPoly(mask,np.rint(h['polygon_px']).astype(int),1)
        radius=math.ceil(heads[hid]['uncertainty_px']+thin)
        blocked|=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    output=np.zeros_like(empty)
    lines=[(2,tips,thin)]+[(3,np.array([tip,anchor]),float(measured[o['extension_id']]['width_px'])) for tip,anchor,o in zip(tips,anchors,options)]
    yy,xx=np.indices(gray.shape);sites=np.stack([xx,yy],axis=-1)
    working=transform((sites+origin).reshape(-1,2),np.linalg.inv(matrix)).reshape(*gray.shape,2)
    q=np.rint(working).astype(int)
    inside=(q[...,0]>=0)&(q[...,1]>=0)&(q[...,0]<roles.shape[1])&(q[...,1]<roles.shape[0])
    allowed=np.zeros_like(ink);allowed[inside]=np.isin(roles[q[...,1][inside],q[...,0][inside]],[255,2,3])
    for role,ends,width in lines:
        a,b=ends;v=b-a;l=float(v@v)
        t=np.clip(((sites-a)@v)/l,0,1)
        selected=(np.linalg.norm(sites-(a+t[...,None]*v),axis=-1)<=width/2+math.sqrt(.5))&ink&~(blocked>0)&allowed
        # Keep thick contour ink and its junction separate from extension bodies.
        c=np.asarray(contour['points_px']);u=c[1]-c[0];u/=np.linalg.norm(u)
        distance=abs((sites-c[0])@np.array([-u[1],u[0]]))
        selected&=distance>contour['width_px']/2+math.sqrt(.5)
        output[selected]=role
    if not all(np.any(output==r) for r in (2,3)):return held('no_safe_source_body')
    row.update(status='source_short_linear_bodies_admitted',tips=(tips+origin).tolist(),
        anchors=(np.array(anchors)+origin).tolist(),dimension=[(tips+origin).tolist()],
        extensions=[(np.array([t,a])+origin).tolist() for t,a in zip(tips,anchors)],
        body_pixels={str(r):int(np.count_nonzero(output==r)) for r in (2,3)},
        contour_owner=contour['id'],roles_are_native_source_bodies=True,
        source_roles_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
        head_protection_sha256=hashlib.sha256(blocked.tobytes()).hexdigest(),
        working_PART_not_modified=True,physical_geometry_approved=False)
    return dict(report=row,roles=output)
