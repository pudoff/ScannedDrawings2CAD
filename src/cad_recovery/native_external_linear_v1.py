"""External short dimensions with distinct measured contacts and a closed wall.

The closed nonrectangular wall witness is independent of dimension semantics.
Existing straight/ARC rules remain in their frozen modules.
"""
from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np
from PIL import Image

from . import native_short_linear_v1 as previous
from .native_dimension_context_v1 import _native_geometry,_span
from .source_arrow_observation_v1 import transform,digest,checked_matrix
from .part_admission_v1 import object_sha

POLICY=dict(maximum_roi_pixels=4_000_000, maximum_scan_in_pens=192,
    minimum_wall_pen_ratio=1.5,minimum_parallel_run_in_spans=6,
    minimum_body_sections=24,maximum_body_pen=16,
    maximum_cap_bands=8,maximum_contours=1024)


def _runs(v):
    d=np.diff(np.r_[False,v,False].astype('int8'))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))


def _section(ink,point,normal,radius):
    offsets=np.arange(-radius,radius+1)
    q=np.rint(np.asarray(point)+offsets[:,None]*normal).astype(int)
    if (q<0).any() or (q>=np.asarray(ink.shape[::-1])).any():return None
    v=ink[q[:,1],q[:,0]];mid=radius
    if not v[mid]:return None
    lo=hi=mid
    while lo and v[lo-1]:lo-=1
    while hi<len(v)-1 and v[hi+1]:hi+=1
    if lo==0 or hi==len(v)-1:return None
    return dict(width=hi-lo+1,center=(q[lo]+q[hi])/2)


def _wall(source,box,tips,pen):
    """Require a real closed hole with a nonconvex wall outline and parallel legs."""
    with Image.open(source) as im:gray=np.array(im.crop(box).convert('L'))
    ink=previous._ink(gray);origin=np.array(box[:2]);p=tips-origin
    radius=math.ceil(pen/2)
    opened=cv2.morphologyEx(ink.astype('uint8'),cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    contours,tree=cv2.findContours(opened,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    report=dict(accepted=False,box=box,source_pixels_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
        opening_is_analysis_only=True,opening_radius_px=radius,kind='closed_nonrectangular_wall',
        inferred_from_arrowheads=False)
    if len(contours)>POLICY['maximum_contours']:return None,dict(report,reason='wall_contour_budget')
    if tree is None:return None,dict(report,reason='closed_wall_missing')
    # Scan transverse bridges below both contacts. Cap must be distinct from the carrier.
    x0,x1=sorted(p[:,0]);xs=np.arange(math.ceil(x0+pen),math.floor(x1-pen)+1)
    low=math.ceil(max(p[:,1])+4*pen)
    high=min(ink.shape[0],math.ceil(max(p[:,1])+POLICY['maximum_scan_in_pens']*pen))
    if len(xs)<2 or high<=low:return None,dict(report,reason='wall_cap_domain_missing')
    ys=np.arange(low,high)
    bridges=ink[ys[:,None],xs[None,:]].mean(1)>=.95
    bands=_runs(bridges)
    if len(bands)>POLICY['maximum_cap_bands']:return None,dict(report,reason='cap_band_budget')
    options=[]
    for lo,hi in bands:
        thickness=int(hi-lo)
        if thickness<POLICY['minimum_wall_pen_ratio']*pen:continue
        y=(ys[lo]+ys[hi-1])/2
        anchors=np.column_stack((p[:,0],np.full(2,y)))
        cap_center=((x0+x1)/2,y)
        matching=[]
        for i,c in enumerate(contours):
            if tree[0,i,3]<0:continue
            parent=int(tree[0,i,3])
            approximation=cv2.approxPolyDP(c,pen,True)
            if len(approximation)<8 or cv2.isContourConvex(approximation):continue
            bounds=cv2.boundingRect(c)
            if bounds[3]<POLICY['minimum_parallel_run_in_spans']*(x1-x0):continue
            # The candidate cap must touch this hole boundary, rather than a remote hole.
            if abs(cv2.pointPolygonTest(c,(float(cap_center[0]),float(cap_center[1])),True))>2*thickness:continue
            if min(abs(y-bounds[1]),abs(y-(bounds[1]+bounds[3])))>2*thickness:continue
            # Both legs persist at the same pen below the cap. A plain bar is insufficient.
            widths=[];valid=True
            for a in anchors:
                samples=[]
                for d in np.linspace(3*thickness,POLICY['minimum_parallel_run_in_spans']*(x1-x0),32):
                    s=_section(ink,a+[0,d],np.array([1.,0.]),math.ceil(4*thickness))
                    if s is None or not .75*thickness<=s['width']<=1.5*thickness:valid=False;break
                    samples.append(s['width'])
                widths.append(samples)
            if valid:matching.append(dict(hole_index=i,parent_index=parent,hole_area_px=float(cv2.contourArea(c)),
                hole_vertices=len(approximation),hole_box=list(bounds),leg_widths_px=widths))
        if len(matching)==1:options.append((anchors+origin,thickness,matching[0]))
    if len(options)!=1:return None,dict(report,reason='unique_closed_wall_cap_missing',cap_bands=len(bands),qualified_caps=len(options))
    anchors,thickness,proof=options[0]
    # Actual cap and both boundary contacts must be on one opened ink component.
    _,labels=cv2.connectedComponents(opened,8)
    q=np.rint(anchors-origin).astype(int);ids=labels[q[:,1],q[:,0]]
    if not ids[0] or ids[0]!=ids[1]:return None,dict(report,reason='cap_contacts_not_connected')
    report.update(accepted=True,cap_width_px=thickness,anchors_px=anchors.tolist(),
        topology=proof,reason='closed_nonrectangular_wall_and_two_persistent_legs')
    return anchors,report


def observe_context(gray,context,heads,geometry,roles,matrix,*,source,source_hash,size):
    bindings=context.get('head_extension_bindings',[])
    choices=[[o for b in bindings if b['head_id']==hid for o in b['extension_options']
              if o['source_shape_rechecked']] for hid in context['head_ids']]
    if any(len(c)!=1 for c in choices):
        return previous.observe_context(gray,context,heads,geometry,roles,matrix,
            source=source,source_hash=source_hash,size=size)
    options=[c[0] for c in choices];origin=np.asarray(context['source_box_px'][:2])
    tips=np.array([o['intersection_px'] for o in options])+origin
    delta=tips[1]-tips[0];span=float(np.linalg.norm(delta))
    axis=delta/span if span else np.zeros(2)
    directions=[np.asarray(o['candidate']['direction']) for o in options]
    if span==0 or abs(axis[0])<.999 or directions[0]@axis<.98 or directions[1]@axis>-.98:
        return previous.observe_context(gray,context,heads,geometry,roles,matrix,
            source=source,source_hash=source_hash,size=size)
    row=dict(id=context['id'],status='withheld',reasons=[],head_ids=context['head_ids'],
        parent_context_sha256=object_sha(context),source_sha256=source_hash,
        coordinate_space='original_source_pixels',geometry_sha256=object_sha(geometry),
        input_roles_sha256=hashlib.sha256(roles.tobytes()).hexdigest(),working_to_source=np.asarray(matrix).tolist(),
        arrangement='external_heads')
    empty=np.zeros_like(gray,dtype='uint8')
    def held(reason):row['reasons'].append(reason);return dict(report=row,roles=empty)
    if digest(source)!=source_hash or hashlib.sha256(gray.tobytes()).hexdigest()!=context['source_window_sha256']:
        raise ValueError('Current source/context mismatch')
    checked_matrix(matrix)
    measured={c['id']:c for c in context['stroke_observation']['lines']}
    # Every alternative must have been shape-tested, not silently discarded.
    alternatives=[]
    for hid,seeds in zip(context['head_ids'],context['extension_seed_ids']):
        observed={o['extension_id']:o for b in bindings if b['head_id']==hid for o in b['extension_options']}
        if not set(seeds)<=observed.keys():return held('external_extension_domain_incomplete')
        for eid in seeds:
            o=observed[eid]
            if not o['source_shape_rechecked']:
                if o['recheck_reason'] not in {'native_triangle_not_filled','apex_without_native_ink'}:
                    return held('external_alternative_not_shape_excluded')
                alternatives.append(dict(head_id=hid,extension_id=eid,reason=o['recheck_reason']))
    # Carrier width is measured independently from the extension/contour mixture.
    pens=[float(o['candidate']['evidence']['shaft_width_px']) for o in options]
    pen=max(pens)
    if pen<=0 or pen>POLICY['maximum_body_pen']:return held('external_pen_budget')
    # Keep the entire inherited head bound as well as the measured extension
    # bound. Separation does not rely on shrinking an uncalibrated head error.
    intervals=[]
    for t,o,hid in zip(tips,options,context['head_ids']):
        e=measured[o['extension_id']];a,b=np.array(e['points_px']);u=(b-a)/np.linalg.norm(b-a)
        sine=abs(np.linalg.det(np.column_stack((axis,u))))
        if sine<.98:return held('unstable_external_contact')
        bound=(float(e['uncertainty_px'])+pen/2+math.sqrt(.5)+float(heads[hid]['uncertainty_px']))/sine
        center=float(t@axis);intervals.append([float(center-bound),float(center+bound)])
    row.update(contact_intervals_along_axis_px=intervals,alternative_checks=alternatives,
        inherited_head_uncertainties_px=[float(heads[h]['uncertainty_px']) for h in context['head_ids']])
    if intervals[0][1]>=intervals[1][0]:return held('external_contacts_not_separated')
    # Source window extends to include both immutable carrier intervals.
    native={c['id']:c for c in _native_geometry(geometry,matrix)}
    carriers=[]
    for o in options:
        ids=o['candidate']['evidence']['geometry_ids']
        if len(ids)!=1 or ids[0] not in native:return held('external_carrier_owner_missing')
        carriers.append(native[ids[0]])
    cloud=np.concatenate([np.asarray(c['points_px']) for c in carriers]+[tips])
    parent=context['source_box_px']
    box=[min(parent[0],math.floor(cloud[:,0].min()-2*pen)),parent[1],
         max(parent[2],math.ceil(cloud[:,0].max()+2*pen+1)),parent[3]]
    box=[max(0,box[0]),max(0,box[1]),min(size[0],box[2]),min(size[1],box[3])]
    if (box[2]-box[0])*(box[3]-box[1])>POLICY['maximum_roi_pixels']:return held('external_body_window_budget')
    witness_box=[max(0,parent[0]-math.ceil(192*pen)),parent[1],min(size[0],parent[2]+math.ceil(192*pen)),parent[3]]
    if (witness_box[2]-witness_box[0])*(witness_box[3]-witness_box[1])>POLICY['maximum_roi_pixels']:return held('external_wall_window_budget')
    anchors,witness=_wall(source,witness_box,tips,pen)
    row['part_witness']=witness
    if anchors is None:return held(witness['reason'])
    with Image.open(source) as im:local=np.array(im.crop(box).convert('L'))
    local_ink=previous._ink(local);origin=np.array(box[:2]);blocked=np.zeros_like(local,dtype='uint8')
    dimension=[]
    for t,o,c in zip(tips,options,carriers):
        polygon=np.asarray(o['candidate']['polygon_px'])+np.array(parent[:2])
        footprint=np.zeros_like(blocked);cv2.fillConvexPoly(footprint,np.rint(polygon-origin).astype(int),1)
        radius=math.ceil(float(heads[o['candidate']['id']]['uncertainty_px'])+pen)
        blocked|=cv2.dilate(footprint,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
        ends=np.asarray(c['points_px']);base=polygon[1:].mean(0)
        # The finite native owner restricts the outer shaft; never extend it.
        outer=ends[np.argmax(np.linalg.norm(ends-t,axis=1))]
        dimension.append(np.array([outer,base]))
    extensions=[np.array([t,a]) for t,a in zip(tips,anchors)]
    paths=[(2,e) for e in dimension]+[(3,e) for e in extensions]
    support=[_span(local_ink.astype('uint8')*255,*(e-origin),pen/2+math.sqrt(.5)) for _,e in paths]
    row['source_span_support']=support
    if not all(s['passed'] for s in support):return held('external_body_gap')
    yy,xx=np.indices(local.shape);sites=np.stack([xx,yy],axis=-1)
    q=np.rint(transform((sites+origin).reshape(-1,2),np.linalg.inv(matrix))).astype(int).reshape(*local.shape,2)
    inside=(q[...,0]>=0)&(q[...,1]>=0)&(q[...,0]<roles.shape[1])&(q[...,1]<roles.shape[0])
    allowed=np.zeros_like(local_ink);allowed[inside]=np.isin(roles[q[...,1][inside],q[...,0][inside]],[255,2,3])
    output=np.zeros_like(local,dtype='uint8')
    for role,e in paths:
        a,b=e-origin;v=b-a;l=float(v@v)
        if l<=0:return held('external_zero_body')
        t=np.clip(((sites-a)@v)/l,0,1)
        selected=(np.linalg.norm(sites-(a+t[...,None]*v),axis=-1)<=pen/2+math.sqrt(.5))&local_ink&allowed&(blocked==0)
        # Entire cap/contact strip is excluded from extension ownership.
        selected&=abs((sites+origin)[...,1]-anchors[0,1])>witness['cap_width_px']/2+math.sqrt(.5)
        output[selected]=role
    if not all(np.any(output==r) for r in [2,3]):return held('external_no_safe_body')
    row.update(status='source_short_linear_bodies_admitted',source_box_px=box,
        source_window_sha256=hashlib.sha256(local.tobytes()).hexdigest(),tips=tips.tolist(),
        anchors=anchors.tolist(),dimension=[e.tolist() for e in dimension],extensions=[e.tolist() for e in extensions],
        body_pixels={str(r):int(np.count_nonzero(output==r)) for r in [2,3]},
        source_roles_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
        head_protection_sha256=hashlib.sha256(blocked.tobytes()).hexdigest(),
        roles_are_native_source_bodies=True,working_PART_not_modified=True,physical_geometry_approved=False)
    return dict(report=row,roles=output)
