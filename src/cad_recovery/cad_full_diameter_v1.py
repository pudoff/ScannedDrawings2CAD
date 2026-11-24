"""Native diameter presentation with a complete shaft and opposed outside heads."""
import math
import cv2
import numpy as np
from .exporter_v2 import dimension_display_text
from .cad_object_assembly_v1 import compact


def validate(packet,annotations):
    heads={h['id']:h for h in annotations['heads']};geometry={c['id']:c for c in packet['geometry']}
    gray=packet['gray'];_,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    distance=cv2.distanceTransform(255-ink,cv2.DIST_L2,5)
    for row in annotations['dimensions']:
        if not row.get('full_span_two_heads'):continue
        if row['kind']!='diameter' or len(row.get('head_ids',[]))!=2 or not row.get('source_candidate_ids'):
            raise ValueError('Full diameter requires two bound heads and source shafts')
        if not set(row.get('auxiliary_source_candidate_ids',[]))<=set(row['source_candidate_ids']):
            raise ValueError('Foreign diameter presentation source')
        c,q=np.asarray(row['points_px'],float);tips=np.array([q,2*c-q]);actual=[]
        for identity in row['head_ids']:
            h=heads[identity];p=np.asarray(h['tip_px'],float);direction=np.asarray(h['direction'],float)
            if min(np.linalg.norm(p-t) for t in tips)>2:raise ValueError('Diameter head is not at a defining endpoint')
            if direction.shape!=(2,) or not np.isfinite(direction).all() or np.linalg.norm(direction)==0:
                raise ValueError('Invalid diameter head direction')
            direction=direction/np.linalg.norm(direction)
            inward=c-p
            if float(direction@inward)/np.linalg.norm(inward)<math.cos(math.radians(20)):
                raise ValueError('Diameter head points away from its supported circle')
            actual.append(p)
        if np.linalg.norm(actual[0]-actual[1])<np.linalg.norm(tips[0]-tips[1])-4:
            raise ValueError('Diameter heads are not opposed')
        samples=np.linspace(tips[0],tips[1],max(3,int(np.linalg.norm(tips[0]-tips[1])*2)))
        if len(samples)>120000:raise ValueError('Diameter support budget')
        pixels=np.rint(samples).astype(int)
        if (pixels<0).any() or (pixels[:,0]>=gray.shape[1]).any() or (pixels[:,1]>=gray.shape[0]).any():
            raise ValueError('Diameter outside source')
        if distance[pixels[:,1],pixels[:,0]].max()>2:raise ValueError('Diameter shaft crosses unsupported paper')
        if not set(row['source_candidate_ids'])<=geometry.keys():raise ValueError('Foreign diameter source')


def render(entity,*,auxiliary_paths=(),height=None,text_location_px=None):
    """Keep DIMENSION definitions native; correct the source-bound display block.

    A CAD editor can regenerate this diameter from its native definitions and
    standard DIMTOFL/DIMTOH properties. This function also works after an edit.
    """
    if entity.dxftype()!='DIMENSION' or entity.dxf.dimtype&7!=3:raise ValueError('Native diameter required')
    override=entity.override();override.update({'dimtofl':1,'dimtoh':1,'dimtih':0,'dimtmove':0,'dimtix':0})
    override.commit();entity.dxf.dimtype &= ~128;entity.render()
    block=entity.get_geometry_block();a=np.array(entity.dxf.defpoint,float);b=np.array(entity.dxf.defpoint4,float);center=(a+b)/2
    arrows=list(block.query('INSERT'))
    if len(arrows)!=2:raise ValueError('Full diameter did not produce two native heads')
    for arrow in arrows:
        p=np.array(arrow.dxf.insert,float);v=center-p
        if min(np.linalg.norm(p-t) for t in (a,b))>1e-7:raise ValueError('Diameter arrow moved off its definition')
        arrow.dxf.rotation=math.degrees(math.atan2(v[1],v[0]))
    for line in list(block.query('LINE')):block.delete_entity(line)
    attrs={'layer':'0','color':0,'lineweight':13}
    block.add_line(a,b,dxfattribs=attrs)
    for path in auxiliary_paths:
        if height is None:raise ValueError('Auxiliary diameter sources require a pixel frame')
        vertices=[(float(p[0]),height-1-float(p[1])) for p in compact(np.asarray(path,int))]
        block.add_lwpolyline(vertices,dxfattribs=attrs)
    if text_location_px is not None:
        if height is None:raise ValueError('Diameter text requires a pixel frame')
        location=(float(text_location_px[0]),height-1-float(text_location_px[1]),0.)
        for text in block.query('MTEXT'):
            text.dxf.insert=location;text.dxf.attachment_point=5
        entity.dxf.text_midpoint=location;entity.dxf.dimtype |= 128
    return {'head_count':2,'full_shaft_length_px':float(np.linalg.norm(b-a)),
        'computed_display':dimension_display_text(entity),'native_diameter_preserved':True}
