"""Reviewed pixel-role replacement and styling; no physical dimension inference."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import cv2
import ezdxf
import numpy as np
from .demo_export_v1 import merged_runs


def render_review_dxf(path,target):
    """Actual reopened DXF, fixed recorded bounds, absolute paper lineweights."""
    from ezdxf.addons.drawing import Frontend,RenderContext,layout
    from ezdxf.addons.drawing.config import Configuration,BackgroundPolicy,ColorPolicy,LineweightPolicy
    from ezdxf.addons.drawing.svg import SVGBackend
    from ezdxf.math import BoundingBox2d
    import fitz
    doc=ezdxf.readfile(path);backend=SVGBackend()
    Frontend(RenderContext(doc),backend,config=Configuration(background_policy=BackgroundPolicy.WHITE,
        color_policy=ColorPolicy.BLACK,lineweight_policy=LineweightPolicy.ABSOLUTE)).draw_layout(doc.modelspace())
    bounds=BoundingBox2d([tuple(doc.header['$EXTMIN'])[:2],tuple(doc.header['$EXTMAX'])[:2]])
    svg=backend.get_string(layout.Page(210,297),render_box=bounds)
    tree=ET.fromstring(svg)
    rules={key:value for e in tree.iter() if e.tag.endswith('style')
           for key,value in re.findall(r'\.([A-Za-z0-9_-]+)\s*\{([^}]+)\}',e.text or '')}
    for element in tree.iter():
        for cls in element.get('class','').split():
            for part in rules.get(cls,'').split(';'):
                if ':' in part:
                    key,value=part.split(':',1);element.set(key.strip(),value.strip())
    with fitz.open(stream=ET.tostring(tree),filetype='svg') as rendered:
        Path(target).write_bytes(rendered[0].get_pixmap(matrix=fitz.Matrix(1.6,1.6),alpha=False).tobytes('png'))


def replace_reviewed_ink(ink,protected,review,source_sha256):
    if ink.dtype!=bool or ink.ndim!=2 or ink.size>1_000_000 or protected.dtype!=bool or protected.shape!=ink.shape:
        raise ValueError('Expected bounded boolean source/protection masks')
    if review['source_sha256']!=source_sha256 or review['size_px']!=[ink.shape[1],ink.shape[0]]:
        raise ValueError('Review does not bind this source')
    items=review['items']
    if len(items)>64 or len({r['id'] for r in items})!=len(items):raise ValueError('Duplicate IDs or item budget exceeded')
    removed=np.zeros_like(ink);accepted=[];ledger=[];used=[]
    for row in items:
        reason=None;kind=row['kind'];points=np.asarray(row['points_px'],float)
        if row.get('status')!='reviewed_pixel_role' or not row.get('reviewer'):
            reason='role_not_reviewed'
        elif kind not in ('line','arrow_open','arrow_filled'):reason='unknown_arrow_or_primitive_type'
        elif row.get('role') not in (('dimension_line','extension_line') if kind=='line' else ('arrowhead',)):
            reason='unsupported_role'
        if reason:
            ledger.append({'id':row['id'],'status':'needs_review','reason':reason});continue
        if points.shape!=((2,2) if kind=='line' else (3,2)) or not np.isfinite(points).all():raise ValueError('Invalid primitive points')
        if np.any(points<0) or np.any(points[:,0]>=ink.shape[1]) or np.any(points[:,1]>=ink.shape[0]):raise ValueError('Points outside source')
        if kind=='line' and np.linalg.norm(points[1]-points[0])<1:raise ValueError('Degenerate line')
        if kind!='line' and abs(cv2.contourArea(points.astype(np.float32)))<.25:raise ValueError('Degenerate arrow')
        l,t,r,b=row['owned_box']
        if any(type(x)!=int for x in [l,t,r,b]) or not (0<=l<r<=ink.shape[1] and 0<=t<b<=ink.shape[0]):raise ValueError('Invalid ownership box')
        owner=np.zeros_like(ink);owner[t:b,l:r]=True
        paint=np.zeros(ink.shape,dtype=np.uint8)
        rounded=np.rint(points).astype('int32')
        if kind=='line':cv2.line(paint,tuple(rounded[0]),tuple(rounded[1]),1,1)
        elif kind=='arrow_filled':cv2.fillPoly(paint,[rounded],1)
        else:
            cv2.line(paint,tuple(rounded[0]),tuple(rounded[1]),1,1)
            cv2.line(paint,tuple(rounded[0]),tuple(rounded[2]),1,1)
        contacts=np.zeros_like(paint)
        for point in row.get('shared_contacts',[]):
            if (len(point)!=2 or not np.isfinite(point).all() or min(point)<0
                    or point[0]>=ink.shape[1] or point[1]>=ink.shape[0]):raise ValueError('Invalid contact')
            cv2.circle(contacts,tuple(np.rint(point).astype(int)),1,1,-1)
        footprint=paint>0
        if np.any(footprint&~owner&~(contacts>0)):reason='vector_outside_reviewed_ownership'
        elif np.any(footprint&protected&~(contacts>0)):reason='protected_text_or_geometry_conflict'
        elif any(np.any(owner&oldowner) for oldowner in used):reason='ambiguous_overlapping_ownership'
        owned_ink=ink&owner&~protected
        if not owned_ink.any():reason=reason or 'no_source_ink'
        reference=(owned_ink|(ink&protected&(contacts>0))).astype('uint8')
        distance=cv2.distanceTransform(1-reference,cv2.DIST_L2,5)
        errors=distance[footprint]
        if errors.size==0 or float(errors.max())>1.5:reason=reason or 'unsupported_or_gap_bridging_primitive'
        if reason:
            ledger.append({'id':row['id'],'status':'needs_review','reason':reason});continue
        # Only ink near the reviewed primitive can be erased, never an entire
        # arbitrary ROI containing other useful objects.
        radius=row.get('erase_radius_px',1)
        if type(radius)!=int or not 1<=radius<=3:raise ValueError('Invalid reviewed erase radius')
        corridor=cv2.dilate(paint,np.ones((2*radius+1,2*radius+1),dtype=np.uint8))>0
        if np.any(owned_ink&~corridor):
            ledger.append({'id':row['id'],'status':'needs_review','reason':'mixed_owned_region'});continue
        erase=owned_ink&corridor;removed|=erase;used.append(owner)
        item=copy.deepcopy(row);item['physical_geometry_approved']=False;accepted.append(item)
        vector_info={}
        if kind!='line':
            direction=points[0]-(points[1]+points[2])/2
            vector_info={'direction':(direction/np.linalg.norm(direction)).tolist(),
                         'length_px':float(np.linalg.norm(direction)),
                         'base_width_px':float(np.linalg.norm(points[2]-points[1]))}
        ledger.append({'id':row['id'],'status':'pixel_replacement_ready','removed_pixels':int(erase.sum()),
            'source_support_max_error_px':float(errors.max()),'shared_contact_ink_retained':int((ink&protected&footprint).sum()),
            **vector_info})
    derivative=ink&~removed
    if np.any(removed&protected) or np.any(derivative&~ink):raise ValueError('Protection invariant failed')
    return derivative,removed,accepted,ledger


def raster_ink_from_dxf(doc,shape):
    height,width=shape;ink=np.zeros(shape,dtype=bool)
    for solid in doc.modelspace().query('SOLID[layer=="SCAN_INK_DRAFT"]'):
        p=np.asarray([tuple(solid.dxf.get(f'vtx{i}'))[:2] for i in range(4)])
        l,r=np.rint(p[:,0].min()+.5),np.rint(p[:,0].max()+.5)
        t,b=np.rint(height-.5-p[:,1].max()),np.rint(height-.5-p[:,1].min())
        l,t,r,b=map(int,[l,t,r,b])
        if not (0<=l<r<=width and 0<=t<b<=height):raise ValueError('Invalid source ink SOLID')
        ink[t:b,l:r]=True
    return ink


def export_styled_review(source_dxf,output,original,derivative,items):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    doc=ezdxf.readfile(source_dxf);model=doc.modelspace();height,width=original.shape
    if not np.array_equal(raster_ink_from_dxf(doc,original.shape),original):raise ValueError('DXF source differs from mask')
    if np.any(derivative&~original):raise ValueError('Unexpected added raster pixels')
    for e in list(model.query('SOLID[layer=="SCAN_INK_DRAFT"]')):model.delete_entity(e)
    for l,t,r,b in merged_runs(derivative):
        model.add_solid([(l-.5,height-.5-b),(r-.5,height-.5-b),(l-.5,height-.5-t),(r-.5,height-.5-t)],
                        dxfattribs={'layer':'SCAN_INK_DRAFT'})
    for layer,weight in [('ANNOTATION_LINES_REVIEW',13),('ANNOTATION_HEADS_REVIEW',13)]:
        if layer in doc.layers:raise ValueError('Review layers already exist')
        doc.layers.new(layer,dxfattribs={'lineweight':weight,'color':7})
    def point(p):return float(p[0]),height-1-float(p[1])
    for row in items:
        points=[point(p) for p in row['points_px']];kind=row['kind']
        attrs={'layer':'ANNOTATION_LINES_REVIEW' if kind=='line' else 'ANNOTATION_HEADS_REVIEW',
               'lineweight':-1,'color':256,'linetype':'BYLAYER'}
        if kind=='line':model.add_line(*points,dxfattribs=attrs)
        elif kind=='arrow_open':
            for p in points[1:]:model.add_line(points[0],p,dxfattribs=attrs)
        elif kind=='arrow_filled':model.add_solid(points,dxfattribs=attrs)
        else:raise ValueError('Unsupported reviewed type')
    doc.header['$LWDISPLAY']=True;doc.header['$FILLMODE']=1
    doc.saveas(output)
    reopened=ezdxf.readfile(output);audit=reopened.audit()
    if audit.errors or audit.fixes:raise ValueError('Styled DXF requires repairs')
    if not np.array_equal(raster_ink_from_dxf(reopened,derivative.shape),derivative):raise ValueError('Derivative roundtrip mismatch')
    if reopened.modelspace().query('DIMENSION IMAGE UNDERLAY'):raise ValueError('Pixel draft must not imply physical dimensions')
    render_review_dxf(output,output.with_suffix('.png'))
    return {'dxf_version':reopened.dxfversion,'dxf_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'status':'needs_review','physical_geometry_approved':False,'native_dimensions':0,
        'mask_roundtrip_exact':True,'lineweight_mm':.13,'model_units':'unitless processed pixels',
        'trueview_gui_verified':False,'audit_errors':0,'audit_fixes':0,
        'lineweight_display':True,'fillmode':1,'source_view_and_layout_preserved':True}


def style_verified_native_dimensions(doc,*,verified_fixture=False):
    """Style an already verified physical fixture; never infer its scale/anchors."""
    if verified_fixture is not True:raise ValueError('Verified physical fixture required')
    from .exporter_v2 import dimension_display_text,rerender_native_dimension
    results=[]
    for entity in doc.modelspace().query('DIMENSION'):
        before_text=dimension_display_text(entity);measurement=entity.get_measurement()
        before_def={k:entity.dxf.get(k) for k in ['defpoint','defpoint2','defpoint3','defpoint4','angle'] if entity.dxf.hasattr(k)}
        override=entity.override();arrow=override.get('dimblk');size=override.get('dimasz')
        override.update({'dimlwd':13,'dimlwe':13});override.commit()
        text=rerender_native_dimension(entity)
        if text!=before_text or not np.allclose(entity.get_measurement(),measurement):raise ValueError('Styling changed dimension semantics')
        if any(entity.dxf.get(k)!=v for k,v in before_def.items()):raise ValueError('Styling changed definition points')
        results.append({'handle':entity.dxf.handle,'text':text,'arrow_type':arrow,'arrow_size':size,
                        'dimlwd':entity.override().get('dimlwd'),'dimlwe':entity.override().get('dimlwe')})
    doc.header['$LWDISPLAY']=True
    return results
