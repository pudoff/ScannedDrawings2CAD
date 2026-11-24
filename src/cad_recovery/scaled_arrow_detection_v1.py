"""Stroke-normalized local profiles, mapped back to original pixel centers."""
from copy import deepcopy
from pathlib import Path
import hashlib
import math
import cv2
import numpy as np
from PIL import Image
from . import dimension_association_v3 as v3
from . import arrow_detection as old

POLICY = {'version':'scaled-arrow-detection/1.0','canonical_stroke_px':2.,
          'canonical_tip_endpoint_width_px':.5,
          'canonical_margin_px':96,'canonical_radius_px':40,
          'canonical_head_length_px':[4.,160.],'canonical_taper_slope':[.12,2.5],
          'stroke_probe_radius_px':64,'maximum_stroke_probe_centers':128,
          'maximum_stroke_width_px':48.,'maximum_lines':256,
          'maximum_samples':12_000_000,'maximum_strip_pixels':1_000_000,
          'maximum_canonical_line_length':8192,'maximum_pixels':12_000_000,
          'production_admitted':False,'physical_geometry_approved':False}


class Limit(Exception):
    pass


def _stroke(foreground, line, account):
    first,second = line['points'];length = line['length'];unit = line['u']
    normal = np.array([-unit[1],unit[0]])
    count = min(POLICY['maximum_stroke_probe_centers'],max(2,math.ceil(length)))
    centers = first + np.linspace(0,length,count)[:,None]*unit
    radius = POLICY['stroke_probe_radius_px'];offsets = np.arange(-radius,radius+1)
    account(count*len(offsets))
    xy = centers[:,None,:]+offsets[None,:,None]*normal
    ix = np.rint(xy).astype(int);h,w = foreground.shape
    valid = (ix>=0).all(axis=2)&(ix[:,:,0]<w)&(ix[:,:,1]<h)
    ink = np.zeros(valid.shape,bool)
    ink[valid] = foreground[ix[:,:,1][valid],ix[:,:,0][valid]]>0
    widths = []
    for row in ink:
        # A local center fitting error is bounded by the line's localization
        # estimate, rather than a distant black component in the same profile.
        uncertainty = line['original'].get('uncertainty_px',1.)
        if not isinstance(uncertainty,(int,float)) or not math.isfinite(uncertainty) or uncertainty<0:
            raise ValueError('Finite nonnegative line uncertainty required')
        allowance = min(radius//4,max(1,math.ceil(uncertainty)))
        candidates = [i for i in range(radius-allowance,radius+allowance+1) if row[i]]
        if not candidates:continue
        center = min(candidates,key=lambda i:abs(i-radius));lo=hi=center
        while lo>0 and row[lo-1]:lo-=1
        while hi+1<len(row) and row[hi+1]:hi+=1
        if lo>0 and hi+1<len(row):widths.append(hi-lo+1)
    if len(widths)<max(4,count*.25):return None
    width = float(np.quantile(widths,.25))
    return width if 1<=width<=POLICY['maximum_stroke_width_px'] else None


def _fit(widths, midpoints):
    supported=widths[(widths>0)&(widths<40)]
    if len(supported)<20:return []
    stroke=max(1.,min(10.,float(np.quantile(supported,.25))))
    threshold=max(stroke+2,stroke*1.6,4.)
    found=[]
    for begin,end in old._runs(widths>=threshold):
        if not 4<=end-begin<=140:continue
        x=np.arange(begin,end,dtype=float);y=widths[begin:end].astype(float)
        keep=y<60
        if keep.sum()<4:continue
        design=np.c_[x-x.mean(),np.ones(len(x))]
        fit=np.linalg.lstsq(design[keep],y[keep],rcond=None)[0]
        for _ in range(2):
            residual=np.abs(y-design@fit)
            keep &= residual<=max(2.,float(np.median(residual[keep]))*2.5)
            if keep.sum()<4:break
            fit=np.linalg.lstsq(design[keep],y[keep],rcond=None)[0]
        if keep.sum()<4 or keep.mean()<.7:continue
        slope,intercept=fit
        variance=float(np.sum((y[keep]-y[keep].mean())**2));error=float(np.sum((y[keep]-(design@fit)[keep])**2))
        r2=1-error/variance if variance else 0.
        low,high=POLICY['canonical_taper_slope']
        if not low<=abs(slope)<=high or r2<.68:continue
        if float(np.ptp(y[keep]))<3 or float(y[keep].max())<stroke+3:continue
        center_slope=float(np.linalg.lstsq(design[keep],midpoints[begin:end][keep],rcond=None)[0][0])
        if abs(center_slope)>max(.075,abs(slope)*.18):continue
        sign=1 if slope<0 else -1
        base=float(x[keep].min() if sign>0 else x[keep].max())
        tip=float(x.mean()+(POLICY['canonical_tip_endpoint_width_px']-intercept)/slope);length=abs(tip-base)
        width=float(intercept+slope*(base-x.mean()))
        if not POLICY['canonical_head_length_px'][0]<=length<=POLICY['canonical_head_length_px'][1] or not .65<=length/width<=8:continue
        back=np.rint(base-sign*np.arange(2,5)).astype(int)
        if ((back<0).any() or (back>=len(widths)).any() or np.median(widths[back])>max(stroke+2,width*.65)):continue
        shaft=max(old._shaft_support(widths,base,-sign,stroke),old._shaft_support(widths,tip,sign,stroke))
        if shaft<.75:continue
        found.append({'tip':tip,'base':base,'width':width,'sign':sign,'stroke':stroke,
            'uncertainty':max(1.5,math.sqrt(error/keep.sum())/abs(slope)+1),
            'r_squared':r2,'shaft_support':shaft,'inlier_fraction':float(keep.mean()),
            'confidence':float(min(.95,.55+.25*r2+.15*shaft))})
    return found


def detect(source_image, *, geometry, text_polygons=()):
    source_image=Path(source_image)
    common={k:geometry.get(k) for k in ('source_sha256','processed_size_px')}
    digest,size,lines,_,_=v3._validate(source_image,geometry,{**common,'lines':[],'notations':[]},{**common,'candidates':[]})
    if size[0]*size[1]>POLICY['maximum_pixels']:raise ValueError('Scaled detector raster limit exceeded')
    with Image.open(source_image) as image:gray=np.array(image.convert('L'))
    foreground=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    polygons=[v3._polygon(p,'OCR exclusion polygon',size=size).astype(np.float32) for p in text_polygons]
    candidates=[];events=[];samples=0;limited=len(lines)>POLICY['maximum_lines']
    def account(n):
        nonlocal samples
        if samples+n>POLICY['maximum_samples']:raise Limit()
        samples+=n
    try:
        if not limited:
            for line in sorted(lines,key=lambda l:l['id']):
                if (line['points']<0).any() or (line['points']>=np.array(size)).any():
                    raise ValueError('Geometry outside source raster')
                stroke=_stroke(foreground,line,account)
                event={'geometry_id':line['id'],'estimated_stroke_width_px':stroke};events.append(event)
                if stroke is None:event['status']='no_supported_stroke';continue
                scale=POLICY['canonical_stroke_px']/stroke
                length=line['length']*scale
                margin=POLICY['canonical_margin_px'];radius=POLICY['canonical_radius_px']
                width=math.ceil(length)+2*margin+1;height=2*radius+1
                if length>POLICY['maximum_canonical_line_length'] or width*height>POLICY['maximum_strip_pixels']:raise Limit()
                account(width*height)
                along=(np.arange(width)-margin)/scale
                across=(np.arange(height)-radius)/scale
                normal=np.array([-line['u'][1],line['u'][0]])
                coordinates=line['points'][0]+along[None,:,None]*line['u']+across[:,None,None]*normal
                q=np.rint(coordinates).astype(int)
                valid=(q>=0).all(axis=2)&(q[:,:,0]<size[0])&(q[:,:,1]<size[1])
                strip=np.zeros((height,width),np.uint8)
                strip[valid]=foreground[q[:,:,1][valid],q[:,:,0][valid]]
                # The old transverse-width primitive is reused in a bounded
                # rectified strip. Its 48-px margin is now an internal offset;
                # total canonical search extent is the explicit margin above.
                _,widths,mids=old._profile(strip,np.array([margin-old.MARGIN,float(radius)]),np.array([1.,0.]),length+2*(margin-old.MARGIN))
                fitted=_fit(widths,mids);event.update(status='searched',canonical_scale=scale,candidates=len(fitted),
                    strip_size_px=[width,height],strip_to_processed=[
                        [float(line['u'][0]/scale),float(normal[0]/scale),float(line['points'][0][0]-(margin*line['u'][0]+radius*normal[0])/scale)],
                        [float(line['u'][1]/scale),float(normal[1]/scale),float(line['points'][0][1]-(margin*line['u'][1]+radius*normal[1])/scale)],
                        [0.,0.,1.]])
                for f in fitted:
                    tip=line['points'][0]+(f['tip']-margin)/scale*line['u']
                    base=line['points'][0]+(f['base']-margin)/scale*line['u']
                    if (tip<0).any() or (tip>=np.array(size)).any():continue
                    polygon=np.array([tip,base+normal*f['width']/scale/2,base-normal*f['width']/scale/2])
                    if any(cv2.pointPolygonTest(p,tuple(tip),False)>=0 or cv2.pointPolygonTest(p,tuple(polygon.mean(axis=0)),False)>=0 for p in polygons):continue
                    row={'tip_px':tip.tolist(),'polygon_px':polygon.tolist(),'direction':(f['sign']*line['u']).tolist(),
                        'confidence':f['confidence'],'uncertainty_px':f['uncertainty']/scale,
                        'evidence':{'method':POLICY['version'],'geometry_ids':[line['id']],
                            'shaft_width_px':f['stroke']/scale,'estimated_source_stroke_width_px':stroke,
                            'arrow_length_px':abs(f['tip']-f['base'])/scale,'base_width_px':f['width']/scale,
                            'taper_r_squared':f['r_squared'],'shaft_support_ratio':f['shaft_support'],
                            'profile_inlier_fraction':f['inlier_fraction'],'confidence_calibrated':False,
                            'normalization':deepcopy(event)},
                        'source_ref':{'sha256':digest,'image':source_image.name,'coordinate_space':'processed_image_pixels'}}
                    duplicate=next((a for a in candidates if np.linalg.norm(np.array(a['tip_px'])-tip)<=max(a['uncertainty_px'],row['uncertainty_px'])
                        and np.dot(a['direction'],row['direction'])>.96),None)
                    if duplicate:
                        duplicate['evidence']['geometry_ids']=sorted(set(duplicate['evidence']['geometry_ids']+[line['id']]))
                    else:candidates.append(row)
    except Limit:
        limited=True;events.append({'status':'unknown','reason':'resource_limit'})
    candidates.sort(key=lambda a:(a['tip_px'][0],a['tip_px'][1],a['direction'][0],a['direction'][1]))
    for i,a in enumerate(candidates,1):a.update(id=f'scaled-arrow-{i:05d}',classification='filled_arrow_candidate')
    return {'schema_version':POLICY['version'],'source_sha256':digest,'processed_size_px':size,
        'candidates':candidates,'analysis':{'search_limited':limited,'profile_samples':samples,
            'lines_examined':sum(e.get('status')=='searched' for e in events),'full_resolution':False},
        'issues':[{'code':'arrow_search_limit'}] if limited else [],'profile_events':events,
        'policy':deepcopy(POLICY),'status':'needs_review','production_admitted':False,'physical_geometry_approved':False}
