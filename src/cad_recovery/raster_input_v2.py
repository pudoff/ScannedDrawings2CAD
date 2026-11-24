"""Shared source-bound overview/context/native-window API, with bounded memory."""
import hashlib
import io
import math
from pathlib import Path
import struct

import cv2
import numpy as np
from PIL import Image,JpegImagePlugin
from .dataset_layout_v1 import encoded
from .demo_input_v1 import _orientation_matrix,_scaled_transform
from .raster_stream_v2 import PNGReader,png_header,bounded_bytes,dimensions,orientation,POLICY,RasterInputError


def oriented(array,value):
    operations={1:lambda a:a,2:np.fliplr,3:lambda a:np.flipud(np.fliplr(a)),4:np.flipud,
        5:lambda a:a.T,6:lambda a:np.rot90(a,-1),7:lambda a:np.flipud(np.fliplr(a.T)),8:lambda a:np.rot90(a)}
    return np.ascontiguousarray(operations[value](array))


def jpeg_header(path,raw=None):
    raw=bounded_bytes(path) if raw is None else raw
    if not raw.startswith(b'\xff\xd8') or not raw.endswith(b'\xff\xd9'):raise RasterInputError('input_jpeg_format','Complete JPEG required')
    p=2;size=None;exif=None;dpi=None;markers=0
    while p<len(raw):
        if raw[p]!=255:raise RasterInputError('input_jpeg_format','Invalid JPEG marker')
        while p<len(raw) and raw[p]==255:p+=1
        if p>=len(raw):raise RasterInputError('input_jpeg_format','Truncated JPEG marker')
        marker=raw[p];p+=1;markers+=1
        if markers>10000:raise RasterInputError('input_jpeg_format','Marker budget')
        if marker==0xda:break
        if marker in (0xd8,0xd9) or 0xd0<=marker<=0xd7:raise RasterInputError('input_jpeg_format','Unexpected standalone marker')
        if p+2>len(raw):raise RasterInputError('input_jpeg_format','Missing marker length')
        n=struct.unpack_from('>H',raw,p)[0]
        if n<2 or p+n>len(raw):raise RasterInputError('input_jpeg_format','Truncated marker payload')
        data=raw[p+2:p+n];p+=n
        if marker in (0xc0,0xc1,0xc2):
            if size is not None or len(data)<6 or data[0]!=8 or data[5] not in (1,3):raise RasterInputError('input_jpeg_format','Only8-bit gray/RGB JPEG supported')
            h,w=struct.unpack_from('>HH',data,1);dimensions(w,h);size=[w,h]
        elif marker==0xe1 and data.startswith(b'Exif\0\0'):
            if exif is not None:raise RasterInputError('input_exif','Duplicate JPEG EXIF')
            exif=data;orientation(exif)
        elif marker==0xe0 and data.startswith(b'JFIF\0') and len(data)>=12:
            unit=data[7];x,y=struct.unpack_from('>HH',data,8)
            if unit in (1,2) and x and y:dpi=[x*(2.54 if unit==2 else 1),y*(2.54 if unit==2 else 1)]
    if size is None:raise RasterInputError('input_jpeg_format','Missing supported JPEG frame')
    return dict(path=str(Path(path).resolve()),sha256=hashlib.sha256(raw).hexdigest(),size_px=size,
        orientation=orientation(exif) if exif else 1,dpi_xy=dpi,bytes=len(raw),format='JPEG')


def inspect(path):
    path=Path(path);suffix=path.suffix.lower()
    if suffix=='.png':
        meta,_=png_header(path);return dict(meta,format='PNG',native_windows_available=True)
    if suffix in ('.jpg','.jpeg'):return dict(jpeg_header(path),native_windows_available=False)
    if suffix=='.pdf':
        raw=bounded_bytes(path)
        if not raw.startswith(b'%PDF-'):raise RasterInputError('input_signature','PDF signature required')
        return dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),format='PDF',bytes=len(raw),
            native_windows_available=False,pdf_page_rendering='existing bounded renderer; embedded-image native tiling is10.4.3')
    raise RasterInputError('input_extension','PNG/JPEG/PDF required')


class AreaOverview:
    """Pixel-area integration with exact fractional vertical footprints."""
    def __init__(self,w,h,target):
        self.w,self.h=w,h;self.ow,self.oh=target
        self.sums=np.zeros((self.oh,self.ow),np.float64);self.cover=np.zeros(self.oh,np.float64)
    def add(self,top,gray):
        horizontal=cv2.resize(gray.astype(np.float32),(self.ow,len(gray)),interpolation=cv2.INTER_AREA)
        y=np.arange(top,top+len(gray),dtype=float);first=np.floor(y*self.oh/self.h).astype(int)
        a=np.minimum(y+1,(first+1)*self.h/self.oh)-y;b=1-a
        np.add.at(self.sums,first,horizontal*a[:,None]);np.add.at(self.cover,first,a)
        second=(b>1e-10)&(first+1<self.oh)
        np.add.at(self.sums,first[second]+1,horizontal[second]*b[second,None]);np.add.at(self.cover,first[second]+1,b[second])
    def finish(self):
        if not np.allclose(self.cover,self.h/self.oh,atol=1e-7,rtol=1e-8):raise RasterInputError('input_coverage','Overview source coverage incomplete')
        return np.clip(np.rint(self.sums/self.cover[:,None]),0,255).astype(np.uint8)


def target_size(w,h,side):
    scale=min(1.,side/w,side/h,math.sqrt(4_000_000/(w*h)))
    return max(1,int(w*scale)),max(1,int(h*scale))


def prepare(source,folder):
    source=Path(source).resolve();folder=Path(folder);meta=inspect(source)
    if folder.exists():raise RasterInputError('input_output_exists','Input workspace must be new')
    folder.mkdir(parents=True);common=dict(schema_version='common-raster-input/2.0',source=meta,
        original_path=str(source),original_name=source.name,sha256=meta['sha256'],physical_geometry_approved=False,
        full_resolution_analysis=False,full_sheet_quality_accepted=False,limits=POLICY)
    if meta['format']=='PDF':
        common.update(mode='bounded_pdf_passthrough',analysis_path=str(source),analysis_sha256=meta['sha256'],resized=False)
        (folder/'manifest.json').write_bytes(encoded(common));return source,common
    w,h=meta['size_px'];value=meta['orientation'];size=target_size(w,h,2000);small=target_size(w,h,512)
    if meta['format']=='PNG':
        reader=PNGReader(source,meta['sha256']);reducers=[AreaOverview(w,h,size),AreaOverview(w,h,small)]
        for top,gray in reader.iter_gray():
            for reducer in reducers:reducer.add(top,gray)
        if not reader.validated:raise RasterInputError('input_integrity','PNG stream not validated')
        work,context=[oriented(r.finish(),value) for r in reducers];decoder='bounded_PNG_strips600MP';full_allocated=False
    else:
        # Direct format plugin after independent limits avoids Image.open's
        # global bomb check; draft configures libjpeg's reduced decoding first.
        raw=bounded_bytes(source)
        with JpegImagePlugin.JpegImageFile(io.BytesIO(raw)) as image:
            if list(image.size)!=meta['size_px']:raise RasterInputError('input_integrity','JPEG size changed')
            image.draft('L',(2000,2000))
            if image.width*image.height>12_000_000:raise RasterInputError('input_jpeg_budget','JPEG reduced decode exceeds12MP')
            image.load();gray=np.array(image.convert('L'));decoded_size=list(image.size)
        work=oriented(cv2.resize(gray,size,interpolation=cv2.INTER_AREA),value)
        context=oriented(cv2.resize(gray,small,interpolation=cv2.INTER_AREA),value)
        decoder='bounded_libjpeg_draft';full_allocated=decoded_size==[w,h]
        common['jpeg_decoded_size_px']=decoded_size
    oh,ow=work.shape;ch,cw=context.shape;left,top=(512-cw)//2,(512-ch)//2
    canvas=np.full((512,512),255,np.uint8);canvas[top:top+ch,left:left+cw]=context
    oriented_size=[h,w] if value>=5 else [w,h];scales=[ow/oriented_size[0],oh/oriented_size[1]]
    matrix=_scaled_transform(value,(w,h),tuple(scales));ctx=np.array(_scaled_transform(value,(w,h),(cw/oriented_size[0],ch/oriented_size[1])))
    ctx[:2,2]+=[left,top]
    dpi=meta['dpi_xy'];oriented_dpi=dpi[::-1] if dpi and value>=5 else dpi
    out_dpi=[a*b for a,b in zip(oriented_dpi,scales)] if oriented_dpi else None
    working=folder/'working.png';Image.fromarray(work).save(working,**({'dpi':tuple(out_dpi)} if out_dpi else {}));Image.fromarray(canvas).save(folder/'context.png')
    with source.open('rb') as stream:current=hashlib.file_digest(stream,'sha256').hexdigest()
    if current!=meta['sha256']:raise RasterInputError('input_integrity','Source changed')
    common.update(mode='bounded_raster_overview_and_source_windows',decoder=decoder,analysis_path=str(working.resolve()),
        analysis_sha256=hashlib.sha256(working.read_bytes()).hexdigest(),original_size_px=[w,h],oriented_size_px=oriented_size,
        analysis_size_px=[ow,oh],resized=[ow,oh]!=[w,h] or value!=1,scale_xy=scales,
        source_pixel_to_analysis_px=matrix,source_to_context=ctx.tolist(),context_path=str((folder/'context.png').resolve()),
        exif_orientation=value,exif_orientation_baked=True,original_dpi_xy=dpi,analysis_dpi_xy=out_dpi,
        coordinate_convention='pixel centres; first centre(0,0)',full_source_allocated=full_allocated,
        native_windows_available=meta['native_windows_available'],source_integrity_validated=True,
        reason='Whole-source overview; full-resolution detail processing/synthesis is10.4.3')
    (folder/'manifest.json').write_bytes(encoded(common));return working,common


def native_windows(source,boxes,expected_sha=None):
    """Read oriented-source integer boxes once; return only after full validation."""
    reader=PNGReader(source,expected_sha);meta=reader.meta;w,h=meta['size_px'];value=meta['orientation']
    ow,oh=(h,w) if value>=5 else (w,h);T=np.array(_orientation_matrix(value,w,h));inverse=np.linalg.inv(T)
    if not 1<=len(boxes)<=64:raise RasterInputError('input_windows','Window count1..64 required')
    if any(len(b)!=4 or any(type(n) is not int for n in b) for b in boxes):raise RasterInputError('input_windows','Integer boxes required')
    if sum((b[2]-b[0])*(b[3]-b[1]) for b in boxes)>16_000_000:raise RasterInputError('input_windows','Window area exceeds16MP')
    items=[]
    for box in boxes:
        if len(box)!=4 or any(type(x) is not int for x in box):raise RasterInputError('input_windows','Integer boxes required')
        l,t,r,b=box
        if not 0<=l<r<=ow or not 0<=t<b<=oh:raise RasterInputError('input_windows','Window outside oriented source')
        points=np.array([[l,t,1],[r-1,t,1],[l,b-1,1],[r-1,b-1,1]])@inverse.T
        x,y=np.rint(points[:,:2].min(0)).astype(int);right,bottom=np.rint(points[:,:2].max(0)).astype(int)+1
        items.append(dict(box=box,raw_box=[x,y,right,bottom],array=np.empty((bottom-y,right-x),np.uint8),
            source_to_tile=(np.array([[1,0,-l],[0,1,-t],[0,0,1]])@T).tolist()))
    for top,gray in reader.iter_gray():
        for item in items:
            x,y,r,b=item['raw_box'];first,last=max(top,y),min(top+len(gray),b)
            if first<last:item['array'][first-y:last-y]=gray[first-top:last-top,x:r]
    if not reader.validated:raise RasterInputError('input_integrity','Native window source incomplete')
    for item in items:item['array']=oriented(item['array'],value)
    return items,dict(source_sha256=meta['sha256'],source_integrity_validated=True,native_resolution=True,limits=POLICY)
