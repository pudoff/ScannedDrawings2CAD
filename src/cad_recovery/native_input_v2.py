"""Bounded raster normalization with explicit encoded-pixel provenance."""
import math
from pathlib import Path
import struct
import zlib
import numpy as np
from PIL import Image
from .family_catalog_v2 import require
from .dataset_layout_v1 import digest,read
from .large_raster import prepare_large_png
from .png_stream import PNGStripReader
from .raster_regions import read_prepared_region

MAX_DECODE_PIXELS=25_000_000

def orientation_matrix(width,height,orientation):
    matrices={1:([[1,0,0],[0,1,0],[0,0,1]],(width,height)),
        2:([[-1,0,width-1],[0,1,0],[0,0,1]],(width,height)),
        3:([[-1,0,width-1],[0,-1,height-1],[0,0,1]],(width,height)),
        4:([[1,0,0],[0,-1,height-1],[0,0,1]],(width,height)),
        5:([[0,1,0],[1,0,0],[0,0,1]],(height,width)),
        6:([[0,-1,height-1],[1,0,0],[0,0,1]],(height,width)),
        7:([[0,-1,height-1],[-1,0,width-1],[0,0,1]],(height,width)),
        8:([[0,1,0],[-1,0,width-1],[0,0,1]],(height,width))}
    require(orientation in matrices,'Unsupported EXIF orientation')
    matrix,size=matrices[orientation];return np.array(matrix,dtype=int),list(size)

def copy_png_without_exif(source,target):
    """Preserve compressed pixels; extract EXIF before the frozen strip decoder.

    Every original chunk CRC is checked. The decoder still validates raster
    structure/IDAT; APNG and transparency chunks are retained for rejection.
    """
    orientation=1;seen=False;chunks=0
    safe_metadata={b'eXIf',b'iCCP',b'sRGB',b'gAMA',b'cHRM',b'tEXt',b'zTXt',b'iTXt',b'bKGD',b'tIME'}
    with Path(source).open('rb') as stream,Path(target).open('xb') as output:
        signature=stream.read(8);require(signature==b'\x89PNG\r\n\x1a\n','Invalid PNG signature');output.write(signature)
        while True:
            header=stream.read(8);require(len(header)==8,'Truncated PNG metadata');size,kind=struct.unpack('>I4s',header)
            chunks+=1;require(chunks<=10000 and size<=50*1024**2,'PNG chunk limit exceeded')
            keep=kind not in safe_metadata;crc=zlib.crc32(kind);payload=bytearray()
            if kind==b'eXIf':require(not seen and size<=1024**2,'Duplicate or oversized EXIF');seen=True
            if keep:output.write(header)
            remaining=size
            while remaining:
                part=stream.read(min(65536,remaining));require(bool(part),'Truncated PNG chunk');remaining-=len(part);crc=zlib.crc32(part,crc)
                if keep:output.write(part)
                elif kind==b'eXIf':payload.extend(part)
            tail=stream.read(4);require(len(tail)==4 and struct.unpack('>I',tail)[0]==crc&0xffffffff,'PNG CRC mismatch')
            if keep:output.write(tail)
            if kind==b'eXIf':
                exif=Image.Exif();exif.load(bytes(payload));orientation=int(exif.get(274,1))
            if kind==b'IEND':require(not stream.read(1),'Unexpected PNG trailer');break
    return orientation

def _chunk(stream,kind,payload):
    stream.write(struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(payload,zlib.crc32(kind))&0xffffffff))

def normalized_png(prepared,target,matrix,size):
    """Write transformed RGB8 strips; no full-size array or changed pixel guards."""
    inverse=np.linalg.inv(matrix).round().astype(int);width,height=size
    with Path(target).open('xb') as stream:
        stream.write(b'\x89PNG\r\n\x1a\n');_chunk(stream,b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
        compressor=zlib.compressobj(3)
        for top in range(0,height,64):
            bottom=min(height,top+64)
            corners=np.array([[0,top,1],[width-1,top,1],[0,bottom-1,1],[width-1,bottom-1,1]])@inverse.T
            lo=corners[:,:2].min(axis=0);hi=corners[:,:2].max(axis=0)+1
            pixels,_=read_prepared_region(prepared,[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])],max_pixels=4_194_304)
            # Integer D4 map: each output row samples exact original RGB pixels.
            xs=np.arange(width)
            for y in range(top,bottom):
                sx=inverse[0,0]*xs+inverse[0,1]*y+inverse[0,2]-lo[0]
                sy=inverse[1,0]*xs+inverse[1,1]*y+inverse[1,2]-lo[1]
                data=compressor.compress(b'\0'+pixels[sy,sx].tobytes())
                if data:_chunk(stream,b'IDAT',data)
        data=compressor.flush()
        if data:_chunk(stream,b'IDAT',data)
        _chunk(stream,b'IEND',b'')

def prepare_image(source,folder,rotation_deg=0):
    source=Path(source);folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    original_sha=digest(source);streaming=False;decoded_path=source
    if source.suffix.lower()=='.png':
        try:PNGStripReader(source);streaming=True
        except ValueError:pass
    if streaming:
        copied=folder/'encoded-source.png';orientation=copy_png_without_exif(source,copied)
        initial=prepare_large_png(copied,folder/'encoded',tile_size=1536,overlap=256)
        size=initial['source_size_px'];conversion='streamed exact RGB8 PNG; EXIF extracted and ancillary display metadata removed without decoding IDAT'
    else:
        with Image.open(source) as image:
            width,height=image.size;require(width*height<=MAX_DECODE_PIXELS,'Non-RGB8 PNG/JPEG exceeds 25M decode pixels; provide RGB8 noninterlaced PNG')
            require(image.format in ('PNG','JPEG') and getattr(image,'n_frames',1)==1,'Supported raster formats: static PNG/JPEG')
            size=[width,height];orientation=int(image.getexif().get(274,1))
            rgba=image.convert('RGBA');background=Image.new('RGBA',image.size,'white');background.alpha_composite(rgba)
            decoded_path=folder/'decoded.png';background.convert('RGB').save(decoded_path)
        initial=prepare_large_png(decoded_path,folder/'encoded',tile_size=1536,overlap=256)
        conversion='decoded JPEG or alpha/gray PNG; RGB white alpha composite, unchanged pixel dimensions'
    exif,display_size=orientation_matrix(*size,orientation)
    explicit,final_size=orientation_matrix(*display_size,{0:1,90:6,180:3,270:8}[rotation_deg]);forward=explicit@exif
    prepared=folder/'encoded'
    if not np.array_equal(forward,np.eye(3)):
        normalized_png(prepared,folder/'oriented.png',forward,final_size)
        prepare_large_png(folder/'oriented.png',folder/'oriented',tile_size=1536,overlap=256);prepared=folder/'oriented'
    require(digest(source)==original_sha,'Original raster changed during preparation')
    return prepared,{'original_sha256':original_sha,'original_size_px':size,'analysis_size_px':final_size,
        'exif_orientation':orientation,'explicit_clockwise_deg':rotation_deg,'original_to_analysis':forward.tolist(),
        'analysis_to_original':np.linalg.inv(forward).tolist(),'conversion':conversion,'source_resized':False,
        'physical_scale_confirmed':False,'analysis_raster_sha256':read(prepared/'preparation.json')['source_sha256']}

def input_pages(source,folder,options):
    source=Path(source);folder=Path(folder)
    require(source.is_file() and 0<source.stat().st_size<=50*1024**2,'Input missing or exceeds 50MiB')
    if source.suffix.lower()!='.pdf':
        require(source.suffix.lower() in ('.png','.jpg','.jpeg'),'Supported inputs: PNG/JPEG/PDF')
        prepared,chain=prepare_image(source,folder/'page-001',options.rotation_deg)
        yield prepared,{**chain,'source_kind':'raster','page_index':0};return
    import fitz
    with fitz.open(source) as doc:
        require(not doc.is_encrypted and 0<len(doc)<=5,'Native PDF analysis supports 1..5 unencrypted pages')
        # Check every page before rendering the first one.
        for page in doc:
            width=math.ceil(page.rect.width*options.render_dpi/72);height=math.ceil(page.rect.height*options.render_dpi/72)
            require(0<width*height<=MAX_DECODE_PIXELS,'Requested PDF DPI exceeds 25M page render pixels; select a lower explicit DPI')
        for index,page in enumerate(doc):
            folder.mkdir(parents=True,exist_ok=True);path=folder/f'render-{index+1:03d}.png'
            pixmap=page.get_pixmap(matrix=fitz.Matrix(options.render_dpi/72,options.render_dpi/72),alpha=False,colorspace=fitz.csRGB)
            require(pixmap.width*pixmap.height<=MAX_DECODE_PIXELS,'Rendered PDF exceeds pixel budget');pixmap.save(path);del pixmap
            prepared,chain=prepare_image(path,folder/f'page-{index+1:03d}',options.rotation_deg)
            scale=72/options.render_dpi
            yield prepared,{**chain,'source_kind':'pdf_render','page_index':index,'render_dpi':options.render_dpi,
                'source_pdf_sha256':digest(source),'original_frame':'rendered page before explicit rotation; PDF display rotation included',
                'rendered_pixel_to_displayed_pdf_points':[[scale,0,page.rect.x0+scale/2],[0,scale,page.rect.y0+scale/2],[0,0,1]]}
