"""Versioned600MP PNG strip decoder; old300MP experiments remain immutable."""
import hashlib
import io
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image
from .png_stream import SIGNATURE,_chunk

POLICY=dict(max_pixels=600_000_000,max_side=32768,max_bytes=50*1024**2,max_chunks=10000,
            strip_rows=64,max_exif_bytes=65536,decoder='raster-strips/2.0')


class RasterInputError(ValueError):
    def __init__(self,code,message):self.code=code;super().__init__(message)


def bounded_bytes(path):
    path=Path(path)
    with path.open('rb') as stream:raw=stream.read(POLICY['max_bytes']+1)
    if not raw or len(raw)>POLICY['max_bytes']:raise RasterInputError('input_bytes','Input exceeds50MiB or is empty')
    return raw


def orientation(exif):
    if exif.startswith(b'Exif\0\0'):exif=exif[6:]
    if len(exif)>65536 or len(exif)<8 or exif[:2] not in (b'II',b'MM'):raise RasterInputError('input_exif','Invalid bounded EXIF')
    order='<' if exif[:2]==b'II' else '>'
    def integer(fmt,offset):
        size=struct.calcsize(fmt)
        if offset<0 or offset+size>len(exif):raise RasterInputError('input_exif','Truncated EXIF')
        return struct.unpack_from(order+fmt,exif,offset)[0]
    if integer('H',2)!=42:raise RasterInputError('input_exif','Unsupported TIFF header')
    start=integer('I',4);count=integer('H',start)
    if count>256:raise RasterInputError('input_exif','EXIF tag budget')
    result=1;seen=False
    for i in range(count):
        p=start+2+12*i;tag=integer('H',p)
        if p+12>len(exif):raise RasterInputError('input_exif','Truncated EXIF entry')
        if tag==274:
            if seen or integer('H',p+2)!=3 or integer('I',p+4)!=1:raise RasterInputError('input_exif','Invalid orientation tag')
            result=integer('H',p+8);seen=True
    if result not in range(1,9):raise RasterInputError('input_exif','Orientation must be1..8')
    return result


def dimensions(w,h):
    if not 0<w<=32768 or not 0<h<=32768 or w*h>600_000_000:
        raise RasterInputError('input_pixels','Input exceeds qualified600MP/32768px envelope')


def png_header(path,expected_sha=None):
    raw=bounded_bytes(path);sha=hashlib.sha256(raw).hexdigest()
    if expected_sha and sha!=expected_sha:raise RasterInputError('input_integrity','Source SHA changed')
    if raw[:8]!=SIGNATURE:raise RasterInputError('input_signature','PNG signature required')
    pos=8;chunks=0;header=None;palette=None;alpha=None;exif=None;dpi=None;pieces=[];state=0;done=False
    while pos<len(raw):
        if pos+12>len(raw):raise RasterInputError('input_png_crc','Truncated chunk')
        n,kind=struct.unpack_from('>I4s',raw,pos);pos+=8;chunks+=1
        if chunks>10000 or n>POLICY['max_bytes'] or pos+n+4>len(raw):raise RasterInputError('input_png_crc','Chunk budget/truncation')
        value=memoryview(raw)[pos:pos+n];pos+=n
        if zlib.crc32(value,zlib.crc32(kind))&0xffffffff!=struct.unpack_from('>I',raw,pos)[0]:raise RasterInputError('input_png_crc','PNG chunk CRC mismatch')
        pos+=4
        if not all(65<=b<=90 or 97<=b<=122 for b in kind) or not 65<=kind[2]<=90:raise RasterInputError('input_png_format','Invalid chunk name')
        if chunks==1 and kind!=b'IHDR':raise RasterInputError('input_png_format','IHDR must be first')
        if kind==b'IHDR':
            if header is not None or n!=13:raise RasterInputError('input_png_format','Duplicate/invalid IHDR')
            header=struct.unpack('>IIBBBBB',value);w,h,depth,color,c,f,interlace=header;dimensions(w,h)
            if (c,f,interlace)!=(0,0,0) or color not in (0,2,3,4,6):raise RasterInputError('input_png_format','Static non-interlaced PNG required')
            if (color in (2,4,6) and depth!=8) or (color==3 and depth not in (1,2,4,8)) or (color==0 and depth not in (1,2,4,8,16)):
                raise RasterInputError('input_png_format','Unqualified PNG bit depth/color combination')
        elif kind==b'PLTE':
            if palette is not None or state or not n or n%3 or n>768 or (color==3 and n>3*2**depth):raise RasterInputError('input_png_format','Invalid palette')
            palette=bytes(value)
        elif kind==b'tRNS':
            if alpha is not None or state or color in (4,6):raise RasterInputError('input_png_format','Invalid transparency')
            if (color==3 and (palette is None or not 0<n<=len(palette)//3)) or (color==0 and n!=2) or (color==2 and n!=6):raise RasterInputError('input_png_format','Invalid transparency payload')
            alpha=bytes(value)
        elif kind==b'eXIf':
            if exif is not None:raise RasterInputError('input_exif','Duplicate EXIF')
            exif=bytes(value);orientation(exif)
        elif kind==b'pHYs':
            if dpi is not None or state or n!=9:raise RasterInputError('input_png_format','Invalid density')
            x,y,unit=struct.unpack('>IIB',value)
            if unit not in (0,1):raise RasterInputError('input_png_format','Invalid density unit')
            dpi=(x*.0254,y*.0254) if unit==1 and x and y else (None,None)
        elif kind==b'IDAT':
            if state==2 or (color==3 and palette is None):raise RasterInputError('input_png_format','Nonconsecutive IDAT/missing palette')
            state=1;pieces.append(value)
        elif kind==b'IEND':
            if n or not pieces or pos!=len(raw):raise RasterInputError('input_png_format','Invalid IEND/trailing bytes')
            done=True;break
        elif kind in (b'acTL',b'fcTL',b'fdAT') or kind[0]<97:raise RasterInputError('input_png_format','Unsupported critical/animated chunk')
        if state==1 and kind!=b'IDAT':state=2
    if not done:raise RasterInputError('input_png_format','Missing IEND')
    return dict(path=str(Path(path).resolve()),sha256=sha,size_px=[w,h],depth=depth,color=color,
                orientation=orientation(exif) if exif else 1,dpi_xy=list(dpi) if dpi and dpi[0] else None,
                bytes=len(raw),chunks=chunks), (raw,pieces,palette,alpha)


class PNGReader:
    def __init__(self,path,expected_sha=None,strip_rows=64):
        if type(strip_rows) is not int or not 1<=strip_rows<=128:raise ValueError('Strip height1..128 required')
        self.meta,self.payload=png_header(path,expected_sha);self.rows=strip_rows;self.validated=False;self.started=False

    def iter_gray(self):
        if self.started:raise ValueError('Reader supports one pass')
        self.started=True;raw,pieces,palette,alpha=self.payload;m=self.meta;w,h=m['size_px'];depth=m['depth'];color=m['color']
        channels={0:1,2:3,3:1,4:2,6:4}[color];packed=(w*channels*depth+7)//8;stride=packed+1
        padded=packed*8//depth if depth<8 else w;previous=bytes(packed);buffer=bytearray();inflater=zlib.decompressobj();decoded=0
        prefix=_chunk(b'PLTE',palette) if palette is not None else b''
        if alpha is not None:prefix+=_chunk(b'tRNS',alpha)
        try:
            for piece in pieces:
                for offset in range(0,len(piece),65536):
                    pending=piece[offset:offset+65536]
                    if inflater.eof and len(pending):raise RasterInputError('input_png_stream','Extra zlib stream')
                    while len(pending):
                        rows=min(self.rows,h-decoded);room=max(1,rows*stride-len(buffer))
                        buffer.extend(inflater.decompress(pending,room));pending=inflater.unconsumed_tail
                        if inflater.unused_data:raise RasterInputError('input_png_stream','Extra zlib bytes')
                        if not rows and buffer:raise RasterInputError('input_png_stream','Too many source rows')
                        if rows and len(buffer)==rows*stride:
                            if any(buffer[k]>4 for k in range(0,len(buffer),stride)):raise RasterInputError('input_png_stream','Invalid filter')
                            blob=(SIGNATURE+_chunk(b'IHDR',struct.pack('>IIBBBBB',padded,rows+1,depth,color,0,0,0))+prefix
                                +_chunk(b'IDAT',zlib.compress(b'\0'+previous+buffer,1))+_chunk(b'IEND',b''))
                            with Image.open(io.BytesIO(blob)) as image:
                                array=np.array(image)
                                if color==3 and np.any(array[1:,:w]>=len(palette)//3):raise RasterInputError('input_png_format','Missing palette entry')
                                if depth<8:
                                    values=array[-1].astype(np.uint16)
                                    if color==0:values=np.rint(values*((2**depth-1)/255)).astype(np.uint16) if image.mode!='1' else values
                                    values=values.reshape(-1,8//depth);shifts=np.arange(8-depth,-1,-depth)
                                    previous=np.bitwise_or.reduce(values<<shifts,axis=1).astype(np.uint8).tobytes()
                                elif depth==16:previous=array[-1].astype('>u2').tobytes()
                                else:previous=array[-1].astype(np.uint8).tobytes()
                                if depth==16:
                                    gray=np.rint(array[1:,:w].astype(float)/257).astype(np.uint8)
                                    if alpha is not None:gray[array[1:,:w]==struct.unpack('>H',alpha)[0]]=255
                                elif color in (4,6) or alpha is not None:
                                    rgba=image.convert('RGBA');white=Image.new('RGBA',rgba.size,'white')
                                    gray=np.array(Image.alpha_composite(white,rgba).convert('L'))[1:,:w]
                                else:gray=np.array(image.convert('L'))[1:,:w]
                            top=decoded;decoded+=rows;buffer.clear();yield top,gray
            if decoded!=h or buffer or not inflater.eof or inflater.unused_data:raise RasterInputError('input_png_stream','Truncated source stream')
        except zlib.error as error:raise RasterInputError('input_png_stream',str(error)) from error
        with Path(m['path']).open('rb') as stream:current=hashlib.file_digest(stream,'sha256').hexdigest()
        if current!=m['sha256']:raise RasterInputError('input_integrity','Source changed during decoding')
        self.validated=True
