"""Source-preserving PNG compatibility for a valid image with a bounded trailer.

The strict decoder is unchanged. The complete original and discarded-byte hash
are retained, and the normalized prefix must pass every strict PNG check.
"""
from pathlib import Path
import struct
import hashlib
from .dataset_layout_v1 import encoded
from .raster_input_v2 import prepare
from .raster_stream_v2 import bounded_bytes,RasterInputError


def png_prefix_length(raw):
    if raw[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('PNG signature required')
    position=8;chunks=0
    while position+12<=len(raw):
        size=struct.unpack_from('>I',raw,position)[0];kind=raw[position+4:position+8];end=position+size+12;chunks+=1
        if end>len(raw) or chunks>10000:raise ValueError('Truncated PNG/chunk budget')
        if kind==b'IEND':
            if size!=0:raise ValueError('Nonempty IEND')
            if len(raw)-end>65536:raise ValueError('PNG trailer exceeds64KiB compatibility budget')
            if raw[end:].startswith((b'\x89PNG\r\n\x1a\n',b'\xff\xd8',b'%PDF-')):raise ValueError('Concatenated image/document is not a trailer')
            return end
        position=end
    raise ValueError('Complete IEND required')


def prepare_compatible(source,folder):
    source=Path(source).resolve();folder=Path(folder)
    if source.suffix.lower()!='.png':return prepare(source,folder)
    raw=bounded_bytes(source);end=png_prefix_length(raw)
    if end==len(raw):return prepare(source,folder)
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True);normalized=folder/'normalized.png';normalized.write_bytes(raw[:end])
    # Strict CRC, zlib length/rows, dimensions, SHA and IEND are still required.
    image,meta=prepare(normalized,folder/'decoded')
    original_sha=hashlib.sha256(raw).hexdigest()
    if hashlib.sha256(source.read_bytes()).hexdigest()!=original_sha:raise ValueError('Original source changed')
    meta.update(sha256=original_sha,original_path=str(source),original_name=source.name,
        compatibility_normalization=dict(kind='valid_PNG_prefix_with_bounded_trailer',
            original_sha256=original_sha,original_bytes=len(raw),normalized_path=str(normalized.resolve()),
            normalized_sha256=hashlib.sha256(raw[:end]).hexdigest(),normalized_bytes=end,
            trailing_bytes=len(raw)-end,trailing_sha256=hashlib.sha256(raw[end:]).hexdigest(),
            original_file_retained=True,decoded_image_modified=False))
    meta['source']=dict(meta['source'],path=str(source),sha256=original_sha,bytes=len(raw))
    (folder/'manifest.json').write_bytes(encoded(meta))
    return image,meta
