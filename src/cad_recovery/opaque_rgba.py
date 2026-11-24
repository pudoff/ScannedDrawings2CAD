"""Prepare a bounded, provably opaque RGBA PNG without changing RGB samples.

This separate adapter preserves the sealed RGB-only preparation implementation.
Transparent pixels, animation and orientation metadata are deliberately refused.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
import zlib

import numpy as np
from PIL import Image

from .large_raster import prepare_large_png

MAX_RGBA_PIXELS = 8_000_000
MAX_FILE_BYTES = 50 * 1024**2
MAX_SIDE = 32768
MAX_CHUNKS = 10000
READ_BYTES = 65536
SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _validate_png(raw):
    """Check structure and bounded inflate independently of Pillow's decoder.

    Return a minimal PNG for pixel decoding. Ancillary payloads have their CRC
    checked but are not handed to Pillow for text/ICC metadata decompression.
    """
    if len(raw) < 33 or raw[:8] != SIGNATURE or raw[8:16] != b'\x00\x00\x00\rIHDR':
        raise ValueError('PNG must begin with its signature and IHDR')
    width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', raw[16:29])
    if not 0 < width <= MAX_SIDE or not 0 < height <= MAX_SIDE or width*height > MAX_RGBA_PIXELS:
        raise ValueError('RGBA PNG dimensions exceed bounded adapter limits')
    if (depth, color, compression, filtering, interlace) != (8, 6, 0, 0, 0):
        raise ValueError('Expected 8-bit RGBA noninterlaced PNG')
    stride, decoded = width*4+1, 0
    expected = stride*height
    inflater = zlib.decompressobj()
    position, chunks, idat_state, palette, physical = 8, 0, 0, False, False
    essential, ignored = [SIGNATURE], []
    while True:
        if position+12 > len(raw):
            raise ValueError('Truncated PNG chunk')
        length, kind = struct.unpack('>I4s', raw[position:position+8])
        chunks += 1
        if chunks > MAX_CHUNKS:
            raise ValueError('PNG chunk count exceeds adapter limit')
        end = position+12+length
        if length > MAX_FILE_BYTES or end > len(raw):
            raise ValueError('PNG chunk length exceeds source bounds')
        if not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in kind) or not 65 <= kind[2] <= 90:
            raise ValueError('Invalid PNG chunk type')
        payload = memoryview(raw)[position+8:position+8+length]
        crc = zlib.crc32(payload, zlib.crc32(kind)) & 0xffffffff
        if crc != struct.unpack('>I', raw[end-4:end])[0]:
            raise ValueError('PNG chunk CRC mismatch')
        if kind == b'IHDR' and (chunks != 1 or length != 13):
            raise ValueError('Duplicate or invalid PNG IHDR')
        if kind in {b'acTL', b'fcTL', b'fdAT', b'eXIf', b'tRNS'}:
            raise ValueError('Unsupported PNG animation, orientation or transparency chunk')
        if kind not in {b'IHDR', b'PLTE', b'IDAT', b'IEND'} and kind[0] < 97:
            raise ValueError('Unknown critical PNG chunk')
        if kind == b'PLTE':
            if palette or idat_state != 0 or not 0 < length <= 768 or length % 3:
                raise ValueError('Invalid or misplaced PNG palette')
            palette = True
        if kind == b'pHYs':
            if physical or idat_state != 0 or length != 9 or payload[-1] not in (0, 1):
                raise ValueError('Invalid or misplaced PNG physical dimensions')
            physical = True
        if kind == b'IDAT':
            if idat_state == 2:
                raise ValueError('PNG IDAT chunks must be consecutive')
            idat_state = 1
            try:
                for offset in range(0, length, READ_BYTES):
                    pending = payload[offset:offset+READ_BYTES]
                    if inflater.eof:
                        raise ValueError('Extra compressed PNG data after zlib stream')
                    while pending:
                        block = inflater.decompress(pending, READ_BYTES)
                        if decoded+len(block) > expected:
                            raise ValueError('PNG zlib output exceeds declared scanlines')
                        first_filter = (-decoded) % stride
                        if any(block[i] > 4 for i in range(first_filter, len(block), stride)):
                            raise ValueError('PNG scanline has invalid filter')
                        decoded += len(block)
                        pending = inflater.unconsumed_tail
                        if inflater.eof:
                            if inflater.unused_data or pending:
                                raise ValueError('Extra or concatenated PNG zlib streams')
                            break
            except zlib.error as error:
                raise ValueError('Invalid PNG zlib stream') from error
        elif idat_state == 1:
            idat_state = 2
        if kind in (b'IHDR', b'IDAT', b'IEND'):
            essential.append(memoryview(raw)[position:end])
        else:
            ignored.append(kind.decode('ascii'))
        position = end
        if kind == b'IEND':
            if length != 0 or idat_state == 0:
                raise ValueError('Invalid PNG IEND or missing IDAT')
            if position != len(raw):
                raise ValueError('Unexpected data after PNG IEND')
            break
    if not inflater.eof or decoded != expected:
        raise ValueError('Truncated PNG zlib stream or incorrect scanline count')
    return b''.join(essential), {'width_px': width, 'height_px': height, 'bit_depth': 8,
        'color_type': 6, 'interlace_method': 0, 'chunk_count': chunks,
        'decoded_scanline_bytes': decoded, 'ignored_ancillary_chunks': ignored,
        'ancillary_payloads_decompressed': False, 'integrity_validated': True}


def _make_derivative(source, output, expected_sha256):
    size = source.stat().st_size
    if not 0 < size <= MAX_FILE_BYTES:
        raise ValueError('RGBA source exceeds 50 MiB file limit')
    with source.open('rb') as stream:
        raw = stream.read(MAX_FILE_BYTES+1)
    if len(raw) != size or len(raw) > MAX_FILE_BYTES:
        raise ValueError('RGBA source size changed while reading')
    source_sha = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and source_sha != expected_sha256:
        raise ValueError('RGBA source SHA-256 differs from expected source')
    minimal, decoder = _validate_png(raw)
    del raw
    with Image.open(io.BytesIO(minimal)) as picture:
        if picture.format != 'PNG' or picture.mode != 'RGBA' or getattr(picture, 'n_frames', 1) != 1:
            raise ValueError('Expected one static RGBA PNG image')
        picture.load()
        with picture.getchannel('A') as alpha:
            extrema = list(alpha.getextrema())
        if extrema != [255, 255]:
            raise ValueError('RGBA adapter requires every alpha sample to equal 255')
        expected = np.asarray(picture)[:, :, :3]
        output.mkdir(parents=True, exist_ok=False)
        derivative = output/'opaque-rgb.png'
        with Image.fromarray(expected) as rgb:
            with derivative.open('xb') as stream:
                rgb.save(stream, format='PNG', compress_level=3)
        with Image.open(derivative) as reloaded:
            if reloaded.mode != 'RGB' or reloaded.size != picture.size or not np.array_equal(np.asarray(reloaded), expected):
                raise ValueError('RGB derivative differs from original RGB channel samples')
        dimensions = list(picture.size)
    # The pixel array and all Pillow buffers go out of scope before tile preparation.
    if _digest(source) != source_sha:
        raise ValueError('RGBA source changed during derivative conversion')
    decoder.update(format='PNG', decoder='bounded_opaque_rgba_adapter_v1',
        source_sha256=source_sha, source_size_bytes=size, decoder_limits_overridden=False,
        alpha_extrema=extrema, rgb_samples_preserved=True,
        limits={'max_file_bytes': MAX_FILE_BYTES, 'max_rgba_pixels': MAX_RGBA_PIXELS,
                'max_side': MAX_SIDE, 'max_chunks': MAX_CHUNKS})
    conversion = {'source_sha256': source_sha, 'alpha_extrema': extrema,
        'method': 'discard_proven_opaque_alpha_only', 'rgb_samples_preserved': True,
        'source_to_derivative': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'rgb_derivative': {'path': derivative.name, 'sha256': _digest(derivative),
                           'size_bytes': derivative.stat().st_size, 'size_px': dimensions}}
    return decoder, conversion


def prepare_opaque_rgba(source: Path, output_dir: Path, *, expected_sha256=None,
                        tile_size=2048, overlap=128, overview_max_side=2400, strip_rows=128) -> dict:
    """Prepare opaque RGBA with an original-source wrapper and pinned RGB child.

    Coordinates and RGB samples are unchanged. Only the proved constant alpha
    channel is omitted. A failed wrapper never presents child tiles as complete.
    """
    started = time.perf_counter()
    source, requested_output = Path(source).resolve(), Path(output_dir)
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError('RGBA output already exists; choose a new directory')
    output = requested_output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError('RGBA output already exists; choose a new directory')
    if source == output or source.is_relative_to(output):
        raise ValueError('RGBA output may not contain the original source')
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or re.fullmatch('[0-9a-f]{64}', expected_sha256) is None):
        raise ValueError('Expected a lowercase SHA-256 pin')
    try:
        decoder, conversion = _make_derivative(source, output, expected_sha256)
        prepared_dir = output/'rgb-prepared'
        child = prepare_large_png(output/'opaque-rgb.png', prepared_dir,
            expected_sha256=conversion['rgb_derivative']['sha256'], tile_size=tile_size,
            overlap=overlap, overview_max_side=overview_max_side, strip_rows=strip_rows)
        if child.get('status') != 'prepared' or child.get('source_integrity_validated') is not True:
            raise ValueError('RGB child preparation did not fully validate')
        if _digest(source) != decoder['source_sha256']:
            raise ValueError('RGBA source changed during tile preparation')
        report = deepcopy(child)
        for record in [report['overview'], report['ink_overview'], *report['tiles']]:
            record['path'] = 'rgb-prepared/'+record['path']
        conversion['rgb_preparation_report'] = {'path': 'rgb-prepared/preparation.json',
                                                'sha256': _digest(prepared_dir/'preparation.json')}
        report.update(source=source.name, source_sha256=decoder['source_sha256'],
            decoder=decoder, rgba_conversion=conversion,
            coverage='all original RGB channel samples; omitted alpha proven constant 255',
            elapsed_seconds=round(time.perf_counter()-started, 4),
            limitations=['Only fully opaque RGBA8 noninterlaced PNG within 8 million pixels is supported.',
                         'Alpha is proved 255 before omission; RGB channels and pixel coordinates are unchanged.',
                         'Ancillary metadata is not interpreted or propagated into RGB pixels.',
                         'Reduced overviews can merge strokes; use full-resolution tiles for geometry.',
                         'Preparation does not approve pair correspondence, dimensions or training labels.'])
        report['resource_estimates']['rgba_max_pixels'] = MAX_RGBA_PIXELS
        report['resource_estimates']['rgba_decode_precedes_rgb_tiling'] = True
        report['resource_estimates']['working_buffer_scope'] = 'RGB child preparation only; RGBA full decode is bounded separately.'
        _write(output/'preparation.json', report)
        return report
    except Exception as error:
        if output.is_dir() and not (output/'preparation.json').exists():
            _write(output/'preparation.json', {'schema_version': '1.0', 'status': 'failed',
                'source': source.name, 'source_integrity_validated': False,
                'source_modified': False, 'error': {'type': type(error).__name__, 'message': str(error)},
                'limitations': ['Any RGB derivative or tiles are provisional; this wrapper is not prepared.']})
        raise
