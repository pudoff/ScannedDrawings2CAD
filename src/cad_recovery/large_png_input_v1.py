"""Explicit large-PNG policy without changing Pillow's process-wide limits.

Header inspection is not image validation. Streaming consumers must exhaust
PNGStripReader before publishing an analysis copy or original-pixel evidence.
"""
from pathlib import Path
import struct
import zlib

import cv2
import numpy as np
from PIL import Image

from . import demo_input_v1 as legacy
from .png_stream import PNGStripReader, SIGNATURE

MAX_SOURCE_PIXELS = 256_000_000
MAX_SIDE = 32768
_legacy_prepare = legacy.prepare_demo_input


def inspect_png(path, *, max_source_pixels=MAX_SOURCE_PIXELS):
    """Read only 33 header bytes; return bounded metadata, never a decoded image."""
    if type(max_source_pixels) is not int or not 0 < max_source_pixels <= MAX_SOURCE_PIXELS:
        raise ValueError('PNG pixel limit must be a positive integer <= 256 MP')
    path = Path(path)
    if not 0 < path.stat().st_size <= legacy.MAX_SOURCE_BYTES:
        raise ValueError('PNG source exceeds the 50 MiB file limit')
    with path.open('rb') as stream:
        header = stream.read(33)
    if (len(header) != 33 or header[:8] != SIGNATURE
            or header[8:16] != b'\x00\x00\x00\rIHDR'):
        raise ValueError('PNG must begin with its signature and IHDR')
    if zlib.crc32(header[12:29]) & 0xffffffff != struct.unpack('>I', header[29:33])[0]:
        raise ValueError('PNG IHDR CRC mismatch')
    w, h, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', header[16:29])
    if not 0 < w <= MAX_SIDE or not 0 < h <= MAX_SIDE or w*h > max_source_pixels:
        raise ValueError('PNG source dimensions exceed declared bounded-input limits')
    depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
    if depth not in depths.get(color, ()) or compression != 0 or filtering != 0 or interlace not in (0, 1):
        raise ValueError('Invalid PNG IHDR format')
    return dict(size_px=[w, h], bit_depth=depth, color_type=color, interlace=interlace,
                header_crc_validated=True, complete_image_validated=False,
                streaming_format_supported=(depth, color, interlace) == (8, 2, 0),
                limits=dict(source_pixels=max_source_pixels, source_bytes=legacy.MAX_SOURCE_BYTES,
                            max_side=MAX_SIDE), pillow_limits_overridden=False)


def reduce_stream(path, expected_sha, target):
    """Bounded grayscale overview: horizontal area + integer source-row bins.

    This is an explicitly reduced diagnostic view, not full-resolution CV.
    Only completed CRC/zlib/hash validation permits returning its pixels.
    """
    reader = PNGStripReader(Path(path), expected_sha, strip_rows=128)
    width, height = target
    if (any(type(n) is not int or n <= 0 for n in target)
            or width > reader.width or height > reader.height or width*height > 4_000_000):
        raise ValueError('Invalid bounded analysis dimensions')
    sums = np.zeros((height, width), np.float64)
    counts = np.zeros(height, np.int64)
    for start, rgb in reader.iter_strips():
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        horizontal = cv2.resize(gray, (width, len(gray)), interpolation=cv2.INTER_AREA)
        bins = np.arange(start, start+len(gray), dtype=np.int64)*height//reader.height
        np.add.at(sums, bins, horizontal)
        np.add.at(counts, bins, 1)
    if not reader.validated or not np.all(counts > 0):
        raise ValueError('PNG reduction is provisional')
    return np.rint(sums/counts[:, None]).astype(np.uint8), reader.metadata


def prepare_demo_input(source, workdir, *, max_native_pixels=25_000_000,
                       max_demo_pixels=4_000_000, max_source_pixels=MAX_SOURCE_PIXELS):
    """Versioned A2 adapter; small PNG/JPEG/PDF retain the legacy route."""
    limits = (max_demo_pixels, max_native_pixels, max_source_pixels)
    if (any(type(n) is not int or n <= 0 for n in limits)
            or not max_demo_pixels <= min(4_000_000, max_native_pixels)
            or not max_native_pixels <= max_source_pixels <= MAX_SOURCE_PIXELS):
        raise ValueError('Invalid bounded demo input limits')
    source = Path(source).resolve()
    if source.suffix.lower() != '.png':
        return _legacy_prepare(source, workdir, max_native_pixels=max_native_pixels,
            max_demo_pixels=max_demo_pixels, max_source_pixels=min(max_source_pixels, legacy.MAX_SOURCE_PIXELS))
    header = inspect_png(source, max_source_pixels=max_source_pixels)
    size = tuple(header['size_px'])
    if size[0]*size[1] <= min(max_native_pixels, legacy.MAX_SOURCE_PIXELS):
        return _legacy_prepare(source, workdir, max_native_pixels=min(max_native_pixels, legacy.MAX_SOURCE_PIXELS),
            max_demo_pixels=max_demo_pixels, max_source_pixels=min(max_source_pixels, legacy.MAX_SOURCE_PIXELS))
    if not header['streaming_format_supported']:
        raise ValueError('Large PNG route requires 8-bit RGB, non-interlaced static PNG')
    workdir = Path(workdir).resolve()
    if workdir.exists():
        raise ValueError('Demo input directory must be new; existing files are never overwritten')
    source_sha = legacy._digest(source)
    target = legacy._target_size(size, max_demo_pixels)
    pixels, metadata = reduce_stream(source, source_sha, target)
    scale = tuple(t/s for t, s in zip(target, size))
    physical = metadata.get('physical_pixel_dimensions')
    dpi = (tuple(physical[k]*.0254 for k in ('pixels_per_unit_x', 'pixels_per_unit_y'))
           if physical and physical['unit'] == 'meter' else None)
    dpi = legacy._dpi(dpi)
    target_dpi = tuple(n*s for n, s in zip(dpi, scale)) if dpi else None
    workdir.mkdir(parents=True, exist_ok=False)
    output = workdir/'analysis.png'
    Image.fromarray(pixels).save(output, format='PNG', **({'dpi': target_dpi} if target_dpi else {}))
    if legacy._digest(source) != source_sha:
        raise ValueError('Original PNG changed while preparing analysis copy')
    return output, dict(schema_version='demo-input/2.0', original_path=str(source), original_name=source.name,
        sha256=source_sha, analysis_path=str(output), analysis_sha256=legacy._digest(output),
        original_size_px=list(size), oriented_size_px=list(size), analysis_size_px=list(target),
        resized=True, scale_xy=list(scale), source_pixel_to_analysis_px=legacy._scaled_transform(1, size, scale),
        coordinate_convention='pixel centres; first centre at (0, 0); extent registration of reduced view',
        exif_orientation=1, exif_orientation_baked=True, orientation_applied_by='streaming adapter (EXIF rejected)',
        original_dpi_xy=list(dpi) if dpi else None, analysis_dpi_xy=list(target_dpi) if target_dpi else None,
        full_resolution_analysis=False, physical_geometry_approved=False,
        resampling='grayscale; horizontal INTER_AREA; integer source-row bins averaged vertically',
        reason='Explicit bounded diagnostic reduction; thin details and color may be lost',
        source_stream_validation=metadata, header=header,
        limits=dict(native_pixels=max_native_pixels, demo_pixels=max_demo_pixels,
                    source_pixels=max_source_pixels, source_bytes=legacy.MAX_SOURCE_BYTES))
