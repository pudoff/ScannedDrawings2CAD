"""Explicit, bounded image reduction for the presentation draft only.

The original remains authoritative. The returned transform maps original stored
pixel centres to analysis pixel centres, including the EXIF orientation that
intake applies (or that is baked into a reduced PNG). This is not physical-scale
calibration or the full-resolution tiled route.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import warnings

from PIL import Image, ImageOps


MAX_SOURCE_BYTES = 50 * 1024 * 1024
MAX_SOURCE_PIXELS = 100_000_000


def _digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _orientation_matrix(orientation: int, width: int, height: int) -> list[list[float]]:
    transforms = {
        1: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        2: [[-1, 0, width - 1], [0, 1, 0], [0, 0, 1]],
        3: [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
        4: [[1, 0, 0], [0, -1, height - 1], [0, 0, 1]],
        5: [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        6: [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
        7: [[0, -1, height - 1], [-1, 0, width - 1], [0, 0, 1]],
        8: [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
    }
    return [[float(value) for value in row] for row in transforms[orientation]]


def _scaled_transform(orientation: int, size: tuple[int, int], scale: tuple[float, float]):
    transform = _orientation_matrix(orientation, *size)
    for axis, factor in enumerate(scale):
        transform[axis] = [value * factor for value in transform[axis]]
        transform[axis][2] += (factor - 1) / 2
    return transform


def _target_size(size: tuple[int, int], limit: int) -> tuple[int, int]:
    factor = min(1., math.sqrt(limit / (size[0] * size[1])))
    width, height = (max(1, math.floor(value * factor)) for value in size)
    # A one-pixel minor axis cannot shrink further; still enforce the area cap.
    if width * height > limit:
        if width >= height:
            width = max(1, limit // height)
        else:
            height = max(1, limit // width)
    return width, height


def _dpi(value) -> tuple[float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    if not all(isinstance(item, (int, float)) and math.isfinite(item) and item > 0 for item in value):
        return None
    return float(value[0]), float(value[1])


def prepare_demo_input(source, workdir, *, max_native_pixels=25_000_000,
                       max_demo_pixels=4_000_000, max_source_pixels=100_000_000):
    """Return ``(input_path, provenance)``; create workdir only for a reduced PNG.

    Limits may be lowered for testing, never raised beyond 100 MP for source
    images. Native images and PDFs retain their original bytes and path. A
    reduction always has an explicit reason, source digest and pixel transform.
    ``workdir`` must be a fresh request-owned directory outside pipeline output.
    """
    limits = (max_native_pixels, max_demo_pixels, max_source_pixels)
    if any(type(value) is not int or value <= 0 for value in limits):
        raise ValueError('Demo pixel limits must be positive integers')
    if not max_demo_pixels <= max_native_pixels <= max_source_pixels <= MAX_SOURCE_PIXELS:
        raise ValueError('Demo pixel limits must satisfy demo <= native <= source <= 100 MP')
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError('Demo source file does not exist')
    if source.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError('Demo input exceeds 50 MiB limit')
    suffix = source.suffix.lower()
    if suffix not in {'.pdf', '.png', '.jpg', '.jpeg'}:
        raise ValueError('Demo input must be PDF, PNG or JPEG')
    source_hash = _digest(source)
    result = {
        'schema_version': 'demo-input/1.0', 'original_path': str(source),
        'original_name': source.name, 'sha256': source_hash,
        'analysis_sha256': source_hash, 'original_size_px': None,
        'oriented_size_px': None, 'analysis_size_px': None, 'resized': False,
        'scale_xy': [1., 1.], 'source_pixel_to_analysis_px': None,
        'coordinate_convention': 'pixel centres; first centre at (0, 0)',
        'exif_orientation': None, 'exif_orientation_baked': False,
        'original_dpi_xy': None, 'analysis_dpi_xy': None,
        'physical_geometry_approved': False, 'full_resolution_analysis': True,
        'limits': {'native_pixels': max_native_pixels, 'demo_pixels': max_demo_pixels,
                   'source_pixels': max_source_pixels, 'source_bytes': MAX_SOURCE_BYTES},
    }
    if suffix == '.pdf':
        with source.open('rb') as stream:
            if not stream.read(1024).lstrip().startswith(b'%PDF-'):
                raise ValueError('Demo PDF signature does not match extension')
        result.update(reason='PDF is passed to bounded page rendering',
                      full_resolution_analysis=None)
        return source, result

    # Pillow's default warning begins below our explicit 100 MP ceiling. Ignore
    # only that warning while inspecting headers; do not change its global cap.
    # Its hard decompression-bomb error remains active above its default limit.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', Image.DecompressionBombWarning)
        original = Image.open(source)
    with original:
        expected = 'PNG' if suffix == '.png' else 'JPEG'
        if original.format != expected:
            raise ValueError('Demo image signature does not match extension')
        size = original.size
        if size[0] * size[1] > max_source_pixels:
            raise ValueError('Demo source exceeds declared pixel limit; full-resolution tiling is required')
        if getattr(original, 'n_frames', 1) != 1:
            raise ValueError('Animated or multi-frame demo images are not supported')
        orientation = original.getexif().get(274, 1)
        if orientation not in range(1, 9):
            raise ValueError('Unsupported EXIF orientation')
        oriented_size = size[::-1] if orientation >= 5 else size
        original_dpi = _dpi(original.info.get('dpi'))
        oriented_dpi = original_dpi[::-1] if original_dpi and orientation >= 5 else original_dpi
        result.update(original_size_px=list(size), oriented_size_px=list(oriented_size),
                      analysis_size_px=list(oriented_size), exif_orientation=orientation,
                      original_dpi_xy=list(original_dpi) if original_dpi else None,
                      analysis_dpi_xy=list(oriented_dpi) if oriented_dpi else None,
                      source_pixel_to_analysis_px=_scaled_transform(orientation, size, (1., 1.)))
        if size[0] * size[1] <= max_native_pixels:
            result.update(reason='Original image is within the native intake limit',
                          orientation_applied_by='downstream image intake')
            return source, result

        workdir = Path(workdir).resolve()
        if workdir.exists():
            raise ValueError('Demo input directory must be new; existing files are never overwritten')
        target_size = _target_size(oriented_size, max_demo_pixels)
        scale = (target_size[0] / oriented_size[0], target_size[1] / oriented_size[1])
        target_dpi = tuple(value * factor for value, factor in zip(oriented_dpi, scale)) if oriented_dpi else None
        with ImageOps.exif_transpose(original) as oriented:
            # Palette/binary resizing otherwise silently selects nearest-neighbour.
            mode = 'RGBA' if 'A' in oriented.getbands() or 'transparency' in oriented.info else 'RGB'
            with oriented.convert(mode) as pixels:
                with pixels.resize(target_size, Image.Resampling.LANCZOS) as reduced:
                    # No inherited EXIF orientation or source DPI can be applied twice.
                    reduced.info.clear()
                    workdir.mkdir(parents=True, exist_ok=False)
                    output = workdir / 'analysis.png'
                    save_options = {'dpi': target_dpi} if target_dpi else {}
                    reduced.save(output, format='PNG', **save_options)
        if _digest(source) != source_hash:
            raise ValueError('Original demo input changed while preparing the analysis copy')
        result.update(
            analysis_path=str(output), analysis_sha256=_digest(output),
            analysis_size_px=list(target_size), resized=True, scale_xy=list(scale),
            source_pixel_to_analysis_px=_scaled_transform(orientation, size, scale),
            exif_orientation_baked=True, orientation_applied_by='demo input adapter',
            analysis_dpi_xy=list(target_dpi) if target_dpi else None,
            full_resolution_analysis=False, resampling='LANCZOS',
            reason='Explicit presentation-only reduction: original exceeds native intake; thin details may be lost',
        )
        return output, result
