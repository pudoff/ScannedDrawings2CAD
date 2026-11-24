"""Frozen observer with a bounded metadata-only seam for large original PNGs.

Used only in a dedicated experimental CLI child. The scoped module facade
changes neither PIL.Image.open nor Pillow's global safeguards. Pixel windows
continue to use the frozen CRC/hash-validated PNGStripReader route.
"""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import source_arrow_observation_v1 as previous
from .large_png_input_v1 import inspect_png


class _MetadataImageAPI:
    def __init__(self, path, size):
        self.path = Path(path).resolve()
        self.size = tuple(size)
        self.original = previous.Image

    def open(self, path, *args, **kwargs):
        if not args and not kwargs and isinstance(path, (str, Path)) and Path(path).resolve() == self.path:
            # Deliberately offers no load/convert/crop: this seam is metadata only.
            return nullcontext(SimpleNamespace(size=self.size))
        return self.original.open(path, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


def observe(source_image, **kwargs):
    path = Path(source_image)
    if path.suffix.lower() != '.png':
        return previous.observe(source_image, **kwargs)
    header = inspect_png(path, max_source_pixels=previous.POLICY['maximum_source_pixels'])
    if header['size_px'][0]*header['size_px'][1] <= previous.POLICY['maximum_small_decode_pixels']:
        return previous.observe(source_image, **kwargs)
    if not header['streaming_format_supported']:
        raise ValueError('Large native observation requires an 8-bit RGB non-interlaced PNG')
    if Path(kwargs.get('geometry_image') or path).resolve() == path.resolve():
        raise ValueError('Large source requires a separate bounded geometry image and explicit transform')
    before = previous.digest(path)
    with patch.object(previous, 'Image', _MetadataImageAPI(path, header['size_px'])):
        result = previous.observe(source_image, **kwargs)
    if previous.digest(path) != before or result['source_sha256'] != before:
        raise ValueError('Source changed during native observation')
    result['source_header'] = header
    result['source_header']['reader'] = 'bounded-IHDR/1.0; native windows retain PNGStripReader validation'
    return result
