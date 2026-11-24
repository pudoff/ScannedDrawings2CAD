"""Reuse bounded PNG compatibility with explicit original/decoding identities."""
from pathlib import Path
import hashlib

from .dataset_layout_v1 import digest, encoded
from .raster_input_v2 import native_windows as strict_windows
from .raster_stream_v2 import bounded_bytes, RasterInputError
from .raster_trailing_normalization_v2 import prepare_compatible, png_prefix_length

VERSION = 'bounded-raster-compat/1.0'


def prepare(source, folder):
    image, meta = prepare_compatible(source, folder)
    normalization = meta.get('compatibility_normalization')
    if normalization:
        # The old strict API cannot decode the original with trailing bytes.
        # Advertising this as ordinary native access would be misleading.
        meta.update(native_windows_available=False,
            compatibility_native_windows_available=True,
            compatibility_native_window_api='raster_input_compat_v1.native_windows',
            input_compatibility_version=VERSION)
        meta['source'].update(native_windows_available=False,
                             compatibility_native_windows_available=True)
        (Path(folder)/'manifest.json').write_bytes(encoded(meta))
    return image, meta


def native_windows(source, boxes, adaptation):
    """Same source coordinates and strict decoder; verify both byte identities."""
    source = Path(source).resolve()
    expected = adaptation['sha256']
    if (Path(adaptation['original_path']).resolve() != source
            or digest(source) != expected):
        raise RasterInputError('input_integrity', 'Original source differs from adaptation')
    normalization = adaptation.get('compatibility_normalization')
    decoding = source
    decoding_sha = expected
    if normalization:
        raw = bounded_bytes(source)
        end = png_prefix_length(raw)
        prefix_sha = hashlib.sha256(raw[:end]).hexdigest()
        tail_sha = hashlib.sha256(raw[end:]).hexdigest()
        if (hashlib.sha256(raw).hexdigest() != expected
                or normalization['original_sha256'] != expected
                or normalization['original_bytes'] != len(raw)
                or normalization['normalized_bytes'] != end
                or normalization['trailing_bytes'] != len(raw)-end
                or normalization['normalized_sha256'] != prefix_sha
                or normalization['trailing_sha256'] != tail_sha):
            raise RasterInputError('input_integrity', 'PNG normalization does not describe original bytes')
        decoding = Path(normalization['normalized_path'])
        decoding_sha = prefix_sha
    items, proof = strict_windows(decoding, boxes, decoding_sha)
    if digest(source) != expected:
        raise RasterInputError('input_integrity', 'Original changed during native window decoding')
    return items, {**proof, 'source_sha256': expected,
        'decoding_source_sha256': decoding_sha,
        'decoding_uses_compatible_prefix': bool(normalization),
        'coordinate_frame': 'oriented original source pixel centres',
        'compatibility_version': VERSION}
