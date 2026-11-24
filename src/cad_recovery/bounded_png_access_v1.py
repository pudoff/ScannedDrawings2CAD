"""Process-local facade for frozen metadata and grayscale-crop consumers.

Only the named original is intercepted. Full decode methods are deliberately
absent. Crops are bounded and validated by the existing original-pixel reader.
"""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from .large_png_input_v1 import inspect_png
from .source_arrow_observation_v1 import source_windows

MAX_CROP_PIXELS = 12_000_000


class _Original:
    def __init__(self, path, sha256, size, allow_crop):
        self.path, self.sha256, self.size = path, sha256, tuple(size)
        self.allow_crop = allow_crop

    def crop(self, box):
        if not self.allow_crop:
            raise ValueError('This original-source access is metadata only')
        if (not isinstance(box, (tuple, list)) or len(box) != 4
                or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) for n in box)):
            raise ValueError('Native crop requires four integer pixel boundaries')
        l, t, r, b = map(int, box)
        if (not 0 <= l < r <= self.size[0] or not 0 <= t < b <= self.size[1]
                or (r-l)*(b-t) > MAX_CROP_PIXELS):
            raise ValueError('Native crop exceeds bounded source window limits')
        gray = source_windows(self.path, self.sha256, list(self.size), [(l, t, r, b)])[0]
        return Image.fromarray(gray)


class NativePNGImageAPI:
    """Use as a module-local Image binding inside a dedicated CLI child only."""
    def __init__(self, path, sha256, *, allow_crop=False):
        self.path = Path(path).resolve()
        header = inspect_png(self.path)
        if not header['streaming_format_supported']:
            raise ValueError('Native streaming access requires an 8-bit RGB non-interlaced PNG')
        self.original = _Original(self.path, sha256, header['size_px'], allow_crop)

    def open(self, path, *args, **kwargs):
        if not args and not kwargs and isinstance(path, (str, Path)) and Path(path).resolve() == self.path:
            return nullcontext(self.original)
        return Image.open(path, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(Image, name)
