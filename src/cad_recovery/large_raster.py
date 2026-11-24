"""Bounded preparation of large RGB PNGs without changing the input raster.

Full-resolution tiles preserve samples; reduced overviews serve navigation and
global analysis. Coordinates are pixel centers, never inferred millimeters.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .png_stream import PNGStripReader

MAX_TILES = 2048
MAX_OUTPUT_RAW_BYTES = 2 * 1024**3
MAX_WORKING_BUFFER_BYTES = 256 * 1024**2


def _integer(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer between {lower} and {upper}")
    return value


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _starts(length, tile_size, overlap):
    # Skip a redundant final tile containing only the preceding overlap.
    return list(range(0, max(1, length - overlap), tile_size - overlap))


def _matrix(scale=1., x=0., y=0.):
    return [[scale, 0., x], [0., scale, y], [0., 0., 1.]]


def _save_png(path, pixels):
    with path.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG", compress_level=3)
    return {"path": path.name, "sha256": _digest(path), "size_bytes": path.stat().st_size,
            "size_px": [int(pixels.shape[1]), int(pixels.shape[0])]}


class _Overview:
    """Exact integer area average and conservative minimum, one row at a time."""

    def __init__(self, width, height, factor):
        self.width, self.height, self.factor = width, height, factor
        self.size = (math.ceil(width / factor), math.ceil(height / factor))
        self.rgb = np.empty((self.size[1], self.size[0], 3), dtype=np.uint8)
        self.ink = np.empty((self.size[1], self.size[0]), dtype=np.uint8)
        self.starts = np.arange(0, width, factor)
        self.sums = np.zeros((self.size[0], 3), dtype=np.uint32)
        self.minimum = np.full(self.size[0], 255, dtype=np.uint8)
        self.rows = 0
        self.output_row = 0

    def add(self, pixels):
        for row in pixels:
            # Cast before summing: an overview block may contain many samples.
            sums = np.add.reduceat(row.astype(np.uint32), self.starts, axis=0, dtype=np.uint32)
            sums[-1] += (self.size[0] * self.factor - self.width) * 255
            self.sums += sums
            self.minimum = np.minimum(self.minimum, np.minimum.reduceat(row.min(axis=1), self.starts))
            self.rows += 1
            if self.rows % self.factor == 0:
                self._finish_row()

    def _finish_row(self):
        denominator = self.factor**2
        self.rgb[self.output_row] = (self.sums + denominator // 2) // denominator
        self.ink[self.output_row] = self.minimum
        self.output_row += 1
        self.sums.fill(0)
        self.minimum.fill(255)

    def finish(self):
        if self.rows != self.height:
            raise ValueError("Overview did not receive the declared source rows")
        if self.rows % self.factor:
            self.sums += (self.factor - self.rows % self.factor) * self.factor * 255
            self._finish_row()
        if self.output_row != self.size[1]:
            raise ValueError("Overview row count is inconsistent")


def prepare_large_png(source: Path, output_dir: Path, *, expected_sha256=None,
                      tile_size=2048, overlap=128, overview_max_side=2400,
                      strip_rows=128) -> dict:
    """Prepare one RGB8 noninterlaced PNG into a new directory.

    A ``prepared`` report is written only after complete PNG integrity checks.
    If decoding fails, provisional tiles remain with a failed report and cannot
    be mistaken for a complete page. The original and decoder limits are intact.
    """
    started = time.perf_counter()
    source, output_dir = Path(source), Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Large-raster output already exists; choose a new directory")
    tile_size = _integer(tile_size, "tile_size", 128, 2048)
    overlap = _integer(overlap, "overlap", 0, tile_size // 2)
    overview_max_side = _integer(overview_max_side, "overview_max_side", 128, 4096)
    strip_rows = _integer(strip_rows, "strip_rows", 1, 256)
    reader = PNGStripReader(source, expected_sha256=expected_sha256, strip_rows=strip_rows)
    width, height = reader.width, reader.height
    factor = max(1, math.ceil(max(width, height) / overview_max_side))
    ow, oh = math.ceil(width / factor), math.ceil(height / factor)
    xs, ys = _starts(width, tile_size, overlap), _starts(height, tile_size, overlap)
    count = len(xs) * len(ys)
    raw_bytes = sum(min(tile_size, width - x) for x in xs) * sum(min(tile_size, height - y) for y in ys) * 3 + ow * oh * 4
    # Explicit upper allowance for the row band, decoder strip copies, pooling,
    # tile compression copies and overlap move. This is an allocation estimate,
    # not a measurement or a claim about total interpreter process RSS.
    working_bytes = width * 3 * (min(height, tile_size) + 8 * strip_rows + 2 * overlap) + ow * oh * 4 + tile_size**2 * 9 + width * 64
    if count > MAX_TILES:
        raise ValueError(f"Preparation would exceed the {MAX_TILES} tile limit")
    if raw_bytes > MAX_OUTPUT_RAW_BYTES:
        raise ValueError("Preparation would exceed the 2 GiB decoded output budget")
    if working_bytes > MAX_WORKING_BUFFER_BYTES:
        raise ValueError("Preparation exceeds the 256 MiB working-buffer budget; reduce tile_size or strip_rows")

    output_dir.mkdir(parents=True, exist_ok=False)
    tiles = []
    try:
        overview = _Overview(width, height, factor)
        band = np.empty((min(tile_size, height), width, 3), dtype=np.uint8)
        band_index, filled, received = 0, 0, 0

        def write_band():
            y = ys[band_index]
            band_height = min(tile_size, height - y)
            if filled != band_height:
                raise ValueError("Incomplete tile band")
            for x in xs:
                pixels = band[:band_height, x:min(x + tile_size, width)]
                identifier = f"tile-{len(tiles) + 1:04d}"
                record = _save_png(output_dir / f"{identifier}.png", pixels)
                record.update(id=identifier, source_box_px=[x, y, min(x + tile_size, width), y + band_height],
                              source_to_tile=_matrix(x=-x, y=-y), tile_to_source=_matrix(x=x, y=y),
                              pixel_encoding="RGB8", source_samples_preserved=True)
                tiles.append(record)

        for y, pixels in reader.iter_strips():
            if y != received or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[1:] != (width, 3):
                raise ValueError("PNG reader returned inconsistent strip coordinates or samples")
            overview.add(pixels)
            received += len(pixels)
            offset = 0
            while offset < len(pixels):
                if band_index >= len(ys):
                    raise ValueError("PNG reader returned excess image rows")
                needed = min(tile_size, height - ys[band_index]) - filled
                take = min(needed, len(pixels) - offset)
                band[filled:filled + take] = pixels[offset:offset + take]
                filled += take
                offset += take
                if filled == min(tile_size, height - ys[band_index]):
                    write_band()
                    band_index += 1
                    if band_index < len(ys):
                        if overlap:
                            band[:overlap] = band[filled - overlap:filled].copy()
                        filled = overlap
                    else:
                        filled = 0
        if not reader.validated or received != height or band_index != len(ys) or len(tiles) != count:
            raise ValueError("PNG stream or tile coverage was not completely validated")
        overview.finish()
        overview_record = _save_png(output_dir / "overview.png", overview.rgb)
        inverse = _matrix(factor, (factor - 1) / 2, (factor - 1) / 2)
        forward = _matrix(1 / factor, (1 / factor - 1) / 2, (1 / factor - 1) / 2)
        overview_record.update(scale_factor=factor, source_to_overview=forward, overview_to_source=inverse,
                               method="integer_area_mean_RGB_white_padding_round_half_up")
        ink_record = _save_png(output_dir / "overview-ink.png", overview.ink)
        ink_record.update(scale_factor=factor, source_to_overview=forward, overview_to_source=inverse,
                          method="minimum_of_RGB_channels_and_block_samples", role="ink_presence_only_not_lossless_geometry")
        report = {
            "schema_version": "1.0", "status": "prepared", "source": source.name,
            "source_sha256": reader.source_sha256, "source_size_px": [width, height],
            "source_integrity_validated": True, "source_modified": False,
            "physical_scale_confirmed": False, "coordinate_system": "pixel centers; top-left origin; x right, y down",
            "source_box_convention": "half-open integer pixel bounds [left, top, right, bottom]",
            "overview": overview_record, "ink_overview": ink_record, "tiles": tiles,
            "tile_count": len(tiles), "coverage": "all original pixels; overlapping tiles preserve RGB samples",
            "overview_padding_px": {"right": ow * factor - width, "bottom": oh * factor - height, "value": [255, 255, 255]},
            "parameters": {"tile_size": tile_size, "overlap": overlap, "overview_max_side": overview_max_side,
                           "strip_rows": strip_rows, "max_tiles": MAX_TILES,
                           "max_output_raw_bytes": MAX_OUTPUT_RAW_BYTES,
                           "max_working_buffer_bytes": MAX_WORKING_BUFFER_BYTES},
            "resource_estimates": {"working_buffer_bytes": working_bytes, "decoded_output_bytes": raw_bytes,
                                   "total_process_memory_measured": False},
            "decoder": reader.metadata,
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "limitations": ["Only RGB8 noninterlaced PNG is supported by this preparation path.",
                            "A reduced overview can merge nearby strokes; use full-resolution tiles for labels and detail.",
                            "Tile-local pipeline coordinates require composition with the recorded source transform.",
                            "Preparation does not recognize dimensions, verify pair correspondence or permit model training."],
        }
        with (output_dir / "preparation.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        return report
    except Exception as error:
        failure = {"schema_version": "1.0", "status": "failed", "source": source.name,
                   "source_sha256": reader.source_sha256, "source_integrity_validated": False,
                   "error": {"type": type(error).__name__, "message": str(error)},
                   "provisional_tile_count": len(tiles),
                   "limitations": ["Any existing image artifacts in this directory are provisional and must not be used as a complete prepared page."]}
        with (output_dir / "preparation.json").open("x", encoding="utf-8") as stream:
            json.dump(failure, stream, ensure_ascii=False, indent=2, allow_nan=False)
        raise
